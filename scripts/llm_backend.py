"""Pass1b(헤더 후보 계층 판단)가 호출하는 LLM을 백엔드 추상화 뒤로 감싼다.

배경: 원래 `classify_headings_pass1.py`의 `classify()`가 Ollama의 `/api/chat` 스키마(응답이
`{"message": {"content": ...}}`, Ollama 전용 `options.num_ctx`/`think` 필드 등)에 직접 강결합돼
있었다. 로컬 GPU/Ollama 없이도 이 파이프라인을 쓸 수 있게 하려면(예: 서비스로 배포할 때 서버에
GPU가 없는 경우) OpenAI 호환(`/chat/completions`, `{"choices":[{"message":{...}}]}`) API도 같은
`classify()` 로직으로 호출할 수 있어야 한다.

Ollama와 OpenAI 호환 API의 실제 차이:
  - 엔드포인트/응답 스키마가 다름(`/api/chat` vs `/chat/completions`)
  - Ollama는 컨텍스트 창 크기(`options.num_ctx`)를 요청자가 직접 지정해야 하고 지정 안 하면 조용히
    작은 기본값(~4096)으로 떨어지는 문제가 실제로 있었다(README 참고) — OpenAI 호환 API는 이런
    설정이 없다(서버가 알아서 처리).
  - JSON 강제 출력 방식이 다름: Ollama는 최상위 `"format": "json"`, OpenAI 호환은
    `response_format={"type": "json_object"}`(단, 모든 제공자가 지원하는 건 아님 — 지원 안 하면
    프롬프트의 "반드시 JSON만 출력" 지시에만 의존).

두 경우 모두 "system/user 프롬프트를 보내고 JSON 객체 하나를 받는다"는 동일한 계약이므로, 그 계약을
`LLMBackend.chat_json()`으로 추상화하고 각 백엔드가 자기 스키마로 변환한다."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request


class LLMBackend:
    """system/user 프롬프트로 LLM을 호출해 JSON 객체 하나를 받아오는 계약."""

    def chat_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int, seed: int | None = 42) -> dict:
        raise NotImplementedError


class RateLimitExceeded(RuntimeError):
    """제공자(OpenAI 호환 API)가 실제로 돌려준 TPM(분당 토큰) 한도 초과 응답 — 일반 `RuntimeError`와
    구분해서, 호출부(`classify_and_merge()`)가 이 정보를 보고 배치 크기를 자동으로 줄여 재시도할 수
    있게 한다(2026-08-27, "실제 연결한 모델의 API"가 알려주는 실측 한도로 자동 조절 — 제공자별로
    미리 값을 추측/하드코딩하는 대신, 매 실패에서 그 제공자가 직접 알려주는 정확한 수치를 그대로 쓴다).

    `is_size_limit=True`면 이 요청 자체가 한 번에 한도를 넘은 것(배치를 줄여 즉시 재시도하면 됨)
    — `False`면 최근 누적 사용량으로 분당 한도 자체가 소진된 것(배치를 줄여도 소용없고
    `retry_after`만큼 기다려야 함). HTTP 상태 코드(413/429)가 아니라 응답에서 파싱한 실제
    limit/requested 수치로 판단한다 — 제공자마다 상태 코드 관례가 다르다(`_parse_rate_limit_error()`
    docstring 참고: Groq는 413/429로 나누지만 OpenAI는 같은 상황도 전부 429로 보낸다)."""

    def __init__(
        self,
        message: str,
        *,
        is_size_limit: bool,
        limit_tokens: int | None = None,
        requested_tokens: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.is_size_limit = is_size_limit
        self.limit_tokens = limit_tokens
        self.requested_tokens = requested_tokens
        self.retry_after = retry_after


_LIMIT_RE = re.compile(r"[Ll]imit[:\s]+(\d+)")
_REQUESTED_RE = re.compile(r"[Rr]equested[:\s]+(\d+)")


def _parse_rate_limit_error(code: int, body_text: str, headers) -> "RateLimitExceeded | None":
    """HTTPError 본문/헤더가 실제로 (TPM) 토큰 한도 초과인지 판별해서 `RateLimitExceeded`로
    구조화한다 — 아니면 None(호출부가 기존처럼 일반 `RuntimeError`로 처리).

    Groq 실측(2026-08-27) 기준 본문 형태: `{"error": {"code": "rate_limit_exceeded", "message":
    "... Limit 8000, Requested 14483 ..."}}`, HTTP 413. 메시지의 "Limit N"/"Requested N"이 있으면
    그 값으로 정확한 축소 비율을 계산하고, 없으면(다른 제공자가 문구를 다르게 쓰는 경우) 폴백으로
    절반씩 줄이게 `limit_tokens`/`requested_tokens`를 None으로 둔다.

    `is_size_limit`(배치를 줄여야 하는지, 아니면 시간만 기다리면 되는지)은 HTTP 상태 코드가 아니라
    파싱한 실제 수치로 판단한다 — Groq는 "요청 자체가 너무 큼"을 413으로 구분해 보내지만, OpenAI는
    같은 상황도 429로 보낸다(웹 검색 확인, 2026-08-27: "Request too large for gpt-4.1... on tokens
    per min (TPM): Limit 30000, Requested 42638"도 429). 상태 코드로 나누면 OpenAI에서는 이
    경우를 "그냥 기다리면 되는 문제"로 오판해 배치를 안 줄이고 똑같은 크기로 계속 재시도하다
    실패한다 — `requested_tokens > limit_tokens`를 직접 확인하는 쪽이 제공자 무관하게 안전하다.
    두 수치를 못 읽으면(다른 제공자가 문구를 또 다르게 쓰는 경우) 축소를 기본값으로 삼는다 —
    실제로는 시간창 문제였을 뿐이면 API 호출이 몇 번 더 느는 정도지만, 반대로 오판하면(진짜
    크기 문제인데 안 줄이면) 재시도 상한까지 같은 실패를 반복하게 되므로 축소 쪽이 더 안전한
    기본값이다."""
    if code not in (413, 429):
        return None
    try:
        err = json.loads(body_text).get("error", {})
    except (json.JSONDecodeError, AttributeError):
        err = {}
    msg = str(err.get("message", ""))
    err_code = str(err.get("code", ""))
    has_rate_limit_headers = any(h.lower().startswith("x-ratelimit-") for h in headers.keys())
    if err_code != "rate_limit_exceeded" and not has_rate_limit_headers:
        return None  # 토큰 한도 초과가 아닌 다른 413/429(예: 요청 자체가 유효하지 않음)

    limit_m = _LIMIT_RE.search(msg)
    req_m = _REQUESTED_RE.search(msg)
    limit_tokens = int(limit_m.group(1)) if limit_m else None
    requested_tokens = int(req_m.group(1)) if req_m else None
    is_size_limit = requested_tokens > limit_tokens if (limit_tokens and requested_tokens) else True

    retry_after_raw = headers.get("retry-after")
    return RateLimitExceeded(
        f"토큰 한도 초과 ({code}): {msg or body_text[:300]}",
        is_size_limit=is_size_limit,
        limit_tokens=limit_tokens,
        requested_tokens=requested_tokens,
        retry_after=float(retry_after_raw) if retry_after_raw else None,
    )


class OllamaBackend(LLMBackend):
    """기존 `classify()`가 쓰던 Ollama `/api/chat` 호출 로직을 그대로 옮김 — 동작 100% 동일."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def chat_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int, seed: int | None = 42) -> dict:
        # num_ctx도 max_tokens(응답 예산)에 비례해서 늘린다 — 8GB VRAM 환경에서 고정 num_ctx로는
        # 후보가 많은 문서의 프롬프트가 조용히 잘리는 문제가 있었음(README 참고).
        num_ctx = min(32768, max(8192, max_tokens + 2000))
        req = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0, "seed": seed, "num_predict": max_tokens, "num_ctx": num_ctx},
        }
        data = json.dumps(req).encode("utf-8")
        r = urllib.request.Request(
            f"{self.host}/api/chat", data=data, headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        with urllib.request.urlopen(r, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0

        content = body["message"]["content"]
        print(
            f"완료 ({dt:.1f}s) prompt_eval={body.get('prompt_eval_count')} "
            f"eval={body.get('eval_count')}"
        )
        return json.loads(content)


class OpenAICompatBackend(LLMBackend):
    """OpenAI `/chat/completions` 스키마를 쓰는 아무 제공자나 호출한다 — OpenAI 자체는 물론,
    같은 스키마를 쓰는 Groq/OpenRouter 등도 base_url만 바꾸면 그대로 동작한다(2026-08-23,
    Groq 무료 티어로 실제 검증). Gemini는 OpenAI 호환 레이어(`/v1beta/openai/`)를 통해서만
    호환되며, `response_format`(JSON 강제) 지원 여부는 모델마다 달라 별도 검증이 필요하다."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int, seed: int | None = 42) -> dict:
        req = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            req["seed"] = seed
        data = json.dumps(req).encode("utf-8")
        r = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Cloudflare가 urllib의 기본 User-Agent("Python-urllib/x.y")를 차단한다(Groq에서
                # 실측 — 유효한 키를 줘도 403 "error code: 1010"만 나옴). 일반적인 User-Agent로
                # 바꾸면 통과한다.
                "User-Agent": "hwp-hierarchical-md/0.1",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(r, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            rate_limit_error = _parse_rate_limit_error(e.code, detail, e.headers)
            if rate_limit_error is not None:
                raise rate_limit_error from e
            raise RuntimeError(f"{self.base_url} 호출 실패 ({e.code}): {detail[:500]}") from e
        dt = time.time() - t0

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        print(f"완료 ({dt:.1f}s) prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}")
        return json.loads(content)
