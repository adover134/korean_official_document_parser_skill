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
import time
import urllib.error
import urllib.request


class LLMBackend:
    """system/user 프롬프트로 LLM을 호출해 JSON 객체 하나를 받아오는 계약."""

    def chat_json(self, system_prompt: str, user_prompt: str, *, max_tokens: int, seed: int | None = 42) -> dict:
        raise NotImplementedError


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
            raise RuntimeError(f"{self.base_url} 호출 실패 ({e.code}): {detail[:500]}") from e
        dt = time.time() - t0

        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        print(f"완료 ({dt:.1f}s) prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}")
        return json.loads(content)
