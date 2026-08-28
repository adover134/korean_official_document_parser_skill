"""`classify_and_merge()`의 자동 배치 축소/재시도, 그리고 그 신호가 되는
`llm_backend._parse_rate_limit_error()`의 회귀 테스트.

Groq 실측(2026-08-27, README "LLM 백엔드 검증 메모" 참고)에서 관찰한 두 가지 실패 유형을
그대로 재현한다:
  - `is_size_limit=True` — 이 요청 자체가 한 번에 한도를 넘음. 제공자가 응답에 실어 보낸
    정확한 `Limit`/`Requested` 값으로 배치를 줄여 즉시 재시도해야 한다.
  - `is_size_limit=False` — 최근 누적 사용량으로 분당 한도 자체가 소진됨. 배치를 줄여도
    소용없고 `retry_after`만큼 기다린 뒤 같은 크기로 재시도해야 한다.

`TestParseRateLimitError`는 이 둘을 HTTP 상태 코드가 아니라 파싱한 실제 수치로 구분해야
하는 이유를 검증한다 — Groq는 전자를 413으로 보내지만, OpenAI는 같은 상황도 429로 보낸다
(웹 검색으로 확인, 2026-08-27). 나머지는 전부 `classify()`를 목(mock)으로 대체해서 LLM
호출 없이 즉시 실행된다(429 재시도 대기도 `retry_after`를 짧게 줘서 실제로 기다리지 않는다)."""

from __future__ import annotations

from unittest import mock

from classify_headings_pass1 import classify_and_merge
from llm_backend import RateLimitExceeded, _parse_rate_limit_error


class TestParseRateLimitError:
    def test_groq_413_size_limit_shrinks(self):
        """Groq 실측 응답(413) — 요청 자체가 한도를 넘었으므로 축소해야 함."""
        body = (
            '{"error":{"message":"Request too large for model `openai/gpt-oss-20b` on tokens per '
            'minute (TPM): Limit 8000, Requested 14483, please reduce your message size.",'
            '"type":"tokens","code":"rate_limit_exceeded"}}'
        )
        err = _parse_rate_limit_error(413, body, {"retry-after": "95"})
        assert err.is_size_limit is True
        assert err.limit_tokens == 8000
        assert err.requested_tokens == 14483
        assert err.retry_after == 95.0

    def test_openai_style_429_same_size_situation_still_shrinks(self):
        """OpenAI는 Groq의 413과 같은 상황(요청 자체가 너무 큼)도 429로 보낸다 — 상태 코드가
        아니라 파싱한 Limit/Requested로 판단해야 이 경우도 올바르게 축소된다."""
        body = (
            '{"error":{"message":"Request too large for gpt-4.1 in organization org-x on tokens '
            'per min (TPM): Limit 30000, Requested 42638.","type":"tokens",'
            '"code":"rate_limit_exceeded"}}'
        )
        err = _parse_rate_limit_error(429, body, {"x-ratelimit-limit-tokens": "30000"})
        assert err.is_size_limit is True, "429여도 requested>limit이면 축소해야 함"

    def test_openai_style_429_genuine_window_exhaustion_waits(self):
        """진짜 분당 누적 한도 소진(이번 요청 자체는 한도보다 작음) — 축소해도 소용없으므로
        대기만 해야 함."""
        body = (
            '{"error":{"message":"Rate limit reached for gpt-4o on tokens per min. Limit: 30000, '
            'Used: 29500, Requested: 800.","type":"tokens","code":"rate_limit_exceeded"}}'
        )
        err = _parse_rate_limit_error(429, body, {})
        assert err.is_size_limit is False, "requested(800) < limit(30000)이면 배치를 줄일 필요 없음"

    def test_non_rate_limit_413_returns_none(self):
        """토큰 한도 초과가 아닌 다른 413/429는 None을 반환해 호출부가 일반 오류로 처리하게 한다."""
        body = '{"error": {"message": "invalid request", "code": "invalid_request"}}'
        assert _parse_rate_limit_error(413, body, {}) is None

    def test_non_413_429_status_returns_none(self):
        assert _parse_rate_limit_error(500, "{}", {}) is None


def _candidates(n: int) -> list[dict]:
    return [{"line": i, "text": f"{i}. 항목", "type": "numbered", "in_box": False} for i in range(1, n + 1)]


def _ok(cands):
    return [{"line": c["line"], "classification": "main_section", "reason": "test"} for c in cands]


def test_size_limit_error_shrinks_using_reported_limit_and_requested():
    """413 + 제공자가 알려준 정확한 Limit/Requested 값 -> 그 비율로 배치를 줄여 재시도."""
    calls: list[int] = []

    def fake_classify(cands, backend):
        calls.append(len(cands))
        if len(cands) > 8:
            requested = len(cands) * 10  # 이 목에서는 후보 1개당 10토큰으로 가정
            raise RateLimitExceeded(
                f"토큰 한도 초과: Limit 100, Requested {requested}",
                is_size_limit=True, limit_tokens=100, requested_tokens=requested,
            )
        return _ok(cands)

    with mock.patch("classify_headings_pass1.classify", fake_classify):
        result = classify_and_merge(_candidates(20), backend=object(), max_candidates_per_call=20)

    assert len(result) == 20
    assert calls[0] == 20, "첫 시도는 요청한 크기 그대로여야 함"
    assert all(c <= 8 for c in calls[1:]), f"축소 후에는 8 이하로만 재시도해야 함: {calls}"


def test_size_limit_error_without_reported_numbers_falls_back_to_halving():
    """Limit/Requested를 못 읽으면(다른 제공자가 문구를 다르게 쓰는 경우) 절반으로 폴백."""
    calls: list[int] = []

    def fake_classify(cands, backend):
        calls.append(len(cands))
        if len(cands) > 5:
            raise RateLimitExceeded("토큰 한도 초과", is_size_limit=True)
        return _ok(cands)

    with mock.patch("classify_headings_pass1.classify", fake_classify):
        result = classify_and_merge(_candidates(10), backend=object(), max_candidates_per_call=10)

    assert len(result) == 10
    assert calls == [10, 5, 5], f"10 -> 절반(5)으로 축소 후 두 배치(5,5)로 재시도해야 함: {calls}"


def test_rolling_window_exhausted_retries_same_size_after_wait(monkeypatch):
    """429(배치 크기와 무관) -> 크기를 그대로 두고 retry_after만큼 대기 후 재시도."""
    sleeps: list[float] = []
    monkeypatch.setattr("classify_headings_pass1.time.sleep", sleeps.append)

    attempts = {"n": 0}

    def fake_classify(cands, backend):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RateLimitExceeded("분당 한도 소진", is_size_limit=False, retry_after=12.5)
        return _ok(cands)

    with mock.patch("classify_headings_pass1.classify", fake_classify):
        result = classify_and_merge(_candidates(5), backend=object(), max_candidates_per_call=5)

    assert len(result) == 5
    assert attempts["n"] == 2
    assert sleeps == [12.5]


def test_persistent_rate_limit_eventually_raises(monkeypatch):
    """계속 실패하면 무한 재시도하지 않고 재시도 상한을 넘기면 예외를 전파한다."""
    monkeypatch.setattr("classify_headings_pass1.time.sleep", lambda *_: None)

    def always_fails(cands, backend):
        raise RateLimitExceeded("계속 실패", is_size_limit=False, retry_after=0.0)

    with mock.patch("classify_headings_pass1.classify", always_fails):
        try:
            classify_and_merge(_candidates(5), backend=object(), max_candidates_per_call=5)
            assert False, "RateLimitExceeded가 전파돼야 함"
        except RateLimitExceeded:
            pass
