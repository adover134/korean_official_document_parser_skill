"""Pass 1 — 제목 후보를 LLM으로 분류 (계층 판단은 여기서, 규칙이 아니라 LLM이 한다).

extract_heading_candidates.py가 뽑은 후보 목록(JSON)을 통째로 LLM에 넘겨서, 각 후보가:
  - main_section: 본문의 진짜 최상위 섹션 (문서의 주된 1..N 번호 계열)
  - sub_section: main_section 밑에 중첩된 하위 제목 (경고 박스, 본문 속 번호 체크리스트가 실은 제목 역할을 하는 경우 등)
  - attachment_section: 붙임/별첨처럼 별도 첨부 문서를 여는 제목 (번호가 새로 시작해도 정상)
  - not_heading: 오탐 — 서명란, 날짜, 본문 문장 등. 제목으로 승격하면 안 됨
인지 판단하게 한다.

핵심 설계: "번호가 이어지는지"로 계층을 정하지 않는다(그 방식은 실패했었음 — README 참고).
후보 전체를 한 번에 보여줘서 LLM이 문맥/의미로 판단하게 한다. 입력이 후보 목록(작음)이라
문서가 아무리 길어도 이 호출의 비용은 거의 안 늘어난다 — 이게 2-패스 설계의 핵심 이점.

사용법:
    python classify_headings_pass1.py <candidates.json> [-o <output.json>] [--model qwen3.5:9b]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from extract_heading_candidates import _match_bracket_marker

SYSTEM_PROMPT = """당신은 한국 공공기관 입찰공고문(RFP)의 구조를 분석하는 전문가입니다.

아래는 문서에서 "제목일 수 있는 줄"을 규칙 기반으로 뽑아낸 후보 목록입니다(순서대로).
이 후보들 중 실제로 헤더(제목)인 것과 아닌 것을 구분하고, 헤더라면 어느 계층인지 분류하세요.

## 중요: 번호를 그대로 믿지 마세요
- 이 문서들은 기관마다, 심지어 한 문서 안에서도 번호 매김 방식이 다릅니다.
- 어떤 문서는 최상위 섹션도 "1. 2. 3."을 쓰고, 그 밑의 체크리스트/하위 항목도 "1. 2. 3."을 씁니다.
  번호가 1로 리셋됐다고 무조건 하위는 아니고, 번호가 이어진다고 무조건 최상위도 아닙니다.
- "붙임", "별첨" 같은 첨부 문서는 번호가 새로 1부터 시작하는 게 정상입니다 — 이건 오류가 아니라
  최상위(또는 그에 준하는) 제목으로 봐야 합니다.
- 번호보다 **문장의 성격**을 보세요: 최상위 섹션 제목은 짧은 명사구입니다(예: "입찰참가자격",
  "입찰의 무효"). 반면 본문 문장은 길고 서술형입니다(예: "제안서는 e-발주시스템을 통해 제출하여야
  하며..." 처럼 주어-서술어를 갖춘 완전한 문장이면 대개 본문이지 제목이 아닙니다).
- 서명란(예: "OO 대표이사", "OO 산학협력단 계약관 귀하")은 헤더가 아닙니다 — 문서 끝의 서명/직인
  자리입니다.
- 같은 텍스트가 문서 안에 두 번 나오면(예: 표제가 서두와 본문에 중복) 그중 실제 섹션 목록이 시작되는
  위치의 것만 진짜 제목으로 보고 나머지는 not_heading으로 판단해도 됩니다.

## 분류
- main_section: 본문의 진짜 최상위 섹션 (문서의 주된 번호 계열, 예: 1.입찰개요 2.입찰방법 ... 10.기타사항)
- sub_section: main_section 밑에 중첩된 하위 제목 (경고/안내 박스, 체크리스트가 실질적으로 소제목 역할을 하는 경우)
- attachment_section: 붙임/별첨 등 별도 첨부 문서를 여는 제목
- not_heading: 오탐 — 서명란, 중복된 표제, 날짜, 완전한 문장 형태의 본문 등

## 출력 형식
반드시 JSON만 출력하세요. 다음 형식의 객체 하나:
{"classifications": [{"line": <원본 줄번호, 정수만, "L" 접두사 붙이지 말 것>, "classification": "<위 4가지 중 하나>", "reason": "<한 문장 이유>"}, ...]}
입력받은 후보 전부에 대해 빠짐없이 하나씩 판단하세요. line 값은 아래 후보 목록에 표시된 "L숫자"에서 숫자만 정수로 쓰세요(예: "L96" -> 96)."""


def build_user_prompt(candidates: list[dict]) -> str:
    lines = ["다음은 한 문서에서 추출한 제목 후보 목록입니다 (문서 순서대로):\n"]
    for c in candidates:
        num = f" 번호={c['number']}" if c.get("number") is not None else ""
        lvl = f" (kordoc 감지 레벨={c['level']})" if c.get("level") else ""
        lines.append(
            f"- L{c['line']} [{c['type']}{num}{lvl}] 텍스트: {c['text'][:120]!r}\n"
            f"  다음 내용: {c.get('context', '')[:100]!r}"
        )
    return "\n".join(lines)


def classify(candidates: list[dict], model: str, host: str = "http://localhost:11434") -> list[dict]:
    # 후보 목록이 "작다"는 설계 전제(모듈 docstring 참고)는 후보가 많은 대형 문서(체크리스트/표가
    # 많은 문서)에서 깨진다 — 고정 num_predict=4000으로는 후보 92개짜리 문서에서 JSON 응답이
    # 중간에 잘려(Unterminated string) 파싱이 실패하는 걸 실제로 확인함(후보 92개짜리 샘플 문서
    # 3건). 후보 개수에 비례해 예산을 늘린다 — 후보가 적은 기존 샘플들(10~22개)에는
    # 영향 없음(원래도 예산 안에서 끝났으므로).
    n = len(candidates)
    num_predict = min(16000, max(4000, n * 120 + 500))
    num_ctx = min(32768, max(8192, num_predict + n * 60 + 2000))
    req = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(candidates)},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0, "seed": 42, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    data = json.dumps(req).encode("utf-8")
    r = urllib.request.Request(f"{host}/api/chat", data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0

    content = body["message"]["content"]
    parsed = json.loads(content)
    result = parsed.get("classifications", [])
    print(
        f"완료 ({dt:.1f}s) prompt_eval={body.get('prompt_eval_count')} "
        f"eval={body.get('eval_count')} classifications={len(result)}/{len(candidates)}"
    )
    return result


def _line_key(v) -> int | None:
    """LLM이 'L123'처럼 접두사를 붙여 돌려줘도 정수로 정규화."""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        digits = "".join(ch for ch in v if ch.isdigit())
        return int(digits) if digits else None
    return None


# 8GB VRAM 환경에서 후보 92개(num_ctx≈16,500)는 성공했지만 103개(num_ctx≈21,000)부터는 Ollama가
# 요청한 num_ctx를 VRAM 부족으로 조용히 못 지키고 자체 기본값(~4096)으로 낮춰버려 프롬프트가 잘리고
# LLM이 사실상 빈 응답(classifications=0/103)을 내는 걸 실제로 확인함(후보 100개 이상인 샘플
# 문서 2건 — 둘 다 prompt_eval이 정확히 4098로 일치, VRAM 기본 컨텍스트에
# 눌린 증거). num_ctx를 더 키우는 방향으로는 VRAM 상한을 못 피하므로, 후보 목록 자체를 여유 있게
# (92개보다 확실히 작게) 청크로 나눠 호출을 여러 번 하는 쪽으로 우회한다.
_MAX_CANDIDATES_PER_CALL = 70


_NUMBER_PREFIX_RE = re.compile(r"^(\d+)[.,]")


def _fill_main_section_number_gaps(merged: list[dict]) -> list[dict]:
    """LLM이 이미 main_section으로 확정한 두 섹션 사이에 번호가 비어 있으면, 그 **좁은 구간**
    (두 확정 섹션의 줄 범위 사이) 안에서 번호가 일치하는 후보를 찾아 main_section으로 승격한다.

    이건 "번호가 연속되면 최상위"라는, 예전에 실패했던 전역 규칙(README 참고 — 붙임에서 번호가
    정당하게 리셋되는 걸 오탐하고 한 번 어긋나면 복구 불가)과는 다르다: 번호를 전역적으로 신뢰하는
    게 아니라, 이미 LLM이 독립적으로 확정한 두 앵커 사이의 좁은 구간에서만, 그것도 "그 사이에 번호가
    비어 있다"는 강한 신호가 있을 때만 작동한다. "붙임" 내부에서 번호가 재시작되는 정상 케이스는
    애초에 main_section 시퀀스 자체가 아니므로 이 로직이 건드리지 않는다.

    실제 배경(2026-08-21): "N. 라벨: 값" 형식(예: "4. 견적서 제출기간: 2026.8.10~8.13")의 섹션이
    LLM에 의해 간헐적으로 not_heading으로 오분류되는 사례가 여러 문서에서 확인됐다. 프롬프트에
    "라벨:값도 헤더일 수 있다"는 가이드를 추가해봤으나 다른 정상 사례에서 새로운 오분류를 만들어내는
    회귀가 발생해 되돌렸고(검증된 프롬프트 유지), 대신 이 결정론적 갭 메우기로 대체함."""
    numbered_mains = []
    for i, c in enumerate(merged):
        if c["classification"] == "main_section":
            m = _NUMBER_PREFIX_RE.match(c["text"].strip())
            if m:
                numbered_mains.append((i, int(m.group(1)), c["line"]))

    for i in range(len(numbered_mains) - 1):
        _, num_a, line_a = numbered_mains[i]
        _, num_b, line_b = numbered_mains[i + 1]
        if num_b - num_a <= 1:
            continue  # 이미 연속 (또는 역순 -> 다른 번호 계열이므로 손대지 않음)
        for expected in range(num_a + 1, num_b):
            for c in merged:
                if c["classification"] == "main_section" or c.get("in_box") or not (line_a < c["line"] < line_b):
                    continue
                m = _NUMBER_PREFIX_RE.match(c["text"].strip())
                if m and int(m.group(1)) == expected:
                    c["classification"] = "main_section"
                    c["reason"] = f"결정론적 후처리: main_section {num_a}번-{num_b}번 사이 번호 공백을 메움"
                    break
    return merged


def _cap_bracket_marker_scopes(merged: list[dict]) -> list[dict]:
    """LLM이 실제로 main_section/attachment_section으로 확정한 bracket-marker 후보(예: "【서식 7】
    이행능력심사...", "[붙임2] 제출 서류")는 그 자체로 서식/붙임 같은 "첨부 스코프"를 여는 앵커로
    본다. 그 앵커부터 다음 앵커(또는 문서 끝) 사이 구간 안에서 main_section/attachment_section으로
    분류된 다른 후보는 전부 sub_section으로 강등한다.

    배경(한 샘플 문서): "【서식 7】 이행능력심사 자기평가 및 심사표"가
    진짜 헤더 앵커인데, 그 직후 kordoc이 중복으로 남긴 자체 "## 이행능력심사 자기평가 및 심사표"
    헤더, 그리고 서식 내부의 평번한 볼드 소라벨("3. 회사를 대표하는 연락책임자 인적사항", "4. 종합평가"
    등)이 문맥 없이 번호만 보고 main_section/attachment_section으로 오분류되어 번호가 중복/역행하는
    문제가 있었다. in_box 규칙(원본 텍스트박스 안 콘텐츠는 항상 하위)과 같은 성격의 문제 —
    "서식/붙임 스코프 안 콘텐츠는 항상 하위"라는, 이미 LLM이 확정한 앵커 사이의 좁은 구간에서만
    작동하는 결정론적 안전망이다. "서식"/"붙임" 같은 특정 단어를 하드코딩하지 않고, bracket-marker
    타입(extract_heading_candidates.py 참고 — 괄호로 감싼 선두 라벨이라는 구조로만 판단)으로 감지된
    앵커라면 어떤 표기든 동일하게 적용된다.

    앵커 판정은 후보의 `type`이 아니라 `text` 내용으로 한다 — 같은 "[붙임 1]" 마커라도 원본에서
    볼드로 감싸져 있으면(`**[붙임 1]**`) Pass1a가 `bracket-marker`가 아니라 `bold` 타입으로 먼저
    추출해버린다(다른 샘플 문서에서 확인 — "[붙임 1]"이 bold 타입이라 캡핑이 아예
    작동 안 해서 그 뒤 "제14조(해석)"이 attachment_section으로 새는 걸 못 잡았음). 어느 경로로
    추출됐든 텍스트가 괄호 마커 구조면 동일하게 앵커로 인정해야 한다.

    스코프의 끝은 "다음으로 LLM이 확정한 앵커"가 아니라 "괄호 마커 구조를 가진 다음 후보"(LLM
    분류 결과와 무관)로 잡는다. 같은 문서 안에서 "[붙임 1]"은 attachment_section으로 정확히
    분류됐지만 바로 다음 "[붙임 2]"는 LLM이 not_heading으로 오분류하는 경우가 실제로 있었다
    (같은 샘플 문서 — "[붙임 2]" 자체는 놓쳤지만, 그렇다고 "[붙임 1]"의 스코프가 "[붙임 2]"
    이후의 완전히 다른 독립 첨부 서식들(사양서, 제안서 양식, 서약서 등)까지 집어삼켜서 전부
    "붙임 1"의 하위로 잘못 눌러버리면 안 된다). "다음 마커부터는 별개 구간"이라는 구조적 경계
    자체는 그 마커의 분류 성공 여부와 무관하게 성립하므로, 마커 "존재"만으로 스코프를 끊는다."""
    marker_lines = sorted(c["line"] for c in merged if _match_bracket_marker(c["text"].strip()))
    anchors = sorted(
        (
            c
            for c in merged
            if _match_bracket_marker(c["text"].strip()) and c["classification"] in ("main_section", "attachment_section")
        ),
        key=lambda c: c["line"],
    )
    for anchor in anchors:
        scope_start = anchor["line"]
        later_markers = [ln for ln in marker_lines if ln > scope_start]
        scope_end = later_markers[0] if later_markers else float("inf")
        for c in merged:
            if c is anchor or not (scope_start < c["line"] < scope_end):
                continue
            if c["classification"] in ("main_section", "attachment_section"):
                c["classification"] = "sub_section"
                c["reason"] = (
                    f"결정론적 후처리: '{anchor['text'][:30]}' 서식/붙임 스코프 내부 콘텐츠는 "
                    "항상 하위 세부사항으로 처리"
                )
    return merged


def classify_and_merge(candidates: list[dict], model: str, host: str = "http://localhost:11434") -> list[dict]:
    """후보를 분류하고 원본 후보 필드에 계층 분류 결과를 병합해 반환 (run_pipeline.py에서도 재사용).

    "OO 귀하" 같은 수신자 살루테이션은 어떤 문서에서도 섹션 제목이 될 수 없는데, 후보가 많은
    대형 문서(후보 92개짜리 샘플 문서)에서 LLM이 같은 텍스트를 4곳 중 1곳만
    attachment_section으로 잘못 분류하는 간헐적 오류를 실제로 확인함 — 결정론적으로 강제 보정."""
    is_top_level = len(candidates) <= _MAX_CANDIDATES_PER_CALL

    if not is_top_level:
        merged: list[dict] = []
        for i in range(0, len(candidates), _MAX_CANDIDATES_PER_CALL):
            merged.extend(classify_and_merge(candidates[i : i + _MAX_CANDIDATES_PER_CALL], model, host))
        return _cap_bracket_marker_scopes(_fill_main_section_number_gaps(merged))

    classifications = classify(candidates, model, host)
    by_line = {_line_key(c.get("line")): c for c in classifications}
    merged = []
    for cand in candidates:
        cls = by_line.get(cand["line"], {"classification": "MISSING", "reason": "LLM 응답에 없음"})
        classification = cls.get("classification")
        reason = cls.get("reason")
        if classification in ("main_section", "attachment_section") and cand["text"].strip().rstrip("*").endswith(
            "귀하"
        ):
            classification = "not_heading"
            reason = "결정론적 후처리: '~귀하' 수신자 살루테이션은 섹션 제목이 될 수 없음"
        if classification in ("main_section", "attachment_section") and cand["text"].strip().rstrip("*").endswith(
            "(인)"
        ):
            # "~귀하"(수신처, 공고 게시 측 대표를 향한 인사말)와는 별개의 패턴 — "(인)"은 응찰자
            # 본인의 서명/날인란("대표자: (인)", "서약자: OOO 대표 OOO (인)" 등)이다. 42개 문서
            # 전수 조사 결과 "(인)"으로 끝나는 줄은 예외 없이 서명란이었다(오탐 0건 — 한 샘플
            # 문서에서 "서약자 : 회사 대표 (인)"이 attachment_section으로
            # 오분류되는 걸 실제로 발견). "~귀하"와 같은 성격이지만 별개 문구라 별도 규칙으로 강제.
            classification = "not_heading"
            reason = "결정론적 후처리: '~(인)' 서명/날인란은 섹션 제목이 될 수 없음"
        if cand.get("in_box") and classification in ("main_section", "attachment_section"):
            # 원본 HWP의 텍스트박스/콜아웃 안에서 발견된 번호/제목 후보 — 9개 문서·20개 이상 인스턴스를
            # 전수 조사한 결과 박스 안 콘텐츠가 스스로 새 최상위 섹션을 여는 사례가 하나도 없었다(항상
            # 직전까지 진행되던 문맥의 하위 세부 내용이었음). LLM이 박스 경계를 모르는
            # 상태로 번호만 보고 main/attachment_section으로 잘못 승격시키는 걸 결정론적으로 되돌린다
            # (한 샘플 문서 — 박스 안 "1.~7." 체크리스트 중 일부가 문서 최상위 섹션 번호와
            # 겹쳐서 번호 충돌을 일으켰던 사고의 근본 해결책).
            classification = "sub_section"
            reason = "결정론적 후처리: 원본 텍스트박스(콜아웃) 안 콘텐츠는 항상 하위 세부사항으로 처리"
        merged.append({**cand, "classification": classification, "reason": reason})
    return _cap_bracket_marker_scopes(_fill_main_section_number_gaps(merged))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="extract_heading_candidates.py가 만든 candidates.json")
    ap.add_argument("-o", "--output", help="분류 결과 JSON 저장 경로")
    ap.add_argument("--model", default="qwen3.5:9b")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]
    merged = classify_and_merge(candidates, args.model)

    counts: dict[str, int] = {}
    for m in merged:
        counts[m["classification"]] = counts.get(m["classification"], 0) + 1

    print(f"분류 분포: {counts}")
    for m in merged:
        print(f"  L{m['line']:>4} [{m['classification']}] {m['text'][:50]!r}  — {m.get('reason', '')[:60]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"source": data.get("source"), "classified": merged}, f, ensure_ascii=False, indent=2)
        print(f"\n저장 -> {args.output}")


if __name__ == "__main__":
    main()
