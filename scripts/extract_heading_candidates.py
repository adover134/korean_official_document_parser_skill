"""제목 후보 추출기 (2-패스 슬라이딩 윈도우 설계의 Pass 1 — 추출 단계만).

목적: 문서 전체를 LLM에 넘기지 않고, 가벼운 규칙으로 "제목일 수 있는 줄"만 뽑아낸다.
이 스크립트는 최종 분류(진짜 헤더인지, 레벨이 몇인지)는 하지 않는다 — 후보를 놓치지 않는지(recall)와
"같은 번호 형식이 여러 계층에서 재사용될 때" 최상위 후보와 중첩된(하위) 후보를 구분하는 것까지만 한다.

감지하는 후보 유형:
  heading        - 이미 Stage1에서 감지된 # 마크다운 헤더
  bold           - 줄 전체가 **볼드**로만 된 줄 (예: **1. 견적제출에 부치는 사항**) — 번호가 있으면 함께 추출
  numbered       - 줄 시작이 "숫자." 인 평문 줄 (예: 1. 입찰에 부치는 사항) — 표 안 줄/날짜 오탐은 제외
  pseudo-table   - 첫 번째 열이 모든 행에서 비어있는 표(라벨/제목 역할, 데이터 값 없음)
                   예1: | 1. | | 입찰 개요 |                      (데이터 행 0개)
                   예2: |  | 《 입찰참가시 유의사항 》 |  \\n |  | 본문... |  (데이터 행 있어도 첫 열이 계속 빈칸)
  bracket-marker - 줄이 괄호류(【】/〈〉/《》/「」/『』/[]/()/（）)로 감싼 짧은 라벨로 시작하는 평문 줄
                   (예: "【서식 7】 이행능력심사 자기평가 및 심사표", "[붙임2] 제출 서류")
                   서식/붙임/별첨 제목이 이 형태로 자주 나오는데, 볼드도 번호(숫자.)도 아니라서
                   기존 규칙엔 안 걸렸다(한 샘플 문서에서 확인) — 다만 이 규칙은
                   "서식"/"붙임" 같은 특정 단어가 아니라 "괄호로 감싼 선두 라벨"이라는 구조만 본다(다른
                   기호 조합의 마커도 있을 수 있다는 점, 그리고 "(예: ~)"처럼 우연히 괄호가 줄 서두에
                   오는 본문도 섞여 들어올 수 있다는 점 둘 다 감안한 설계). 오탐은 여기서 걸러내지
                   않는다 — 다른 유형과 마찬가지로 recall만 하고, 실제 헤더 여부/계층 판단은 Pass1b(LLM)가
                   문맥으로 한다.

번호(숫자) 계층 추론은 여기서 하지 않는다: "연속되면 최상위, 끊기면 중첩"류의 규칙은 실제로 시도해봤으나
붙임/별첨처럼 정당하게 번호가 리셋되는 구간을 전부 오탐하고, 한 번 시퀀스가 어긋나면 복구가 안 되어
헤더가 대거 누락되는 결과를 냈다(2026-08-20 확인). 규칙으로 계층을 고정하려는 시도 자체가 문제였다는
결론 — 이 스크립트는 후보 "추출"(recall)까지만 하고, 계층/중첩 판단은 Pass 1 LLM에게 전체 맥락과 함께
맡긴다.

사용법:
    python extract_heading_candidates.py <input.md> [--context N]
"""

from __future__ import annotations

import argparse
import re

# 줄 전체가 볼드인 경우뿐 아니라, 밑줄(<u>)까지 함께 감싼 "<u>**...**</u>" 형태도 잡는다 —
# 실제로 원본이 이 조합으로 최상위 섹션 제목을 강조한 사례가 있었음(한 샘플 문서의
# "<u>**12. 안전 및 보건 확보 의무사항 준수**</u>" — 밑줄 태그 때문에 줄이 "**"로 시작하지
# 않아서 기존 정규식(^\*\*...\*\*$)에 안 걸려 후보 추출 자체가 안 됐었음). `^...$`로 줄 전체를
# 앵커링하므로, 문장 중간에 부분적으로 <u>**...**</u>가 쓰인 경우(강조 표시)는 여전히 안 걸림.
_BOLD_ONLY_RE = re.compile(r"^(?:<u>)?\*\*(.+?)\*\*(?:</u>)?$")
_NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.+)$")
_BOLD_NUMBERED_RE = re.compile(r"^(\d+)\.\s*(.+)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_TABLE_SEP_RE = re.compile(r"^\|[-:\s|]+\|$")
# render_from_kordoc_json.py가 원본 HWP의 "텍스트박스/콜아웃"(kordoc JSON에서 1x1 표로 확인된 요소)
# 경계에 심어두는 마커. 박스 안에서 발견되는 번호/헤더 후보는 그 자체로 새 최상위 섹션을 여는 사례가
# 지금까지(9개 문서, 20개 이상 인스턴스) 단 한 건도 없었다 — 항상 그 직전까지 진행되던 문맥의 하위
# 세부 내용이었다(2026-08-21 확인). 그래서 in_box 표시만 하고, 실제 강제 강등은
# classify_headings_pass1.py의 결정론적 후처리에서 한다(판단은 여전히 LLM이 하되, 박스 안이면
# main_section/attachment_section으로 나온 결과를 sub_section으로 되돌림).
_BOX_START_MARKER = "<!--box-start-->"
_BOX_END_MARKER = "<!--box-end-->"
# 날짜 오탐 방지: "8." 뒤가 숫자/공백/마침표뿐이면(예: "5." "5.(금)" 앞의 "8.") 번호 매김 제목이 아님
_DATE_LIKE_TAIL_RE = re.compile(r"^[\d.\s()().:월화수목금토일]*$")

# 괄호류로 감싼 선두 라벨 — "서식"/"붙임" 같은 특정 단어가 아니라 이 구조 자체를 본다(모듈 docstring
# 참고). 소괄호/전각소괄호도 포함하되, 오탐("(예: ~)"처럼 본문 중 우연히 줄 서두에 걸리는 경우)은
# 여기서 막지 않고 Pass1b(LLM)의 문맥 판단에 맡긴다.
_BRACKET_MARKER_PAIRS = [
    ("【", "】"),
    ("〈", "〉"),
    ("《", "》"),
    ("「", "」"),
    ("『", "』"),
    ("[", "]"),
    ("(", ")"),
    ("（", "）"),
]


def _match_bracket_marker(stripped: str) -> bool:
    for open_ch, close_ch in _BRACKET_MARKER_PAIRS:
        if stripped.startswith(open_ch):
            close_idx = stripped.find(close_ch, 1)
            if close_idx > 0:
                return True
    return False


def _following_snippet(lines: list[str], start_idx: int, max_chars: int = 80) -> str:
    buf = []
    total = 0
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        buf.append(stripped)
        total += len(stripped)
        if total >= max_chars:
            break
    return " ".join(buf)[:max_chars]


def _is_date_like(text: str) -> bool:
    """번호 뒤 텍스트가 날짜/숫자 파편뿐인지(오탐 방지)."""
    return bool(_DATE_LIKE_TAIL_RE.match(text.strip())) or len(text.strip()) < 2


def extract_candidates(text: str, context_chars: int = 80) -> list[dict]:
    lines = text.split("\n")
    candidates: list[dict] = []

    i = 0
    n = len(lines)
    in_box = False
    while i < n:
        stripped = lines[i].strip()

        if stripped == _BOX_START_MARKER:
            in_box = True
            i += 1
            continue
        if stripped == _BOX_END_MARKER:
            in_box = False
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # 1) 이미 감지된 마크다운 헤더
        m = _HEADING_RE.match(stripped)
        if m:
            candidates.append(
                {
                    "line": i + 1,
                    "type": "heading",
                    "level": len(m.group(1)),
                    "number": None,
                    "text": m.group(2).strip(),
                    "context": _following_snippet(lines, i, context_chars),
                    "in_box": in_box,
                }
            )
            i += 1
            continue

        # 2) 표 블록: 표 안의 "서로 다른 값" 개수가 2개 이하면 → pseudo-table(라벨/안내 박스) 후보
        #    (열 위치가 아니라 값 개수로 판단 — colspan 정규화로 특정 열이 채워져도 흔들리지 않음)
        if _TABLE_ROW_RE.match(stripped) and not _TABLE_SEP_RE.match(stripped):
            block_start = i
            block = [stripped]
            j = i + 1
            while j < n and _TABLE_ROW_RE.match(lines[j].strip()):
                block.append(lines[j].strip())
                j += 1

            has_sep = len(block) > 1 and _TABLE_SEP_RE.match(block[1])
            header_row = block[0]
            data_rows = block[2:] if has_sep else block[1:]

            def _cells(row: str) -> list[str]:
                return [c.strip() for c in row.strip("|").split("|")]

            all_cells = _cells(header_row) + [c for r in data_rows for c in _cells(r)]
            distinct_values = {c for c in all_cells if c}

            if has_sep and 0 < len(distinct_values) <= 2:
                # 값이 여러 개면 가장 긴 것을 라벨/제목으로, 숫자만 있는 짧은 값은 번호로 취급
                values_sorted = sorted(distinct_values, key=len, reverse=True)
                label = values_sorted[0]
                number = None
                for v in distinct_values:
                    vm = re.match(r"^(\d+)\.?$", v)
                    if vm:
                        number = int(vm.group(1))
                if number is not None:
                    # numbered/bold/heading 타입과 동일하게 text에 번호를 포함해서 남긴다
                    label = f"{number}. {label}"
                if label:
                    candidates.append(
                        {
                            "line": block_start + 1,
                            "type": "pseudo-table",
                            "level": None,
                            "number": number,
                            "text": label,
                            "raw_row": header_row,
                            "context": _following_snippet(lines, j - 1, context_chars),
                            "in_box": in_box,
                        }
                    )
            i = j
            continue

        # 3) 표 안이 아닌 볼드 단독 줄 (번호 있으면 함께 추출)
        m = _BOLD_ONLY_RE.match(stripped)
        if m and not stripped.startswith("|"):
            inner = m.group(1).strip()
            num_m = _BOLD_NUMBERED_RE.match(inner)
            number = int(num_m.group(1)) if num_m else None
            body = num_m.group(2).strip() if num_m else inner
            if number is None or not _is_date_like(body):
                candidates.append(
                    {
                        "line": i + 1,
                        "type": "bold",
                        "level": None,
                        "number": number,
                        "text": inner,
                        "context": _following_snippet(lines, i, context_chars),
                        "in_box": in_box,
                    }
                )
            i += 1
            continue

        # 4) 번호 매김 평문 줄 (표 안이 아닌 경우만, 날짜 오탐 제외)
        m = _NUMBERED_RE.match(stripped)
        if m and not stripped.startswith("|") and not _is_date_like(m.group(2)):
            # text에는 원문 그대로 번호를 포함해서 남긴다 (heading/bold 타입과 동일하게) —
            # 예전엔 번호를 떼어내서 'text'가 "물품구매 개요"처럼 나왔는데, 같은 문서의
            # heading 타입 후보("3. 입찰일정")는 번호가 안 떼어져서 최상위 섹션 제목 표시가
            # 후보 타입에 따라 들쭉날쭉했다(2026-08-20 확인). number 필드는 프로그램적 판단용으로 별도 유지.
            candidates.append(
                {
                    "line": i + 1,
                    "type": "numbered",
                    "level": None,
                    "number": int(m.group(1)),
                    "text": stripped,
                    "context": _following_snippet(lines, i, context_chars),
                    "in_box": in_box,
                }
            )
            i += 1
            continue

        # 5) 괄호류(【】/〈〉/《》/「」/『』/[]/()/（）)로 감싼 선두 라벨로 시작하는 평문 줄
        #    (표 안 줄은 이미 위에서 소비되어 여기 도달하지 않음)
        if not stripped.startswith("|") and _match_bracket_marker(stripped):
            candidates.append(
                {
                    "line": i + 1,
                    "type": "bracket-marker",
                    "level": None,
                    "number": None,
                    "text": stripped,
                    "context": _following_snippet(lines, i, context_chars),
                    "in_box": in_box,
                }
            )
            i += 1
            continue

        i += 1

    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--context", type=int, default=80)
    ap.add_argument("-o", "--output", help="JSON으로 저장할 경로 (미지정 시 표준출력에 사람이 읽기 좋은 형식으로만 출력)")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    candidates = extract_candidates(text, args.context)

    print(f"=== {args.input} — 후보 {len(candidates)}개 (계층/중첩 판단 없음 — 추출만) ===")
    for c in candidates:
        lvl = f" lv{c['level']}" if c.get("level") else ""
        num = f" #{c['number']}" if c.get("number") is not None else ""
        print(f"  L{c['line']:>4} [{c['type']}{lvl}{num}] {c['text'][:55]!r}")
        if c.get("context"):
            print(f"         └─ 다음 내용: {c['context'][:70]!r}...")

    if args.output:
        import json

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {"source": args.input, "candidate_count": len(candidates), "candidates": candidates},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nJSON 저장 -> {args.output}")


if __name__ == "__main__":
    main()
