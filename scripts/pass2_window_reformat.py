"""Pass 2 — Pass1이 "실제 헤더다/레벨이 몇이다"라고 판단한 결과를, kordoc이 만든 markdown 초안(Stage1
산출물)에 그대로 적용해 최종 문서를 만든다.

핵심 설계: 문서 복원(원문 추출)은 이미 Stage1(kordoc)이 끝냈다. Pass1(추출+LLM 분류)은 "이 후보가
진짜 헤더인지, 어느 레벨인지"를 판단만 한다 — 이 판단은 여전히 LLM이 맡는다(번호 연속성 같은 규칙만
으로는 계층을 못 정한다는 게 이미 실증됨: 같은 번호 표기도 문서마다 의미가 다르고, 붙임에서 번호가
정당하게 리셋되는 등). Pass2는 그 판단 결과를 markdown 초안에 **꽂아넣기만** 한다 — 본문을 LLM이
다시 쓰지 않는다.

이전 설계(윈도우 단위로 LLM이 섹션 본문을 통째로 재작성)의 문제: LLM에게 "본문 재작성"까지 시키니
재작성 과정에서 문단이 통째로 빠지거나(청렴계약 안내문, 안전보건 서약서 등 실제로 손실된 사례들),
지시된 헤더 자체를 빠뜨리거나, 없던 헤더를 만들어내는 등 콘텐츠 손실/왜곡 위험이 계속 발생했다.
LLM은 "판단"만 하고 "복원/재작성"은 절대 하지 않아야 한다는 원칙(2026-08-21 확정)에 따라, Pass2를
순수 결정론적 적용 단계로 재작성한다:
  - main_section, attachment_section -> 해당 위치를 `## {판정된 헤더 텍스트}`로 치환
  - sub_section -> `### {판정된 헤더 텍스트}`로 치환
  - not_heading, MISSING, 그리고애초에 후보로도 안 뽑힌 모든 텍스트("가.나.다." 하위 항목 포함)는
    **한 글자도 안 건드림** — 원문 그대로 유지
  - pseudo-table(표로 위장된 라벨) 타입은 Pass1a가 이미 라벨/번호를 계산해뒀으므로, 그 표 블록
    전체(라벨 산출에 쓰인 연속된 `|...|` 행들)를 헤더 한 줄로 치환

이 설계로 사라지는 문제들(전부 "LLM이 본문을 재작성한다"는 전제에서 나온 문제였음):
  - 헤더 누락/통째 빠뜨림, 없는 헤더 생성 -> 애초에 LLM이 헤더를 "쓰지" 않고 우리가 "삽입"하므로 불가능
  - 첫 섹션 이전 서두가 사라지는 문제 -> 윈도우 개념 자체가 없어져서(문서 전체를 한 번에 처리) 불가능
  - MAX_WINDOW_CHARS 분할, 표 원자성 보호, 대형 문서 VRAM/컨텍스트 문제 -> Pass2에 LLM 호출이 아예
    없으므로 전부 무관해짐

사용법:
    python pass2_window_reformat.py <stage1.md> <classified.json> [-o <output.md>] [--title TITLE]
"""

from __future__ import annotations

import argparse
import json
import os
import re

_TABLE_ROW_RE = re.compile(r"^\|.+\|$")

# render_from_kordoc_json.py가 원본 텍스트박스/콜아웃 경계에 심어두는 마커 — extract_heading_candidates.py
# /classify_headings_pass1.py가 그 경계 정보를 판단에 쓰고 나면, 최종 사용자용 출력에는 남길 이유가
# 없으므로 여기서 제거한다.
_BOX_MARKERS = {"<!--box-start-->", "<!--box-end-->"}

_LEVEL_BY_CLASSIFICATION = {
    "main_section": 2,
    "attachment_section": 2,
    "sub_section": 3,
}


def load_stage1_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().split("\n")


def _pseudo_table_block_end(lines: list[str], start_idx: int) -> int:
    """extract_heading_candidates.py의 pseudo-table 블록 탐지와 동일한 규칙으로 표 블록의 끝
    (배타적 인덱스)을 다시 스캔한다 — 후보 추출 시 이미 계산됐던 블록 범위를 그대로 재현."""
    j = start_idx
    n = len(lines)
    while j < n and _TABLE_ROW_RE.match(lines[j].strip()):
        j += 1
    return j


def render_markdown(lines: list[str], classified: list[dict], title: str) -> str:
    """Pass1 판단 결과를 markdown 초안(lines)에 그대로 적용해 최종 문서를 만든다.

    승인된(main_section/attachment_section/sub_section) 후보의 위치만 헤더 줄로 치환하고,
    그 외 모든 텍스트는 원문 그대로 보존한다 — LLM 호출 없음, 순수 문자열 조작.

    예외 1건: kordoc이 원본 스타일 메타데이터를 잘못 해석해서 서명란·수신인 같은 줄에 자체적으로
    `#` 마크를 붙여놓는 경우가 실제로 있다(한 샘플 문서 — "OO장
    귀하"에 kordoc이 `##`/`###`를 붙임). Pass1a는 이런 줄도 'heading' 타입 후보로 뽑고 Pass1b가
    정확히 not_heading으로 판정하는데, "판정 안 된 건 안 건드린다"는 원칙만 따르면 kordoc이 이미
    붙여둔 `#`가 그대로 통과해버린다 — 이건 "우리가 새로 헤더를 만드는" 게 아니라 "Pass1이 이미
    헤더가 아니라고 판정한 걸 원문(kordoc)의 실수까지 그대로 살려주는" 경우라 예외적으로 되돌린다."""
    replacements: list[tuple[int, int, str]] = []  # (시작 idx, 끝 idx 배타적, 헤더 줄)
    for c in classified:
        level = _LEVEL_BY_CLASSIFICATION.get(c.get("classification"))
        start_idx = c["line"] - 1
        if not (0 <= start_idx < len(lines)):
            continue
        if level is not None:
            end_idx = (
                _pseudo_table_block_end(lines, start_idx) if c.get("type") == "pseudo-table" else start_idx + 1
            )
            replacements.append((start_idx, end_idx, f"{'#' * level} {c['text']}"))
        elif c.get("type") == "heading":
            # Pass1이 헤더가 아니라고 판정했는데 kordoc이 이미 '#'를 붙여놓은 경우 — 평문으로 되돌림
            replacements.append((start_idx, start_idx + 1, c["text"]))

    replacements.sort(key=lambda r: r[0])

    out_lines: list[str] = []
    i = 0
    ridx = 0
    n = len(lines)
    while i < n:
        if ridx < len(replacements) and replacements[ridx][0] == i:
            start_idx, end_idx, header_line = replacements[ridx]
            out_lines.append(header_line)
            i = end_idx
            ridx += 1
        elif lines[i].strip() in _BOX_MARKERS:
            i += 1
        else:
            out_lines.append(lines[i])
            i += 1

    frontmatter = f"---\ntitle: {title}\n---\n"
    body = "\n".join(out_lines).strip()
    return f"{frontmatter}\n{body}\n"


def derive_title_from_filename(path: str) -> str:
    """문서 제목을 본문 H1이 아니라 파일명에서 유도 (중복 표제/애매한 H1 승격 문제 회피)."""
    base = os.path.basename(path)
    if base.endswith(".md"):
        base = base[: -len(".md")]
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage1_md")
    ap.add_argument("classified_json")
    ap.add_argument("-o", "--output")
    ap.add_argument("--title", help="문서 제목 (미지정 시 stage1_md 파일명에서 유도)")
    args = ap.parse_args()

    lines = load_stage1_lines(args.stage1_md)
    with open(args.classified_json, encoding="utf-8") as f:
        data = json.load(f)
    classified = data["classified"]

    title = args.title or derive_title_from_filename(args.stage1_md)

    counts: dict[str, int] = {}
    for c in classified:
        if c.get("classification") in _LEVEL_BY_CLASSIFICATION:
            counts[c["classification"]] = counts.get(c["classification"], 0) + 1
    print(f"적용됨 (제목: {title!r}): {counts}")

    final = render_markdown(lines, classified, title)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"저장 -> {args.output}")
    else:
        print("\n=== 결과 ===\n")
        print(final)


if __name__ == "__main__":
    main()
