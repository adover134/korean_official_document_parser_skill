"""kordoc의 `--format json` 출력(구조화된 `blocks` 배열)에서 직접 markdown을 조립한다.

배경: kordoc의 기본 markdown 출력 경로는 "1행×1열짜리 표(원본 HWP에서 텍스트박스/콜아웃으로 쓰인
요소)"를 표 마크업 없이 그냥 평문 단락으로 펼쳐버린다 — 그 안에 있던 "이건 박스 안에 중첩된
내용"이라는 경계 정보가 완전히 사라진다(한 샘플 문서 — 박스 안의 "1.~7."
체크리스트가 문서 최상위 섹션 번호(1~10)와 겹치는 범위라 Pass1b가 일부를 최상위 섹션으로
오분류하는 사고로 이어짐).

kordoc의 JSON 출력(`blocks`)에는 이 정보가 그대로 남아있다 — 1x1 표는 `type: "table",
table.rows==1, table.cols==1`로 명확히 구분된다. 이 스크립트는 kordoc의 markdown을 그대로 쓰는
대신, JSON `blocks`를 순회하며 우리가 직접 markdown을 조립하고, 1x1 텍스트박스는 `<!--box-start-->`
/`<!--box-end-->` 마커로 감싸서 이 경계 정보를 markdown에 실어 보낸다 — 이 마커를
`extract_heading_candidates.py`가 읽어서 "박스 안 후보는 최상위가 될 수 없다"는 결정론적 판단에 쓴다
(`classify_headings_pass1.py`). 최종 사용자용 출력(Pass2)에서는 이 마커를 제거한다.

블록 타입:
  - heading: `#`*level + text
  - paragraph: style.bold면 `**text**`, 아니면 text 그대로
  - table: 1x1이면 텍스트박스(마커로 감쌈), 그 외는 GFM pipe-table
    (`normalize_html_tables.rows_to_pipe()` 재사용 — rowspan 복제/colspan 비복제/경계행 분리 로직 동일)

사용법:
    python render_from_kordoc_json.py <input.hwp|hwpx> [-o <output.md>] [--kordoc-version 4.9.0]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize_html_tables import json_table_to_pipe  # noqa: E402
from strip_underline_tags import strip_underline_tags  # noqa: E402

BOX_START = "<!--box-start-->"
BOX_END = "<!--box-end-->"

# 유니코드 기본 다국어 평면 사용자 정의 영역(Private Use Area, U+E000~U+F8FF) — 원본 HWP가 표준에
# 없는 특수기호(예: 강조용 괄호)를 폰트 전용 글리프로 매핑해서 쓴 잔재가 JSON의 원시 텍스트에는
# 그대로 남아있다(한 샘플 문서 — "하도급지킴이" 앞뒤에 U+F0850/U+F0851 등장).
# kordoc의 markdown 출력 경로는 이런 문자를 걸러내는데 JSON 경로엔 없어서, 렌더러가 직접 제거한다.
_PUA_RANGES = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
_PUA_RE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _PUA_RANGES) + "]")


def _strip_pua(text: str) -> str:
    return _PUA_RE.sub("", text)


def run_kordoc_json(input_path: str, kordoc_version: str = "4.9.0") -> dict:
    """kordoc을 JSON 출력 모드로 실행해서 파싱된 결과를 반환."""
    result = subprocess.run(
        ["npx", f"kordoc@{kordoc_version}", input_path, "--format", "json", "--silent"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _table_block_to_pipe(table: dict) -> str:
    """kordoc JSON의 table 블록(이미 colspan이 반영된 완전한 사각 그리드)을 GFM pipe-table로 렌더링.

    HTML 파싱 경로(_TableHTMLParser)는 `<br>` 태그를 리터럴 "<br>" 문자열로 셀 텍스트에 남기는데,
    JSON 경로의 셀 텍스트는 줄바꿈이 실제 개행 문자로 들어있다 — 미리 "<br>"로 바꿔주지 않으면
    `_grid_to_pipe_lines`가 진짜 개행을 공백으로 지워버려서 그 줄바꿈 정보가 사라진다(2026-08-21
    확인, 기존 markdown과 대조해서 발견) — 기존 파이프라인과 동일하게 "<br>"를 논리적 줄바꿈
    마커로 유지하기 위해 여기서 치환한다. colspan/rowspan 자체의 그리드 처리는
    `normalize_html_tables.json_table_to_pipe()`가 담당(HTML과 다른 표현 방식이라 별도 구현 —
    해당 함수 docstring 참고)."""
    cells_2d = [
        [{"text": c.get("text", "").replace("\n", "<br>"), "rowSpan": c.get("rowSpan", 1)} for c in row]
        for row in table.get("cells", [])
    ]
    return json_table_to_pipe(cells_2d, bool(table.get("hasHeader")))


def render_blocks(blocks: list[dict]) -> str:
    """kordoc JSON의 blocks 배열을 순회하며 markdown을 조립한다."""
    parts: list[str] = []
    for b in blocks:
        btype = b.get("type")
        if btype == "heading":
            level = b.get("level") or 2
            parts.append(f"{'#' * level} {b.get('text', '')}")
        elif btype == "paragraph":
            text = b.get("text", "")
            if b.get("style", {}).get("bold"):
                text = f"**{text}**"
            parts.append(text)
        elif btype == "table":
            table = b.get("table", {})
            rows_n, cols_n = table.get("rows", 0), table.get("cols", 0)
            if rows_n == 1 and cols_n == 1:
                # 텍스트박스(원본에서 박스/콜아웃) — 경계 마커로 감싸서 하위 콘텐츠임을 표시
                text = table.get("cells", [[{}]])[0][0].get("text", "")
                parts.append(f"{BOX_START}\n{text}\n{BOX_END}")
            else:
                pipe = _table_block_to_pipe(table)
                if pipe:
                    parts.append(pipe)
        # 그 외 타입은 현재 발견된 게 없음(heading/paragraph/table 3종만 확인됨, 2026-08-21) —
        # 새 타입이 나타나면 여기서 조용히 누락되므로 별도로 로그를 남길 필요가 있으면 추가할 것
    return "\n\n".join(p for p in parts if p)


def render(input_path: str, kordoc_version: str = "4.9.0") -> str:
    data = run_kordoc_json(input_path, kordoc_version)
    md = render_blocks(data["blocks"])
    md = _strip_pua(md)
    # kordoc JSON의 text 필드 자체에 이미 리터럴 "<u>...</u>" HTML 태그가 박혀 있는 경우가 있다
    # (style.underline과 별개로 이중 표현됨 — 검증 대상 42개 문서 중 11건에서 발견). 헤더
    # 후보 추출 정규식을 깨뜨리는 원인이므로(strip_underline_tags.py와 동일한 문제) 여기서도 제거한다.
    md = strip_underline_tags(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--kordoc-version", default="4.9.0")
    args = ap.parse_args()

    md = render(args.input, args.kordoc_version)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"저장 -> {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
