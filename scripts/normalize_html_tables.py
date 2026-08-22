"""kordoc이 병합 셀 표에 쓰는 raw HTML <table>을 GFM pipe-table(`| ... |`)로 정규화한다.

배경: 팀 기존 파서(src/parsers/table_flattener.py)의 _TABLE_ROW_RE는 `|`로 시작/끝나는 줄만
표로 인식하고 raw HTML <table>은 전혀 인식하지 못한다. kordoc은 rowspan/colspan이 있는 표를
HTML로 출력하므로(단순 표는 이미 `|` 문법), 두 표기가 섞여 나온다 — 이후 파이프라인(청킹/평탄화)이
일관되게 동작하려면 전부 `|` 문법으로 통일해야 한다.

병합 셀 처리 방식 — rowspan과 colspan을 다르게 다룬다(2026-08-20, 실제 문서에서 발견한 문제로 수정):
  - **rowspan(행 방향, 위→아래 병합)은 복제한다.** row-by-row로 나중에 개별 처리될 걸 감안하면,
    각 행이 자기 자신의 셀만 보고도 값을 알 수 있어야 하기 때문 — 복제 안 하면 이후 key-value
    평탄화 단계에서 그 값이 어느 행에 속하는지 알 수 없어 정보가 소실된다.
  - **colspan(열 방향, 좌→우 병합)은 복제하지 않는다.** 같은 행 안이라 어차피 한 번에 같이
    처리되므로 복제가 불필요하고, 오히려 문제를 일으킨다 — 실제로 한 문서에서 `colspan="15"`
    (표 전체 너비를 차지하는 배너/구분용 셀)을 곧이곧대로 복제했더니 44자짜리 원본 한 줄이
    497자로(~11배) 부풀고, 체크리스트 항목처럼 `colspan="12"`인 긴 문장이 12번씩 복제되는 등
    문서 전체가 걷잡을 수 없이 커지는 사고가 있었다. 값은 스팬의 첫 칸에만 넣고 나머지는 빈 칸.
  - **"경계 행"(원래 선언된 셀 중 값이 있는 게 0~1개뿐인 행) 은 표에서 분리해 별도 줄로 뺀다.**
    큰 colspan을 쓰는 행은 대개 배너 제목이거나(예: `<td colspan="15">【붙임2】</td>`), 원본
    HWP에서 "표를 두 개로 나누면 오류가 나서" 대신 빈 줄처럼 쓴 여백 행(`<td colspan="15"></td>`)
    이다 — 어느 쪽도 진짜 다열 데이터가 아니다. 값이 있으면 독립된 볼드 줄로, 완전히 비어있으면
    아무것도 안 넣고 **그 지점에서 표를 두 개로 나누는 경계로만** 사용한다.

사용법:
    python normalize_html_tables.py <input.md> [-o <output.md>]
    (미지정 시 표준출력)
"""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser


class _TableHTMLParser(HTMLParser):
    """<table>...</table> 하나를 파싱해 (is_header, cells) 튜플의 행 리스트로 만든다.

    셀 안에 표가 통째로 중첩되는 경우(한 샘플 문서의 [붙임1] "안전·보건 관리 준수
    서약서"에서 발견 — 서약서 안내문 뒤에 <br>로 이어서 <table>이 또 나옴)가 실제로 있다. 중첩
    depth를 추적해 depth>1(중첩 표 내부)인 동안의 tr/td/th는 별도 표로 만들지 않고, 그 텍스트를
    바깥 셀의 텍스트에 그대로 흡수시킨다 — 중첩 표를 별 구조로 재구성하려다 파싱이 꼬여서 바깥
    셀 텍스트 전체가 통째로 사라지는 사고(중첩 <th>가 아직 안 닫힌 바깥 셀의 _cell_text를 덮어씀)가
    있었음. 완벽한 표 구조 복원보다 텍스트 손실 없음을 우선한다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[bool, list[dict]]] = []  # (is_header_row, [{"text":..., "rowspan":.., "colspan":..}])
        self._in_row = False
        self._in_cell = False
        self._cell_is_header = False
        self._cell_text: list[str] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._row_cells: list[dict] = []
        self._row_has_th = False
        self._table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth > 1:
            # 중첩 표 내부 — 별도 표로 만들지 않고 바깥 셀 텍스트에 흡수(공백/<br>로만 구분)
            if tag in ("tr", "td", "th") and self._in_cell:
                self._cell_text.append(" ")
            elif tag == "br" and self._in_cell:
                self._cell_text.append("<br>")
            return

        attrs_d = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._row_cells = []
            self._row_has_th = False
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_is_header = tag == "th"
            self._cell_text = []
            self._cell_rowspan = int(attrs_d.get("rowspan") or 1)
            self._cell_colspan = int(attrs_d.get("colspan") or 1)
        elif tag == "br" and self._in_cell:
            self._cell_text.append("<br>")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth > 1:
            return

        if tag in ("td", "th") and self._in_cell:
            text = "".join(self._cell_text).strip()
            text = re.sub(r"\s*\n\s*", " ", text)
            self._row_cells.append(
                {"text": text, "rowspan": self._cell_rowspan, "colspan": self._cell_colspan}
            )
            if self._cell_is_header:
                self._row_has_th = True
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            self.rows.append((self._row_has_th, self._row_cells))
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)


def _cells_to_grid(rows: list[tuple[bool, list[dict]]]) -> tuple[list[list[str]], bool, int | None]:
    """(is_header_row, cells) 행 리스트를 사각 그리드로 변환.

    rowspan은 값을 복제해서 아래 행들도 채우고, colspan은 복제하지 않고 첫 칸에만 값을 넣는다
    (나머지 스팬 칸은 빈 칸) — 이유는 모듈 docstring 참고.

    Returns:
        (grid, has_header, header_row_index)
    """
    grid: list[list[str]] = []
    # carry[col] = (남은 행 수, 값) — 이전 행의 rowspan이 이 칸을 계속 채우는 중
    carry: dict[int, list] = {}
    has_header = any(is_th for is_th, _ in rows)
    header_row_index = next((i for i, (is_th, _) in enumerate(rows) if is_th), None)

    for row_cells in rows:
        _, cells = row_cells
        grid_row: list[str] = []
        col = 0

        def _fill_carry() -> None:
            nonlocal col
            while col in carry:
                remaining, value = carry[col]
                grid_row.append(value)
                remaining -= 1
                if remaining <= 0:
                    del carry[col]
                else:
                    carry[col] = [remaining, value]
                col += 1

        for cell in cells:
            _fill_carry()
            text = cell["text"]
            colspan = max(1, cell["colspan"])
            rowspan = max(1, cell["rowspan"])
            for k in range(colspan):
                cell_text = text if k == 0 else ""  # colspan은 복제 안 함 — 첫 칸에만
                grid_row.append(cell_text)
                if rowspan > 1:
                    carry[col] = [rowspan - 1, cell_text]
                col += 1

        _fill_carry()
        grid.append(grid_row)

    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        while len(r) < max_cols:
            r.append("")

    return grid, has_header, header_row_index


def _json_rows_to_grid(rows_cells: list[list[dict]]) -> list[list[str]]:
    """이미 colspan이 반영된 JSON 행 목록(연속된 "표" 구간)에 대해 rowspan만 복제해서 그리드로
    변환. colspan은 JSON이 이미 그리드로 펼쳐뒀으므로 복제하지 않는다(HTML과 다른 점 — 아래
    `json_table_to_pipe` docstring 참고)."""
    grid: list[list[str]] = []
    carry: dict[int, list] = {}
    for row in rows_cells:
        grid_row: list[str] = []
        for col_idx, cell in enumerate(row):
            text = cell.get("text", "")
            rowspan = max(1, cell.get("rowSpan", 1))
            if not text and col_idx in carry:
                remaining, value = carry[col_idx]
                grid_row.append(value)
                remaining -= 1
                if remaining <= 0:
                    del carry[col_idx]
                else:
                    carry[col_idx] = [remaining, value]
            else:
                grid_row.append(text)
                if rowspan > 1:
                    carry[col_idx] = [rowspan - 1, text]
        grid.append(grid_row)
    return grid


def json_table_to_pipe(cells_2d: list[list[dict]], has_header: bool) -> str:
    """kordoc JSON `--format json`의 표 셀(cells, 이미 colspan이 반영된 완전한 사각 그리드)을
    GFM pipe-table(들) + 경계 텍스트로 변환.

    HTML `<table>`과 근본적으로 다른 표현 방식: HTML은 `<td colspan="3">`처럼 병합된 칸 뒤의
    셀 자체가 문서에 없어서 우리가 복제해서 채워야 했지만(`_cells_to_grid`), kordoc JSON은
    colspan으로 병합된 칸 뒤에도 빈 셀 객체가 이미 명시적으로 존재하는 완전한 그리드다(실측
    확인, 2026-08-21) — 그대로 `_cells_to_grid`(HTML용 복제 로직)에 넘기면 colspan 값이 중복
    복제돼 열 개수가 부풀어 오르는 버그가 생긴다(한 샘플 문서에서 4열짜리 표가 6열로 깨짐).

    경계 행(선언된 값 0~1개) 판정은 **rowspan 복제 이전(원본 cells)** 레벨에서 먼저 해야 한다 —
    복제 후(그리드) 레벨에서 판정하면 rowspan으로 복제된 값도 "그 행에 새 값이 있다"고 잘못
    카운트해서, 하나의 원본 값이 여러 개의 중복된 경계 텍스트로 뻥튀기되는 사고가 났다(다른
    샘플 문서 — "《 입찰참가시 유의사항 》" 텍스트가 rowspan=2라서 두 번 출력됨).
    HTML 경로(`rows_to_pipe`)는 애초에 파서가 만드는 `rows`가 원본(미복제) 레벨이라 이 문제가
    없었는데, JSON은 처음부터 완전 그리드라 순서를 명시적으로 맞춰야 했다."""
    if not cells_2d:
        return ""

    header_row_index = 0 if has_header else None

    segments: list[tuple[str, object]] = []
    current: list[tuple[int, list[dict]]] = []

    def _flush() -> None:
        if current:
            segments.append(("table", list(current)))
            current.clear()

    for i, row in enumerate(cells_2d):
        non_empty = [c.get("text", "") for c in row if c.get("text", "").strip()]
        if len(non_empty) <= 1:
            _flush()
            if non_empty:
                segments.append(("text", f"**{non_empty[0]}**"))
        else:
            current.append((i, row))
    _flush()

    if not segments:
        return ""

    out_parts: list[str] = []
    for kind, payload in segments:
        if kind == "text":
            out_parts.append(str(payload))
        else:
            sub_grid = _json_rows_to_grid([cells for _, cells in payload])
            sub_header_idx = 0
            for local_i, (orig_i, _) in enumerate(payload):
                if orig_i == header_row_index:
                    sub_header_idx = local_i
                    break
            out_parts.append("\n".join(_grid_to_pipe_lines(sub_grid, sub_header_idx)))
    return "\n\n".join(out_parts)


def _grid_to_pipe_lines(grid: list[list[str]], header_row_index: int | None) -> list[str]:
    """그리드를 GFM pipe-table 줄 목록으로 렌더링."""

    def _escape(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    header_idx = header_row_index if header_row_index is not None else 0
    header = grid[header_idx]
    body_rows = grid[:header_idx] + grid[header_idx + 1 :]

    lines = ["| " + " | ".join(_escape(c) for c in header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body_rows:
        lines.append("| " + " | ".join(_escape(c) for c in row) + " |")
    return lines


def _row_non_empty_texts(cells: list[dict]) -> list[str]:
    return [c["text"] for c in cells if c["text"].strip()]


def rows_to_pipe(rows: list[tuple[bool, list[dict]]]) -> str:
    """(is_header_row, cells) 행 리스트 하나를 GFM pipe-table(들) + 경계 텍스트로 변환.

    HTML 파싱 경로(`html_table_to_pipe`)와 kordoc JSON `blocks` 경로(`render_from_kordoc_json.py`)
    가 공통으로 쓰는 핵심 로직 — 입력 형식만 다르고(HTML 파싱 결과 vs JSON cells) 표 자체의 구조는
    동일(rowspan/colspan/cells)하므로 여기서 한 번만 구현한다.

    선언된 값이 0~1개뿐인 "경계 행"(배너/여백용)을 표에서 분리해서, 그 지점을 기준으로
    표를 여러 조각으로 나눈다. 값이 있으면 볼드 줄로 남기고, 완전히 비어있으면 그냥
    분리 지점으로만 쓰고 아무 텍스트도 안 남긴다.
    """
    if not rows:
        return ""

    segments: list[tuple[str, object]] = []  # ("table", [(is_header, cells), ...]) | ("text", str)
    current: list[tuple[bool, list[dict]]] = []

    def _flush() -> None:
        if current:
            segments.append(("table", list(current)))
            current.clear()

    for is_header, cells in rows:
        non_empty = _row_non_empty_texts(cells)
        if len(non_empty) <= 1:
            _flush()
            if non_empty:
                segments.append(("text", f"**{non_empty[0]}**"))
            # 완전히 빈 경계 행은 텍스트 없이 분리 지점으로만 소비
        else:
            current.append((is_header, cells))
    _flush()

    if not segments:
        return ""

    out_parts: list[str] = []
    for kind, payload in segments:
        if kind == "text":
            out_parts.append(str(payload))
        else:
            grid, _has_header, header_idx = _cells_to_grid(payload)  # type: ignore[arg-type]
            if grid:
                out_parts.append("\n".join(_grid_to_pipe_lines(grid, header_idx)))
    return "\n\n".join(out_parts)


def html_table_to_pipe(html: str) -> str:
    """<table>...</table> 문자열 하나를 GFM pipe-table(들) + 경계 텍스트로 변환 (HTML 파싱 후 `rows_to_pipe` 호출)."""
    parser = _TableHTMLParser()
    parser.feed(html)
    parser.close()
    if not parser.rows:
        return html
    result = rows_to_pipe(parser.rows)
    return result if result else html


_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)


def normalize_html_tables(text: str) -> str:
    """텍스트 안의 모든 <table>...</table> 블록을 GFM pipe-table로 치환한다.

    비탐욕 정규식(`<table>.*?</table>`)은 표 안에 표가 중첩된 경우(한 샘플 문서에서
    발견) 가장 가까운(안쪽) `</table>`에서 매칭이 끝나버려 바깥 표의 나머지 내용이 그대로 원문에
    남는 사고가 있었다 — 대신 depth를 세면서 실제로 짝이 맞는 바깥 `</table>`을 직접 찾는다."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _TABLE_OPEN_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])

        depth = 1
        pos = m.end()
        while depth > 0:
            next_open = _TABLE_OPEN_RE.search(text, pos)
            next_close = _TABLE_CLOSE_RE.search(text, pos)
            if not next_close:
                # 닫히지 않은 표 -- 더 진행할 수 없으니 나머지는 그대로 둔다
                pos = n
                depth = 0
                break
            if next_open and next_open.start() < next_close.start():
                depth += 1
                pos = next_open.end()
            else:
                depth -= 1
                pos = next_close.end()

        block = text[m.start() : pos]
        out.append(html_table_to_pipe(block))
        i = pos

    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="입력 마크다운 파일")
    ap.add_argument("-o", "--output", help="출력 파일 경로 (미지정 시 표준출력)")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    result = normalize_html_tables(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"완료 -> {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
