"""`normalize_html_tables.py`의 회귀 테스트.

이 모듈은 이번 세션에서만 실제 문서로 두 번(2단 헤더 병합, JSON 경로 sub_header_idx)
버그가 났던 순수 문자열/데이터 변환 로직이다 — LLM/네트워크 호출 없이 결정론적으로
테스트 가능하므로, 각 docstring에 적힌 "실제로 발견한 문제"를 회귀 케이스로 고정한다."""

from __future__ import annotations

from normalize_html_tables import (
    html_table_to_pipe,
    json_table_to_pipe,
    normalize_html_tables,
)


def _pipe_rows(md: str) -> list[list[str]]:
    """렌더링된 GFM pipe-table 텍스트를 셀 값의 2차원 리스트로 파싱(헤더/구분선 포함)."""
    rows = []
    for line in md.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


class TestRowspanColspan:
    def test_rowspan_replicates_value_down_rows(self):
        # 경계 행 판정("원래 선언된 셀 중 값이 있는 게 0~1개뿐")은 rowspan으로 채워지기 전,
        # 원본 선언 셀 개수만 본다 — 그래서 두 번째 행도 선언 셀이 2개 이상이어야
        # (rowspan으로 이어받는 칸 제외하고) 경계 행으로 안 빠지고 표에 남는다.
        html = (
            "<table>"
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            '<tr><td rowspan="2">공통값</td><td>1행-B</td><td>1행-C</td></tr>'
            "<tr><td>2행-B</td><td>2행-C</td></tr>"
            "</table>"
        )
        rows = _pipe_rows(html_table_to_pipe(html))
        # header, separator, row1, row2
        assert rows[2] == ["공통값", "1행-B", "1행-C"]
        assert rows[3] == ["공통값", "2행-B", "2행-C"], "rowspan 값이 아래 행에도 복제돼야 함"

    def test_colspan_not_replicated_in_data_row(self):
        html = (
            "<table>"
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            '<tr><td colspan="2">병합값</td><td>일반</td></tr>'
            "</table>"
        )
        rows = _pipe_rows(html_table_to_pipe(html))
        assert rows[2] == ["병합값", "", "일반"], "colspan은 첫 칸에만 값, 나머지는 빈 칸"

    def test_full_width_colspan_row_becomes_boundary_text(self):
        html = (
            "<table>"
            "<tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            '<tr><td colspan="2">【붙임2】</td></tr>'
            "<tr><td>3</td><td>4</td></tr>"
            "</table>"
        )
        result = html_table_to_pipe(html)
        assert "**【붙임2】**" in result
        # 경계 행 앞뒤로 표가 두 조각으로 나뉜다 — 뒤쪽 조각은 자기 헤더 행(<th>)이 없으므로
        # (경계 행 이전의 <th>는 앞쪽 조각에만 속함) GFM pipe-table이 아니라 "3: 4" 같은
        # label:value 텍스트로 떨어진다(TestHeaderHandling의 헤더 없는 표 규칙과 동일).
        separator_lines = [ln for ln in result.split("\n") if ln.strip().startswith("| ---")]
        assert len(separator_lines) == 1, "앞쪽 조각만 진짜 pipe-table이어야 함"
        assert "3: 4" in result

    def test_fully_empty_boundary_row_produces_no_text(self):
        html = (
            "<table>"
            "<tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            '<tr><td colspan="2"></td></tr>'
            "<tr><td>3</td><td>4</td></tr>"
            "</table>"
        )
        result = html_table_to_pipe(html)
        assert "**" not in result, "완전히 빈 경계 행은 텍스트를 남기면 안 됨"
        separator_lines = [ln for ln in result.split("\n") if ln.strip().startswith("| ---")]
        assert len(separator_lines) == 1, "그래도 분리 지점으로는 작동해 표가 두 조각으로 나뉨"
        assert "3: 4" in result


class TestHeaderHandling:
    def test_consecutive_header_rows_merge_without_echo(self):
        html = (
            "<table>"
            "<tr><th>공사예정금액(A=B+E)</th><th>세부내역</th></tr>"
            "<tr><th>공사예정금액(A=B+E)</th><th>산출근거</th></tr>"
            "<tr><td>100</td><td>근거1</td></tr>"
            "</table>"
        )
        rows = _pipe_rows(html_table_to_pipe(html))
        header = rows[0]
        # 두 헤더 행에서 같은 값("공사예정금액(A=B+E)")은 한 번만 남아야 함(에코 방지)
        assert header[0] == "공사예정금액(A=B+E)", header[0]
        assert "공사예정금액(A=B+E) 공사예정금액(A=B+E)" not in header[0]
        assert header[1] == "세부내역 산출근거"
        assert len(rows) == 3, "병합된 헤더 1행 + 구분선 + 데이터 1행"

    def test_headerless_table_renders_as_label_value_text_not_pipe_table(self):
        html = "<table><tr><td>이름</td><td>홍길동</td></tr><tr><td>연락처</td><td>010-0000-0000</td></tr></table>"
        result = html_table_to_pipe(html)
        assert "|" not in result, "실제 <th>가 없으면 GFM pipe-table을 강제로 만들면 안 됨"
        assert "이름: 홍길동" in result
        assert "연락처: 010-0000-0000" in result


class TestNestedTable:
    def test_nested_table_text_absorbed_into_outer_cell(self):
        # 열이 1개뿐이면 모든 행이 "선언 셀 0~1개" 경계 행 규칙에 걸려버리므로, 경계 행
        # 판정과 무관하게 진짜 표로 남는지 보려면 열을 2개 이상으로 구성해야 한다.
        html = (
            "<table><tr><th>A</th><th>B</th></tr>"
            "<tr><td>바깥 텍스트<table><tr><td>안쪽</td></tr></table></td><td>일반</td></tr></table>"
        )
        result = html_table_to_pipe(html)
        assert "바깥 텍스트" in result
        assert "안쪽" in result, "중첩 표 텍스트도 유실 없이 바깥 셀에 흡수돼야 함"
        # 중첩 표가 별도 pipe-table로 새어나오면 안 됨(구분선 "행"이 하나만 있어야 함 —
        # 컬럼 수만큼 "| ---"가 한 줄 안에서 반복되므로 부분 문자열이 아니라 줄 단위로 센다)
        separator_lines = [ln for ln in result.split("\n") if ln.strip().startswith("| ---")]
        assert len(separator_lines) == 1

    def test_normalize_html_tables_handles_nested_table_in_full_document(self):
        text = (
            "앞 문단\n\n"
            "<table><tr><th>A</th></tr>"
            "<tr><td>바깥<table><tr><td>안쪽</td></tr></table></td></tr></table>\n\n"
            "뒤 문단"
        )
        result = normalize_html_tables(text)
        assert "앞 문단" in result
        assert "뒤 문단" in result, "비탐욕 정규식이었다면 중첩 표 이후 내용(뒤 문단)이 유실됐을 것"
        assert "<table>" not in result


class TestFallback:
    def test_returns_original_html_when_no_rows_parsed(self):
        html = "<table></table>"
        assert html_table_to_pipe(html) == html


class TestJsonTableToPipe:
    def _cell(self, text: str, row_span: int = 1) -> dict:
        return {"text": text, "rowSpan": row_span}

    def test_headerless_does_not_default_to_index_zero(self):
        """sub_header_idx 버그 회귀: has_header=False면 어떤 세그먼트도 헤더로 잡히면 안 됨."""
        cells = [
            [self._cell("이름"), self._cell("홍길동")],
            [self._cell("연락처"), self._cell("010-0000-0000")],
        ]
        result = json_table_to_pipe(cells, has_header=False)
        assert "|" not in result
        assert "이름: 홍길동" in result

    def test_header_present_renders_pipe_table(self):
        cells = [
            [self._cell("A"), self._cell("B")],
            [self._cell("1"), self._cell("2")],
        ]
        result = json_table_to_pipe(cells, has_header=True)
        rows = _pipe_rows(result)
        assert rows[0] == ["A", "B"]
        assert rows[2] == ["1", "2"]

    def test_rowspan_banner_not_duplicated_as_boundary_text(self):
        """rowspan=2 배너 텍스트 중복 출력 버그 회귀 — 원본(미복제) 레벨에서 경계 행을
        판정해야 "《 입찰참가시 유의사항 》" 같은 배너가 두 번 나오면 안 됨."""
        cells = [
            [self._cell("《 입찰참가시 유의사항 》", row_span=2), self._cell("", row_span=0)],
            [self._cell("", row_span=0), self._cell("", row_span=0)],
            [self._cell("A"), self._cell("B")],
        ]
        result = json_table_to_pipe(cells, has_header=False)
        assert result.count("입찰참가시 유의사항") == 1

    def test_colspan_already_expanded_not_double_expanded(self):
        """kordoc JSON은 colspan이 이미 완전한 그리드로 펼쳐져 있다 — HTML용 복제 로직을
        그대로 타면 열 개수가 부풀어 오르는 버그(4열 표가 6열로 깨짐) 회귀 테스트."""
        cells = [
            [self._cell("A"), self._cell("B"), self._cell("C"), self._cell("D")],
            [self._cell("병합값"), self._cell(""), self._cell("1"), self._cell("2")],
        ]
        result = json_table_to_pipe(cells, has_header=True)
        rows = _pipe_rows(result)
        assert len(rows[0]) == 4, f"열 개수가 부풀면 안 됨: {rows[0]}"
