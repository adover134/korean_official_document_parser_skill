"""말머리 번호/기호 뒤에 마침표(.) 대신 쉼표(,)가 잘못 쓰인 원문 오탈자를 교정한다.

배경: kordoc이 원문을 충실히 그대로 옮기기 때문에, 원본 HWP 자체에 있는 오탈자도 그대로 넘어온다.
실제로 발견된 사례(한 샘플 문서): "5. 예정가격 및 낙찰자 결정 방법"이어야
할 자리가 "5, 예정가격 및 낙찰자 결정 방법"으로 적혀 있어서, `extract_heading_candidates.py`의 번호
패턴(`^\\d+\\.\\s`)에 안 걸려 후보로 추출조차 안 됐다 — 결과적으로 해당 섹션 전체가 헤더 승격 대상에서
누락됨(내용 자체는 원문 그대로 살아있으니 콘텐츠 손실은 아니지만, 구조가 깨짐).

맞춤법 검사 API(korector, py-hanspell 등)로 교정하는 방안도 검토했으나, 둘 다 Naver의 비공식 내부
API를 스크래핑하는 방식이라 (a) 지금 이 환경에서 실제로 안 됨(korector는 인증 토큰 획득 실패,
py-hanspell은 오래된 pip 내부 모듈에 의존해서 설치 자체가 안 됨), (b) 설사 됐더라도 이런 말머리
구두점 문제까지 잡아줄지 불확실함(일반 맞춤법 검사기는 보통 띄어쓰기/단어 철자가 주력) — 그래서
이 특정 패턴만 겨냥한 정규식 기반 결정론적 교정으로 대체한다.

이 스크립트는 **원문(Stage1 markdown)을 실제로 고쳐서 반영**한다 — 하위 단계에서 매번 다른 패턴을
허용하도록 넓히는 게 아니라, 원인(오탈자)을 원본 위치에서 바로잡는다.

사용법:
    python fix_bullet_punctuation.py <input.md> [-o <output.md>] [--dry-run]
"""

from __future__ import annotations

import argparse
import re

# 줄 시작(공백 들여쓰기 허용)에 숫자/한글 한 글자/알파벳 한 글자 말머리가 오고, 그 뒤에 쉼표+공백이면
# 오탈자로 간주 — 정상적인 말머리는 마침표를 쓴다("1. ", "가. ", "A. "). 쉼표+공백 뒤에 곧바로 또
# 숫자가 오면(예: "1, 2, 3" 같은 나열) 말머리가 아니라 진짜 쉼표 나열일 가능성이 높으므로 제외한다.
BULLET_COMMA_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>\d{1,3}|[가-힣]|[A-Za-z]),(?P<space>[ \t]+)(?!\d)",
    re.MULTILINE,
)


def find_issues(text: str) -> list[dict]:
    issues = []
    for m in BULLET_COMMA_PATTERN.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line_end = len(text) if line_end == -1 else line_end
        line_start = text.rfind("\n", 0, m.start()) + 1
        issues.append(
            {
                "line": line_no,
                "marker": m.group("marker"),
                "context": text[line_start:line_end][:80],
            }
        )
    return issues


def fix_bullet_punctuation(text: str) -> str:
    return BULLET_COMMA_PATTERN.sub(r"\g<indent>\g<marker>.\g<space>", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="출력 경로 (미지정 시 입력 파일을 덮어씀)")
    ap.add_argument("--dry-run", action="store_true", help="교정하지 않고 발견된 오탈자만 출력")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    issues = find_issues(text)
    for issue in issues:
        print(f"  L{issue['line']}: {issue['marker']!r} 뒤 쉼표 -> 마침표 교정 : {issue['context']!r}")

    if args.dry_run:
        print(f"발견: {len(issues)}건 (--dry-run, 교정 안 함)")
        return

    fixed = fix_bullet_punctuation(text)
    out_path = args.output or args.input
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fixed)
    print(f"교정 {len(issues)}건 완료 -> {out_path}")


if __name__ == "__main__":
    main()
