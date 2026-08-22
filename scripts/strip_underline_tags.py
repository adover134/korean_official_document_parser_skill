"""kordoc이 원본 HWP의 밑줄 서식을 그대로 옮긴 `<u>`/`</u>` 태그를 markdown 원문에서 제거한다.

배경: `<u>`는 순수한 서식 정보(강조 표시)일 뿐인데, 이 태그가 헤더 후보 줄의 맨 앞/뒤를 감싸거나
문장 중간에 끼어드는 바람에 여러 문제를 일으켰다(2026-08-21 확인):
  - `extract_heading_candidates.py`의 정규식이 줄 전체 패턴(`^\\*\\*...\\*\\*$` 등)을 기대하는데
    `<u>**12. 안전 및 보건 확보 의무사항 준수**</u>`처럼 태그가 줄 앞뒤를 감싸면 매치가 안 돼서
    후보 추출 자체가 누락됨(한 샘플 문서, "12." 섹션 전체가 후보 목록에서 빠짐)
  - 문장 중간에 `<u>**모두**</u>`처럼 끼어들면 헤더 텍스트가 지저분해짐(다른 샘플 문서)

`<br>`(줄바꿈 의미)과 `**`(볼드, 헤더 후보 판별에 쓰임)는 그대로 둔다 — `<u>`/`</u>` 태그 자체만
제거하고 감싸인 텍스트는 그대로 보존한다(콘텐츠 손실 없음, 서식 정보만 제거).

사용법:
    python strip_underline_tags.py <input.md> [-o <output.md>] [--dry-run]
"""

from __future__ import annotations

import argparse
import re

_UNDERLINE_TAG_RE = re.compile(r"</?u>")


def find_issues(text: str) -> list[dict]:
    issues = []
    for m in _UNDERLINE_TAG_RE.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line_end = len(text) if line_end == -1 else line_end
        line_start = text.rfind("\n", 0, m.start()) + 1
        issues.append({"line": line_no, "tag": m.group(0), "context": text[line_start:line_end][:80]})
    return issues


def strip_underline_tags(text: str) -> str:
    return _UNDERLINE_TAG_RE.sub("", text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="출력 경로 (미지정 시 입력 파일을 덮어씀)")
    ap.add_argument("--dry-run", action="store_true", help="제거하지 않고 발견된 태그만 출력")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    issues = find_issues(text)
    for issue in issues:
        print(f"  L{issue['line']}: {issue['tag']} 제거 : {issue['context']!r}")

    if args.dry_run:
        print(f"발견: {len(issues)}건 (--dry-run, 제거 안 함)")
        return

    fixed = strip_underline_tags(text)
    out_path = args.output or args.input
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fixed)
    print(f"제거 {len(issues)}건 완료 -> {out_path}")


if __name__ == "__main__":
    main()
