"""HWP/HWPX -> Markdown 최종 파이프라인 (Stage1 kordoc 변환 -> Pass1 후보 추출+LLM 계층 판단 ->
Pass2 판단 결과를 markdown 초안에 결정론적으로 적용)을 한 번에 실행한다.

Stage1(라이브러리 변환)·Pass1(후보 추출+LLM 계층 판단)·Pass2(판단 결과의 결정론적 적용) 세 단계를
하나의 명령으로 묶은 최종 구현이다. 설계 배경은 저장소 루트의 README.md·SKILL.md 참고.

## Stage1 라이브러리로 kordoc을 쓰는 이유
`hwp2md`(Go)와 비교 검증한 결과, kordoc은 표 병합 셀(rowspan/colspan)·표 위치·읽기 순서 보존이
더 우수했고, hwp2md는 번호를 헤더(#)로 승격하는 능력이 더 우수했다(원문 스타일 메타데이터가 없으면
kordoc은 헤더를 아예 못 만듦) — 당시엔 상호보완 관계라 Stage1 최종 선택을 미뤄뒀다.
그런데 이 파이프라인은 Stage1의 헤더 승격 결과에 애초에 의존하지 않는다 — Pass1이 Stage1 산출물의
평문 텍스트를 다시 스캔해서 헤더 구조를 자체적으로 판단하기 때문에, hwp2md의 강점은 이 아키텍처에서
의미가 없어진다. 반면 kordoc의 강점(표/순서 보존)은 이후 단계가 복구할 수 없는 손실이라 그대로
남는다. 그래서 kordoc으로 확정.

## 각 단계가 하는 일 (자세한 설계 근거는 pipeline/README.md 참고, LLM은 판단만/복원은 라이브러리만
— 2026-08-21 원칙 확정)
- **Stage1**: `kordoc` CLI(npx)로 원문 추출 + `normalize_html_tables.py`로 raw HTML 표를 GFM
  pipe-table로 정규화. LLM 없음 — 문서 전체를 한 번에 처리(문서 "복원"은 여기서 100% 끝남).
- **Pass1**: `extract_heading_candidates.py`(규칙 기반, recall만)로 "제목일 수 있는 줄"을 뽑고,
  `classify_headings_pass1.py`(LLM)가 그 후보 목록을 보고 "진짜 헤더인지, 어느 레벨인지"만
  **판단**한다(main_section/sub_section/attachment_section/not_heading) — 번호가 이어지는지 같은
  규칙만으로는 판단이 안 되는 걸 이미 여러 번 실증했으므로 이 판단은 LLM이 계속 맡는다. 입력이
  후보 목록(작음)뿐이라 문서가 길어도 비용이 거의 안 늘어남.
- **Pass2**: `pass2_window_reformat.py`가 Pass1의 판단 결과를 kordoc이 만든 markdown 초안에
  그대로 적용한다 — 판단된 위치에 헤더 마크(`##`/`###`)만 삽입하고 **본문은 한 글자도 안 건드림**.
  LLM 호출 없음, 순수 문자열 조작. (예전엔 이 단계도 LLM이 섹션 본문을 통째로 재작성했는데, 그
  과정에서 문단이 통째로 빠지거나 헤더가 누락되는 사고가 반복됐음 — 재작성 자체를 없애서 그 위험군을
  통째로 제거함)

사용법:
    python run_pipeline.py <input.hwp|hwpx|pdf> [--pipeline-root pipeline]
        [--model qwen3.5:9b] [--host http://localhost:11434] [--kordoc-version 4.9.0]
        [--title TITLE] [-o OUTPUT] [--skip-existing-stage1]
        [--backend ollama|openai|groq|gemini] [--api-key KEY] [--base-url URL]
        [--max-candidates-per-call N]

Pass1b(헤더 계층 판단) LLM 호출은 기본이 로컬 Ollama지만, --backend로 OpenAI 호환 API(OpenAI/
Groq/Gemini)로 바꿀 수 있다(자세한 배경은 llm_backend.py 참고) — GPU 없는 환경에서도 쓸 수 있게
하는 확장점.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify_headings_pass1 import classify_and_merge  # noqa: E402
from extract_heading_candidates import extract_candidates  # noqa: E402
from fix_bullet_punctuation import fix_bullet_punctuation, find_issues as find_bullet_issues  # noqa: E402
from llm_backend import LLMBackend, OpenAICompatBackend  # noqa: E402
from pass2_window_reformat import derive_title_from_filename, render_markdown  # noqa: E402
from render_from_kordoc_json import render as render_stage1_from_json  # noqa: E402

_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
_API_KEY_ENV_VARS = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}

STAGE1_DIR = "01-stage1-kordoc"
PASS1_DIR = "02-pass1-heading-candidates-kordoc"
PASS2_DIR = "03-pass2-window-reformat-kordoc"


def run_stage1(input_path: Path, out_md: Path, kordoc_version: str) -> str:
    """kordoc의 `--format json` blocks를 직접 렌더링해서(원본 텍스트박스/콜아웃 경계를
    `<!--box-start/end-->` 마커로 보존, 2026-08-21 확정 — pipeline/README.md 참고) markdown을
    만들고, 말머리 쉼표/마침표 오탈자를 교정해서 out_md에 저장하고 내용을 반환.

    이전엔 kordoc의 markdown 출력을 그대로 쓰고 `normalize_html_tables.py`/`strip_underline_tags.py`
    로 후처리했는데, JSON 렌더러(`render_from_kordoc_json.py`)가 애초에 표/밑줄 태그 없는 깨끗한
    markdown을 만들어내므로 그 두 후처리 단계는 더 이상 필요 없다."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_stage1_from_json(str(input_path), kordoc_version)

    bullet_issues = find_bullet_issues(rendered)
    if bullet_issues:
        print(f"      말머리 쉼표->마침표 오탈자 {len(bullet_issues)}건 교정")
        rendered = fix_bullet_punctuation(rendered)

    out_md.write_text(rendered, encoding="utf-8")
    return rendered


def run_pass1(
    stage1_text: str,
    source_name: str,
    out_json: Path | None,
    model: str | None = None,
    host: str = "http://localhost:11434",
    backend: LLMBackend | None = None,
    max_candidates_per_call: int | None = None,
) -> list[dict]:
    """제목 후보 추출(규칙) + 계층 분류(LLM)를 실행하고, 지정 시 결과를 JSON으로 저장.

    `backend`를 주면 Ollama 대신 그 백엔드(예: OpenAI 호환 API)로 분류한다 —
    `classify_headings_pass1.classify_and_merge()`/`llm_backend.py` 참고. `max_candidates_per_call`은
    그 함수의 같은 이름 인자로 그대로 전달되며, 미지정 시
    `classify_headings_pass1._MAX_CANDIDATES_PER_CALL`(Ollama VRAM 기준 기본값)을 쓴다."""
    candidates = extract_candidates(stage1_text)
    if not candidates:
        return []

    merged = classify_and_merge(candidates, model, host, backend=backend, max_candidates_per_call=max_candidates_per_call)

    counts: dict[str, int] = {}
    for m in merged:
        counts[m["classification"]] = counts.get(m["classification"], 0) + 1
    print(f"      분류 분포: {counts}")

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"source": source_name, "classified": merged}, f, ensure_ascii=False, indent=2)

    return merged


def run_pass2(stage1_text: str, classified: list[dict], title: str) -> str:
    """Pass1 판단 결과를 markdown 초안에 결정론적으로 적용 — LLM 호출 없음."""
    lines = stage1_text.split("\n")
    final = render_markdown(lines, classified, title)
    counts: dict[str, int] = {}
    for c in classified:
        if c.get("classification") in ("main_section", "attachment_section", "sub_section"):
            counts[c["classification"]] = counts.get(c["classification"], 0) + 1
    print(f"      헤더 적용됨: {counts}")
    return final


def _build_backend(args: argparse.Namespace) -> LLMBackend | None:
    """--backend 선택에 따라 LLMBackend 인스턴스를 만든다. ollama(기본)면 None을 반환해서
    run_pass1/classify_and_merge이 기존처럼 model/host로 OllamaBackend를 알아서 만들게 둔다."""
    if args.backend == "ollama":
        return None
    api_key = args.api_key or os.environ.get(_API_KEY_ENV_VARS.get(args.backend, ""))
    if not api_key:
        env_name = _API_KEY_ENV_VARS.get(args.backend, "?")
        raise SystemExit(f"--backend {args.backend}는 --api-key 또는 환경변수 {env_name}가 필요합니다.")
    base_url = args.base_url or _DEFAULT_BASE_URLS.get(args.backend)
    if not base_url:
        raise SystemExit(f"--backend {args.backend}는 --base-url을 직접 지정해야 합니다.")
    return OpenAICompatBackend(model=args.model, api_key=api_key, base_url=base_url)


def _load_dotenv_if_present() -> None:
    """cwd에 .env가 있으면 로드(예: GROQ_API_KEY). python-dotenv 없으면 조용히 건너뜀 —
    --api-key로 직접 줘도 되므로 필수 의존성으로 만들지 않는다."""
    if not Path(".env").exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def main() -> None:
    _load_dotenv_if_present()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="입력 HWP/HWPX/PDF 파일 경로")
    ap.add_argument("--pipeline-root", default="pipeline", help="Stage1/Pass1/Pass2 출력 루트 (기본: pipeline/)")
    ap.add_argument("--model", default="qwen3.5:9b", help="ollama 모델명, 또는 다른 backend일 때 그 제공자의 모델명")
    ap.add_argument("--host", default="http://localhost:11434", help="--backend ollama일 때만 사용")
    ap.add_argument(
        "--backend", choices=["ollama", "openai", "groq", "gemini"], default="ollama",
        help="Pass1b(헤더 계층 판단) LLM 호출 방식. ollama(기본, 로컬) 외에는 --api-key 필요",
    )
    ap.add_argument("--api-key", help="--backend가 ollama가 아닐 때 필요한 API 키 (또는 OPENAI_API_KEY/GROQ_API_KEY/GEMINI_API_KEY 환경변수)")
    ap.add_argument("--base-url", help="OpenAI 호환 엔드포인트 base URL (openai/groq/gemini는 기본값 있음)")
    ap.add_argument(
        "--max-candidates-per-call", type=int, default=None,
        help=(
            "Pass1 LLM 호출 하나당 넘길 헤더 후보 최대 개수 (미지정 시 Ollama VRAM 기준 기본값 사용). "
            "Groq 등 TPM 한도가 낮은 클라우드 백엔드로 --backend를 바꿨을 때 이 값을 낮춰야 할 수 있음"
        ),
    )
    ap.add_argument("--kordoc-version", default="4.9.0")
    ap.add_argument("--title", help="문서 제목 (미지정 시 파일명에서 유도)")
    ap.add_argument("-o", "--output", help="최종 결과 저장 경로 (미지정 시 pipeline-root/03-.../<파일명>.md)")
    ap.add_argument(
        "--skip-existing-stage1",
        action="store_true",
        help="이미 Stage1 결과 파일이 있으면 kordoc 재실행 없이 재사용",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"입력 파일을 찾을 수 없음: {input_path}")

    backend = _build_backend(args)

    base = input_path.name
    root = Path(args.pipeline_root)
    stage1_path = root / STAGE1_DIR / f"{base}.md"
    pass1_path = root / PASS1_DIR / f"{base}.classified.json"
    pass2_path = Path(args.output) if args.output else root / PASS2_DIR / f"{base}.md"

    print(f"[1/3] Stage1 (kordoc {args.kordoc_version}) -> {stage1_path}")
    if args.skip_existing_stage1 and stage1_path.exists():
        stage1_text = stage1_path.read_text(encoding="utf-8")
        print("      기존 결과 재사용 (--skip-existing-stage1)")
    else:
        stage1_text = run_stage1(input_path, stage1_path, args.kordoc_version)
        print(f"      완료 ({len(stage1_text)}자)")

    print(f"[2/3] Pass1 (후보 추출 + LLM 계층 분류) -> {pass1_path}")
    classified = run_pass1(
        stage1_text, str(stage1_path), pass1_path, args.model, args.host, backend=backend,
        max_candidates_per_call=args.max_candidates_per_call,
    )
    if not classified:
        print("      경고: 제목 후보 0개 — 구조 분류 없이 Stage1 결과를 그대로 사용합니다")

    print(f"[3/3] Pass2 (판단 결과를 markdown 초안에 적용) -> {pass2_path}")
    title = args.title or derive_title_from_filename(str(stage1_path))
    # classified가 비어 있거나 헤더로 승격될 후보가 하나도 없어도 render_markdown()은 원문을
    # 그대로 보존한 채(헤더만 없이) frontmatter를 붙여 반환하므로 별도 분기 불필요
    final_md = run_pass2(stage1_text, classified, title)

    pass2_path.parent.mkdir(parents=True, exist_ok=True)
    pass2_path.write_text(final_md, encoding="utf-8")
    print(f"\n완료 -> {pass2_path}")


if __name__ == "__main__":
    main()
