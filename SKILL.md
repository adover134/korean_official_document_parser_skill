---
name: hwp-hierarchical-md
description: Converts hierarchical Korean official documents in HWP/HWPX format (government RFPs, bid announcements, contracts, notices) into structurally faithful Markdown — preserving heading levels, tables, and attachment/exhibit boundaries. Use when the user asks to convert a .hwp or .hwpx file to Markdown, especially when the document has a numbered section hierarchy (1./2./3. main sections, sub-items, 붙임/별첨/서식 attachments) that a plain converter would flatten. Requires Node.js (npx); the heading-judgment LLM call defaults to a local Ollama server with a pulled model, or can use an OpenAI-compatible cloud API (OpenAI/Groq/Gemini) via --backend instead.
---

# HWP 계층 구조 Markdown 변환기

한글(HWP/HWPX) 형식의 계층 구조 공문서(입찰공고문, 계약서, 공고 등)를 Markdown으로 변환하되,
원문의 섹션 계층(대분류/중분류/첨부)을 최대한 보존한다.

## 왜 필요한가

기존 HWP→Markdown 변환 라이브러리는 표/서식/읽기 순서는 잘 보존하지만, "이 줄이 최상위 섹션 제목인지,
하위 세부항목인지, 서명란처럼 헤더가 아닌지"를 판단하지 못한다. 원문 스타일 메타데이터(폰트 크기,
볼드 여부)만으로는 계층을 안정적으로 재구성할 수 없다 — 같은 "1. 2. 3." 번호 매김이 문서 최상위
섹션에도, 그 안의 체크리스트에도, 별도 첨부(붙임/서식)에도 반복해서 쓰이기 때문이다.

## 설계: LLM은 판단만, 변환은 라이브러리가

이 스킬은 "문서를 markdown 텍스트로 바꾸는 것"과 "그 텍스트 중 어디가 헤더인지 판단하는 것"을
분리한다.

1. **Stage1 (라이브러리, LLM 없음)** — [`kordoc`](https://www.npmjs.com/package/kordoc)의
   `--format json` 구조화 출력을 직접 조립해 Markdown 초안을 만든다. 원본의 1×1 텍스트박스(콜아웃)
   는 `<!--box-start/end-->` 마커로 경계를 표시해, 이후 단계가 "이건 박스 안에 중첩된 내용"이라는
   정보를 잃지 않게 한다.
2. **Pass1a (규칙 기반, recall만)** — 헤더일 수 있는 줄(볼드 단독 줄, 번호 매김 줄, 괄호로 감싼
   선두 라벨 등)을 폭넓게 후보로 추출한다. 오탐은 허용하고 놓치지 않는 것을 우선한다.
3. **Pass1b (로컬 LLM, 판단만)** — 후보 목록 전체를 로컬 Ollama 모델에 한 번에 보여주고, 각
   후보가 진짜 헤더인지·어느 계층인지(`main_section`/`sub_section`/`attachment_section`/
   `not_heading`)만 판단하게 한다. 본문을 다시 쓰게 하지 않는다.
4. **결정론적 안전망** — LLM이 반복적으로 놓치는 패턴(텍스트박스 안 콘텐츠, 서식/붙임 스코프 내부
   콘텐츠, "~귀하"/"~(인)" 서명란 등)은 이미 확정된 판단에만 반응하는 좁은 범위의 후처리 규칙으로
   보정한다. 전역 규칙(예: "번호가 이어지면 최상위")은 의도적으로 쓰지 않는다 — 붙임/별첨에서
   번호가 정당하게 리셋되는 경우를 오탐하기 때문이다.
5. **Pass2 (결정론적 적용, LLM 없음)** — Pass1의 판단 결과를 Stage1 Markdown 초안에 헤더 마크만
   삽입한다. 본문은 한 글자도 다시 쓰지 않는다.

이 분리 덕분에 LLM이 문단을 통째로 빠뜨리거나 없는 헤더를 만들어내는 위험이 구조적으로 사라진다 —
LLM 호출은 오직 "이 후보가 헤더인가?"라는 좁은 판단에만 쓰인다.

## 요구사항

- Node.js (`npx`로 `kordoc`을 실행 — 별도 설치 불필요, 최초 실행 시 자동 다운로드)
- Pass1b(헤더 계층 판단) LLM 호출 — 아래 둘 중 하나:
  - 로컬에서 실행 중인 [Ollama](https://ollama.com) 서버 + 풀받은 모델(기본값 `qwen3.5:9b`,
    `--model`로 변경 가능) — 기본값, GPU 필요
  - OpenAI 호환 클라우드 API(OpenAI/Groq/Gemini) — `--backend`로 선택, API 키 필요. GPU 없이도
    쓸 수 있고, Groq는 무료 티어로도 동작 확인함(`scripts/llm_backend.py` 참고)
- Python 3.10+ (표준 라이브러리만 사용, pip 설치 불필요 — `.env` 자동 로드는 `python-dotenv`가
  있으면 쓰고 없으면 조용히 건너뜀)

## 사용법

```bash
python scripts/run_pipeline.py <입력.hwp|hwpx> [옵션]
```

주요 옵션:
- `--pipeline-root DIR` — Stage1/Pass1/Pass2 중간 산출물 저장 위치 (기본: `pipeline/`)
- `--model NAME` — Ollama 모델명, 또는 다른 backend일 때 그 제공자의 모델명 (기본: `qwen3.5:9b`)
- `--host URL` — Ollama 서버 주소 (기본: `http://localhost:11434`, `--backend ollama`일 때만 사용)
- `--backend {ollama,openai,groq,gemini}` — Pass1b LLM 호출 방식 (기본: `ollama`)
- `--api-key KEY` — `--backend`가 ollama가 아닐 때 필요 (또는 `OPENAI_API_KEY`/`GROQ_API_KEY`/
  `GEMINI_API_KEY` 환경변수, 또는 cwd의 `.env`)
- `--base-url URL` — OpenAI 호환 엔드포인트 base URL (openai/groq/gemini는 기본값 있음)
- `--max-candidates-per-call N` — Pass1b LLM 호출 하나당 넘길 헤더 후보 최대 개수 (기본: Ollama
  VRAM 기준 70). Groq 등 TPM 한도가 낮은 백엔드는 이 값을 낮춰야 할 수 있음 — 아래 "한계" 참고
- `-o, --output PATH` — 최종 Markdown 저장 경로
- `--skip-existing-stage1` — 이미 Stage1 결과가 있으면 kordoc 재실행 없이 재사용

Claude가 이 스킬을 트리거하는 경우, 사용자가 지정한 `.hwp`/`.hwpx` 파일 경로를 `run_pipeline.py`에
그대로 전달하고, 결과 Markdown 경로를 사용자에게 보고한다. `--backend ollama`(기본)인데 Ollama가
꺼져 있거나 대상 모델이 없으면 `ollama serve` 및 `ollama pull <model>` 안내를 먼저 제공한다.

## 한계

- 8GB급 VRAM 환경에서는 문서당 후보가 많을 경우(체크리스트/표가 많은 문서) 자동으로 70개 단위로
  분할 호출한다 — 컨텍스트 제한이 다른 환경(또는 Groq처럼 TPM 한도가 낮은 클라우드 백엔드)이면
  소스 수정 없이 `--max-candidates-per-call`로 조정.
- 결정론적 안전망은 실제로 관찰된 실패 패턴(한국 정부/공공기관 입찰공고문 corpus 기준)에서 도출된
  것이라, 완전히 다른 장르의 문서에서는 재조정이 필요할 수 있다.
