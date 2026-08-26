"""이 스킬은 패키지가 아니라 `scripts/`에 놓인 평범한 스크립트들(정식 pip 설치 없이 stdlib만으로
동작하는 게 설계 원칙, SKILL.md "요구사항" 참고)이라 `scripts/`를 직접 import 경로에 추가한다 —
`run_pipeline.py` 등 스크립트 자체가 쓰는 것과 같은 `sys.path.insert()` 방식."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
