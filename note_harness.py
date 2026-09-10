"""note_harness.py — 패스 오케스트레이션(Plan → Synthesis → Gates → Critic → Repair).

계약: `.handoff/rounds/round-04-harness-sweep-contract.md` §4(T2) ·
`.handoff/rounds/round-05-pipe-harness-production-contract.md` §3(T1 —
generate_fn/critic_fn 팩토리 승격 + truncation 가드) · 설계:
`docs/05-note-harness-design.md` §2(파이프라인 ①~⑥).

이 모듈은 모델 무관이다 — `generate_fn`/`critic_fn` 콜러블을 주입받을 뿐,
provider(free_llm.py 등) 구체 타입을 임포트하지 않는다(계약 §4 "모델 무관"
원칙). 호출자(예: `poc/run_sweep.py`, `note_pipe.py`)가 free_llm.generate()가
반환하는 dict-shaped 응답을 만들어 넘긴다.

`make_generate_fn`/`make_free_critic_fn`(모듈 하단)은 round-05 계약 §3
P0-1로 이 모듈에 승격된 공용 팩토리다 — **client를 duck-type 파라미터로만
받는다**(`generate(prompt, *, system_prompt, stream=True) -> dict` 형태를
만족하는 아무 객체나 가능). 이 파일은 어떤 이유로도 `free_llm`을 임포트하지
않는다 — 위반 시 모델-무관 경계가 깨져 G5 블로커다.

## 공개 API

    run_harness(
        transcript: str,
        generate_fn: Callable[..., dict],
        *,
        plan: bool = False,
        critic_fn: Callable[[str, str], dict] | None = None,
        repair_budget: int = 1,
        initial_note: str | None = None,
        prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    ) -> HarnessResult

- `generate_fn(prompt: str, *, system_prompt: str) -> dict` — free_llm.generate()
  형태 dict(최소 `text` 키. `usage`/`finish_reason`/`truncated_suspected` 등은
  있으면 토큰 집계·경고에 활용, 없어도 동작).
- `critic_fn(transcript: str, note: str) -> dict` — `{"findings": [...],
  "usage": dict|None, "source": str}`. None이면 critic 패스를 건너뛴다.
- `initial_note`가 주어지면 plan+synthesis를 건너뛰고 게이트부터 시작한다
  (V0+G/V2/V3처럼 기존 노트에서 파생하는 변형용, 계약 §5).

## 파이프라인 (docs/05 §2 요약)

Plan(토글) → Synthesis(생략 시 initial_note로 대체) → 게이트(항상, 0토큰)
→ Critic(토글) → 조건부 Repair(게이트/critic이 트리거성 finding을 냈을 때만,
예산 `repair_budget`회) → 재게이트 → 품질메타 조립.

**Repair 트리거 규칙(계약 §4)**: `ts_removed`·`completeness`·`structure`
kind의 finding, 또는 critic finding이 있으면 repair 발화. `ts_normalized`·
`uncertain_ts` 단독으로는 트리거하지 않는다 — 정규화는 이미 노트 본문에
적용된 상태이므로 재작성이 필요 없다.

**메시지 배열 규약(캐싱 친화, docs/05 §2)**: 모든 LLM 패스의 user 메시지는
전사를 동일 바이트 prefix로 시작한다 — `전사 → (패스별 추가)` 순서.

**Critic 재실행 없음**: repair 이후 게이트는 재실행하지만 critic은 1회만
호출한다(repair가 critic finding을 신뢰하고 고치고, 게이트가 재검증한다).

**품질메타 헤더**: `render_note_with_meta()`가 단일 선두 HTML 주석 블록으로
조립한다 — `poc/prepare_blind_set.strip_provider_traces`의 `count=1` 선두
블록 스트립과 호환(계약 §4 P1-4).
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from note_validate import Finding, run_gates

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = REPO_ROOT / "prompts"
TEXT_PROMPTS_DIR = DEFAULT_PROMPTS_DIR / "text"

_SYNTHESIS_PROMPT_NAME = "note_synthesis.md"
_PLAN_PROMPT_NAME = "harness/plan.md"
_SYNTHESIS_WITH_PLAN_PROMPT_NAME = "harness/synthesis_with_plan.md"
_CRITIC_PROMPT_NAME = "harness/critic.md"
_REPAIR_PROMPT_NAME = "harness/repair.md"

#: JSON 파싱 전 방어적으로 벗겨낼 코드펜스(```json ... ``` 또는 ``` ... ```).
_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n(?P<body>.*?)\n```\s*$", re.DOTALL | re.IGNORECASE
)

#: repair를 트리거하는 finding kind(계약 §4). ts_normalized/uncertain_ts는
#: 제외 — 정규화는 이미 노트 본문에 반영됐고, 불확실 마커는 값을 지어낼 수
#: 없어 repair로 고칠 수 있는 종류가 아니다.
_REPAIR_TRIGGER_GATE_KINDS = frozenset(
    {"ts_removed", "completeness", "structure", "core_artifact", "source_url", "source_enumeration"}
)

#: usage가 없을 때 문자수 기반 추정 비율(계약 §3 "chars/4 폴백").
_ESTIMATE_CHARS_PER_TOKEN = 4

#: critic usage를 프리미엄 버킷으로 분류하는 source 접두어(계약 §4 V3 2단계
#: 실행 — agent_report:opus 같은 source 문자열).
_PREMIUM_SOURCE_PREFIXES = ("premium", "opus", "agent_report")

# text_post candidate prompt가 첫 줄에 내보내는 비렌더 계측 표식. 합성 본문에서
# 즉시 제거하고 HarnessResult.meta로 옮겨 단일 선두 HTML 메타 블록 계약을 지킨다.
_SOURCE_BLOCK_DENSITY_RE = re.compile(
    r"^\s*<!--\s*source-block-density:\s*"
    r"body=(?P<body>D[123]|NA);\s*"
    r"author_comments=(?P<author_comments>D[123]|NA);\s*"
    r"ocr=(?P<ocr>D[123]|NA)\s*-->\s*",
    re.IGNORECASE,
)

_COVERAGE_TOKEN_RE = re.compile(r"[가-힣A-Za-z]+|\d+(?:[.,]\d+)*")

# P0-16 grammar lens. korean-grammar-checker의 SKILL/rules/common-errors에서
# 기계 판정 가능한 규칙만 옮겼다. 세 번째 필드는 선택지형 확신도이며, 어느
# finding도 노트를 수정하거나 verified/gate 값을 바꾸지 않는다.
_GRAMMAR_RULES: tuple[
    tuple[str, re.Pattern[str], str, str, str], ...
] = (
    (
        "duplicated-particle",
        # '이가/가이'는 '차이가/가이드' 같은 정상 어절 내부에서도 생겨 제외한다.
        re.compile(r"(?:은는|는은|을를|를을|과와|와과|으로로|에서에서)"),
        "✅ 확실한 오류",
        "서로 다른 조사 또는 같은 조사가 연속되어 문장 성분 연결이 어색함",
        "korean-grammar-checker/common-errors:조사 사용 오류",
    ),
    (
        "duplicated-ending",
        re.compile(r"(?:합니다|됩니다|있습니다|없습니다){2,}"),
        "✅ 확실한 오류",
        "서술 종결어미가 붙어서 반복됨",
        "korean-grammar-checker/rules:문법 구조 규칙",
    ),
    ("spelling-doeyo", re.compile(r"되요"), "✅ 확실한 오류", "'되어요'의 준말은 '돼요'임", "korean-grammar-checker/common-errors:되/돼"),
    ("spelling-dwaet", re.compile(r"됬(?:어|다|습니다|어요)"), "✅ 확실한 오류", "'되었-'의 준말은 '됐-'임", "korean-grammar-checker/common-errors:되/돼"),
    ("spelling-saisiot-examples", re.compile(r"(?<![가-힣])(?:나무잎|차잔)(?=$|[\s,.;:!?\)\]]|[은는이가을를도과와])"), "✅ 확실한 오류", "문서 예시의 합성어에는 사이시옷을 써서 '나뭇잎·찻잔'으로 적음", "korean-grammar-checker/rules:사이시옷"),
    ("spelling-adjective-neunji", re.compile(r"(?:좋|많)는지"), "✅ 확실한 오류", "형용사 '좋다·많다'에는 '-ㄴ지'가 결합함", "korean-grammar-checker/common-errors:-ㄴ지/-는지"),
    ("spelling-lge", re.compile(r"하를게(?:요)?"), "✅ 확실한 오류", "의지·약속의 어미는 '-ㄹ게'이므로 '할게(요)'로 적음", "korean-grammar-checker/common-errors:-ㄹ게/-를게"),
    ("spelling-an-anh", re.compile(r"(?:(?:하지|좋지|먹지|가지)\s+안(?:다|아요|습니다)|않\s+(?:해|가)(?:요)?|좋지\s+안아요)"), "✅ 확실한 오류", "부사 '안'과 보조 용언 '-지 않다'를 구분해야 함", "korean-grammar-checker/common-errors:안/않"),
    ("spelling-wen-waen", re.compile(r"(?<![가-힣])(?:왠(?!지)|웬지)"), "✅ 확실한 오류", "'왠'은 거의 '왠지'에만 쓰고, 그 밖에는 '웬'을 씀", "korean-grammar-checker/common-errors:웬/왠"),
    ("spacing-dependent-su", re.compile(r"[가-힣]+(?:ㄹ|을)수(?:있|없)"), "✅ 확실한 오류", "의존 명사 '수'는 앞말과 띄어 씀", "korean-grammar-checker/rules:의존명사"),
    ("spacing-dependent-geot", re.compile(r"(?:하는|되는|좋은|갈|볼)것(?:은|이|을|이다|입니다)?"), "✅ 확실한 오류", "의존 명사 '것'은 앞말과 띄어 씀", "korean-grammar-checker/rules:의존명사"),
    ("spacing-dependent-mankeum", re.compile(r"(?:할|먹을|될)만큼"), "✅ 확실한 오류", "의존 명사 '만큼'은 앞말과 띄어 씀", "korean-grammar-checker/rules:의존명사"),
    ("spacing-dependent-ppun", re.compile(r"(?:갈|할|볼)뿐(?:이다|입니다|이었다)?"), "✅ 확실한 오류", "의존 명사 '뿐'은 앞말과 띄어 씀", "korean-grammar-checker/rules:의존명사"),
    ("spacing-dependent-jul", re.compile(r"(?:할|먹을|갈)줄\s*(?:알|모르)"), "✅ 확실한 오류", "의존 명사 '줄'은 앞말과 띄어 씀", "korean-grammar-checker/common-errors:의존명사"),
    ("particle-eul-reul", re.compile(r"(?:사과을|차을|책를|밥를)"), "✅ 확실한 오류", "받침 유무에 맞지 않는 목적격 조사가 쓰임", "korean-grammar-checker/common-errors:-을/를"),
    ("particle-i-ga", re.compile(r"(?:사과이|책가|밥가)"), "✅ 확실한 오류", "받침 유무에 맞지 않는 주격 조사가 쓰임", "korean-grammar-checker/common-errors:-이/가"),
    ("particle-eun-neun", re.compile(r"(?:사과은|책는|밥는)"), "✅ 확실한 오류", "받침 유무에 맞지 않는 보조사가 쓰임", "korean-grammar-checker/common-errors:-은/는"),
    ("particle-wa-gwa", re.compile(r"(?:사과과|책와|밥와)"), "✅ 확실한 오류", "받침 유무에 맞지 않는 접속 조사가 쓰임", "korean-grammar-checker/common-errors:-와/과"),
    ("ending-eupnida", re.compile(r"(?:먹읍니다|있읍니다)"), "✅ 확실한 오류", "현대 표준어 종결형은 '-습니다'임", "korean-grammar-checker/common-errors:-ㅂ니다/습니다"),
    ("ending-a-eoyo-examples", re.compile(r"(?:가아요|서어요|하아요|오아요)"), "✅ 확실한 오류", "문서에 명시된 '-아요/-어요' 축약 활용이 잘못됨", "korean-grammar-checker/common-errors:-아요/-어요"),
    # 문서 예시 중 '그 가'는 정상 구절 '그 가설/가치'를 오탐하므로 제외한다.
    ("spacing-particle-examples", re.compile(r"(?:나\s+는|너\s+를|집\s+에서)"), "✅ 확실한 오류", "조사는 앞말에 붙여 씀", "korean-grammar-checker/rules:조사 띄어쓰기"),
    ("spacing-adverb-examples", re.compile(r"(?:매우좋다|너무나빨리|정말로좋다)"), "✅ 확실한 오류", "부사는 뒤의 용언과 띄어 씀", "korean-grammar-checker/common-errors:부사 오류"),
    ("subject-predicate-accord", re.compile(r"(?:장점|단점|특징|문제점|핵심)(?:은|는)\s+[^.!?\n]{2,80}(?:[가-힣]+\s+수\s+있(?:다|습니다)|필요(?:하|합니)다)"), "⚠️ 권장", "명사형 주어와 가능·필요 서술어의 호응을 확인해야 함", "korean-grammar-checker/rules:주어와 서술어 호응"),
    ("spacing-auxiliary-juda", re.compile(r"[가-힣]+(?:해|하여|어|아)주세요"), "⚠️ 권장", "격식체에서는 보조 용언 '주다'를 띄어 쓰는 표기를 권장함", "korean-grammar-checker/rules:보조용언"),
    ("punctuation-exclamation", re.compile(r"!{2,}"), "💡 제안", "격식 문서에서는 느낌표 반복을 줄일 수 있음", "korean-grammar-checker/rules:느낌표"),
    ("punctuation-subject-comma", re.compile(r"^(?:나는|저는|우리는|이것은|이 기능은),"), "💡 제안", "한국어의 짧은 주제부 뒤 쉼표는 대개 생략 가능함", "korean-grammar-checker/common-errors:과도한 쉼표"),
)

# 형태·의미·문체 판단 없이는 안전하게 확정할 수 없는 grammar 규칙. 새 추론을
# 호출하지 않고, 이미 실행된 critic이 lens와 이 ID를 명시한 경우에만 결속한다.
_GRAMMAR_CONTEXT_RULE_LEVELS = {
    "grammar-saisiot-context": "✅ 확실한 오류",
    "grammar-contracted-form": "✅ 확실한 오류",
    "grammar-foreign-word": "⚠️ 권장",
    "grammar-duum-law": "✅ 확실한 오류",
    "grammar-nji-ending": "✅ 확실한 오류",
    "grammar-deon-deun": "✅ 확실한 오류",
    "grammar-rosseo-roseo": "✅ 확실한 오류",
    "grammar-an-anh": "✅ 확실한 오류",
    "grammar-eotteoke-eotteokhae": "✅ 확실한 오류",
    "grammar-dependent-noun": "✅ 확실한 오류",
    "grammar-auxiliary-verb": "⚠️ 권장",
    "grammar-unit-noun": "⚠️ 권장",
    "grammar-compound-word": "⚠️ 권장",
    "grammar-particle-spacing": "✅ 확실한 오류",
    "grammar-tense-agreement": "✅ 확실한 오류",
    "grammar-subject-predicate-agreement": "⚠️ 권장",
    "grammar-double-negative": "⚠️ 권장",
    "grammar-particle-selection": "✅ 확실한 오류",
    "grammar-honorific-consistency": "⚠️ 권장",
    "grammar-ending-selection": "✅ 확실한 오류",
    "grammar-punctuation-comma": "💡 제안",
    "grammar-punctuation-period": "💡 제안",
    "grammar-punctuation-question": "💡 제안",
    "grammar-punctuation-quotes": "✅ 확실한 오류",
    "grammar-punctuation-middle-dot": "⚠️ 권장",
}
_GRAMMAR_CONTEXT_RULE_IDS = frozenset(_GRAMMAR_CONTEXT_RULE_LEVELS)

# im-not-ai/humanize-korean ai-tell-taxonomy v2.0의 active pattern ID 70개를
# 그대로 보존한다(A-17은 정본 자체가 hold라 제외). 아래 정규식/metric으로
# 안전하게 판정할 수 없는 ID는 _HUMANIZER_CONTEXT_RULE_IDS에 남겨 기존 critic
# 결과가 명시적으로 해당 lens/rule을 반환할 때만 보조 결속한다.
_HUMANIZER_RULE_CATALOG: dict[str, tuple[str, ...]] = {
    "A": tuple(f"A-{i}" for i in (*range(1, 17), 18, 19)),
    "B": tuple(f"B-{i}" for i in range(1, 5)),
    "C": tuple(f"C-{i}" for i in range(1, 13)),
    "D": tuple(f"D-{i}" for i in range(1, 8)),
    "E": tuple(f"E-{i}" for i in range(1, 8)),
    "F": tuple(f"F-{i}" for i in range(1, 6)),
    "G": tuple(f"G-{i}" for i in range(1, 4)),
    "H": tuple(f"H-{i}" for i in range(1, 5)),
    "I": tuple(f"I-{i}" for i in range(1, 7)),
    "J": tuple(f"J-{i}" for i in range(1, 5)),
}
_HUMANIZER_CONTEXT_RULE_IDS = frozenset(
    {"A-13", "A-15", "A-18", "B-3", "C-4", "C-6", "C-8", "D-5", "E-6", "E-7", "F-2", "J-2", "J-4"}
)

# rule_id, pattern, 최소 출현 수, taxonomy severity, reason. 문서 단위 임계는
# 정본의 threshold를 보수적으로 따른다. DOTALL 패턴도 finding 위치를 남긴다.
_HUMANIZER_REGEX_RULES: tuple[
    tuple[str, re.Pattern[str], int, str, str], ...
] = (
    ("A-1", re.compile(r"에\s*대해(?:서|서는|서도)?"), 1, "S1", "'~에 대해' 번역투"),
    ("A-2", re.compile(r"[을를]\s*통해(?:서|서는|서도)?"), 1, "S1", "'~를 통해' 번역투"),
    ("A-3", re.compile(r"에\s*있어(?:서)?"), 1, "S1", "'~에 있어서' 번역투"),
    ("A-4", re.compile(r"(?:다는|라는)\s*점에서"), 3, "S2", "'~라는 점에서' 반복"),
    ("A-5", re.compile(r"(?:와|과)\s*관련(?:하여|된)"), 3, "S2", "'~와 관련하여' 반복"),
    ("A-6", re.compile(r"(?:에\s*기반하여|을\s*바탕으로)"), 3, "S2", "근거 표현 번역투 반복"),
    ("A-7", re.compile(r"(?:가지고\s*있|회의를\s*가졌|결정을\s*만들었)"), 1, "S1", "have/make light-verb 직역"),
    ("A-8", re.compile(r"(?:되어진|하여진|지게\s*된다)"), 1, "S1", "이중 피동 표현"),
    ("A-9", re.compile(r"에\s*의해"), 3, "S2", "영어식 by-passive 반복"),
    ("A-10", re.compile(r"(?:할|될|볼|높일|줄일)\s*수\s*있"), 3, "S2", "가능형 서술 반복"),
    ("A-11", re.compile(r"[을를]\s*위해"), 3, "S2", "목적절 '~을 위해' 반복"),
    ("A-12", re.compile(r"(?:만들어지|이루어지)"), 3, "S2", "자동 피동 반복"),
    ("A-14", re.compile(r"^\s*그리고\b", re.MULTILINE), 3, "S2", "문두 '그리고' 반복"),
    ("A-16", re.compile(r"(?:그는|그녀는|그들은|그의|그녀의)"), 3, "S1", "영어 대명사 직역 밀도"),
    ("A-19", re.compile(r"(?:에서의|에로의|으로의|에의|으로부터의|로부터의)"), 3, "S2", "이중 조사 결합 반복"),
    ("B-1", re.compile(r"[가-힣]{2,}\s*\([A-Za-z][A-Za-z0-9 -]{1,24}\)"), 3, "S2", "영어 괄호 병기 반복"),
    ("B-2", re.compile(r"\b(?:framework|leverage|seamless|robust|scalable|holistic)\b", re.IGNORECASE), 1, "S2", "번역 가능한 영어 용어 잔존"),
    ("B-4", re.compile(r"(?:라고\s*알려진|로\s*일컬어지는)"), 1, "S3", "known-as 직역"),
    ("C-1", re.compile(r"첫째[,.\s].*?둘째[,.\s].*?셋째[,.\s]", re.DOTALL), 1, "S1", "기계적 3단 병렬 열거"),
    ("C-2", re.compile(r"^(?:\s*[-*+]\s+[^\n]+\n){3,}", re.MULTILINE), 1, "S2", "연속 불릿 블록"),
    ("C-3", re.compile(r"^#{2,}\s*(?:도입|서론|본론|결론)\s*$", re.MULTILINE), 2, "S2", "도식적 섹션 헤딩 반복"),
    ("C-5", re.compile(r"[✅🚀💡⚠️📊❌👇]"), 3, "S1", "이모지 장식 반복"),
    ("C-7", re.compile(r"^\s*(?:먼저|반면|결국|첫째|둘째|마지막으로)\b", re.MULTILINE), 3, "S2", "기계적 문단 전환 3단 공식"),
    ("C-9", re.compile(r"\b1\).*?\b2\).*?\b3\)", re.DOTALL), 1, "S2", "숫자 괄호 3단 인덱싱"),
    ("C-10", re.compile(r"^#{2,}\s+[^\n:]{1,40}:\s*[^\n]+$", re.MULTILINE), 2, "S2", "콜론 부제 헤딩 반복"),
    ("C-11", re.compile(r"(?:고|며|지만|면서|아서|어서|는데),"), 3, "S2", "연결어미 직후 쉼표 반복"),
    ("D-1", re.compile(r"(?:결론적으로|요약하면|종합하면|정리하자면|라고\s*(?:할|볼)\s*수\s*있다|라\s*하겠다|에\s*다름\s*아니다|이를\s*통해|그러므로)"), 1, "S1", "AI 결산·요약 상투구"),
    ("D-2", re.compile(r"(?:매우\s*중요하다|반드시\s*기억해야|시사하는\s*바가\s*크|주목할\s*만하|간과할\s*수\s*없|무시할\s*수\s*없|지평을\s*연|방점을\s*찍)"), 1, "S1", "의의·중요성 과장"),
    ("D-3", re.compile(r"(?:크게\s*세\s*가지로|다음과\s*같은\s*특징|다음과\s*같이\s*요약)"), 1, "S1", "열거 도입 상투구"),
    ("D-4", re.compile(r"(?:혁신적인|획기적인|전례\s*없는|압도적|막강한|폭발적|파격적|대대적|새로운\s*장을\s*열|시대가\s*도래)"), 1, "S1", "AI hype 어휘"),
    ("D-6", re.compile(r"(?:해야\s*할\s*때입니다|나아갈\s*시점입니다|할\s*순간입니다)"), 1, "S2", "완결 공식형 결말"),
    ("D-7", re.compile(r"(?:['\"가-힣A-Za-z ]{2,30}에서\s*['\"가-힣A-Za-z ]{2,30}(?:으)?로|['\"가-힣A-Za-z ]{2,30}(?:을|를)\s*넘어\s*['\"가-힣A-Za-z ]{2,30}(?:으)?로)"), 2, "S2", "'X에서 Y로' 변환 공식 반복"),
    ("F-1", re.compile(r"\b(?:매우|정말|진짜로|대단히|극히)\b"), 3, "S2", "정도부사 반복"),
    ("F-3", re.compile(r"(?:역할과\s*기능|의미와\s*가치|로서의\s*역할과\s*기능)"), 1, "S2", "기능·역할 중복구"),
    ("F-4", re.compile(r"[가-힣A-Za-z]+(?:성|적|화)\b"), 13, "S2", "명사화 접미사 밀도"),
    ("F-5", re.compile(r"[가-힣A-Za-z]+적\s+[가-힣A-Za-z]+"), 3, "S2", "'~적 N' 추상어 체인"),
    ("G-1", re.compile(r"(?:할\s*수\s*있을\s*것으로\s*보인다|인\s*것으로\s*판단된다|라고\s*여겨진다|인\s*듯하다)"), 1, "S2", "추측·관측형 종결"),
    ("G-2", re.compile(r"(?:가능성이\s*있을\s*수\s*있|보여질\s*수\s*있)"), 1, "S2", "이중·삼중 완곡"),
    ("G-3", re.compile(r"(?:양쪽\s*모두|두\s*가지\s*모두|장점도\s*있지만|신중하게|균형)"), 5, "S2", "안전 균형 lexicon 밀도"),
    ("H-1", re.compile(r"^\s*(?:또한|따라서|즉|나아가|아울러|게다가|더욱이)\b", re.MULTILINE), 3, "S2", "문두 접속사 반복"),
    ("H-2", re.compile(r"^\s*(?:하지만|그러나)\b", re.MULTILINE), 3, "S2", "역접 접속사 반복"),
    ("H-3", re.compile(r"(?:이는\s+[^.!?\n]{1,40}(?:의미|뜻)|이\s*점에서|이\s*관점에서|이\s*말은)"), 2, "S2", "지시·메타 진입 반복"),
    ("H-4", re.compile(r"(?:^|[.!?]\s*)즉\b"), 3, "S2", "재정의 접속사 '즉' 반복"),
    ("I-1", re.compile(r"(?:것이다|것입니다)"), 3, "S2", "'것이다' 종결 반복"),
    ("I-2", re.compile(r"(?:주목할\s*점은|나아갈\s*바는|할\s*수가\s*있|하는\s*데에|는\s*점에\s*있)"), 2, "S2", "형식·의존 명사 반복"),
    ("I-3", re.compile(r"(?:다는\s*것이다|라는\s*것이다|다는\s*뜻이다|다는\s*점이다)"), 3, "S2", "형식명사 결산 반복"),
    ("I-4", re.compile(r"(?:할\s*필요가\s*있|해야\s*(?:한다|합니다))"), 6, "S2", "권고형 결말 반복"),
    ("I-5", re.compile(r"[가-힣A-Za-z]+(?:이|가)\s*필요하다"), 3, "S2", "추상적 필요 서술 반복"),
    ("I-6", re.compile(r"[가-힣A-Za-z]+\s*능력"), 3, "S2", "'~능력' 추상명사 연쇄"),
    ("J-1", re.compile(r"\*\*[^*\n]+\*\*"), 6, "S2", "본문 볼드 장식 과다"),
    ("J-3", re.compile(r"—"), 3, "S2", "em-dash 장식 반복"),
)
_HUMANIZER_METRIC_RULE_IDS = frozenset({"C-12", "E-1", "E-2", "E-3", "E-4", "E-5"})
_HUMANIZER_ACTIVE_RULE_IDS = frozenset(
    rule_id for ids in _HUMANIZER_RULE_CATALOG.values() for rule_id in ids
)
_HUMANIZER_EXECUTABLE_RULE_IDS = frozenset(
    rule_id for rule_id, *_ in _HUMANIZER_REGEX_RULES
) | _HUMANIZER_METRIC_RULE_IDS
assert _HUMANIZER_ACTIVE_RULE_IDS == (
    _HUMANIZER_EXECUTABLE_RULE_IDS | _HUMANIZER_CONTEXT_RULE_IDS
), "humanizer taxonomy 이식 목록에 누락/중복 ID가 있습니다"


class HarnessError(RuntimeError):
    """하네스 패스가 복구 불가능하게 실패했을 때 발생(예: plan JSON 파싱 실패
    1회 재시도 후에도 실패). 호출자(스윕 러너)가 변형 단위로 catch한다."""


class TruncationSuspectedError(HarnessError):
    """생성/critic 응답이 잘린 정황(truncated_suspected=True)으로 하드-스톱될 때
    발생(round-05 계약 §3 P0-7 truncation 가드, FIX-4).

    `HarnessError`의 서브클래스이므로 기존 `except HarnessError` 호출자는
    변경 없이 계속 catch한다 — 다만 호출자가 원인을 구분하고 싶을 때(예:
    note_pipe.py가 TruncatedNoteError vs NotePipeError로 정직하게 라벨링)는
    이 구체 타입으로 먼저 catch할 수 있다. `make_generate_fn`/
    `make_free_critic_fn`의 truncation 하드-스톱이 이 예외를 raise한다 —
    plan/critic JSON 파싱 실패 같은 non-truncation 실패는 여전히 base
    `HarnessError`로 남아, "truncated"라는 단어가 진짜 truncation이 아닌
    실패에도 붙는 부정직한 라벨링을 피한다."""


@dataclass(frozen=True)
class PassUsage:
    """패스 1회의 토큰 사용량 기록.

    Attributes:
        pass_name: "plan" | "synthesis" | "critic" | "repair" 등.
        usage: provider가 보고한 usage dict(원본 그대로) 또는 None.
        estimated: True면 usage가 chars/4 추정치(estimated_tokens)로 대체됨.
        estimated_tokens: estimated=True일 때만 의미 있는 추정 토큰 수.
        premium: True면 프리미엄(유료) 토큰 버킷, False면 무료 버킷.
    """

    pass_name: str
    usage: dict | None
    estimated: bool
    estimated_tokens: int | None
    premium: bool = False
    role: str = "writer"
    requested_provider: str | None = None
    requested_model: str | None = None
    actual_provider: str | None = None
    actual_model: str | None = None
    billing_tier: str = "unknown"


class EditKind(str, Enum):
    """P0-8의 허용 편집 단위. GRPO/학습 상태는 의도적으로 포함하지 않는다."""

    CREATE = "Create"
    UPDATE = "Update"
    MERGE = "Merge"
    PRUNE = "Prune"
    NOOP = "Noop"


@dataclass(frozen=True)
class EditEvaluationArtifact:
    """같은 task와 dual-frontier gold로 채점된 편집 전/후 artifact 요약."""

    artifact_id: str
    task_id: str
    gold_frontier_ids: tuple[str, str]
    fabrication_count: int
    coverage: float
    coverage_input_sha256: str


@dataclass(frozen=True)
class EditRewardDecision:
    """P0-8 rollback reward의 재현 가능한 판정."""

    edit_id: str
    edit_kind: EditKind
    before_artifact_id: str
    after_artifact_id: str
    reward_vector: tuple[int, float]
    regeneration_band: float
    disposition: str
    rollback: bool
    dead_instruction: bool = False
    reason: str = ""


@dataclass(frozen=True)
class CopyQualityFinding:
    """P0-16 세 렌즈 중 하나의 report-only finding."""

    lens: str
    rule_id: str
    level: str
    line: int
    excerpt: str
    reason: str
    source_ref: str


@dataclass(frozen=True)
class SourceClaimMapping:
    """노트 claim과 근거가 겹친 source block의 결정론적 매핑."""

    claim_id: str
    line: int
    claim: str
    source_block_ids: tuple[str, ...]
    matched_tokens: tuple[str, ...]


@dataclass
class HarnessResult:
    """`run_harness`의 반환값.

    Attributes:
        note: 최종 노트 본문(품질메타 헤더 미포함 — `render_note_with_meta`가
            별도로 헤더를 붙인다).
        meta: 품질메타 요약 dict(게이트 meta + passes_run + verified 등).
        findings: 최종(재게이트 이후) findings 리스트.
        tokens: 토큰 집계 dict — `{"passes": [...], "total_known": int,
            "free_total": int, "premium_total": int}`.
        verified: True면 남은 트리거성 finding이 없음(repair 예산 소진 후에도
            남아 있으면 False).
        passes_run: 실행된 패스 이름 순서 리스트(예: ["plan", "synthesis",
            "gate", "critic", "repair", "gate"]).
    """

    note: str
    meta: dict[str, object]
    findings: list[Finding]
    tokens: dict[str, object]
    verified: bool
    passes_run: list[str] = field(default_factory=list)


def _strip_fence(text: str) -> str:
    """방어적 코드펜스 스트립. 펜스가 없으면 원문을 그대로 strip해 반환."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match["body"].strip()
    return stripped


def extract_json_object(text: str) -> str:
    """모델 응답에서 최상위 JSON 객체만 꺼낸다.

    코드펜스로 감싸는 모델이 있고, 지시문을 복창한 뒤 JSON을 붙이는 모델도 있다
    (2026-09-05 실측: 같은 gemma-4-31b가 NIM 경유로는 ```json 펜스를, Gemini API
    직결로는 "* Task: Grounding Critic..." 서두를 앞에 달았다). 펜스만 벗기면
    후자가 `line 1 column 1`로 실패해, 유효한 JSON을 낸 모델이 능력 미달로
    잘못 판정된다.

    중괄호 균형을 세어 첫 객체의 끝을 찾는다. 문자열 안의 중괄호와 이스케이프는
    세지 않는다.
    """
    body = _strip_fence(text)
    start = body.find("{")
    if start < 0:
        return body
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(body)):
        char = body[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return body[start:index + 1]
    return body


def coverage_input_tokens(text: str) -> frozenset[str]:
    """coverage에 넘길 내용 토큰 집합.

    P0-16 report 토글의 비간섭을 검증하기 위한 공개 순수 함수다. report는 이
    집합을 읽기만 하며 원문이나 노트를 바꾸지 않는다.
    """

    return frozenset(token.casefold() for token in _COVERAGE_TOKEN_RE.findall(text))


def coverage_input_sha256(text: str) -> str:
    """정렬된 coverage 입력 토큰 집합의 안정적 SHA-256."""

    canonical = "\n".join(sorted(coverage_input_tokens(text))).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _match_excerpt(text: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - 32)
    end = min(len(text), match.end() + 32)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _humanizer_level(severity: str) -> str:
    # AI 티는 맞춤법 오류가 아니다. S1도 강한 권장으로만 보고한다.
    return "⚠️ 권장" if severity == "S1" else "💡 제안"


def report_grammar(note: str) -> list[CopyQualityFinding]:
    """비문·맞춤법 lens만 보고한다. humanizer/source 판정은 섞지 않는다."""

    findings: list[CopyQualityFinding] = []
    in_fence = False
    for line_no, raw_line in enumerate(note.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith(("#", "<!--")):
            continue
        for rule_id, pattern, level, reason, source_ref in _GRAMMAR_RULES:
            match = pattern.search(stripped)
            if match is None:
                continue
            findings.append(
                CopyQualityFinding(
                    lens="grammar",
                    rule_id=rule_id,
                    level=level,
                    line=line_no,
                    excerpt=_match_excerpt(stripped, match),
                    reason=reason,
                    source_ref=source_ref,
                )
            )
    return findings


def report_sentence_quality(note: str) -> list[CopyQualityFinding]:
    """이전 API 호환용 grammar-only 별칭. 세 렌즈 합산 함수가 아니다."""

    return report_grammar(note)


def _sentence_spans(text: str) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip()
        if len(sentence) >= 8 and not sentence.startswith(("#", "<!--")):
            spans.append((match.start(), sentence))
    return spans


def _metric_finding(
    note: str, rule_id: str, reason: str, excerpt: str, *, severity: str = "S2"
) -> CopyQualityFinding:
    offset = note.find(excerpt)
    return CopyQualityFinding(
        lens="humanizer",
        rule_id=rule_id,
        level=_humanizer_level(severity),
        line=_line_for_offset(note, max(0, offset)),
        excerpt=re.sub(r"\s+", " ", excerpt)[:120],
        reason=reason,
        source_ref=f"im-not-ai/humanize-korean ai-tell-taxonomy:{rule_id}",
    )


def _report_humanizer_metrics(note: str) -> list[CopyQualityFinding]:
    findings: list[CopyQualityFinding] = []
    sentences = _sentence_spans(note)
    sentence_texts = [sentence for _, sentence in sentences]
    if len(sentence_texts) >= 5:
        comma_ratio = sum("," in sentence for sentence in sentence_texts) / len(sentence_texts)
        if comma_ratio > 0.5:
            findings.append(_metric_finding(note, "C-12", f"쉼표 포함 문장 비율 {comma_ratio:.0%} > 50%", sentence_texts[0]))

        lengths = [len(re.sub(r"\s+", "", sentence)) for sentence in sentence_texts]
        if statistics.pstdev(lengths) < 8 and 20 <= statistics.mean(lengths) <= 60:
            findings.append(_metric_finding(note, "E-1", f"문장 길이 표준편차 {statistics.pstdev(lengths):.1f} < 8", sentence_texts[0]))

        endings = []
        for sentence in sentence_texts:
            match = re.search(r"(입니다|습니다|한다|된다|이다|있다|없다)[.!?]*$", sentence)
            endings.append(match.group(1) if match else "")
        for idx in range(max(0, len(endings) - 3)):
            if endings[idx] and len(set(endings[idx:idx + 4])) == 1:
                findings.append(_metric_finding(note, "E-2", f"같은 종결어미 '{endings[idx]}'가 4문장 연속됨", sentence_texts[idx]))
                break

        complex_count = sum(
            bool(re.search(r"(?:고|며|지만|면서|는데|어서|아서|하면|하는)\b", sentence))
            for sentence in sentence_texts
        )
        if complex_count / len(sentence_texts) < 0.25:
            findings.append(_metric_finding(note, "E-4", f"복문·중문 신호 문장 비율 {complex_count / len(sentence_texts):.0%} < 25%", sentence_texts[0]))

    paragraphs = [p for p in re.split(r"\n\s*\n", note) if p.strip() and not p.lstrip().startswith("#")]
    paragraph_counts = [len(_sentence_spans(p)) for p in paragraphs]
    nonzero_counts = [count for count in paragraph_counts if count]
    if len(nonzero_counts) >= 3 and all(3 <= count <= 4 for count in nonzero_counts):
        findings.append(_metric_finding(note, "E-3", "모든 prose 문단이 3~4문장으로 균일함", paragraphs[0]))

    comma_segments = [segment.strip() for sentence in sentence_texts for segment in sentence.split(",")]
    if len(comma_segments) >= 4:
        word_lengths = [len(segment.split()) for segment in comma_segments if segment]
        if word_lengths and statistics.mean(word_lengths) > 7:
            findings.append(_metric_finding(note, "E-5", f"쉼표 분절 평균 {statistics.mean(word_lengths):.1f}어절 > 7", comma_segments[0]))
    return findings


def report_humanizer(note: str) -> list[CopyQualityFinding]:
    """10대 taxonomy의 실행 가능 규칙을 적용한다. 문장을 다시 쓰지 않는다."""

    findings: list[CopyQualityFinding] = []
    for rule_id, pattern, threshold, severity, reason in _HUMANIZER_REGEX_RULES:
        matches = list(pattern.finditer(note))
        if len(matches) < threshold:
            continue
        for match in matches[:3]:
            findings.append(
                CopyQualityFinding(
                    lens="humanizer",
                    rule_id=rule_id,
                    level=_humanizer_level(severity),
                    line=_line_for_offset(note, match.start()),
                    excerpt=_match_excerpt(note, match),
                    reason=f"{reason} (문서 {len(matches)}회; 임계 {threshold}회)",
                    source_ref=f"im-not-ai/humanize-korean ai-tell-taxonomy:{rule_id}",
                )
            )
    findings.extend(_report_humanizer_metrics(note))
    return findings


_SOURCE_BLOCK_LABELS = {
    "본문": "body",
    "원저자 연속글": "author_thread",
    "원저자 댓글": "author_comments",
    "OCR 텍스트": "ocr",
}
_SOURCE_LABEL_RE = re.compile(r"^\[(?P<label>본문|원저자 연속글|원저자 댓글|OCR 텍스트)\]\s*$")
_SOURCE_FIDELITY_EXECUTABLE_RULE_IDS = frozenset(
    {
        "SF-1-unsupported-claim",
        "SF-2-quantifier-strengthening",
        "SF-3-polarity-reversal",
        "SF-4-assertion-level-change",
    }
)
_SOURCE_FIDELITY_CONTEXT_RULE_IDS = frozenset(
    {
        "SF-5-causal-reversal",
        "SF-A2-self-contradiction",
        "SF-A3-false-causality",
        "SF-A9-source-traceability",
    }
)
_STRONG_QUANTIFIER_RE = re.compile(
    r"(?:모든|전부|항상|모두)(?=$|[\s,.;:!?\)\]\"'”]|[은는이가을를도만])"
)
_WEAK_QUANTIFIER_RE = re.compile(
    r"(?:대부분|일부|상당수)(?=$|[\s,.;:!?\)\]\"'”]|[은는이가을를도만])"
)
_HEDGE_RE = re.compile(
    r"(?:인\s*듯(?:하(?:다|며|고|지만|다)|합니다)?|"
    r"(?:으)?로\s+보(?:인|입니|였|이)|"
    r"것으로\s+(?:보(?:인|입니|였|이)|추정)|"
    r"추정(?:된|되|하)|추측(?:된|되|하)|가능성이\s+있)"
)
_ASSERTIVE_END_RE = re.compile(
    r"(?:이다|입니다|한다|합니다|된다|됩니다|있다|있습니다|없다|없습니다)[.!?]*$"
)
_COMPARISON_RE = re.compile(
    r"(?P<number>\d[\d,.]*)\s*(?:[%가-힣]{0,8}\s*)?(?P<operator>이상|미만)"
)
_CLAIM_STOPWORDS = frozenset(
    {"그리고", "그러나", "하지만", "또한", "이는", "이것은", "대한", "통해", "있다", "있는", "합니다", "됩니다", "입니다", "것이다", "같은", "위한", "해당", "본문", "설명", "내용"}
)


def _normalize_claim_token(token: str) -> str:
    value = token.casefold()
    if re.fullmatch(r"[가-힣]+", value):
        for suffix in ("으로부터", "에서는", "에게서", "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "입니다", "한다", "했다", "된다", "하고", "하며", "은", "는", "이", "가", "을", "를", "와", "과", "도", "만"):
            if value.endswith(suffix) and len(value) >= len(suffix) + 2:
                value = value[:-len(suffix)]
                break
    return value


def _content_tokens(text: str) -> frozenset[str]:
    tokens = {_normalize_claim_token(token) for token in _COVERAGE_TOKEN_RE.findall(text)}
    return frozenset(token for token in tokens if len(token) >= 2 and token not in _CLAIM_STOPWORDS)


_FIDELITY_MARKER_FRAGMENTS = (
    "모든",
    "전부",
    "항상",
    "모두",
    "대부분",
    "일부",
    "상당수",
    "가능",
    "불가",
    "이상",
    "미만",
    "듯",
    "보인다",
    "보입니다",
    "추정",
    "추측",
)

# prompts/text/note_synthesis.md의 [근거 조건] 정본. 유사 표현은 일부러
# 허용하지 않는다. sentinel 형식 위반은 source fidelity가 아닌 별도 축이다.
_FIDELITY_SENTINELS = (
    "텍스트 글에 명시된 이유 없음",
    "구체적 절차 미제시",
)
_FIDELITY_SOURCE_REFERENCE_RE = re.compile(
    r"(?:본문|원저자\s*(?:댓글|연속글)|댓글|(?:D\d+\s*)?OCR|텍스트\s*글|원문)",
    re.IGNORECASE,
)
_FIDELITY_META_SCAFFOLD_FRAGMENTS = (
    "항목",
    "일부",
    "끝부분",
    "강조",
    "정리",
    "흐름",
    "다음",
    "같다",
    "관련",
    "아래",
    "목록",
    "보존",
)
_FIDELITY_META_CONTENT_TOKEN_LIMIT = 3
_LATIN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]{2,}")


def _fidelity_alignment_tokens(text: str) -> frozenset[str]:
    """대조 표지 자체를 빼고 두 문장이 같은 대상을 말하는지 보수적으로 맞춘다."""

    return frozenset(
        token
        for token in _content_tokens(text)
        if not any(fragment in token for fragment in _FIDELITY_MARKER_FRAGMENTS)
    )


def _is_exact_fidelity_sentinel(text: str) -> bool:
    """레이블은 허용하되 sentinel 본문은 production prompt와 정확히 맞춘다."""

    value = text.strip().rstrip(".!?").strip()
    if value in _FIDELITY_SENTINELS:
        return True
    if ":" not in value and "：" not in value:
        return False
    payload = re.split(r"[:：]", value, maxsplit=1)[1].strip()
    return any(
        payload == sentinel or payload == f"텍스트 글에 {sentinel}"
        for sentinel in _FIDELITY_SENTINELS
    )


def _is_fidelity_source_meta(text: str) -> bool:
    """출처 참조를 걷어냈을 때 내용이 없는 인용·부재·처리 메타만 제외한다."""

    if _FIDELITY_SOURCE_REFERENCE_RE.search(text) is None:
        return False

    without_references = _FIDELITY_SOURCE_REFERENCE_RE.sub(" ", text)
    raw_tokens = _content_tokens(without_references)
    scaffold_fragments = {
        fragment
        for fragment in _FIDELITY_META_SCAFFOLD_FRAGMENTS
        if any(fragment in token for token in raw_tokens)
    }
    residual_tokens = {
        token
        for token in raw_tokens
        if not any(
            fragment in token for fragment in _FIDELITY_META_SCAFFOLD_FRAGMENTS
        )
    }
    if not residual_tokens or (
        len(scaffold_fragments) >= 2
        and len(residual_tokens) < _FIDELITY_META_CONTENT_TOKEN_LIMIT
    ):
        return True

    # "정의/설명/언급/제시가 소스에 따로 없다"는 내용 주장이 아니라
    # source scope의 부재 공시다. 임의 sentinel 유사문구를 허용하지 않고,
    # 출처 참조와 부재 서술이 함께 있을 때만 이 경로를 탄다.
    absence_disclosure = re.search(
        r"(?:정의|설명|언급|제시).*(?:따로|별도로).*(?:하지\s*않|없)",
        text,
    ) or re.search(
        r"(?:따로|별도로).*(?:정의|설명|언급|제시).*(?:하지\s*않|없)",
        text,
    )
    if absence_disclosure:
        return True

    # 원문을 보존하면서 새 기능·인과로 해석하지 않겠다는 문장은 노트의
    # 자기 처리 방침이다. 두 동작을 모두 요구해 실제 OCR 내용 진술과 구분한다.
    preserves_source = re.search(
        r"원문.{0,12}(?:순서대로\s*)?(?:보존|인용|전재|옮기)", text
    )
    rejects_inference = re.search(
        r"(?:별도|새로운).{0,16}(?:기능|인과).{0,16}"
        r"(?:해석|추론).{0,12}(?:하지\s*않|않는|금지)",
        text,
    )
    return bool(preserves_source and rejects_inference)


def _rare_source_latin_terms(source: str) -> frozenset[str]:
    counts: dict[str, int] = {}
    for term in _LATIN_TERM_RE.findall(source):
        normalized = term.casefold()
        counts[normalized] = counts.get(normalized, 0) + 1
    return frozenset(term for term, count in counts.items() if count == 1)


def _short_claim_rare_term_evidence(
    claim: str,
    source_block: str,
    rare_source_latin: frozenset[str],
) -> frozenset[str]:
    """짧은 `레이블: 값`에서 희귀 라틴어와 같은 문장의 한국어 단서를 확인한다."""

    if not rare_source_latin or not re.search(r"[:：]", claim):
        return frozenset()
    if len(_content_tokens(claim)) > 3:
        return frozenset()
    claim_latin = {term.casefold() for term in _LATIN_TERM_RE.findall(claim)}
    rare_matches = claim_latin & rare_source_latin
    if not rare_matches:
        return frozenset()

    korean_clues = {
        _normalize_claim_token(token)
        for token in re.findall(r"[가-힣]+", claim)
        if len(_normalize_claim_token(token)) >= 2
        and _normalize_claim_token(token) not in _CLAIM_STOPWORDS
    }
    for _, sentence in _sentence_spans(source_block):
        sentence_latin = {
            term.casefold() for term in _LATIN_TERM_RE.findall(sentence)
        }
        anchored_terms = rare_matches & sentence_latin
        if anchored_terms and any(clue in sentence for clue in korean_clues):
            return frozenset(anchored_terms)
    return frozenset()


def _source_sentence_records(
    blocks: list[tuple[str, str]],
) -> list[tuple[str, str, frozenset[str]]]:
    records: list[tuple[str, str, frozenset[str]]] = []
    for block_id, text in blocks:
        for _, sentence in _sentence_spans(text):
            tokens = _fidelity_alignment_tokens(sentence)
            if len(tokens) >= 2:
                records.append((block_id, sentence, tokens))
    return records


def _aligned_source_sentences(
    claim: str, records: list[tuple[str, str, frozenset[str]]]
) -> list[tuple[str, str]]:
    """내용 토큰 2개 이상·짧은 쪽의 절반 이상이 겹치는 최상위 문장만 반환."""

    claim_tokens = _fidelity_alignment_tokens(claim)
    if len(claim_tokens) < 2:
        return []
    candidates: list[tuple[int, str, str]] = []
    for block_id, sentence, source_tokens in records:
        overlap = len(claim_tokens & source_tokens)
        if overlap < 2 or overlap * 2 < min(len(claim_tokens), len(source_tokens)):
            continue
        candidates.append((overlap, block_id, sentence))
    if not candidates:
        return []
    best_overlap = max(item[0] for item in candidates)
    return [(block_id, sentence) for overlap, block_id, sentence in candidates if overlap == best_overlap]


def _possibility_polarity(text: str) -> str | None:
    negative = bool(
        re.search(
            r"(?:불가능|할\s+수\s+없|가능하지\s+않|불가(?=$|[\s,.;:!?\)\]]|[하했한]))",
            text,
        )
    )
    positive = bool(
        re.search(
            r"(?:할\s+수\s+있|(?<!불)가능(?!하지\s+않)(?:하|했|한|합니|성|$))",
            text,
        )
    )
    if negative == positive:
        return None
    return "negative" if negative else "positive"


def _comparison_polarities(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    conflicts: set[str] = set()
    for match in _COMPARISON_RE.finditer(text):
        number = match.group("number").replace(",", "")
        operator = match.group("operator")
        if number in values and values[number] != operator:
            conflicts.add(number)
        values[number] = operator
    return {number: operator for number, operator in values.items() if number not in conflicts}


def _source_comparison_finding(
    *,
    rule_id: str,
    line_no: int,
    claim: str,
    block_id: str,
    source_sentence: str,
    reason: str,
    source_ref: str,
) -> CopyQualityFinding:
    return CopyQualityFinding(
        lens="source_fidelity",
        rule_id=rule_id,
        level="⚠️ 권장",
        line=line_no,
        excerpt=claim[:120],
        reason=f"{reason}; source {block_id}: {source_sentence[:100]}",
        source_ref=source_ref,
    )


def _report_source_meaning_changes(
    line_no: int,
    claim: str,
    aligned: list[tuple[str, str]],
) -> list[CopyQualityFinding]:
    """명시적 표지의 방향만 비교한다. 일반 의미 판정은 context-only로 남긴다."""

    findings: list[CopyQualityFinding] = []

    note_strong = tuple(_STRONG_QUANTIFIER_RE.findall(claim))
    if note_strong and not any(_STRONG_QUANTIFIER_RE.search(sentence) for _, sentence in aligned):
        weaker = next(
            ((block_id, sentence) for block_id, sentence in aligned if _WEAK_QUANTIFIER_RE.search(sentence)),
            None,
        )
        if weaker is not None:
            findings.append(
                _source_comparison_finding(
                    rule_id="SF-2-quantifier-strengthening",
                    line_no=line_no,
                    claim=claim,
                    block_id=weaker[0],
                    source_sentence=weaker[1],
                    reason=f"소스의 약한 양화를 노트가 강한 양화({', '.join(note_strong)})로 바꾼 후보",
                    source_ref="content-fidelity-auditor checklist 10:양화·한정",
                )
            )

    note_possibility = _possibility_polarity(claim)
    if note_possibility is not None:
        source_states = [
            (block_id, sentence, _possibility_polarity(sentence))
            for block_id, sentence in aligned
        ]
        if not any(state == note_possibility for _, _, state in source_states):
            opposite = next(
                ((block_id, sentence) for block_id, sentence, state in source_states if state is not None and state != note_possibility),
                None,
            )
            if opposite is not None:
                findings.append(
                    _source_comparison_finding(
                        rule_id="SF-3-polarity-reversal",
                        line_no=line_no,
                        claim=claim,
                        block_id=opposite[0],
                        source_sentence=opposite[1],
                        reason="가능/불가능 극성이 소스와 반대인 후보",
                        source_ref="content-fidelity-auditor checklist 11:긍정·부정 극성",
                    )
                )

    note_comparisons = _comparison_polarities(claim)
    for number, note_operator in note_comparisons.items():
        source_comparisons = [
            (block_id, sentence, _comparison_polarities(sentence).get(number))
            for block_id, sentence in aligned
        ]
        if any(operator == note_operator for _, _, operator in source_comparisons):
            continue
        opposite = next(
            (
                (block_id, sentence, operator)
                for block_id, sentence, operator in source_comparisons
                if operator is not None and operator != note_operator
            ),
            None,
        )
        if opposite is not None:
            findings.append(
                _source_comparison_finding(
                    rule_id="SF-3-polarity-reversal",
                    line_no=line_no,
                    claim=claim,
                    block_id=opposite[0],
                    source_sentence=opposite[1],
                    reason=f"같은 수치 {number}의 경계가 {opposite[2]}→{note_operator}으로 반전된 후보",
                    source_ref="content-fidelity-auditor checklist 7·11:주장 방향·극성",
                )
            )

    if _ASSERTIVE_END_RE.search(claim) and not _HEDGE_RE.search(claim):
        if not any(not _HEDGE_RE.search(sentence) for _, sentence in aligned):
            hedged = next(
                ((block_id, sentence) for block_id, sentence in aligned if _HEDGE_RE.search(sentence)),
                None,
            )
            if hedged is not None:
                findings.append(
                    _source_comparison_finding(
                        rule_id="SF-4-assertion-level-change",
                        line_no=line_no,
                        claim=claim,
                        block_id=hedged[0],
                        source_sentence=hedged[1],
                        reason="소스의 추측·관측 표현이 노트에서 단정으로 바뀐 후보",
                        source_ref="content-fidelity-auditor checklist 7·10:주장 층위",
                    )
                )
    return findings


def _parse_source_blocks(source: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_id = "source:1"
    current_lines: list[str] = []
    counts: dict[str, int] = {}
    saw_label = False
    for raw_line in source.splitlines():
        match = _SOURCE_LABEL_RE.match(raw_line.strip())
        if match is None:
            current_lines.append(raw_line)
            continue
        if current_lines and (not saw_label or "".join(current_lines).strip()):
            blocks.append((current_id, "\n".join(current_lines).strip()))
        saw_label = True
        base = _SOURCE_BLOCK_LABELS[match.group("label")]
        counts[base] = counts.get(base, 0) + 1
        current_id = f"{base}:{counts[base]}"
        current_lines = []
    if current_lines or not blocks:
        blocks.append((current_id, "\n".join(current_lines).strip()))
    if saw_label:
        blocks = [block for block in blocks if block[0] != "source:1" or block[1]]
    return blocks


def _extract_note_claims(note: str) -> list[tuple[int, str]]:
    claims: list[tuple[int, str]] = []
    in_fence = False
    in_html_comment = False
    for line_no, raw_line in enumerate(note.splitlines(), start=1):
        line = raw_line
        if in_html_comment:
            if "-->" not in line:
                continue
            line = line.split("-->", 1)[1]
            in_html_comment = False
        while "<!--" in line:
            before, after = line.split("<!--", 1)
            if "-->" not in after:
                line = before
                in_html_comment = True
                break
            line = before + after.split("-->", 1)[1]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith(("#", "**출처:**")):
            continue
        prose = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", stripped)
        prose = re.sub(r"[*_`]+", "", prose)
        for _, sentence in _sentence_spans(prose):
            if (
                len(_content_tokens(sentence)) >= 3
                and not _is_exact_fidelity_sentinel(sentence)
                and not _is_fidelity_source_meta(sentence)
            ):
                claims.append((line_no, sentence))
    return claims


def report_source_fidelity(
    source: str, note: str
) -> tuple[list[SourceClaimMapping], list[CopyQualityFinding]]:
    """외부 웹 없이 claim의 lexical 근거와 보수적 의미변형 표지를 대조한다."""

    blocks = _parse_source_blocks(source)
    block_tokens = {block_id: _content_tokens(text) for block_id, text in blocks}
    block_text = dict(blocks)
    rare_source_latin = _rare_source_latin_terms(source)
    source_sentences = _source_sentence_records(blocks)
    mappings: list[SourceClaimMapping] = []
    findings: list[CopyQualityFinding] = []
    for index, (line_no, claim) in enumerate(_extract_note_claims(note), start=1):
        claim_tokens = _content_tokens(claim)
        # 한 단어 우연 일치는 근거로 세지 않는다. 짧은 claim도 최소 두 개의
        # 내용 토큰이 같은 block에 있어야 매핑한다.
        threshold = max(2, (len(claim_tokens) + 4) // 5)
        evidence: list[tuple[str, frozenset[str]]] = []
        for block_id, tokens in block_tokens.items():
            overlap = claim_tokens & tokens
            if len(overlap) >= threshold:
                evidence.append((block_id, overlap))
                continue
            rare_term_evidence = _short_claim_rare_term_evidence(
                claim, block_text[block_id], rare_source_latin
            )
            if rare_term_evidence:
                evidence.append((block_id, overlap | rare_term_evidence))
        matched_tokens = tuple(sorted({token for _, overlap in evidence for token in overlap}))
        block_ids = tuple(block_id for block_id, _ in evidence)
        claim_id = f"claim-{index:03d}"
        mappings.append(
            SourceClaimMapping(
                claim_id=claim_id,
                line=line_no,
                claim=claim,
                source_block_ids=block_ids,
                matched_tokens=matched_tokens,
            )
        )
        aligned = _aligned_source_sentences(claim, source_sentences)
        if aligned:
            findings.extend(_report_source_meaning_changes(line_no, claim, aligned))
        if block_ids:
            continue
        findings.append(
            CopyQualityFinding(
                lens="source_fidelity",
                rule_id="SF-1-unsupported-claim",
                level="⚠️ 권장",
                line=line_no,
                excerpt=claim[:120],
                reason="노트 claim과 충분히 겹치는 source block 근거 토큰을 찾지 못함",
                source_ref="docs/09 C-1 fact/claim check → NoteFactory source-block comparison",
            )
        )
    return mappings, findings


def _finding_to_dict(finding: CopyQualityFinding) -> dict[str, object]:
    return {
        "lens": finding.lens,
        "rule_id": finding.rule_id,
        "level": finding.level,
        "line": finding.line,
        "excerpt": finding.excerpt,
        "reason": finding.reason,
        "source_ref": finding.source_ref,
    }


def _attach_existing_critic_findings(
    note: str,
    critic_findings: list[dict],
    grammar: list[CopyQualityFinding],
    humanizer: list[CopyQualityFinding],
    source_fidelity: list[CopyQualityFinding],
) -> None:
    """이미 실행된 critic 결과만 재사용한다. 이 함수는 provider를 호출하지 않는다."""

    for raw in critic_findings:
        if not isinstance(raw, dict):
            continue
        claim = str(raw.get("claim") or raw.get("text_span") or "").strip()
        if not claim or claim not in note:
            continue
        lens = str(raw.get("lens") or "").replace("-", "_")
        rule_id = str(raw.get("rule_id") or raw.get("category") or "critic-context")
        target: list[CopyQualityFinding] | None = None
        if lens == "grammar" and rule_id in _GRAMMAR_CONTEXT_RULE_IDS:
            target = grammar
        elif lens == "humanizer" and rule_id in _HUMANIZER_CONTEXT_RULE_IDS:
            target = humanizer
        elif lens == "source_fidelity" and rule_id in (
            _SOURCE_FIDELITY_EXECUTABLE_RULE_IDS | _SOURCE_FIDELITY_CONTEXT_RULE_IDS
        ):
            target = source_fidelity
        else:
            verdict = str(raw.get("verdict", "")).lower()
            if verdict in {"ungrounded", "unsupported", "fabricated", "uncertain"}:
                target = source_fidelity
                lens = "source_fidelity"
                rule_id = (
                    "SF-critic-uncertain" if verdict == "uncertain" else "SF-critic-ungrounded"
                )
        if target is None:
            continue
        target.append(
            CopyQualityFinding(
                lens=lens,
                rule_id=rule_id,
                level=(
                    _GRAMMAR_CONTEXT_RULE_LEVELS[rule_id]
                    if lens == "grammar" and rule_id in _GRAMMAR_CONTEXT_RULE_LEVELS
                    else str(raw.get("level") or "⚠️ 권장")
                ),
                line=_line_for_offset(note, note.find(claim)),
                excerpt=claim[:120],
                reason=str(raw.get("reason") or raw.get("evidence") or "기존 critic의 문맥 판정"),
                source_ref="existing critic result (no additional inference)",
            )
        )


def report_copy_quality(
    source: str, note: str, *, critic_findings: list[dict] | None = None
) -> dict[str, object]:
    """grammar/humanizer/source-fidelity를 독립 report로 반환한다."""

    grammar = report_grammar(note)
    humanizer = report_humanizer(note)
    mappings, source_fidelity = report_source_fidelity(source, note)
    _attach_existing_critic_findings(
        note, critic_findings or [], grammar, humanizer, source_fidelity
    )
    return {
        "schema": "notefactory-copy-quality-report/v1",
        "report_only": True,
        "standalone_gate": False,
        "auto_rewrite": False,
        "external_web_fact_check": False,
        "fact_check_interpretation": "note claims mapped to provided source blocks",
        "rule_counts": {
            "grammar_executable": len(_GRAMMAR_RULES),
            "grammar_context_via_existing_critic": len(_GRAMMAR_CONTEXT_RULE_IDS),
            "humanizer_catalog": len(_HUMANIZER_ACTIVE_RULE_IDS),
            "humanizer_categories": len(_HUMANIZER_RULE_CATALOG),
            "humanizer_executable": len(_HUMANIZER_EXECUTABLE_RULE_IDS),
            "humanizer_context_via_existing_critic": len(_HUMANIZER_CONTEXT_RULE_IDS),
            "source_fidelity_executable": len(_SOURCE_FIDELITY_EXECUTABLE_RULE_IDS),
            "source_fidelity_context_via_existing_critic": len(_SOURCE_FIDELITY_CONTEXT_RULE_IDS),
        },
        "lenses": {
            "grammar": {
                "findings": [_finding_to_dict(finding) for finding in grammar],
                "context_rule_ids": sorted(_GRAMMAR_CONTEXT_RULE_IDS),
            },
            "humanizer": {
                "findings": [_finding_to_dict(finding) for finding in humanizer],
                "context_rule_ids": sorted(_HUMANIZER_CONTEXT_RULE_IDS),
            },
            "source_fidelity": {
                "findings": [_finding_to_dict(finding) for finding in source_fidelity],
                "context_rule_ids": sorted(_SOURCE_FIDELITY_CONTEXT_RULE_IDS),
                "claim_mappings": [
                    {
                        "claim_id": mapping.claim_id,
                        "line": mapping.line,
                        "claim": mapping.claim,
                        "source_block_ids": list(mapping.source_block_ids),
                        "matched_tokens": list(mapping.matched_tokens),
                    }
                    for mapping in mappings
                ],
            },
        },
    }


def extract_source_block_densities(note: str) -> tuple[str, dict[str, str]]:
    """합성 표식을 본문에서 제거하고 세 블록 등급을 반환한다.

    정확한 세 필드가 첫 줄에 없으면 빈 dict와 원문을 반환한다. 호출자는
    text_post에서 이 상태를 fail-closed할 수 있다.
    """

    match = _SOURCE_BLOCK_DENSITY_RE.match(note)
    if match is None:
        return note, {}
    densities = {key: value.upper() for key, value in match.groupdict().items()}
    return note[match.end():].lstrip(), densities


def evaluate_prompt_edit(
    *,
    edit_id: str,
    edit_kind: EditKind,
    before: EditEvaluationArtifact,
    after: EditEvaluationArtifact,
    regeneration_band: float,
) -> EditRewardDecision:
    """P0-8 편집 단위 rollback reward를 계산한다.

    fabrication 증가는 크기와 무관하게 rollback한다. coverage 감소는 같은 노트의
    regeneration band를 초과할 때만 rollback한다. Prune은 두 하드 축이 나빠지지
    않으면 죽은 지시 제거로 채택한다. 입력 task/gold/token set 정체가 다르면 비교
    자체를 거부해 서로 다른 과제를 같은 A/B처럼 세탁하지 못하게 한다.
    """

    if before.task_id != after.task_id:
        raise ValueError("편집 전후 task_id가 다릅니다")
    if before.gold_frontier_ids != after.gold_frontier_ids:
        raise ValueError("편집 전후 dual-frontier gold가 다릅니다")
    if before.coverage_input_sha256 != after.coverage_input_sha256:
        raise ValueError("편집 전후 coverage 입력 토큰 집합이 다릅니다")
    if regeneration_band < 0:
        raise ValueError("regeneration_band는 0 이상이어야 합니다")

    fabrication_delta = after.fabrication_count - before.fabrication_count
    coverage_delta = after.coverage - before.coverage
    rollback_reasons: list[str] = []
    if fabrication_delta > 0:
        rollback_reasons.append(f"fabrication +{fabrication_delta}")
    if coverage_delta < -regeneration_band:
        rollback_reasons.append(
            f"coverage {coverage_delta:+.3f} < -band({regeneration_band:.3f})"
        )

    if rollback_reasons:
        return EditRewardDecision(
            edit_id=edit_id,
            edit_kind=edit_kind,
            before_artifact_id=before.artifact_id,
            after_artifact_id=after.artifact_id,
            reward_vector=(-fabrication_delta, coverage_delta),
            regeneration_band=regeneration_band,
            disposition="ROLLBACK",
            rollback=True,
            reason="; ".join(rollback_reasons),
        )

    noop = fabrication_delta == 0 and coverage_delta == 0
    dead_instruction = (
        edit_kind is EditKind.PRUNE
        and fabrication_delta <= 0
        and coverage_delta >= -regeneration_band
    )
    disposition = "NOOP" if edit_kind is EditKind.NOOP or noop else "ADOPT"
    if dead_instruction:
        disposition = "ADOPT_PRUNE"
    return EditRewardDecision(
        edit_id=edit_id,
        edit_kind=edit_kind,
        before_artifact_id=before.artifact_id,
        after_artifact_id=after.artifact_id,
        reward_vector=(-fabrication_delta, coverage_delta),
        regeneration_band=regeneration_band,
        disposition=disposition,
        rollback=False,
        dead_instruction=dead_instruction,
        reason=(
            "두 하드 축 비악화 — 죽은 지시로 기록"
            if dead_instruction
            else "rollback 조건 없음"
        ),
    )


def _estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(t) for t in texts)
    return total_chars // _ESTIMATE_CHARS_PER_TOKEN


def _is_premium_source(source: str) -> bool:
    lowered = source.lower()
    return any(lowered.startswith(prefix) for prefix in _PREMIUM_SOURCE_PREFIXES)


def _usable_usage_tokens(usage: dict | None) -> int | None:
    """usage dict에서 실측 토큰 수를 도출한다 — 도출 불가면 None.

    total_tokens 우선, 없으면 prompt+completion 합산. 셋 다 없거나(빈/무키
    dict 포함) 값이 전부 falsy면 None — 유령 0을 실측으로 위장하지 않는다.
    """
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if total:
        return total
    prompt_t = usage.get("prompt_tokens") or 0
    completion_t = usage.get("completion_tokens") or 0
    if prompt_t or completion_t:
        return prompt_t + completion_t
    return None


def _record_usage(
    pass_name: str,
    *,
    usage: dict | None,
    prompt_text: str,
    output_text: str,
    premium: bool = False,
    response: Mapping[str, object] | None = None,
    role: str | None = None,
) -> PassUsage:
    """usage에서 실측 토큰을 도출할 수 있으면 실측, 아니면 chars/4 추정치.

    "usable한 숫자가 있는가"를 여기(record 시점)서 판정한다 — usage가 non-None
    이어도 빈/무키 dict면 추정으로 정직하게 분류한다(G5 iter2 신규 P1:
    usage={} 유령 0 방지). 추정은 이 계층(하네스)의 책임이다 — free_llm.py는
    절대 추정치를 usage에 채우지 않는다(계약 §3).
    """
    evidence = response if isinstance(response, Mapping) else {}
    billing_tier = evidence.get("billing_tier")
    if billing_tier not in {"free", "paid"}:
        billing_tier = "paid" if premium else "free"
    premium = billing_tier == "paid"
    route_fields = {
        "role": role or ("critic" if pass_name == "critic" else "writer"),
        "requested_provider": str(evidence["requested_provider"]) if evidence.get("requested_provider") else None,
        "requested_model": str(evidence["requested_model"]) if evidence.get("requested_model") else None,
        "actual_provider": str(evidence["provider"]) if evidence.get("provider") else None,
        "actual_model": str(evidence["model"]) if evidence.get("model") else None,
        "billing_tier": str(billing_tier),
    }
    estimated_tokens = _estimate_tokens(prompt_text, output_text)
    if _usable_usage_tokens(usage) is not None:
        return PassUsage(
            pass_name=pass_name,
            usage=usage,
            estimated=False,
            estimated_tokens=estimated_tokens,  # 참고용 보존(폴백 아님)
            premium=premium,
            **route_fields,
        )
    return PassUsage(
        pass_name=pass_name,
        usage=usage,  # 원본 보존(빈 dict였다는 사실도 감사 대상)
        estimated=True,
        estimated_tokens=estimated_tokens,
        premium=premium,
        **route_fields,
    )


def _summarize_tokens(passes: list[PassUsage]) -> dict[str, object]:
    total_known = 0
    free_total = 0
    premium_total = 0

    for p in passes:
        if not p.estimated:
            tokens = _usable_usage_tokens(p.usage) or 0
        else:
            tokens = p.estimated_tokens or 0
        total_known += tokens

        if p.premium:
            premium_total += tokens
        else:
            free_total += tokens

    return {
        "passes": [
            {
                "pass": p.pass_name,
                "usage": p.usage,
                "estimated": p.estimated,
                "est_tokens": p.estimated_tokens,
                "premium": p.premium,
                "role": p.role,
                "requested_provider": p.requested_provider,
                "requested_model": p.requested_model,
                "actual_provider": p.actual_provider,
                "actual_model": p.actual_model,
                "billing_tier": p.billing_tier,
            }
            for p in passes
        ],
        "total_known": total_known,
        "free_total": free_total,
        "premium_total": premium_total,
    }


def load_prompt(name: str, *, prompts_dir: Path = DEFAULT_PROMPTS_DIR) -> str:
    """`prompts_dir`(기본 리포 루트 `prompts/`) 아래 상대 경로 `name`의 프롬프트를 읽는다."""
    path = prompts_dir / name
    if not path.exists():
        raise HarnessError(f"프롬프트 파일을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def _parse_plan_markdown(text: str) -> dict:
    """plan 출력을 md로 읽어 JSON 경로와 **같은 dict 모양**으로 돌려준다.

    JSON은 round-04 계약이 고른 운반 수단이지 plan의 목적이 아니다. 목적은
    "주제 목록과 그 개수"이고, 그게 게이트의 완전성 하한이 된다(round-04 결과:
    plan이 뽑은 11 topics → 정확히 11섹션, 얕게 줄이기가 구조적으로 차단됨).
    md로 받아도 그 계약은 그대로 산다.

    JSON을 강제하면 **글은 잘 쓰는데 JSON 형식을 못 맞추는 모델**이 작가 후보에서
    탈락한다. 노트 품질과 무관한 이유다. 형식은 두 가지를 다 받고, 둘 다 아니면
    그때 하드 에러를 낸다 — round-04의 "silent 진행 금지"는 그대로 지킨다.

    받는 형식::

        ## 주제 이름
        - 핵심: 한 문장
        - 수치: 6500원, 8할
        - 고유명사: Claude

    `## ` 헤딩이 하나도 없으면 최상위 목록 줄을 주제 이름으로 받는다 — 약한 모델이
    헤딩 대신 목록으로 내놓는 경우를 살리기 위해서다.
    """
    lines = [ln.rstrip() for ln in _strip_fence(text).splitlines()]
    field_map = {"핵심": "gist", "수치": "key_numbers", "고유명사": "proper_nouns"}
    topics: list[dict] = []

    def new_topic(title: str) -> dict:
        return {"title": title.strip(), "gist": "", "key_numbers": [],
                "proper_nouns": [], "evidence_ts": []}

    for ln in lines:
        if ln.startswith("## "):
            topics.append(new_topic(ln[3:]))
            continue
        if not topics:
            continue
        m = re.match(r"\s*[-*]\s*(핵심|수치|고유명사)\s*[:：]\s*(.*)$", ln)
        if not m:
            continue
        key, raw = field_map[m.group(1)], m.group(2).strip()
        if key == "gist":
            topics[-1]["gist"] = raw
        else:
            topics[-1][key] = [v.strip() for v in raw.split(",") if v.strip()]

    if not topics:
        for ln in lines:
            m = re.match(r"\s*(?:[-*]|\d+[.)])\s+(.+)$", ln)
            if m:
                topics.append(new_topic(m.group(1)))

    if not topics:
        raise ValueError("plan md에서 주제를 하나도 찾지 못했습니다")
    return {"has_timestamps": False, "topics": topics}


#: 소스의 열거 표지 종류. 값은 정규식 패턴이며, 이스케이프를 피하려고 코드포인트로
#: 조립한다. 우선순위를 미리 정하지 않는다 — 어느 층이 바깥인지는 §A의 중첩 판정이
#: 실측으로 가른다.
_VS16 = chr(0xFE0F)
_KEYCAP = chr(0x20E3)
_SEGMENT_MARKERS: dict[str, str] = {
    "heading": "(?m)^#{1,6} ",
    "emoji_digit": "[0-9]" + _VS16 + "?" + _KEYCAP,
    "circled": "[" + chr(0x2460) + "-" + chr(0x2473) + "]",
    "numbered": "(?m)^[ " + chr(9) + "]*[0-9]+[.)] ",
}

#: 표지가 이 개수 미만이면 "열거"로 보지 않는다. 하나짜리는 우연이다.
_SEGMENT_MIN_COUNT = 2


def detect_source_segments(source: str) -> tuple[int, str] | None:
    """소스의 **최상위** 열거 표지를 세어 `(개수, 표지이름)`을 돌려준다.

    plan이 topic을 몇 개로 쪼갤지는 지금까지 모델이 정했다. 그래서 같은 모델·같은
    소스가 7개와 14개를 오갔고(2026-08-24 실측), 그 개수가 그대로 완전성 게이트의
    분모가 되어 **게이트의 엄격도가 실행마다 흔들렸다.** 채울 게 없는데 섹션 수를
    맞추라고 하면 지어내는 수밖에 없다 — round-33이 창작의 원인으로 지목한 경로다.

    개수는 소스가 정해야 한다. 원저자가 직접 매긴 번호가 그 답이다.

    **가장 바깥 층 판정**: 우선순위를 미리 정하지 않고 중첩으로 가른다. A 표지들
    사이 구간 하나에 B 표지가 전부 들어 있으면 B는 A의 안쪽이다. 어느 것의 안쪽도
    아닌 표지가 최상위다. `source_A`에서 `①~⑥`은 전부 `1️⃣`과 `2️⃣` 사이에 있으므로
    최상위는 `1️⃣` 4개다.

    표지가 없거나(줄글) 최상위를 하나로 좁히지 못하면 `None`을 돌려준다 — 그때는
    개수를 강제하지 않는다. 억지로 세느니 안 세는 편이 낫다.
    """
    found: dict[str, list[int]] = {}
    for name, pattern in _SEGMENT_MARKERS.items():
        positions = [m.start() for m in re.finditer(pattern, source)]
        if len(positions) >= _SEGMENT_MIN_COUNT:
            found[name] = positions
    if not found:
        return None

    def nested_in(inner: list[int], outer: list[int]) -> bool:
        """inner 표지가 outer 표지의 **한 구간 안**에 전부 들어 있는가."""
        bounds = outer + [len(source) + 1]
        for i in range(len(bounds) - 1):
            if all(bounds[i] < pos < bounds[i + 1] for pos in inner):
                return True
        return False

    outermost = [name for name, pos in found.items()
                 if not any(nested_in(pos, other)
                            for other_name, other in found.items()
                            if other_name != name)]
    if len(outermost) != 1:
        return None
    name = outermost[0]
    return len(found[name]), name


#: 구간 하나가 이 길이 미만이면 "라벨"로 본다. round-33의 D1 정의("각 항목에
#: 설명·이유·정의가 붙어 있지 않다")를 길이로 근사한다. 실측 간극이 커서 경계값이
#: 예민하지 않다 — `source_A`의 구간 중앙값은 수백 자이고, UI 라벨 나열은 수십 자다.
_LABEL_SEGMENT_MAX_CHARS = 80


def classify_source_density(source: str) -> str:
    """소스가 **라벨 나열(D1)**인지 아닌지만 가른다. `"D1"` 또는 `"D2+"`.

    round-33은 창작을 부르는 지시로 "topic 수를 세어 1:1 커버, 빠지면 FAIL"을 지목했다
    — 라벨 8개를 topic 8개로 바꾸면 섹션 8개를 채워야 하고, 라벨에는 채울 내용이 없으니
    정의를 지어내게 된다. 그 압력은 **소스가 나열형일 때만** 생긴다.

    등급을 모델에게 묻지 않는다. D1을 선언하면 게이트가 느슨해지므로, 자기 신고로 두면
    빠져나갈 구멍이 된다. 기계로 확정할 수 있는 축은 기계가 잰다.

    판정은 보수적이다 — 구간이 전부 짧을 때만 D1이라 부른다. 애매하면 `"D2+"`로 두어
    기존 계약(섹션 1:1)을 그대로 적용한다. 잘못 D1이라 부르면 게이트가 통째로 꺼진다.
    """
    segments = detect_source_segments(source)
    if segments is None:
        return "D2+"
    _count, kind = segments
    positions = [m.start() for m in re.finditer(_SEGMENT_MARKERS[kind], source)]
    bounds = positions + [len(source)]
    lengths = [len(source[bounds[i]:bounds[i + 1]].strip())
               for i in range(len(bounds) - 1)]
    if not lengths:
        return "D2+"
    return "D1" if max(lengths) < _LABEL_SEGMENT_MAX_CHARS else "D2+"


def _run_plan_pass(
    transcript: str,
    generate_fn: Callable[..., dict],
    *,
    prompts_dir: Path,
    plan_prompt_override: str | None = None,
) -> tuple[dict, PassUsage]:
    """Plan 패스 — 구조 개요를 추출한다. JSON·md 둘 다 받고, 둘 다 실패하면
    1회 재시도 후 하드 에러(round-04 "silent 진행 금지" 유지)."""
    system_prompt = (
        plan_prompt_override
        if plan_prompt_override is not None
        else load_prompt(_PLAN_PROMPT_NAME, prompts_dir=prompts_dir)
    )

    # §A — topic 개수를 소스가 정하게 한다. 표지가 없으면 앵커 없이 간다.
    segments = detect_source_segments(transcript)
    user_message = transcript
    if segments is not None:
        count, _kind = segments
        user_message = (
            transcript
            + chr(10) * 2 + "---" + chr(10) * 2
            + "[구조 앵커 — 기계가 센 값]" + chr(10)
            + f"이 텍스트 글은 원저자가 매긴 최상위 구분 표지 {count}개로 나뉩니다. "
            + f"topic은 **그 {count}개 단위**를 따르십시오. "
            + "그 안쪽의 하위 번호는 topic이 아니라 해당 topic의 내용입니다."
        )

    errors: list[str] = []
    last_error: Exception | None = None
    for attempt in (1, 2):
        response = generate_fn(user_message, system_prompt=system_prompt)
        raw_text = response.get("text", "")
        try:
            try:
                plan_json = json.loads(extract_json_object(raw_text))
            except json.JSONDecodeError:
                # JSON이 아니면 md로 읽는다 — 운반 형식이 둘일 뿐 계약은 같다.
                plan_json = _parse_plan_markdown(raw_text)
            # 유효 JSON이어도 형상이 계약(dict + topics 리스트)과 다르면 파싱 실패와
            # 동일 취급한다(G5 silent-failure P1: AttributeError raw traceback 방지).
            if not isinstance(plan_json, dict) or not isinstance(
                plan_json.get("topics"), list
            ):
                raise ValueError(
                    f"plan 형상이 계약과 다릅니다(dict+topics 리스트 필요): "
                    f"{type(plan_json).__name__}"
                )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            errors.append(f"시도 {attempt}: {exc}")
            logger.warning(
                "plan 패스 JSON 파싱/형상 실패(시도 %d/2): %s", attempt, exc
            )
            continue

        usage = _record_usage(
            "plan", usage=response.get("usage"), prompt_text=user_message + system_prompt,
            output_text=raw_text, response=response,
        )
        return plan_json, usage

    raise HarnessError(
        "plan 패스 파싱이 재시도(1회) 후에도 실패했습니다(JSON·md 둘 다): "
        + " / ".join(errors)
    ) from last_error


def _build_synthesis_system_prompt(
    *,
    plan: dict | None,
    prompts_dir: Path,
    synthesis_prompt_override: str | None = None,
    synthesis_with_plan_prompt_override: str | None = None,
) -> str:
    base = (
        synthesis_prompt_override
        if synthesis_prompt_override is not None
        else load_prompt(_SYNTHESIS_PROMPT_NAME, prompts_dir=prompts_dir)
    )
    if plan is None:
        return base
    addendum = (
        synthesis_with_plan_prompt_override
        if synthesis_with_plan_prompt_override is not None
        else load_prompt(_SYNTHESIS_WITH_PLAN_PROMPT_NAME, prompts_dir=prompts_dir)
    )
    return base + "\n\n" + addendum


def _run_synthesis_pass(
    transcript: str,
    generate_fn: Callable[..., dict],
    *,
    plan: dict | None,
    prompts_dir: Path,
    structure_profile: str,
    synthesis_prompt_override: str | None = None,
    synthesis_with_plan_prompt_override: str | None = None,
    source_contract_prompt: str | None = None,
) -> tuple[str, PassUsage]:
    """Synthesis 패스. user 메시지는 전사가 항상 첫 prefix(캐싱 친화 규약)."""
    system_prompt = _build_synthesis_system_prompt(
        plan=plan,
        prompts_dir=prompts_dir,
        synthesis_prompt_override=synthesis_prompt_override,
        synthesis_with_plan_prompt_override=synthesis_with_plan_prompt_override,
    )
    if source_contract_prompt:
        system_prompt = f"{system_prompt}\n\n{source_contract_prompt}"

    user_message = transcript
    if plan is not None:
        plan_json_text = json.dumps(plan, ensure_ascii=False, indent=2)
        plan_contract_label = (
            "[구조 개요(plan JSON) — plan topic 커버 계약]"
            if structure_profile == "functional"
            else "[구조 개요(plan JSON) — Learning Path 섹션 수 하한 계약]"
        )
        user_message = (
            f"{transcript}\n\n---\n\n"
            f"{plan_contract_label}\n"
            f"{plan_json_text}"
        )

    response = generate_fn(user_message, system_prompt=system_prompt)
    raw_note = _strip_fence(response.get("text", ""))
    usage = _record_usage(
        "synthesis",
        usage=response.get("usage"),
        prompt_text=user_message + system_prompt,
        output_text=raw_note,
        response=response,
    )
    return raw_note, usage


def _repair_findings_trigger(gate_findings: list[Finding], critic_findings: list[dict]) -> bool:
    """repair를 발화시킬 트리거성 finding이 있는지 판정한다(계약 §4)."""
    has_gate_trigger = any(f.kind in _REPAIR_TRIGGER_GATE_KINDS for f in gate_findings)
    has_critic_trigger = bool(critic_findings)
    return has_gate_trigger or has_critic_trigger


def _render_findings_for_repair(gate_findings: list[Finding], critic_findings: list[dict]) -> str:
    lines: list[str] = []
    for f in gate_findings:
        if f.kind not in _REPAIR_TRIGGER_GATE_KINDS:
            continue
        context_part = f" (문맥: {f.context})" if f.context else ""
        lines.append(f"- [{f.kind}] {f.detail}{context_part}")
    for cf in critic_findings:
        claim = cf.get("claim", "")
        verdict = cf.get("verdict", "")
        evidence = cf.get("evidence", "")
        lines.append(f"- [critic:{verdict}] {claim} — {evidence}")
    return "\n".join(lines)


def _run_repair_pass(
    transcript: str,
    note: str,
    gate_findings: list[Finding],
    critic_findings: list[dict],
    generate_fn: Callable[..., dict],
    *,
    prompts_dir: Path,
    source_contract_prompt: str | None = None,
) -> tuple[str, PassUsage]:
    system_prompt = load_prompt(_REPAIR_PROMPT_NAME, prompts_dir=prompts_dir)
    if source_contract_prompt:
        system_prompt = f"{system_prompt}\n\n{source_contract_prompt}"
    findings_block = _render_findings_for_repair(gate_findings, critic_findings)

    user_message = (
        f"{transcript}\n\n---\n\n"
        f"[현재 노트]\n{note}\n\n---\n\n"
        f"[수리할 findings]\n{findings_block}"
    )

    response = generate_fn(user_message, system_prompt=system_prompt)
    repaired_note = _strip_fence(response.get("text", ""))
    usage = _record_usage(
        "repair",
        usage=response.get("usage"),
        prompt_text=user_message + system_prompt,
        output_text=repaired_note,
        response=response,
    )
    return repaired_note, usage


def _run_critic_pass(
    transcript: str, note: str, critic_fn: Callable[[str, str], dict]
) -> tuple[list[dict], PassUsage]:
    critic_response = critic_fn(transcript, note)
    findings = critic_response.get("findings", [])
    source = critic_response.get("source", "")
    usage_dict = critic_response.get("usage")
    premium = _is_premium_source(source)

    usage = _record_usage(
        "critic",
        usage=usage_dict,
        prompt_text=transcript + note,
        output_text=json.dumps(findings, ensure_ascii=False),
        premium=premium,
        response=critic_response,
        role="arbiter" if critic_response.get("role") == "arbiter" else "critic",
    )
    return findings, usage


def _extract_plan_topic_items(topics_list: list) -> list[tuple[str, list[str]]]:
    """topic마다 `(제목, key_numbers + proper_nouns)` 쌍을 뽑는다.

    plan은 topic별 수치·고유명사를 이미 내놓는데 게이트는 제목만 쓰고 이 둘을
    버려왔다. 완전성을 이름이 아니라 내용으로 재려면 이 값이 게이트까지 가야 한다.
    """
    out: list[tuple[str, list[str]]] = []
    for topic in topics_list:
        if not isinstance(topic, dict):
            continue
        title = str(topic.get("title") or "").strip()
        if not title:
            continue
        items: list[str] = []
        for key in ("key_numbers", "proper_nouns"):
            value = topic.get(key)
            if isinstance(value, list):
                items.extend(str(v).strip() for v in value if str(v).strip())
        out.append((title, items))
    return out


def _extract_plan_topic_labels(topics_list: list) -> list[str]:
    """plan JSON의 topics 리스트에서 topic **라벨** 리스트를 뽑는다(G5 P1-2).

    반환 리스트 길이는 `len(topics_list)`와 **정확히 일치**한다(lockstep) —
    title이 없거나 비어 있거나 항목이 비-dict여도 drop하지 않고
    `(topic #N, 제목 없음)` placeholder를 부여한다. 그렇지 않으면
    `plan_topic_count`(전체 개수)와 이름 리스트 길이가 발산해, 하필 결측
    topic이 전부 title-less일 때 `check_completeness`가 missing==[]로
    카운트-only 폴백해 T3 수정이 무력화된다.

    placeholder를 실제로 커버 판정에 쓰면 우연 매칭 위험이 있으나,
    `_topic_covered`는 토큰 집합 기반이라 "제목 없음" placeholder는 실제
    노트 섹션 제목과 토큰이 거의 겹치지 않아 자연히 "누락"으로 표면화된다
    (원하는 동작 — 제목 없는 topic도 커버 여부를 사람이 확인하도록).
    """
    labels: list[str] = []
    divergence_seen = False
    for idx, topic in enumerate(topics_list):
        title = topic.get("title") if isinstance(topic, dict) else None
        if isinstance(title, str) and title.strip():
            labels.append(title.strip())
        else:
            divergence_seen = True
            labels.append(f"(topic #{idx + 1}, 제목 없음)")
    if divergence_seen:
        logger.info(
            "plan topics 중 title 누락 항목을 placeholder로 대체했습니다"
            "(count=%d, lockstep 유지) — repair 완전성 판정이 카운트-only로 "
            "폴백하지 않도록 함(G5 P1-2)",
            len(topics_list),
        )
    return labels


def run_harness(
    transcript: str,
    generate_fn: Callable[..., dict],
    *,
    plan: bool = False,
    critic_fn: Callable[[str, str], dict] | None = None,
    repair_budget: int = 1,
    initial_note: str | None = None,
    plan_topic_count_override: int | None = None,
    plan_topics_override: list[str] | None = None,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
    plan_prompt_override: str | None = None,
    synthesis_prompt_override: str | None = None,
    synthesis_with_plan_prompt_override: str | None = None,
    require_source_block_densities: bool = False,
    copy_quality_report_enabled: bool = True,
    sentence_quality_report_enabled: bool | None = None,
    source_contract_prompt: str | None = None,
    contract_findings_fn: Callable[[str], list[Finding]] | None = None,
) -> HarnessResult:
    """전사(또는 기존 노트)로부터 계측형 하네스 전체 파이프를 실행한다.

    Args:
        transcript: 전사 원문.
        generate_fn: `(prompt, *, system_prompt) -> dict` 콜러블(free_llm.generate
            형태 dict를 반환). plan/synthesis/repair 패스에서 재사용된다.
        plan: True면 Plan 패스를 먼저 실행하고 완전성 게이트의 하한 계약으로
            쓴다. `initial_note`가 주어지면 이 값과 무관하게 plan/synthesis는
            건너뛴다.
        critic_fn: `(transcript, note) -> dict` 콜러블. None이면 critic 패스
            생략.
        repair_budget: repair 재시도 예산(회). 트리거성 finding이 남아 있는데
            예산이 소진되면 `verified=False`로 전파.
        initial_note: 주어지면 plan+synthesis를 건너뛰고 이 노트로 게이트부터
            시작한다(V0+G/V2/V3처럼 기존 노트에서 파생하는 변형용).
        plan_topic_count_override: `initial_note` 경로에서 완전성 게이트가
            참조할 plan topics 수를 명시적으로 넘길 때 사용(예: V2/V3가
            V1의 저장된 plan JSON에서 topic 수를 로드해 넘기는 경우).
        plan_topics_override: `initial_note` 경로에서 완전성 게이트가 결측
            topic 이름을 계산할 때 참조할 plan topic 제목 리스트(round-06 T3,
            수렴 iter1 P0-2). `plan_topic_count_override`와 함께 쓰인다 —
            개수만 있고 이름이 없으면 게이트는 카운트 기반 detail로 폴백한다
            (`note_validate.check_completeness` 참조).
        prompts_dir: 프롬프트 파일을 찾을 루트 디렉토리(기본: 리포 `prompts/`).
        plan_prompt_override: 후보 계측에서 plan prompt만 바꿔 끼운다.
        synthesis_prompt_override: 계측에서 synthesis prompt 후보만 바꿔 끼운다.
            plan/critic/repair와 구조 profile은 `prompts_dir`의 프로덕션 계약을
            그대로 유지하므로 prompt A/B가 다른 하네스를 비교하지 않게 한다.
        synthesis_with_plan_prompt_override: 후보 plan schema에 맞춘 addendum만
            바꿔 끼운다. critic/repair와 구조 profile은 그대로 유지한다.
        require_source_block_densities: True면 합성 출력 첫 줄의 세 블록 등급
            표식을 요구하고 누락/형식 오류를 fail-closed한다(text_post P0-15).
        copy_quality_report_enabled: P0-16 grammar/humanizer/source-fidelity
            세 report-only 축. 꺼도 노트·gate·repair·verified·coverage 입력은
            바뀌지 않는다.
        sentence_quality_report_enabled: 이전 호출자 호환 별칭. 명시하면
            `copy_quality_report_enabled`를 덮어쓰며 단일 축을 되살리지는 않는다.

    Returns:
        HarnessResult — 최종 노트(메타 헤더 미포함), 품질메타, findings,
        토큰 집계, verified bool.
    """
    if sentence_quality_report_enabled is not None:
        copy_quality_report_enabled = sentence_quality_report_enabled
    structure_profile = "functional" if prompts_dir == TEXT_PROMPTS_DIR else "legacy"
    passes_run: list[str] = []
    pass_usages: list[PassUsage] = []
    plan_topic_count: int | None = plan_topic_count_override
    plan_topics: list[str] | None = plan_topics_override
    plan_topic_items: list[tuple[str, list[str]]] = []
    # §B — 라벨 나열(D1)이면 topic 1:1 섹션 계약을 적용하지 않는다.
    enforce_topic_sections = classify_source_density(transcript) != "D1"
    source_block_densities: dict[str, str] = {}

    if initial_note is not None:
        note = initial_note
    else:
        plan_json: dict | None = None
        if plan:
            plan_json, plan_usage = _run_plan_pass(
                transcript,
                generate_fn,
                prompts_dir=prompts_dir,
                plan_prompt_override=plan_prompt_override,
            )
            passes_run.append("plan")
            pass_usages.append(plan_usage)
            topics_list = plan_json.get("topics", [])
            plan_topic_count = len(topics_list)
            # round-06 T3(수렴 iter1 P0-2 확정 진단): 기존 코드는 여기서
            # len()만 취하고 topics_list 자체를 버려 repair가 어느 topic이
            # 빠졌는지 알 방법이 없었다. title을 뽑아 완전성 게이트까지
            # 관통시킨다. G5 P1-2: title 없는 항목을 drop하면 count와 리스트
            # 길이가 발산해, 하필 결측 topic이 전부 title-less면 missing==[]로
            # 카운트-only 폴백해 수정이 무력화된다 — placeholder로 lockstep 유지.
            plan_topics = _extract_plan_topic_labels(topics_list)
            plan_topic_items = _extract_plan_topic_items(topics_list)

        note, synthesis_usage = _run_synthesis_pass(
            transcript,
            generate_fn,
            plan=plan_json,
            prompts_dir=prompts_dir,
            structure_profile=structure_profile,
            synthesis_prompt_override=synthesis_prompt_override,
            synthesis_with_plan_prompt_override=synthesis_with_plan_prompt_override,
            source_contract_prompt=source_contract_prompt,
        )
        passes_run.append("synthesis")
        pass_usages.append(synthesis_usage)

    note, source_block_densities = extract_source_block_densities(note)
    if require_source_block_densities and set(source_block_densities) != {
        "body",
        "author_comments",
        "ocr",
    }:
        raise HarnessError(
            "text_post 합성 출력에 source-block-density 세 블록 등급이 "
            "없거나 형식이 올바르지 않습니다"
        )

    gate_result = run_gates(
        note,
        transcript,
        plan_topic_count=plan_topic_count,
        plan_topics=plan_topics,
        plan_topic_items=plan_topic_items,
        enforce_topic_sections=enforce_topic_sections,
        structure_profile=structure_profile,
    )
    passes_run.append("gate")
    note = gate_result.note

    contract_findings = contract_findings_fn(note) if contract_findings_fn else []
    gate_result.meta["source_contract_omissions"] = [finding.detail for finding in contract_findings]

    critic_findings: list[dict] = []
    critic_findings_for_copy_report: list[dict] = []
    critic_findings_count = 0
    if critic_fn is not None:
        critic_findings, critic_usage = _run_critic_pass(transcript, note, critic_fn)
        critic_findings_for_copy_report = list(critic_findings)
        passes_run.append("critic")
        pass_usages.append(critic_usage)
        critic_findings_count = len(critic_findings)

    findings = [*gate_result.findings, *contract_findings]
    verified = not _repair_findings_trigger(findings, critic_findings)

    remaining_budget = repair_budget
    critic_repair_attempted_count = 0
    # round-35 §E — 회차마다 무엇이 남았는지 기록한다. 이게 없으면 "예산을 늘리면
    # 수렴하는가"를 물을 수가 없다 — 지금까지는 마지막 상태만 남고 과정이 버려졌다.
    repair_iterations: list[dict[str, object]] = []
    while not verified and remaining_budget > 0:
        _repair_started = time.monotonic()
        _findings_before = sorted({f.kind for f in findings})
        note, repair_usage = _run_repair_pass(
            transcript, note, findings, critic_findings, generate_fn,
            prompts_dir=prompts_dir,
            source_contract_prompt=source_contract_prompt,
        )
        passes_run.append("repair")
        pass_usages.append(repair_usage)
        remaining_budget -= 1

        repaired_note, repaired_densities = extract_source_block_densities(note)
        note = repaired_note
        if repaired_densities:
            if source_block_densities and repaired_densities != source_block_densities:
                raise HarnessError("repair가 source-block-density 판정을 바꿨습니다")
            source_block_densities = repaired_densities

        gate_result = run_gates(
            note,
            transcript,
            plan_topic_count=plan_topic_count,
            plan_topics=plan_topics,
            plan_topic_items=plan_topic_items,
            enforce_topic_sections=enforce_topic_sections,
            structure_profile=structure_profile,
        )
        passes_run.append("gate")
        note = gate_result.note
        contract_findings = contract_findings_fn(note) if contract_findings_fn else []
        gate_result.meta["source_contract_omissions"] = [
            finding.detail for finding in contract_findings
        ]
        findings = [*gate_result.findings, *contract_findings]

        # repair는 1회만 신뢰하고 critic은 재실행하지 않는다(모듈 docstring
        # "Critic 재실행 없음") — 게이트 트리거만으로 재판정한다. critic
        # findings는 repair 시도에 반영을 "시도"한 것이지 해소가 검증된 것이
        # 아니므로, 해소(resolved)가 아니라 시도(repair_attempted)로만 기록한다
        # (G5 silent-failure P1: 검증 없는 성공 신호 위장 방지).
        if critic_findings:
            critic_repair_attempted_count = len(critic_findings)
        critic_findings = []
        verified = not _repair_findings_trigger(findings, critic_findings)
        repair_iterations.append({
            "iteration": len(repair_iterations) + 1,
            "before": _findings_before,
            "after": sorted({f.kind for f in findings}),
            "after_detail": [f"{f.kind}: {f.detail[:120]}" for f in findings],
            "verified": verified,
            "seconds": round(time.monotonic() - _repair_started, 1),
        })

    tokens = _summarize_tokens(pass_usages)

    meta: dict[str, object] = dict(gate_result.meta)
    meta["grounding_flags"] = critic_findings_count
    meta["grounding_repair_attempted"] = critic_repair_attempted_count
    meta["passes_run"] = passes_run
    meta["verified"] = verified
    # round-06 G5 P0-B: plan topic 라벨을 meta에 노출한다 — poc/run_sweep.py의
    # V1 분기가 이 이름을 v1_plan.json에 보존해 V2/V3가 plan_topics_override로
    # 재전달할 수 있게 한다(sweep 경로에서 T3 완전성 수정이 무력화되지 않도록).
    # plan을 돌리지 않은 경로(initial_note 등)에서는 None.
    meta["plan_topics"] = list(plan_topics) if plan_topics is not None else None
    meta["source_block_densities"] = source_block_densities
    meta["repair_iterations"] = repair_iterations
    copy_report = (
        report_copy_quality(
            transcript,
            note,
            critic_findings=critic_findings_for_copy_report,
        )
        if copy_quality_report_enabled
        else report_copy_quality("", "")
    )
    copy_report["enabled"] = copy_quality_report_enabled
    meta["copy_quality_report"] = copy_report
    meta["copy_quality_report_enabled"] = copy_quality_report_enabled
    meta["coverage_input_sha256"] = coverage_input_sha256(note)

    return HarnessResult(
        note=note,
        meta=meta,
        findings=findings,
        tokens=tokens,
        verified=verified,
        passes_run=passes_run,
    )


def render_note_with_meta(result: HarnessResult) -> str:
    """`HarnessResult`로부터 단일 선두 HTML 주석 품질메타 헤더 + 노트 본문을 조립한다.

    `poc/prepare_blind_set.strip_provider_traces`의 `count=1` 선두 블록
    스트립과 호환되도록 헤더는 정확히 하나의 `<!-- ... -->` 블록이어야
    한다(계약 §4 P1-4).
    """
    meta = result.meta
    completeness = meta.get("completeness")
    ts_total = meta.get("ts_total", 0)
    ts_verified = meta.get("ts_verified", 0)
    ts_removed = meta.get("ts_removed", 0)
    literal_artifacts = meta.get("literal_artifacts", "0/0")
    source_urls = meta.get("source_urls", "0/0")
    source_enumeration = meta.get("source_enumeration", "0/0")
    grounding_flags = meta.get("grounding_flags", 0)
    grounding_repair_attempted = meta.get("grounding_repair_attempted", 0)
    source_block_densities = meta.get("source_block_densities") or {}
    copy_quality_report = meta.get("copy_quality_report") or {}
    copy_lenses = copy_quality_report.get("lenses") or {}
    grammar_findings = (copy_lenses.get("grammar") or {}).get("findings") or []
    humanizer_findings = (copy_lenses.get("humanizer") or {}).get("findings") or []
    source_fidelity_findings = (copy_lenses.get("source_fidelity") or {}).get("findings") or []
    tokens = result.tokens

    lines = [
        "<!--",
        f"completeness: {completeness}",
        f"literal_artifacts: {literal_artifacts}",
        f"source_urls: {source_urls}",
        f"source_enumeration: {source_enumeration}",
        f"ts_verified: {ts_verified}/{ts_total} (removed: {ts_removed})",
        f"grounding_flags: {grounding_flags} (repair_attempted: {grounding_repair_attempted})",
        (
            "source_block_densities: "
            + ", ".join(
                f"{key}={source_block_densities[key]}"
                for key in ("body", "author_comments", "ocr")
                if key in source_block_densities
            )
        ),
        f"grammar_findings: {len(grammar_findings)} (report_only: True)",
        f"humanizer_findings: {len(humanizer_findings)} (report_only: True)",
        f"source_fidelity_findings: {len(source_fidelity_findings)} (report_only: True)",
        f"coverage_input_sha256: {meta.get('coverage_input_sha256')}",
        f"passes_run: {result.passes_run}",
        f"tokens_total_known: {tokens.get('total_known')}",
        f"tokens_free_total: {tokens.get('free_total')}",
        f"tokens_premium_total: {tokens.get('premium_total')}",
        f"verified: {result.verified}",
        "-->",
        "",
    ]
    return "\n".join(lines) + result.note


# ---------------------------------------------------------------------------
# 공용 팩토리 — generate_fn/critic_fn (round-05 계약 §3 P0-1 승격)
# ---------------------------------------------------------------------------
#
# poc/run_sweep.py의 `_make_generate_fn`·`_make_free_critic_fn`을 그대로
# 이곳으로 승격했다 — note_pipe.py(프로덕션 CLI)와 run_sweep.py(PoC 러너)가
# 동일 래퍼를 공유한다. client 파라미터는 duck-type만 요구한다:
#   client.generate(prompt: str, *, system_prompt: str, stream: bool) -> dict
# free_llm.FreeLLMClient가 이 계약을 만족하지만, 이 모듈은 그 타입을 알지
# 못한다(임포트 없음) — 어떤 duck-type 호환 객체도 주입 가능하다.


def make_generate_fn(client: object) -> Callable[..., dict]:
    """`run_harness`가 기대하는 `(prompt, *, system_prompt) -> dict` 시그니처로
    duck-typed client.generate를 감싼다(스트리밍 고정 — LONG_FORM 원칙).

    truncation 하드-스톱(계약 §3 P0-7): `HarnessResult`/`PassUsage`는 패스별
    `truncated_suspected`를 전파하지 않으므로, 이 래퍼 계층에서 응답의
    `truncated_suspected`가 참이면 즉시 `HarnessError`를 raise한다 — 부분
    산출을 다음 패스(synthesis→gate→critic→repair)로 조용히 흘려보내지
    않는다.
    """

    def _generate(prompt: str, *, system_prompt: str) -> dict:
        response = client.generate(prompt, system_prompt=system_prompt, stream=True)
        if response.get("truncated_suspected"):
            raise TruncationSuspectedError(
                "생성 응답이 잘린 정황(truncated_suspected=True)입니다 — "
                f"finish_reason={response.get('finish_reason')}. 부분 산출을 다음 "
                "패스로 넘기지 않고 즉시 중단합니다."
            )
        return response

    return _generate


def make_free_critic_fn(
    client: object, prompts_dir: Path = DEFAULT_PROMPTS_DIR
) -> Callable[[str, str], dict]:
    """V2(무료 비판)용 — critic.md를 system prompt로, 전사+노트를 user
    메시지로 보내 findings JSON을 파싱한다. 무료 critic의 일시적인 형식 편차는
    최대 4회까지 같은 고정 모델로 재시도한 뒤 하드 에러로 남긴다. truncation
    하드-스톱은 `make_generate_fn`과
    동일하게 이 래퍼에서 수행한다(계약 §3 P0-7).

    반환 dict의 `model_reported`(round-05 G5 iter2 FIX-B1): 내부 `_critic`
    클로저는 raw client 응답을 `{"findings", "usage", "source"}`로 재조립해
    반환하는데, 여기서 raw 응답의 `model_reported`를 누락하면 note_pipe.py의
    `_wrap_critic_fn_capturing_model_reported`가 아무리 이 함수의 반환값에서
    `response.get("model_reported")`를 읽어도 항상 None이라 critic 경로의
    model_reported 캡처가 구조적으로 불가능해진다. 이 값을 그대로 전달해
    파이프 계층의 집계(`aggregate_model_reported`)가 critic 패스도 반영하게
    한다.

    `prompts_dir`(round-09 계약 §3 [수렴 fold iter2 C-P0-1]): 기본값은 기존
    상수 `DEFAULT_PROMPTS_DIR`이므로 이 파라미터를 넘기지 않는 기존 호출자
    (transcript 경로)는 byte-identical로 동작한다. text_post 경로가
    `prompts/text`를 넘기면 이 팩토리가 `prompts/text/harness/critic.md`를
    로드한다 — 이 함수는 프롬프트 로딩 팩토리일 뿐 게이트/캘리브레이션
    로직이 아니므로, 경로 파라미터화가 다른 봉인된 불변식을 건드리지
    않는다.
    """
    system_prompt = load_prompt(_CRITIC_PROMPT_NAME, prompts_dir=prompts_dir)

    def _critic(transcript: str, note: str) -> dict:
        user_message = f"{transcript}\n\n---\n\n[노트]\n{note}"
        errors: list[str] = []
        last_error: Exception | None = None
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            response = client.generate(user_message, system_prompt=system_prompt, stream=True)
            if response.get("truncated_suspected"):
                raise TruncationSuspectedError(
                    "critic 응답이 잘린 정황(truncated_suspected=True)입니다 — "
                    f"finish_reason={response.get('finish_reason')}. 부분 findings를 "
                    "신뢰하지 않고 즉시 중단합니다."
                )
            raw_text = response.get("text", "")
            try:
                parsed = json.loads(extract_json_object(raw_text))
                if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
                    raise ValueError(
                        f"critic JSON 형상이 계약과 다릅니다(dict+findings 리스트 필요): "
                        f"{type(parsed).__name__}"
                    )
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                errors.append(f"시도 {attempt}: {exc}")
                logger.warning(
                    "critic 패스 JSON 파싱/형상 실패(시도 %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                continue
            return {
                "findings": parsed.get("findings", []),
                "usage": response.get("usage"),
                "source": response.get("route_source") or "free:nvidia_nim:deepseek-ai/deepseek-v4-pro",
                # FIX-B1: raw 응답의 model_reported를 보존해 note_pipe의
                # 캡처 래퍼(_wrap_critic_fn_capturing_model_reported)가
                # 실제로 값을 읽을 수 있게 한다.
                "model_reported": response.get("model_reported"),
                # round-34 provenance: free critic 응답을 재조립할 때 실제 route를
                # 버리지 않는다. note_pipe가 생성·critic 전체 응답의 실제
                # provider/model을 정직하게 집계한다.
                "provider": response.get("provider"),
                "model": response.get("model"),
                "requested_provider": response.get("requested_provider"),
                "requested_model": response.get("requested_model"),
                "billing_tier": response.get("billing_tier"),
                "role": response.get("role") or "critic",
            }
        raise HarnessError(
            f"critic 패스 JSON 파싱이 재시도({max_attempts - 1}회) 후에도 실패했습니다: "
            + " / ".join(errors)
        ) from last_error

    return _critic
