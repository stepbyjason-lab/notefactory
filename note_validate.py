"""note_validate.py — 결정적 게이트(0토큰) 순수 함수 모음.

계약: `.handoff/rounds/round-04-harness-sweep-contract.md` §4(T2, note_validate
불릿) · 설계: `docs/05-note-harness-design.md` §2-③ (a)~(d).

이 모듈은 LLM을 호출하지 않는다. 네트워크·파일 I/O도 하지 않는다 — 모든
핵심 함수는 문자열/파싱된 구조를 인자로 받아 문자열/구조를 반환하는 순수
함수다(호출자가 전사·노트를 어디서 읽어오는지는 이 모듈의 관심사가 아니다).

게이트 4종(docs/05 §2-③, `run_gates`가 순서대로 실행):
  (a) 타임스탬프 정규화 — 모든 `[t=...]` 표기를 `[t=HH:MM:SS]` 단일 형식으로 통일
  (b) 타임스탬프 실재 대조 — 전사(SRT 블록 시각/인라인 표기)에 실존하는지 ±2s
      허용 대조. 미실존 → 표기 제거 + finding 기록
  (c) 완전성 proxy — legacy는 Learning Path 섹션 수, functional은 plan topic과
      자유 `## ` 콘텐츠 헤딩의 최대-cardinality 1:1 매칭(개요 없으면 생략)
  (d) 구조 린트 — legacy는 고정 8섹션, functional은 제목 + 콘텐츠 섹션 존재

(e) 팽창률 게이트(`check_inflation_ratio`, round-10 계약 §3) — `run_gates`의
    일부가 아니다. `note_pipe.py::run()`이 `source_kind == "text_post"`일 때만
    하네스 실행 직후 직접 호출하는 별도 post-hoc 게이트다(원문 대비 산출물
    길이 배율 상한 판정).

round-03 4방향 비교(comparison.md)에서 실측된 타임스탬프 형식 편차:
  - deepseek(무료): `[t=32:08:849]` 같은 `MM:SS:ms`를 `:` 구분자로 오기
    (콜론 3개, 세 번째 그룹이 ms) — `note_nvidia_nim_deepseek-ai-deepseek-v4-pro.md`
  - codex-5.5: `[t=00:15:20,159]` — `HH:MM:SS,ms`(쉼표 앞 ms 명시)
    — `note_codex-5.5.md`
  - opus: `[t=00:13:43]` — `HH:MM:SS`(밀리초 없음) + 불확실 시각을
    `[t=01:55:xx? — 아래 Caveats 참조]` 형태로 남김 — `note_opus.md`

**MM:SS:ms 휴리스틱(문서화 필수 결정, 계약 §4)**: `[t=...]` 안에 콜론(:)으로
구분된 그룹이 정확히 3개일 때, 세 번째 그룹의 값으로 분기한다.
  - 세 번째 그룹 > 59 → 초 단위일 수 없으므로 밀리초로 간주하고
    `MM:SS,ms`로 해석한다(deepseek 케이스: `32:08:849` → 32분 08초 849ms).
  - 세 번째 그룹 <= 59 → 정상 초 값이므로 `HH:MM:SS`로 해석한다
    (opus/codex 케이스: `00:13:43` → 0시 13분 43초).
콜론 3그룹이면서 쉼표로 명시적 ms가 더 붙은 경우(`HH:MM:SS,ms`)는 애초에
쉼표가 구분자이므로 이 휴리스틱 분기 이전에 별도 패턴으로 먼저 매치된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 상수 / 정규식
# ---------------------------------------------------------------------------

#: 노트 구조 계약(prompts/note_synthesis.md 노트 구조 계약 블록)의 `## ` 7섹션.
#: 제목(`# <노트 제목>`, level-1) + 이 7개 = 8섹션(계약 §4 "8섹션").
REQUIRED_SECTIONS: tuple[str, ...] = (
    "TLDR",
    "Core Thesis",
    "Concept Map",
    "Learning Path",
    "Frameworks, Methods, and Decision Rules",
    "Examples and Applications",
    "Caveats and Open Questions",
)

_LEARNING_PATH_HEADING = "Learning Path"

#: SRT 블록의 타임코드 한 줄: `HH:MM:SS,ms --> HH:MM:SS,ms`. 시작 시각만 수집.
_SRT_CUE_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{1,3})\s*-->\s*"
    r"\d{1,2}:\d{2}:\d{2},\d{1,3}"
)

#: 전사 인라인 표기 `[MM:SS]` / `[HH:MM:SS]`(note_synthesis.md 프롬프트가
#: 문서화하는 원시 전사 타임스탬프 스타일). `[t=...]` 노트 표기와는 다른
#: 패턴이므로 별도로 매치한다.
_INLINE_TS_RE = re.compile(r"\[(?P<body>\d{1,2}(?::\d{2}){1,2})\]")

#: 노트 안의 `[t=...]` 토큰 전체(정규화 대상 원본). 대괄호 안에 `]`가 없다는
#: 전제로 non-greedy 없이 `[^\]]*`를 사용한다.
_NOTE_TS_TOKEN_RE = re.compile(r"\[t=(?P<body>[^\]]*)\]")

#: 불확실 마커(opus 스타일): `xx` 플레이스홀더가 포함된 시각 표기.
#: 예: `[t=01:55:xx? — 아래 Caveats 참조]`, `[t=00:26:xx?]`.
_UNCERTAIN_MARKER_RE = re.compile(r"xx", re.IGNORECASE)

#: `HH:MM:SS,ms` 명시적 쉼표 구분 밀리초(codex 스타일).
_HMS_COMMA_MS_RE = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{1,3})$"
)

#: 콜론 3그룹(`A:B:C`) — deepseek(MM:SS:ms 오기) / opus·codex(HH:MM:SS) 공용
#: 패턴. 세 번째 그룹 값으로 실제 의미를 휴리스틱 분기한다(모듈 docstring 참조).
_THREE_COLON_GROUPS_RE = re.compile(
    r"^(?P<g1>\d{1,2}):(?P<g2>\d{2}):(?P<g3>\d{2,3})$"
)

#: 콜론 2그룹(`MM:SS`).
_TWO_COLON_GROUPS_RE = re.compile(r"^(?P<m>\d{1,3}):(?P<s>\d{2})$")

#: 캔버스 정규 표기 `HH:MM:SS`(정규화 결과 + 이미 정규화된 입력 판별용).
_CANONICAL_TS_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})$")

#: 정규화 이후 노트 본문에서 `[t=HH:MM:SS]`를 다시 찾기 위한 패턴(verify 단계).
_CANONICAL_TOKEN_RE = re.compile(r"\[t=(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\]")

#: Learning Path 하위 섹션(`### `) 헤딩.
_H3_RE = re.compile(r"^### .+$", re.MULTILINE)

#: `## ` 레벨 헤딩(제목 텍스트만 캡처).
_H2_RE = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)

#: `# ` 레벨(level-1) 제목 헤딩. `## `까지 매치되지 않도록 뒤에 공백이
#: `#`가 아님을 보장.
_H1_RE = re.compile(r"^# (?!#).+$", re.MULTILINE)

#: 닫힌(paired) fenced block만 제거한다. 미닫힌 여는 펜스를 EOF까지
#: 제거하지 않는 것은 round-13 부록 A의 의도적 회귀 방지 결정이다.
_FENCE_BLOCK_RE = re.compile(r"(?m)^\s*```.*?^\s*```", re.DOTALL)

# 원문에 그대로 써야 하는 자료가 있을 때, 요약이 그 실물을 덮어버리지 않게
# 잡는 결정론 게이트다. 프롬프트·템플릿·코드·체크리스트라는 말만 언급한 글은
# 대상이 아니다. 실제 지시문 본문이 충분한 길이로 수집됐을 때만 발동한다.
_LITERAL_ARTIFACT_CUE_RE = re.compile(
    r"(?i)(?:프롬프트|prompt|템플릿|template|체크리스트|checklist|코드|code)"
)
_LITERAL_ARTIFACT_START_RE = re.compile(
    r"(?im)^(?:please|create|write|generate|make|build|use)\b"
)
_LITERAL_ARTIFACT_STOP_RE = re.compile(
    r"(?im)^\s*(?:더 보기|show more|편집)\s*$"
)
_LITERAL_ARTIFACT_SECTION_RE = re.compile(
    r"(?im)^##\s+.*(?:프롬프트|prompt|템플릿|template|체크리스트|checklist|코드|code)"
)
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_LITERAL_ARTIFACT_MIN_CHARS = 160
_LITERAL_ARTIFACT_TOKEN_RE = re.compile(r"[가-힣A-Za-z]{3,}|\d{2,}")
_LITERAL_ARTIFACT_TOKEN_COVERAGE = 0.6
_SOURCE_URL_RE = re.compile(
    r"(?:https?://[^\s<>()\[\]{}\"']+|"
    r"(?<![@A-Za-z0-9.-])(?:[A-Za-z0-9-]+\.)+(?:com|org|net|dev|io|ai|co|kr)"
    r"[^\s<>()\[\]{}\"']*)",
    re.IGNORECASE,
)
_URL_SECTION_RE = re.compile(r"(?im)^##\s+.*(?:링크|url|URL|자료|repository|저장소)")
_SOURCE_NUMBERED_ITEM_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?P<number>\d+)[.)]\s+(?P<body>\S.+)$"
)
_NOTE_NUMBERED_ITEM_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?P<number>\d+)[.)]\s+(?P<body>\S.+)$"
)
_SOURCE_NUMBERED_MIN_ITEMS = 2


@dataclass(frozen=True)
class Finding:
    """게이트가 남기는 단일 발견 사항.

    Attributes:
        kind: 발견 종류. "ts_normalized" | "uncertain_ts" | "ts_removed" |
            "completeness" | "structure".
        detail: 사람이 읽을 수 있는 상세 설명(값·개수 등).
        context: 해당 발견의 주변 문맥(발췌). 없으면 빈 문자열.
    """

    kind: str
    detail: str
    context: str = ""


@dataclass(frozen=True)
class GateResult:
    """`run_gates`의 반환값 — 정규화+정제된 노트와 전체 findings·메타.

    Attributes:
        note: 게이트 4종을 모두 거친 노트 본문(정규화 + 미실존 타임스탬프 제거).
        findings: 게이트 실행 중 수집된 모든 Finding(순서 보존).
        meta: 품질메타 요약 dict. 키:
            ts_total, ts_verified, ts_removed, ts_normalized(모두 int),
            completeness(str "N/M" | None), structure_ok(bool).
    """

    note: str
    findings: list[Finding]
    meta: dict[str, object]


def _strip_fences(note: str) -> str:
    """헤딩 추출 전에 닫힌(paired) fenced block만 제거한다."""
    return _FENCE_BLOCK_RE.sub("", note)


def _seconds_from_groups(hours: int, minutes: int, seconds: int) -> int:
    return hours * 3600 + minutes * 60 + seconds


def parse_transcript_timestamps(transcript: str) -> list[int]:
    """전사 문자열에서 모든 타임스탬프 인스턴트를 초 단위로 추출한다.

    다음 두 형식을 처리한다:
      - SRT 큐 블록: `HH:MM:SS,ms --> HH:MM:SS,ms` (시작 시각만 수집)
      - 인라인 `[MM:SS]` / `[HH:MM:SS]` (note_synthesis.md가 문서화하는
        원시 전사 스타일)

    Returns:
        중복 제거·오름차순 정렬된 초 단위 리스트.
    """
    seconds_set: set[int] = set()

    for match in _SRT_CUE_RE.finditer(transcript):
        h, m, s = int(match["h"]), int(match["m"]), int(match["s"])
        seconds_set.add(_seconds_from_groups(h, m, s))

    for match in _INLINE_TS_RE.finditer(transcript):
        parts = match["body"].split(":")
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            seconds_set.add(_seconds_from_groups(0, m, s))
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            seconds_set.add(_seconds_from_groups(h, m, s))

    return sorted(seconds_set)


def _classify_ts_body(body: str) -> tuple[int, int, int] | None:
    """`[t=...]` 안쪽 문자열을 (h, m, s) 튜플로 분류한다.

    파싱 불가·불확실 마커면 None을 반환한다(호출자가 uncertain_ts로 처리).
    """
    stripped = body.strip()

    comma_match = _HMS_COMMA_MS_RE.match(stripped)
    if comma_match:
        return int(comma_match["h"]), int(comma_match["m"]), int(comma_match["s"])

    three_group_match = _THREE_COLON_GROUPS_RE.match(stripped)
    if three_group_match:
        g1, g2, g3 = (
            int(three_group_match["g1"]),
            int(three_group_match["g2"]),
            int(three_group_match["g3"]),
        )
        # MM:SS:ms 휴리스틱(모듈 docstring 참조): 세 번째 그룹 > 59 → ms로 간주.
        if g3 > 59:
            return 0, g1, g2
        return g1, g2, g3

    two_group_match = _TWO_COLON_GROUPS_RE.match(stripped)
    if two_group_match:
        return 0, int(two_group_match["m"]), int(two_group_match["s"])

    return None


def normalize_timestamps(note: str) -> tuple[str, list[Finding]]:
    """노트 안의 모든 `[t=...]` 토큰을 `[t=HH:MM:SS]`로 재작성한다.

    불확실 마커(`xx` 포함, 예: `[t=01:55:xx? ...]`)는 그대로 두고
    kind="uncertain_ts" finding만 남긴다(값을 지어내지 않는다).
    파싱 가능한 값은 재작성 후 kind="ts_normalized" finding을 남긴다.
    """
    findings: list[Finding] = []

    def _replace(match: re.Match[str]) -> str:
        original = match.group(0)
        body = match["body"]

        if _UNCERTAIN_MARKER_RE.search(body):
            findings.append(
                Finding(kind="uncertain_ts", detail=original, context=original)
            )
            return original

        parsed = _classify_ts_body(body)
        if parsed is None:
            # 파싱 불가능한 값도 창작하지 않고 그대로 둔다 — uncertain 취급.
            findings.append(
                Finding(kind="uncertain_ts", detail=original, context=original)
            )
            return original

        h, m, s = parsed
        canonical = f"[t={h:02d}:{m:02d}:{s:02d}]"
        if canonical != original:
            findings.append(
                Finding(kind="ts_normalized", detail=f"{original} -> {canonical}")
            )
        return canonical

    normalized = _NOTE_TS_TOKEN_RE.sub(_replace, note)
    return normalized, findings


def _context_around(text: str, start: int, end: int, radius: int = 40) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi]


def verify_timestamps(
    note: str, transcript_seconds: list[int], tolerance_s: int = 2
) -> tuple[str, list[Finding]]:
    """정규화된 `[t=HH:MM:SS]` 각각이 전사에 실존하는지 대조한다.

    ±tolerance_s 이내에 전사 인스턴트가 하나도 없으면 해당 표기를 노트에서
    제거하고(kind="ts_removed") 주변 40자 문맥을 기록한다. 값을 지어내지
    않고 그저 제거만 한다는 점이 게이트 (b)의 계약이다. 불확실 마커
    (`normalize_timestamps`가 이미 uncertain_ts로 남긴 것)는 이 단계에서
    건드리지 않는다 — `_CANONICAL_TOKEN_RE`가 canonical 형식만 매치하므로
    자동으로 skip된다.
    """
    findings: list[Finding] = []

    if not transcript_seconds:
        # 대조할 전사 시각이 아예 없으면 모든 canonical 표기가 미실존 —
        # 하지만 이는 전사 파싱 실패 가능성이 높으므로 빈 리스트를
        # "전부 제거 대상"으로 취급하지 않고, 호출자가 판단하도록 그대로
        # verify를 진행한다(빈 리스트 = 모든 대조가 실패하는 것이 맞는 동작).
        pass

    def _tolerant_match(seconds: int) -> bool:
        return any(abs(seconds - ts) <= tolerance_s for ts in transcript_seconds)

    def _replace(match: re.Match[str]) -> str:
        h, m, s = int(match["h"]), int(match["m"]), int(match["s"])
        seconds = _seconds_from_groups(h, m, s)
        if _tolerant_match(seconds):
            return match.group(0)

        context = _context_around(note, match.start(), match.end())
        findings.append(
            Finding(kind="ts_removed", detail=match.group(0), context=context)
        )
        return ""

    verified_note = _CANONICAL_TOKEN_RE.sub(_replace, note)
    return verified_note, findings


def _extract_learning_path_titles(note: str) -> list[str]:
    """`## Learning Path` 아래 `### ` 헤딩의 제목 텍스트만 순서대로 추출한다.

    round-06 T3 — 결측 topic 이름을 계산하려면 최종 노트가 실제로 커버한
    섹션 "제목"이 필요하다(기존 `_count_learning_path_sections`는 개수만
    반환). 정확 문자열 일치가 아니라 완화 매칭(`_topic_covered`)에 쓰이므로
    여기서는 원문 그대로(strip만) 반환한다.
    """
    h2_matches = list(_H2_RE.finditer(note))
    lp_start: int | None = None
    lp_end = len(note)

    for idx, match in enumerate(h2_matches):
        if match["title"].strip() == _LEARNING_PATH_HEADING:
            lp_start = match.end()
            if idx + 1 < len(h2_matches):
                lp_end = h2_matches[idx + 1].start()
            break

    if lp_start is None:
        return []

    segment = note[lp_start:lp_end]
    return [
        h3.group(0).removeprefix("### ").strip() for h3 in _H3_RE.finditer(segment)
    ]


#: topic 커버 판정 임계(round-06 G5 P1-1). 두 제목의 토큰 집합 **Jaccard
#: 유사도**(교집합/합집합)가 이 값 이상이면 커버로 본다. 0.6은:
#:   - "결제 퍼널 최적화" vs "결제 퍼널 최적화 전략"(재서술) = 3/4 = 0.75 → 커버
#:   - "전략" vs "가격 전략 심화"(우연 1토큰 겹침) = 1/3 = 0.33 → 미커버
#:   - "첫 번째 개념" vs "두 번째 개념"(구분 토큰 하나 다름) = 2/4 = 0.5 → 미커버
#: 를 모두 올바르게 가르는 균형값이다.
_TOPIC_COVER_JACCARD_THRESHOLD = 0.6


def _normalize_topic_label(title: str) -> str:
    """topic 제목을 완화 비교용으로 정규화한다(공백 축약 + 소문자화)."""
    return re.sub(r"\s+", " ", title).strip().lower()


def _significant_tokens(title: str) -> set[str]:
    """정규화된 제목을 토큰 집합으로 분해한다(round-06 G5 P1-1).

    영문/숫자 연속과 한글 음절 연속을 토큰으로 본다. 한국어는 "첫/두/세"처럼
    1음절 토큰이 topic을 구분하는 유의 정보일 수 있으므로 길이로 버리지
    않는다 — 대신 판정을 raw substring이 아니라 Jaccard 유사도로 바꿔
    우연 매칭을 억제한다(`_topic_covered`).
    """
    normalized = _normalize_topic_label(title)
    return set(re.findall(r"[0-9a-z]+|[가-힣]+", normalized))


def _topic_covered(topic_title: str, section_titles: list[str]) -> bool:
    """`topic_title`이 최종 노트의 Learning Path 섹션 제목 중 하나로 커버됐는지
    **토큰 집합 Jaccard 유사도 임계**로 판정한다(round-06 G5 P1-1).

    기존 구현은 정규화 후 raw 양방향 substring 포함(`a in b or b in a`)을
    썼는데, 이는 두 방향 모두에서 오매칭했다:
      (a) 짧은 generic topic("전략")이 무관한 긴 섹션("가격 전략 심화")에
          우연 포함돼 실제 누락을 "커버됨"으로 숨김 → 거짓 verified=True.
      (b) 짧은 섹션 제목이 긴 topic에 포함돼 오매칭.
    LLM 생성 제목은 도메인 어휘 반복이 흔해 실전 위험이 크다.

    대신 여기서는 양쪽을 토큰 집합으로 분해하고 **Jaccard 유사도(교집합/
    합집합)가 `_TOPIC_COVER_JACCARD_THRESHOLD` 이상일 때만** 커버로 본다.
    Jaccard는 대칭적이라 "짧은 쪽/긴 쪽" 방향 편향이 없고, 우연히 흔한
    토큰 하나만 겹치는 경우(합집합이 커서 비율이 낮음)를 자연스럽게
    걸러낸다.
    """
    topic_tokens = _significant_tokens(topic_title)
    if not topic_tokens:
        return False

    for section_title in section_titles:
        section_tokens = _significant_tokens(section_title)
        if not section_tokens:
            continue
        union = topic_tokens | section_tokens
        jaccard = len(topic_tokens & section_tokens) / len(union)
        if jaccard >= _TOPIC_COVER_JACCARD_THRESHOLD:
            return True
    return False


def check_completeness(
    note: str,
    plan_topic_count: int | None,
    *,
    plan_topics: list[str] | None = None,
) -> list[Finding]:
    """`## Learning Path` 아래 `### ` 섹션 수를 개요 topics 수와 대조한다.

    `plan_topic_count`가 None이면 개요가 없다는 뜻이므로 검사를 생략한다
    (docs/05 §2-③(c) "개요 없으면 검사 생략").

    round-06 T3(수렴 iter1 P0-2): `plan_topics`가 주어지면(plan topic 이름
    리스트) 단순 개수 미달뿐 아니라 **어느 topic이 최종 노트에서 커버되지
    않았는지**까지 계산해 `Finding.detail`에 이름을 명시한다. repair 패스가
    "15/16" 같은 카운트 문자열만 받으면 무엇을 추가해야 하는지 전혀 알 수
    없다 — 이 이름 목록이 `_render_findings_for_repair`(note_harness.py)를
    거쳐 repair user message에 그대로 노출돼야 실제로 고칠 수 있다.

    `plan_topics`가 없으면(하위 호환 — 예: V0G처럼 plan 자체가 없는 경로,
    또는 개수만 아는 옛 호출자) 기존처럼 카운트 기반 detail로 폴백한다.
    """
    if plan_topic_count is None:
        return []

    section_count = _count_learning_path_sections(note)
    if section_count >= plan_topic_count:
        return []

    if not plan_topics:
        return [
            Finding(
                kind="completeness",
                detail=f"{section_count}/{plan_topic_count}",
            )
        ]

    section_titles = _extract_learning_path_titles(note)
    missing_topics = [
        title for title in plan_topics if not _topic_covered(title, section_titles)
    ]

    if not missing_topics:
        # 이름 매칭상 전부 커버됐지만 개수는 여전히 미달(예: plan_topic_count가
        # plan_topics보다 큰 특수 케이스) — 카운트 기반 detail로 폴백한다.
        return [
            Finding(
                kind="completeness",
                detail=f"{section_count}/{plan_topic_count}",
            )
        ]

    missing_list = ", ".join(missing_topics)
    return [
        Finding(
            kind="completeness",
            detail=f"{section_count}/{plan_topic_count} — 누락된 topic: {missing_list}",
        )
    ]


#: 한 topic 섹션이 담아야 할 「원문 근거 항목」의 최소 비율. plan이 뽑은
#: key_numbers·proper_nouns 중 **원문에 실제로 있는 것만** 요구 대상으로 삼고,
#: 그중 이 비율 이상이 해당 섹션 본문에 있으면 그 topic은 커버된 것으로 본다.
#: 값은 2026-08-24 실측으로 잡았다(`tools/completeness_calib.py`, 표본 12개 —
#: 최소 0.778 · 중앙 1.000). 정상 노트를 떨어뜨리지 않으면서 "헤딩만 만들고 속을
#: 비운" 회피(비율 0)는 잡는 값이다.
#:
#: **앵커(§A) 적용 후에 다시 쟀다.** 그 전 측정(표본 27개, 최소 0.857 → 0.8)은
#: 무효다 — 그때는 plan이 topic을 4·7·14개로 제멋대로 쪼갰고, 잘게 쪼갤수록 topic당
#: 요구 항목이 1~2개라 비율이 자동으로 1.0이 됐다. 개수를 소스에 고정하자 topic당
#: 항목이 늘면서 최소값이 0.857에서 0.778로 내려갔다. 흔들리는 하네스 위에서 잰
#: 값으로 임계를 잡으면 안 된다는 round-33의 교훈이 이 자리에서도 그대로였다.
_TOPIC_CONTENT_COVER_RATIO = 0.7


#: 눈으로는 같은 글자인데 코드포인트가 다른 문장부호. 모델은 `ugc-character`를
#: `ugc‑character`(U+2011 비분리 하이픈)로 옮겨 적고, ASCII 따옴표를 활자 따옴표로
#: 바꾼다. 접어두지 않으면 **노트가 담고 있는 내용을 "안 담았다"고 판정**한다 —
#: 2026-08-24 실측에서 repair가 4회 연속 같은 지적을 못 지운 원인이 이것이었다.
#: 고칠 게 없는데 계속 지적당하니 수렴할 수가 없었다.
_PUNCT_FOLD = str.maketrans({
    chr(0x2010): "-", chr(0x2011): "-", chr(0x2012): "-", chr(0x2013): "-",
    chr(0x2014): "-", chr(0x2015): "-", chr(0x2212): "-",
    chr(0x2018): "'", chr(0x2019): "'", chr(0x201C): '"', chr(0x201D): '"',
})


def _norm_frag(text: str) -> str:
    """대조용 정규화 — 공백을 지우고, 문장부호 변형을 접고, 소문자화한다.

    공백을 지우는 이유는 원문이 줄바꿈·OCR로 쪼개져 있어도 같은 문자열로 보기
    위해서다(note_metrics의 인용 대조와 같은 규칙). 문장부호를 접는 이유는
    `_PUNCT_FOLD` 주석 참조.
    """
    return re.sub(r"\s+", "", text or "").translate(_PUNCT_FOLD).lower()


def _section_bodies(note: str) -> list[tuple[str, str]]:
    """`## ` 헤딩마다 (제목, 다음 헤딩 전까지의 본문) 쌍을 만든다."""
    matches = list(_H2_RE.finditer(note))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        out.append((m["title"].strip(), note[start:end]))
    return out


#: 게이트 지적의 처리 층(round-35 §D). 실패를 한 칸으로 뭉뚱그리면 "완벽하지 않다"가
#: "노트가 없다"가 된다. 세 층으로 가른다.
#:
#:   block  내보내면 안 되는 것 — 폴백한다
#:   retry  시간 예산 안에서 고칠 것 — 예산을 다 써도 **차단으로 올라가지 않는다**
#:   flag   남아도 내보내되 표시할 것
#:
#: 형식이 좀 어긋난 노트는 "없는 노트"보다 낫다. 그래서 completeness·structure는
#: retry에 둔다 — 고쳐보되, 못 고쳤다고 산출을 버리지는 않는다.
#:
#: **block이 지금 비어 있는 것은 누락이 아니다.** 내보내면 안 되는 사유(절단·팽창·
#: 파싱 실패)는 게이트가 아니라 각자의 경로에서 이미 하드-스톱된다. 그리고 기계 창작
#: 신호는 여기 못 들어온다 — 오탐이 있기 때문이다(2026-08-24 실측: 원문 "8할"을
#: 노트가 "80%"로 환산한 것이 창작으로 잡혔다). 게이트에 넣을 수 있는 것은 오탐이
#: 없는 신호뿐이다.
GATE_FINDING_TIER: dict[str, str] = {
    "ts_removed": "retry",
    "completeness": "retry",
    "structure": "retry",
    "core_artifact": "retry",
    "source_url": "retry",
    "source_enumeration": "retry",
    "uncertain_ts": "flag",
    "ts_normalized": "flag",
    "transcript_no_timestamps": "flag",
}


def blocking_findings(findings: list[Finding]) -> list[Finding]:
    """내보내면 안 되는 지적만 골라낸다. 비어 있으면 노트를 낸다.

    표에 없는 kind는 **block으로 본다** — 새 게이트를 추가하고 층을 정하지 않았을 때
    조용히 통과시키지 않기 위해서다(fail-closed).
    """
    return [f for f in findings if GATE_FINDING_TIER.get(f.kind, "block") == "block"]


def _completeness_content(
    note: str, plan_topic_items: list[tuple[str, list[str]]], source: str
) -> list[Finding]:
    """각 topic 섹션이 **그 topic의 원문 근거 항목**을 담았는지 본다.

    제목 대조(`_completeness_functional`)는 "섹션을 만들었는가"만 잰다. 제목을
    plan 그대로 쓰게 하면(round-35 ⓐ) 그 검사는 통과가 기정사실이 되므로, 헤딩만
    개수 맞춰 만들고 속을 비우는 회피가 열린다. 이 함수가 그 자리를 메운다 —
    round-04가 "얕게 줄이기 차단"으로 의도한 것을 이름이 아니라 내용으로 잰다.

    plan이 지어낸 항목은 요구 대상에서 빠진다(원문에 없는 것은 애초에 요구하지
    않는다). 그래서 plan의 창작이 게이트를 통해 노트에 강요되지 않는다.
    """
    if not plan_topic_items:
        return []

    src = _norm_frag(source)
    sections = _section_bodies(note)
    by_title = {_norm_frag(t): body for t, body in sections}

    short: list[str] = []
    for title, items in plan_topic_items:
        required = [x for x in items if len(_norm_frag(x)) >= 2 and _norm_frag(x) in src]
        if not required:
            continue
        body = by_title.get(_norm_frag(title))
        if body is None:
            # 제목이 안 맞으면 이 함수가 판정하지 않는다 — 그건 제목 대조의 몫이다.
            continue
        nb = _norm_frag(body)
        hit = sum(1 for x in required if _norm_frag(x) in nb)
        ratio = hit / len(required)
        if ratio < _TOPIC_CONTENT_COVER_RATIO:
            short.append(f"{title}({hit}/{len(required)})")

    if not short:
        return []
    return [
        Finding(
            kind="completeness",
            detail=("섹션이 원문 근거 항목을 충분히 담지 않았습니다 — "
                    + ", ".join(short)),
        )
    ]


def _content_titles(note: str, plan_topics: list[str]) -> list[str]:
    """topic 후보 = 모든 `## ` 헤딩(요약-마커 휴리스틱 없음)."""
    return [match["title"].strip() for match in _H2_RE.finditer(note)]


def _matched_functional_topics(note: str, plan_topics: list[str]) -> set[int]:
    """plan topic과 `## ` 제목의 최대-cardinality 1:1 매칭 인덱스 집합."""
    titles = _content_titles(_strip_fences(note), plan_topics)
    adj: dict[int, list[int]] = {}
    for ti, topic in enumerate(plan_topics):
        topic_tokens = _significant_tokens(topic)
        candidates: list[tuple[float, int]] = []
        if topic_tokens:
            for hi, title in enumerate(titles):
                section_tokens = _significant_tokens(title)
                if section_tokens:
                    jaccard = len(topic_tokens & section_tokens) / len(
                        topic_tokens | section_tokens
                    )
                    if jaccard >= _TOPIC_COVER_JACCARD_THRESHOLD:
                        candidates.append((jaccard, hi))
        candidates.sort(reverse=True)
        adj[ti] = [hi for _jaccard, hi in candidates]

    match_heading: dict[int, int] = {}

    def _augment(ti: int, seen: set[int]) -> bool:
        for hi in adj[ti]:
            if hi in seen:
                continue
            seen.add(hi)
            if hi not in match_heading or _augment(match_heading[hi], seen):
                match_heading[hi] = ti
                return True
        return False

    matched_topics: set[int] = set()
    for ti in range(len(plan_topics)):
        if _augment(ti, set()):
            matched_topics.add(ti)
    return matched_topics


def _completeness_functional(
    note: str, plan_topics: list[str]
) -> list[Finding]:
    """plan topic을 모든 `## ` 헤딩과 Kuhn 최대매칭으로 대조한다."""
    if not plan_topics:
        return []

    matched_topics = _matched_functional_topics(note, plan_topics)
    missing = [
        plan_topics[index]
        for index in range(len(plan_topics))
        if index not in matched_topics
    ]
    if missing:
        return [
            Finding(
                kind="completeness",
                detail=(
                    f"{len(plan_topics) - len(missing)}/{len(plan_topics)}"
                    f" — 누락: {', '.join(missing)}"
                ),
            )
        ]
    return []


def _count_learning_path_sections(note: str) -> int:
    """`## Learning Path` 헤딩부터 다음 `## ` 헤딩(또는 문서 끝) 사이의
    `### ` 헤딩 개수를 센다."""
    h2_matches = list(_H2_RE.finditer(note))
    lp_start: int | None = None
    lp_end = len(note)

    for idx, match in enumerate(h2_matches):
        if match["title"].strip() == _LEARNING_PATH_HEADING:
            lp_start = match.end()
            if idx + 1 < len(h2_matches):
                lp_end = h2_matches[idx + 1].start()
            break

    if lp_start is None:
        return 0

    segment = note[lp_start:lp_end]
    return len(_H3_RE.findall(segment))


def _lint_structure_legacy(note: str) -> list[Finding]:
    """8섹션(제목 1 + `## ` 7종) 존재 및 Learning Path 필드 완전성을 검사한다."""
    findings: list[Finding] = []

    if not _H1_RE.search(note):
        findings.append(
            Finding(kind="structure", detail="레벨-1 제목(`# <노트 제목>`) 누락")
        )

    present_titles = {match["title"].strip() for match in _H2_RE.finditer(note)}
    for required in REQUIRED_SECTIONS:
        if required not in present_titles:
            findings.append(
                Finding(kind="structure", detail=f"'## {required}' 섹션 누락")
            )

    findings.extend(_lint_learning_path_fields(note))
    return findings


def _lint_structure_functional(note: str) -> list[Finding]:
    """제목·콘텐츠 섹션·실제 본문으로 문서 붕괴를 검사한다.

    짧은 단일-topic 노트는 허용하되, 제목/헤딩만 있거나 서로 무관한
    목록만 나열한 출력을 성공으로 보지 않는다.
    """
    note = _strip_fences(note)
    findings: list[Finding] = []
    if not _H1_RE.search(note):
        findings.append(Finding(kind="structure", detail="제목 없음"))

    h2 = _H2_RE.findall(note)
    h3 = _H3_RE.findall(note)
    if len(h2) < 1 and len(h3) < 1:
        findings.append(
            Finding(
                kind="structure",
                detail=f"붕괴(섹션 0개: ## {len(h2)} / ### {len(h3)})",
            )
        )

    content_lines = []
    for raw_line in note.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```") or line in {"<!--", "-->"}:
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9_-]*:\s*.*$", line):
            # 생성 노트 앞부분의 기계 판독용 provenance 메타는 본문이 아니다.
            continue
        content_lines.append(line)

    if not content_lines:
        findings.append(Finding(kind="structure", detail="본문 없음"))
    elif all(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line) for line in content_lines):
        findings.append(
            Finding(kind="structure", detail="무관 나열 의심(서술 본문 없음)")
        )
    return findings


def _lint_learning_path_fields(note: str) -> list[Finding]:
    """Learning Path의 각 `### ` 서브섹션에 Why/How/Traceability 필드가
    있는지 검사한다."""
    findings: list[Finding] = []

    h2_matches = list(_H2_RE.finditer(note))
    lp_start: int | None = None
    lp_end = len(note)
    for idx, match in enumerate(h2_matches):
        if match["title"].strip() == _LEARNING_PATH_HEADING:
            lp_start = match.end()
            if idx + 1 < len(h2_matches):
                lp_end = h2_matches[idx + 1].start()
            break

    if lp_start is None:
        # 섹션 자체가 없다는 findings는 이미 REQUIRED_SECTIONS 검사가 남긴다.
        return findings

    segment = note[lp_start:lp_end]
    h3_matches = list(_H3_RE.finditer(segment))
    if not h3_matches:
        return findings

    required_fields = ("**Why it matters:**", "**How to apply it:**", "**Traceability:**")
    for idx, h3 in enumerate(h3_matches):
        body_start = h3.end()
        body_end = h3_matches[idx + 1].start() if idx + 1 < len(h3_matches) else len(segment)
        body = segment[body_start:body_end]
        title = h3.group(0).removeprefix("### ").strip()

        missing = [field_name for field_name in required_fields if field_name not in body]
        if missing:
            findings.append(
                Finding(
                    kind="structure",
                    detail=f"'{title}' 섹션에 필드 누락: {', '.join(missing)}",
                )
            )

    return findings


#: round-10 계약 §3 — text_post 팽창률 게이트 기본 임계(원문 대비 배율 상한).
_INFLATION_MAX_RATIO = 4.0

#: round-10 계약 §3 — 초단문 원문에도 8섹션 서식 여유를 주는 floor(문자 수).
_INFLATION_MIN_CHARS = 4000


def check_inflation_ratio(
    source_text: str,
    note_text: str,
    *,
    max_ratio: float = _INFLATION_MAX_RATIO,
    min_chars: int = _INFLATION_MIN_CHARS,
) -> Finding | None:
    """`text_post` 산출물이 원문 대비 팽창 임계를 넘는지 결정적으로 판정한다.

    round-10 계약 §3 — `run_gates`(전사 경로 4종 게이트)의 일부가 **아니다**.
    `note_pipe.py::run()`이 `source_kind == "text_post"`일 때만 하네스 실행
    직후 직접 호출하는 post-hoc 게이트다(transcript 경로는 호출 자체가 없음).

    round-09에서 deepseek 계열이 원문 대비 6~15배 팽창(대량 창작 확정)한 것이
    임계 산정 근거다. `max(source_chars * max_ratio, min_chars)`를 초과하면
    hard fail — floor(`min_chars`)가 없으면 초단문 원문에서 정상적인 8섹션
    서식 비용까지 과차단할 위험이 있다.

    Args:
        source_text: 합성 소스 원문(text_post 경로의 `transcript` 파라미터와
            동일 — body_text+원저자 댓글+OCR 병기 문자열).
        note_text: 하네스 최종 산출물(`HarnessResult.note`).
        max_ratio: 원문 대비 배율 상한.
        min_chars: 원문이 짧아도 보장되는 최소 허용 문자 수(floor).

    Returns:
        `Finding`(`kind="invalid_source"` 또는 `kind="inflation"`) 또는
        통과 시 `None`.
    """
    if not source_text.strip():
        # source_text가 비어 있으면 배율 계산 자체가 무의미하다(0 * ratio는
        # 항상 min_chars 이하를 "통과"시켜버리는 논리적 허점 — 사전 패널
        # A-P1-1 fold). 현재 호출 경로(build_text_post_source)는 빈 body_text를
        # 이미 상류에서 막지만, 이 함수는 독립 테스트 가능한 순수 함수이므로
        # 방어적으로 처리한다.
        return Finding(
            kind="invalid_source",
            detail="source_text가 비어 있거나 공백뿐입니다 — 팽창률을 계산할 수 없습니다.",
        )

    threshold = max(len(source_text) * max_ratio, min_chars)
    note_chars = len(note_text)
    if note_chars > threshold:
        return Finding(
            kind="inflation",
            detail=(
                f"산출물 {note_chars}자가 임계 {threshold:.0f}자를 초과했습니다 "
                f"(원문 {len(source_text)}자 × {max_ratio} 배율, floor {min_chars}자)."
            ),
        )
    return None


def _literal_artifact_blocks(source: str) -> list[str]:
    """수집된 원문 중 요약 대신 실물 보존이 필요한 지시문 블록을 찾는다.

    단순히 "prompt"라는 단어가 있다고 발동하지 않는다. 실제 지시문 형태의
    충분히 긴 본문이 있을 때만 반환한다. 이 경계가 없으면 프롬프트를 소개만 한
    짧은 게시물에도 원문 전문을 요구하는 거짓 실패가 생긴다.
    """
    if not _LITERAL_ARTIFACT_CUE_RE.search(source):
        return []

    artifacts: list[str] = []
    for match in _LITERAL_ARTIFACT_START_RE.finditer(source):
        tail = source[match.start():]
        stop = _LITERAL_ARTIFACT_STOP_RE.search(tail)
        if stop is not None:
            tail = tail[:stop.start()]
        # 다음 소스 블록·메타데이터로 흘러가 한 자료가 과도하게 커지는 것을 막는다.
        tail = tail[:1600].strip()
        if len(tail) < _LITERAL_ARTIFACT_MIN_CHARS:
            continue
        if artifacts and match.start() < source.find(artifacts[-1]) + len(artifacts[-1]):
            continue
        artifacts.append(tail)
    return artifacts


def _literal_artifact_tokens(text: str) -> set[str]:
    return {token.casefold() for token in _LITERAL_ARTIFACT_TOKEN_RE.findall(text)}


def check_literal_artifacts(source: str, note: str) -> tuple[list[Finding], str]:
    """핵심 지시문이 번호 붙은 원문 보존 블록으로 남았는지 확인한다.

    반환 두 번째 값은 ``covered/total`` 메타다. 대상 자료가 없으면 ``0/0``이며
    게이트는 발동하지 않는다. 자료의 일부만 수집됐을 때도 확보된 부분만 요구한다.
    누락된 전문을 지어내게 만들지 않기 위해서다.
    """
    artifacts = _literal_artifact_blocks(source)
    if not artifacts:
        return [], "0/0"

    findings: list[Finding] = []
    section = _LITERAL_ARTIFACT_SECTION_RE.search(note)
    if section is None:
        return [
            Finding(
                kind="core_artifact",
                detail=(
                    f"원문 핵심 자료 {len(artifacts)}개를 '## 원문 프롬프트/템플릿' "
                    "섹션의 번호 목록으로 보존하지 않았습니다"
                ),
                context=artifacts[0][:180],
            )
        ], f"0/{len(artifacts)}"

    next_h2 = re.search(r"(?m)^##\s+", note[section.end():])
    section_end = section.end() + next_h2.start() if next_h2 is not None else len(note)
    section_body = note[section.end():section_end]
    numbered_count = len(_NUMBERED_ITEM_RE.findall(section_body))
    if numbered_count < len(artifacts):
        findings.append(
            Finding(
                kind="core_artifact",
                detail=(
                    f"원문 핵심 자료 {len(artifacts)}개 중 번호가 붙은 보존 항목은 "
                    f"{numbered_count}개입니다"
                ),
                context=artifacts[0][:180],
            )
        )

    note_tokens = _literal_artifact_tokens(section_body)
    covered = 0
    for index, artifact in enumerate(artifacts, start=1):
        artifact_tokens = _literal_artifact_tokens(artifact)
        shared = len(artifact_tokens & note_tokens)
        ratio = shared / len(artifact_tokens) if artifact_tokens else 1.0
        if ratio >= _LITERAL_ARTIFACT_TOKEN_COVERAGE:
            covered += 1
            continue
        findings.append(
            Finding(
                kind="core_artifact",
                detail=(
                    f"{index}번 핵심 자료 실물 보존 부족: 중요 토큰 {shared}/"
                    f"{len(artifact_tokens)} ({ratio:.0%})"
                ),
                context=artifact[:180],
            )
        )
    return findings, f"{covered}/{len(artifacts)}"


def _source_urls(text: str) -> list[str]:
    """원문 URL을 첫 등장 순서로 중복 없이 꺼낸다."""
    urls: list[str] = []
    canonical_urls: set[str] = set()
    for raw in _SOURCE_URL_RE.findall(text):
        url = raw.rstrip(".,;:!?，。`")
        canonical = _canonical_url(url)
        if url and canonical not in canonical_urls:
            urls.append(url)
            canonical_urls.add(canonical)
    return urls


def _canonical_url(url: str) -> str:
    """Compare source and note URLs without treating their optional scheme as content."""
    return re.sub(r"^https?://", "", url, flags=re.IGNORECASE).rstrip("/").casefold()


def _matching_url_position(text: str, source_url: str) -> tuple[int, str]:
    """Return the first textual URL equivalent to source_url, preserving its spelling."""
    target = _canonical_url(source_url)
    for match in _SOURCE_URL_RE.finditer(text):
        candidate = match.group(0).rstrip(".,;:!?，。`")
        if _canonical_url(candidate) == target:
            return match.start(), candidate
    return -1, ""


def _url_section_bodies(note: str) -> list[str]:
    """링크 섹션으로 볼 수 있는 `## ` 구간의 본문을 모두 돌려준다.

    제목 정규식은 "링크"뿐 아니라 "자료"·"저장소"도 잡는다. 노트가 그런 제목을
    여러 개 쓰면 첫 매칭이 진짜 링크 섹션이 아닐 수 있으므로 전부 모은다.
    """
    bodies: list[str] = []
    for section in _URL_SECTION_RE.finditer(note):
        next_h2 = re.search(r"(?m)^##\s+", note[section.end():])
        end = section.end() + next_h2.start() if next_h2 is not None else len(note)
        bodies.append(note[section.end():end])
    return bodies


def _explanation_below(section_body: str, line_end: int) -> str:
    """URL 줄 다음에 이어지는 같은 항목의 설명 줄을 모은다.

    빈 줄이나 새 URL을 만나면 항목이 끝난 것으로 본다.
    """
    collected: list[str] = []
    for line in section_body[line_end:].splitlines()[1:]:
        stripped = line.strip()
        if not stripped or _SOURCE_URL_RE.search(stripped):
            break
        collected.append(stripped.lstrip("-—* ").strip())
        if len(collected) >= 3:
            break
    return " ".join(collected)


def check_source_urls(source: str, note: str) -> tuple[list[Finding], str]:
    """원문 URL과 그 URL의 원문 근거 설명이 노트에 남았는지 확인한다."""
    urls = _source_urls(source)
    if not urls:
        return [], "0/0"

    sections = _url_section_bodies(note)
    if not sections:
        return [
            Finding(
                kind="source_url",
                detail=f"원문 URL {len(urls)}개를 설명과 함께 남긴 링크 섹션이 없습니다",
                context=urls[0],
            )
        ], f"0/{len(urls)}"

    # 제목만 보고 첫 매칭을 고르면 "## 복사용 핵심 자료"처럼 URL이 없는 섹션이
    # 진짜 "## 링크와 원문 설명"보다 앞에 있을 때 0/N이 된다(2026-09-04 실측:
    # NIM Gemma 노트가 링크 섹션을 갖고도 0/3으로 떨어졌고, 재수리 2회를 돌고도
    # 같은 지적이 반복돼 예산만 소모했다). 후보 전부에서 URL을 찾는다.
    section_body = "\n".join(sections)
    findings: list[Finding] = []
    covered = 0
    for url in urls:
        position, matched_url = _matching_url_position(section_body, url)
        if position < 0:
            findings.append(
                Finding(
                    kind="source_url",
                    detail=f"원문 URL 누락: {url}",
                    context=url,
                )
            )
            continue
        line_start = section_body.rfind("\n", 0, position) + 1
        line_end = section_body.find("\n", position)
        if line_end < 0:
            line_end = len(section_body)
        line = section_body[line_start:line_end]
        explanation = line.replace(matched_url, "").replace("-", "").replace("—", "").strip()
        if len(re.sub(r"[^0-9A-Za-z가-힣]", "", explanation)) < 4:
            # 설명을 URL 아래 줄에 다는 노트가 실재한다(2026-09-04 실측: NIM 5종이
            # URL 3개를 모두 싣고도 0/3으로 떨어졌다). 같은 항목에 속한 다음 줄까지
            # 본다 — 다음 URL이나 빈 줄이 나오면 그 항목은 끝난 것으로 본다.
            explanation += " " + _explanation_below(section_body, line_end)
        if len(re.sub(r"[^0-9A-Za-z가-힣]", "", explanation)) < 4:
            findings.append(
                Finding(
                    kind="source_url",
                    detail=f"원문 URL 설명 누락: {url}",
                    context=url,
                )
            )
            continue
        covered += 1
    return findings, f"{covered}/{len(urls)}"


def check_source_enumeration(source: str, note: str) -> tuple[list[Finding], str]:
    """원문이 명시한 번호 sequence가 노트에서 사라지거나 뒤집히지 않았는지 확인한다.

    항목 제목의 단어 일치는 의미 평가이지 literal 계약이 아니다. 자연스러운
    패러프레이즈를 하드 실패시키지 않고, 번호의 존재와 원문 순서만 결정론적으로
    판정한다.
    """
    source_items = [
        (match.group("number"), match.group("body").strip())
        for match in _SOURCE_NUMBERED_ITEM_RE.finditer(source)
    ]
    if len(source_items) < _SOURCE_NUMBERED_MIN_ITEMS:
        return [], "0/0"

    note_items = [
        (match.group("number"), match.start())
        for match in _NOTE_NUMBERED_ITEM_RE.finditer(note)
    ]
    findings: list[Finding] = []
    covered = 0
    previous_position = -1
    for number, body in source_items:
        later_positions = [
            position for candidate_number, position in note_items
            if candidate_number == number and position > previous_position
        ]
        if not later_positions:
            exists_elsewhere = any(candidate_number == number for candidate_number, _ in note_items)
            findings.append(
                Finding(
                    kind="source_enumeration",
                    detail=(
                        f"원문 {number}번 항목의 순서가 노트에서 뒤집혔습니다"
                        if exists_elsewhere
                        else f"원문 {number}번 항목의 번호가 노트에서 사라졌습니다"
                    ),
                    context=body[:180],
                )
            )
            continue
        previous_position = later_positions[0]
        covered += 1
    return findings, f"{covered}/{len(source_items)}"


def run_gates(
    note: str,
    transcript: str,
    plan_topic_count: int | None = None,
    *,
    plan_topics: list[str] | None = None,
    plan_topic_items: list[tuple[str, list[str]]] | None = None,
    enforce_topic_sections: bool = True,
    structure_profile: str = "legacy",
) -> GateResult:
    """게이트 (a)~(d)를 순서대로 실행하고 정규화+정제된 노트와 findings를 묶는다.

    실행 순서: normalize_timestamps → verify_timestamps → profile별 completeness
    → profile별 structure(정규화+검증 완료된 노트 기준).

    Args:
        plan_topic_count: 완전성 게이트 하한(개요 topics 수). None이면 검사 생략.
        plan_topics: round-06 T3 — plan topic **이름** 리스트(선택). 주어지면
            `check_completeness`가 결측 topic 이름을 finding detail에 명시한다
            (수렴 iter1 P0-2, `check_completeness` 참조).
        enforce_topic_sections: False면 topic 1:1 섹션 계약을 적용하지 않는다.
            소스가 라벨 나열(D1)일 때 쓴다 — 라벨마다 섹션을 강제하면 채울 내용이
            없어 정의를 지어내게 된다(round-33이 지목한 창작 경로). 나열은 나열로
            두는 것이 정답이다.
        plan_topic_items: topic마다 `(제목, key_numbers+proper_nouns)` 쌍.
            주어지면 내용 기반 완전성(`_completeness_content`)을 추가로 돌린다 —
            제목 대조는 "섹션을 만들었는가", 이쪽은 "속을 채웠는가"를 잰다.
        structure_profile: `"functional"`이면 자유 `## ` 구조의 조직화·topic
            커버 게이트, 그 외에는 기존 Learning Path 게이트를 실행한다.
    """
    findings: list[Finding] = []

    normalized_note, normalize_findings = normalize_timestamps(note)
    findings.extend(normalize_findings)

    transcript_seconds = parse_transcript_timestamps(transcript)
    # "전사에서 타임스탬프를 하나도 못 찾음"(파싱 실패/무타임스탬프 전사)과
    # "노트 타임스탬프가 전부 틀림"은 후속 조치가 다르다 — 전자를 구분 가능하게
    # meta에 전사 측 개수를 기록하고, 노트에 [t=]가 있는데 전사 측이 0이면
    # 별도 finding으로 표면화한다(G5 silent-failure P2).
    if not transcript_seconds and _CANONICAL_TOKEN_RE.search(normalized_note):
        findings.append(
            Finding(
                kind="transcript_no_timestamps",
                detail="전사에서 타임스탬프를 찾지 못했는데 노트에 [t=] 표기가 있음 — "
                "전사 파싱 실패 또는 무타임스탬프 전사일 수 있음(제거된 표기 해석 주의)",
                context="",
            )
        )
    verified_note, verify_findings = verify_timestamps(normalized_note, transcript_seconds)
    findings.extend(verify_findings)

    if structure_profile == "functional":
        if enforce_topic_sections:
            # 제목 대조가 "섹션을 만들었는가"를, 내용 대조가 "속을 채웠는가"를 잰다.
            completeness_findings = _completeness_functional(
                verified_note, plan_topics or []
            ) + _completeness_content(
                verified_note, plan_topic_items or [], transcript
            )
        else:
            completeness_findings = []
    else:
        completeness_findings = check_completeness(
            verified_note, plan_topic_count, plan_topics=plan_topics
        )
    findings.extend(completeness_findings)

    if structure_profile == "functional":
        structure_findings = _lint_structure_functional(verified_note)
    else:
        structure_findings = _lint_structure_legacy(verified_note)
    findings.extend(structure_findings)

    literal_artifact_findings, literal_artifacts_meta = check_literal_artifacts(
        transcript, verified_note
    )
    source_url_findings, source_urls_meta = check_source_urls(transcript, verified_note)
    source_enumeration_findings, source_enumeration_meta = check_source_enumeration(
        transcript, verified_note
    )
    findings.extend(literal_artifact_findings)
    findings.extend(source_url_findings)
    findings.extend(source_enumeration_findings)

    ts_normalized_count = sum(1 for f in normalize_findings if f.kind == "ts_normalized")
    ts_removed_count = sum(1 for f in verify_findings if f.kind == "ts_removed")
    ts_total = len(_CANONICAL_TOKEN_RE.findall(normalized_note))
    ts_verified = ts_total - ts_removed_count

    if plan_topic_count is None:
        completeness_meta: str | None = None
    elif structure_profile == "functional":
        matched_count = len(
            _matched_functional_topics(verified_note, plan_topics or [])
        )
        completeness_meta = f"{matched_count}/{plan_topic_count}"
    else:
        section_count = _count_learning_path_sections(verified_note)
        completeness_meta = f"{section_count}/{plan_topic_count}"

    meta: dict[str, object] = {
        "ts_total": ts_total,
        "ts_verified": ts_verified,
        "ts_removed": ts_removed_count,
        "ts_normalized": ts_normalized_count,
        "transcript_ts_count": len(transcript_seconds),
        "completeness": completeness_meta,
        "structure_ok": len(structure_findings) == 0,
        "literal_artifacts": literal_artifacts_meta,
        "source_urls": source_urls_meta,
        "source_enumeration": source_enumeration_meta,
    }

    return GateResult(note=verified_note, findings=findings, meta=meta)
