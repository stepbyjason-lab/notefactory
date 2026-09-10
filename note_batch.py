"""note_batch.py — sipher fetch → note_pipe.py 배치 실행 러너.

계약: `.handoff/rounds/round-06-batch-calibration-contract.md` §3(T1) ·
§4(T2).

경계(오케스트레이터 제3 레포 금지, docs/00-overview.md §2와 동일 원칙):
이 모듈은 `sipher`를 import하지 않는다. fetch는 외부 command(기본
`python -m core fetch {url} --json --with-transcript`, `--fetch-cmd`로
오버라이드 가능)를 subprocess로 실행해 stdout의 8-key JSON만 소비한다.

## 책임 분리

- **단건 처리는 `note_pipe.py`에 위임한다.** 이 모듈은 그 CLI를 subprocess로
  호출해 기존 exit code 계약(§8 forbidden — 단건 semantic 변경 금지)을
  그대로 재사용한다.
- **이 모듈의 책임**: orchestration(URL/큐 순회), 항목 격리(각 항목 독립
  작업 디렉터리), manifest 기록, retry(제한된 대상만), vault 안전장치
  (`--commit-vault`+`--max-vault-writes`+dry-run).

## 항목 결과 분류(계약 §3, 수렴 iter1 P0-1)

`note_pipe.py`의 실측 exit 시맨틱은 `verified=False`와 hard failure를
모두 exit 1로 합류시킨다(`UnverifiedNoteError`⊂`NotePipeError`→
`SystemExit(1)`, `note_pipe.py main()` 참조) — exit code만으로는 두 상황을
구분할 수 없다. 따라서 이 모듈은 다음 규칙으로 분류한다:

- `exit 0`: `verified=True`(성공).
- `exit 1` + 예상 산출 노트 파일 존재: 노트는 기록됐고 헤더의
  `verified: False`를 파싱해 재확인 — unverified.
- `exit 1` + 노트 파일 부재: hard failure(입력/provider/truncation 등,
  노트가 기록되지 않음).
- `exit 2` 또는 그 외: 배치 구성 버그(usage 오류) — hard failure.

예상 산출 경로는 배치가 `--out`/`--name`(또는 `--vault`/`--name`)을
결정적으로 넘기므로 배치 스스로 계산 가능하다(`note_pipe.resolve_output_path`
호출 없이, note_pipe.py와 동일한 명명 규칙을 이 모듈이 재현한다 — 실제로는
note_pipe에 항상 `--name`을 넘기므로 `<out_dir>/<sanitized-name>.md`로
결정적이다).

## Retry 정책(계약 §4)

기본 retry 횟수는 **1**(즉 최초 시도 + 1회 재시도 = 최대 2 attempt)로 한다
— fetch 실패는 흔히 일시적 네트워크 문제이므로 0보다는 안전하지만, 무한
재시도는 배치 전체를 느리게 만들 위험이 있어 1로 제한한다(계약 §4 "0 또는 1
중 구현자가 선택하되 문서화").

retry 대상: fetch command nonzero exit, fetch timeout, transient subprocess
launch failure(예: `FileNotFoundError` — 실행 파일 자체를 못 찾음).

retry 비대상(계약 §4): `note_pipe.py exit 1`(verified=False 포함),
deterministic gate failure, truncation suspected, malformed 8-key JSON —
이들은 재시도해도 같은 결과가 나올 결정적 실패이므로 attempt를 낭비하지
않는다.

## Vault 안전장치(계약 §4)

배치 기본 실행은 실볼트에 쓰지 않는다(`--out` staging이 기본). `--vault`
모드로 실볼트에 쓰려면 다음을 모두 통과해야 한다:

- `--commit-vault` 명시(단순 `--vault`만으로는 dry-run/preview에 머문다).
- `--max-vault-writes N` 명시(기본 1 — 대량 오염 방지).
- queue JSONL 각 항목에 `name` 필수(URL 목록 입력은 실볼트 모드에서 거부).
- `verified=False` 항목은 `--allow-unverified-vault` 없이는 실볼트에
  쓰지 않는다(이 플래그가 있어도 `--max-vault-writes` cap은 우회 불가).

`--max-vault-writes` 캡은 이 배치 계층이 강제한다(note_pipe의 TOCTOU
방어는 건수 개념이 없는 per-invocation 덮어쓰기 방지일 뿐이다, 수렴 iter1
P1-4). 캡 검사는 각 항목의 실볼트 subprocess 기동 **직전**에 수행하고,
카운터는 **확인된 신규 파일 실쓰기 성공 시에만** 증가한다(fetch/pipe hard
failure는 증가시키지 않되, retry로 인한 실쓰기 성공은 증가시킨다).

⚠️ 실볼트(`D:\\SecondBrain\\0-inbox`) 쓰기는 lens/reviewer/subagent가 직접
수행하지 않는다 — orchestrator가 사용자 승인 범위 안에서만 수행한다
(docs/06-deploy.md §7, controls).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from note_pipe import VaultNameError, sanitize_for_filename, sanitize_vault_note_name  # noqa: E402

logger = logging.getLogger(__name__)

#: note_pipe.py의 --vault 고정 인박스와 동일 상수(계약 §4 — 배치도 같은
#: 실볼트 경로를 참조해야 preview 경로가 실제 쓰기 경로와 일치한다).
DEFAULT_VAULT_INBOX = Path(r"D:\SecondBrain\0-inbox")

#: 기본 fetch command 템플릿(계약 §3) — {url} 자리에 항목 URL이 들어간다.
DEFAULT_FETCH_CMD_TEMPLATE = ["python", "-m", "core", "fetch", "{url}", "--json", "--with-transcript"]

#: fetch subprocess 타임아웃(초). sipher 수집은 트랜스크립션을 포함하면
#: 수 분이 걸릴 수 있어 넉넉히 잡는다.
FETCH_TIMEOUT_SECONDS = 900.0

#: note_pipe.py subprocess 타임아웃(초) — LONG_FORM 설정(note_pipe.py) 준용,
#: 배치는 하네스 전체 패스(plan+synthesis+critic+repair)를 감안해 더 넉넉히.
PIPE_TIMEOUT_SECONDS = 1800.0

#: retry 정책(모듈 docstring 참조) — 최초 시도 + 이 값만큼 추가 재시도.
DEFAULT_RETRY_COUNT = 1

#: --max-vault-writes 기본값(계약 §4 "권장 1").
DEFAULT_MAX_VAULT_WRITES = 1

_NOTE_PIPE_SCRIPT = REPO_ROOT / "note_pipe.py"

_VERIFIED_HEADER_RE = re.compile(r"^verified:\s*(True|False)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# 예외
# ---------------------------------------------------------------------------


class BatchConfigError(RuntimeError):
    """배치 구성 자체가 잘못된 경우(usage 오류) — 배치 시작 전 하드 실패."""


class QueueParseError(RuntimeError):
    """JSONL 큐 파싱 실패. line number를 포함해 진단 가능하게 한다(계약 §3)."""


# ---------------------------------------------------------------------------
# 큐 항목
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueItem:
    """배치 처리 대상 항목 하나(URL 목록 또는 JSONL 큐에서 파생).

    Attributes:
        index: 1-based 순번(item_id/작업 디렉터리 명명에 사용).
        url: fetch 대상 URL.
        name: 산출 노트 파일명 stem(선택). URL 목록 입력은 None —
            item_id 기반 stable id로 대체한다. 실볼트 모드는 필수.
        profile: note_pipe.py `--profile`(기본 "default").
        source_kind: 선택적 기대값(`transcript|text_post`). fetch payload에서
            추론한 실제 값과 일치할 때만 note_pipe.py로 전파한다.
    """

    index: int
    url: str
    name: str | None
    profile: str
    source_kind: str | None = None


#: round-07 G5 리뷰 [P0] 수정(계약 [수렴 fold F-P1-2] 강제화): 이번 라운드에
#: 신규 추가된 카탈로그 provider는 --providers 분산 후보에서 절대 제외한다
#: — "백업 슬롯"은 free_llm.py 내부 폴백 체인 최하위만을 의미하며, 분산
#: CLI를 통한 배정은 그 경계를 위반한다. gemini(기술 비호환)와는 별도 사유이므로
#: 에러 메시지도 구분한다.
_ROUND07_BACKUP_ONLY_PROVIDERS = frozenset({"github_models", "zhipu"})

#: 저RPM 백업 provider 목록(round-07 계약 §scope 1 [수렴 fold A-P0-3]) —
#: --providers에 넣으면 배치 전체 처리 속도가 이 RPM으로 하향 평준화된다.
#: 하드 블록 대상은 아니다(기술 비호환인 gemini만 하드 블록) — 경고만 출력.
_LOW_RPM_WARNING_PROVIDERS = frozenset({"mistral"})


class ProviderSpecError(RuntimeError):
    """`--providers` 파싱 실패(usage 오류) — 배치 시작 전 하드 실패."""


def parse_providers_spec(spec: str) -> list[tuple[str, str]]:
    """`--providers "provider:model,provider:model"` 문자열을 파싱한다.

    round-07 계약 §scope P0-1: 모델 ID 안에 `:`가 포함될 수 있으므로
    (예: `openrouter:google/gemma-4-31b-it:free`) 각 항목에서
    **첫 번째 `:`만** provider/model 구분자로 쓴다(`str.split(":", 1)`).

    빈 항목, provider 누락, model 누락은 명확한 `ProviderSpecError`로
    거부한다. Gemini도 현행 통합 dispatcher streaming 경로를 사용하므로 허용한다.
    """
    if not spec or not spec.strip():
        raise ProviderSpecError("--providers 값이 비어 있습니다.")

    entries: list[tuple[str, str]] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            raise ProviderSpecError(
                f"--providers에 빈 항목이 있습니다(콤마 연속 또는 trailing comma): {spec!r}"
            )
        if ":" not in item:
            raise ProviderSpecError(
                f"--providers 항목은 'provider:model' 형식이어야 합니다(':' 없음): {item!r}"
            )
        provider, model = item.split(":", 1)
        provider = provider.strip()
        model = model.strip()
        if not provider:
            raise ProviderSpecError(f"--providers 항목에 provider가 비어 있습니다: {item!r}")
        if not model:
            raise ProviderSpecError(f"--providers 항목에 model이 비어 있습니다: {item!r}")
        if provider.lower() in _ROUND07_BACKUP_ONLY_PROVIDERS:
            raise ProviderSpecError(
                f"--providers에 {provider!r}는 지정할 수 없습니다(round-07 계약 §scope [수렴 fold F-P1-2] — "
                "이번 라운드에 신규 추가된 카탈로그 provider는 free_llm.py 내부 폴백 체인의 백업 슬롯 "
                f"전용이며 --providers 분산 후보로 승격하지 않습니다): {item!r}"
            )
        entries.append((provider, model))

    if not entries:
        raise ProviderSpecError(f"--providers에 유효한 provider:model 항목이 없습니다: {spec!r}")

    return entries


def low_rpm_warning_message(providers: list[tuple[str, str]]) -> str | None:
    """저RPM 백업 provider가 --providers 목록에 있으면 경고 문구를 반환한다
    (round-07 계약 §scope 1 [수렴 fold A-P0-3]). 없으면 None."""
    flagged = sorted({p for p, _ in providers if p.lower() in _LOW_RPM_WARNING_PROVIDERS})
    if not flagged:
        return None
    return (
        f"[경고] --providers에 저RPM 백업 provider({', '.join(flagged)})가 포함되어 있습니다 — "
        "배치 전체 처리 속도가 해당 provider의 RPM으로 하향 평준화됩니다."
    )


def assign_providers_round_robin(
    items: list[QueueItem], providers: list[tuple[str, str]]
) -> dict[int, tuple[str, str]]:
    """항목별 provider/model을 입력 순서 기준 라운드로빈으로 결정적 배정한다
    (round-07 계약 §scope 2 [수렴 fold A-P0-2] — 배정 방식 고정, 구현 선택지
    없음). 결정성 스코프: 같은 큐 파일 + 같은 --providers 목록 -> 같은 배정.

    반환: {item.index: (provider, model)} — item.index(1-based)를 키로 써서
    큐 순서가 재구성돼도 항목별 배정 결과를 추적 가능하게 한다.
    """
    if not providers:
        return {}
    return {item.index: providers[i % len(providers)] for i, item in enumerate(items)}


def next_provider_in_cycle(
    providers: list[tuple[str, str]], current: tuple[str, str]
) -> tuple[str, str] | None:
    """재배정 대상 provider를 목록에서 다음 순번(순환)으로 결정적으로 정한다
    (round-07 계약 §scope 3 [수렴 fold A-P2-6]). provider 목록이 1개뿐이면
    재배정 대상이 없으므로 None을 반환한다."""
    if len(providers) <= 1:
        return None
    try:
        current_idx = providers.index(current)
    except ValueError:
        return None
    return providers[(current_idx + 1) % len(providers)]


def _stable_id_from_url(url: str, index: int) -> str:
    """URL 목록 입력(이름 없음)의 기본 stable id를 만든다.

    ASCII-safe하고 파일시스템 안전해야 하므로 note_pipe의 ASCII sanitizer를
    재사용한다(계약 §3 "URL 기반 stable id").
    """
    safe = sanitize_for_filename(url)
    # 파일시스템 세그먼트 길이 방어 — 매우 긴 URL이 그대로 디렉터리명이 되는
    # 것을 막는다(Windows MAX_PATH 여유 확보).
    safe = safe[:80] if safe else "unknown-url"
    return f"{index:04d}-{safe}"


def parse_url_list(path: Path) -> list[QueueItem]:
    """`--urls` 파일(한 줄에 URL 하나, 빈 줄/`#` 주석 무시)을 큐 항목으로 변환한다."""
    if not path.exists():
        raise BatchConfigError(f"--urls 파일을 찾을 수 없습니다: {path}")

    items: list[QueueItem] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        index += 1
        items.append(QueueItem(index=index, url=stripped, name=None, profile="default"))

    if not items:
        raise BatchConfigError(f"--urls 파일에 유효한 URL이 없습니다: {path}")

    return items


def parse_queue_jsonl(path: Path) -> list[QueueItem]:
    """`--queue` JSONL 파일을 큐 항목으로 변환한다.

    각 줄은 `{"url": str, "name": str(선택), "profile": str(선택)}` 형태의
    JSON object여야 한다. 파싱/스키마 실패는 line number를 포함한
    `QueueParseError`로 즉시 중단한다(계약 §3 "line number 포함 오류") —
    이는 배치 시작 전 구성 검증이므로 부분 진행을 허용하지 않는다.
    """
    if not path.exists():
        raise BatchConfigError(f"--queue 파일을 찾을 수 없습니다: {path}")

    items: list[QueueItem] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise QueueParseError(f"{path}:{line_no}: 유효한 JSON이 아닙니다: {exc}") from exc

        if not isinstance(payload, dict):
            raise QueueParseError(
                f"{path}:{line_no}: 최상위가 object가 아닙니다: {type(payload).__name__}"
            )
        if "url" not in payload or not isinstance(payload["url"], str) or not payload["url"].strip():
            raise QueueParseError(f"{path}:{line_no}: 'url'(non-empty str)이 없습니다.")

        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise QueueParseError(f"{path}:{line_no}: 'name'은 문자열이어야 합니다.")

        profile = payload.get("profile", "default")
        if profile not in ("default", "light"):
            raise QueueParseError(
                f"{path}:{line_no}: 'profile'은 'default' 또는 'light'여야 합니다: {profile!r}"
            )

        source_kind = payload.get("source_kind")
        if source_kind is not None and source_kind not in {"transcript", "text_post"}:
            raise QueueParseError(
                f"{path}:{line_no}: 'source_kind'는 'transcript' 또는 'text_post'여야 합니다: "
                f"{source_kind!r}"
            )

        index += 1
        items.append(
            QueueItem(
                index=index,
                url=payload["url"].strip(),
                name=name,
                profile=profile,
                source_kind=source_kind,
            )
        )

    if not items:
        raise BatchConfigError(f"--queue 파일에 유효한 항목이 없습니다: {path}")

    return items


# ---------------------------------------------------------------------------
# manifest 레코드
# ---------------------------------------------------------------------------


@dataclass
class ManifestRecord:
    """`manifest.jsonl`의 항목 1개(계약 §4 필드 목록)."""

    item_id: str
    url: str
    name: str | None
    profile: str
    attempt: int
    fetch_status: str
    pipe_exit_code: int | None
    verified: bool | None
    note_path: str | None
    error_kind: str | None
    error_message: str | None
    usage_measured: bool
    usage_estimated: bool
    started_at: str
    finished_at: str
    vault_written: bool = False
    #: round-07 계약 §scope 3 [수렴 fold C-P1-2/A-P0-4]: provider 가로분산
    #: 배정 필드. 기본값 None의 Optional로 추가해 하위호환을 보존한다 —
    #: `--providers` 미지정 시(기존 배치 동작) 항상 None으로 기록된다.
    assigned_provider: str | None = None
    assigned_model: str | None = None
    #: 재배정(1회 한정) 발생 시에만 채운다. `attempt`는 fetch retry 전용
    #: 의미를 유지하고, 재배정은 이 별도 필드로 기록한다([수렴 fold F-P1-3]).
    reassigned_provider: str | None = None
    reassigned_model: str | None = None
    #: round-07 G5 리뷰 [P1] 수정: 재배정이 실제 발생한 경우에만 최초 시도의
    #: pipe exit code/error_message를 보존한다(계약 §scope 3 재배정 기록 요건
    #: "최초 배정, 실패, 재배정 provider/model, 최종 상태를 기록"). 재배정
    #: 없이 하드 실패로 끝난 항목은 두 번째 필드가 필요 없으므로 None을
    #: 유지한다(기존 필드가 이미 최초=유일 시도 정보를 담고 있다). 기본값
    #: None의 Optional로 추가해 하위호환을 보존한다.
    initial_pipe_exit_code: int | None = None
    initial_error_message: str | None = None
    source_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "url": self.url,
            "name": self.name,
            "profile": self.profile,
            "attempt": self.attempt,
            "fetch_status": self.fetch_status,
            "pipe_exit_code": self.pipe_exit_code,
            "verified": self.verified,
            "note_path": self.note_path,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "usage_measured": self.usage_measured,
            "usage_estimated": self.usage_estimated,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "vault_written": self.vault_written,
            "assigned_provider": self.assigned_provider,
            "assigned_model": self.assigned_model,
            "reassigned_provider": self.reassigned_provider,
            "reassigned_model": self.reassigned_model,
            "initial_pipe_exit_code": self.initial_pipe_exit_code,
            "initial_error_message": self.initial_error_message,
            "source_kind": self.source_kind,
        }


#: 항목 최종 분류(계약 §3) — manifest/summary 집계에 쓰인다.
_OUTCOME_VERIFIED = "verified"
_OUTCOME_UNVERIFIED = "unverified"
_OUTCOME_HARD_FAILURE = "hard_failure"


@dataclass
class ItemResult:
    """항목 1개의 최종 처리 결과(재시도 전부 소진 후)."""

    item: QueueItem
    item_id: str
    outcome: str  # _OUTCOME_* 중 하나
    note_path: Path | None
    records: list[ManifestRecord] = field(default_factory=list)
    vault_written: bool = False


# ---------------------------------------------------------------------------
# fetch 실행
# ---------------------------------------------------------------------------


class FetchError(RuntimeError):
    """fetch subprocess가 실패했을 때(retry 대상 — 모듈 docstring 참조)."""


def build_fetch_argv(template: list[str], url: str) -> list[str]:
    """fetch command 템플릿에 URL을 안전하게 치환한다.

    list argv 기반이라 shell injection 위험이 없다(계약 §3 "가능하면 list
    argv 기반으로 구현") — `{url}` 토큰이 있는 원소만 URL로 치환하고, 그 외
    원소는 그대로 둔다. 셸을 거치지 않으므로 URL에 어떤 특수문자가 있어도
    별도 인자로 그대로 전달된다.
    """
    return [part.replace("{url}", url) for part in template]


def run_fetch(argv: list[str], *, timeout_seconds: float = FETCH_TIMEOUT_SECONDS) -> dict:
    """fetch subprocess를 실행하고 stdout의 8-key JSON을 파싱해 반환한다.

    Raises:
        FetchError: nonzero exit, timeout, 실행 파일 부재, 또는 stdout이
            유효한 JSON이 아닌 경우(retry 대상 — malformed JSON은 8-key
            스키마 위반과 달리 "fetch 자체가 덜 끝났을 수 있다"는 신호라
            retry 대상으로 분류한다. sipher가 완전한 8-key dict를 내지만
            내용이 스키마 위반인 경우는 note_pipe.py가 이미 하드 에러로
            잡는다 — 여기서는 순수 JSON 파싱 가능 여부만 본다).
    """
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_seconds,
            text=False,
        )
    except FileNotFoundError as exc:
        raise FetchError(f"fetch 실행 파일을 찾을 수 없습니다: {argv[0]} ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"fetch가 {timeout_seconds}초 내에 끝나지 않았습니다: {argv}") from exc

    if completed.returncode != 0:
        stderr_tail = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise FetchError(
            f"fetch가 exit {completed.returncode}로 실패했습니다: {argv}\nstderr: {stderr_tail}"
        )

    try:
        stdout_text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FetchError(f"fetch stdout이 UTF-8이 아닙니다: {exc}") from exc

    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise FetchError(f"fetch stdout이 유효한 JSON이 아닙니다: {exc}") from exc

    if not isinstance(payload, dict):
        raise FetchError(f"fetch stdout 최상위가 dict가 아닙니다: {type(payload).__name__}")

    return payload


# ---------------------------------------------------------------------------
# note_pipe.py 호출
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipeInvocationResult:
    """`note_pipe.py` subprocess 1회 호출 결과."""

    exit_code: int
    stdout: str
    stderr: str


def build_pipe_argv(
    *,
    input_json_path: Path,
    profile: str,
    out_dir: Path,
    name: str,
    python_executable: str = sys.executable,
    provider: str | None = None,
    model: str | None = None,
    source_kind: str | None = None,
) -> list[str]:
    """note_pipe.py subprocess 호출 argv를 구성한다.

    round-06 G5 P0-A(stage-then-promote): 배치는 vault 모드에서도 note_pipe.py를
    **항상 `--out <staging>`으로** 호출한다 — `--vault`는 절대 넘기지 않는다
    (note_pipe.py가 verified 판정 전에 실볼트에 쓰는 것을 원천 차단). 실볼트
    승격은 outcome 판정 후 `promote_to_vault`가 원자적으로 수행한다.

    항상 `--name`을 넘긴다 — 배치는 예상 산출 경로를 결정적으로 계산해야
    항목 분류(계약 §3)가 가능하므로, provider/model 기반 기본 stem에
    의존하지 않는다. staging은 machine path이므로 note_pipe.py의 ASCII
    sanitizer가 name을 처리한다(`_expected_note_path`가 동일 규칙 재현).

    round-07 계약 §scope 2: `provider`/`model`이 주어지면(가로분산 모드)
    `note_pipe.py --provider --model`로 그대로 관통시킨다. 둘 다 None이면
    (=`--providers` 미지정 — 기존 동작 보존) 기존과 동일하게 note_pipe.py의
    기본값(DEFAULT_PROVIDER/DEFAULT_MODEL)에 맡기고 argv에 추가하지 않는다.
    """
    argv = [
        python_executable,
        str(_NOTE_PIPE_SCRIPT),
        str(input_json_path),
        "--profile",
        profile,
        "--name",
        name,
        "--out",
        str(out_dir),
    ]
    if provider is not None:
        argv.extend(["--provider", provider])
    if model is not None:
        argv.extend(["--model", model])
    if source_kind is not None:
        if source_kind not in {"transcript", "text_post"}:
            raise BatchConfigError(f"유효하지 않은 source_kind: {source_kind!r}")
        argv.extend(["--source-kind", source_kind])
    return argv


def run_note_pipe(argv: list[str], *, timeout_seconds: float = PIPE_TIMEOUT_SECONDS) -> PipeInvocationResult:
    """note_pipe.py subprocess를 실행하고 exit code/stdout/stderr를 반환한다.

    이 함수 자체는 예외를 raise하지 않는다(timeout/실행 파일 부재 제외) —
    note_pipe.py의 exit code(0/1/2)는 배치 계층이 노트 파일 존재 여부와
    함께 해석해야 할 정상적 신호이지, 배치의 예외 상황이 아니다.
    """
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        # timeout은 note_pipe.py 자체의 실패이지 fetch의 transient 실패가
        # 아니다 — retry 비대상(계약 §4, "note_pipe.py exit 1"과 동일 취급).
        return PipeInvocationResult(exit_code=1, stdout="", stderr=f"note_pipe.py timeout: {exc}")
    except FileNotFoundError as exc:
        return PipeInvocationResult(exit_code=2, stdout="", stderr=f"note_pipe.py 실행 실패: {exc}")

    return PipeInvocationResult(
        exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
    )


def parse_verified_header(note_text: str) -> bool | None:
    """노트 파일 헤더의 `verified: True|False` 줄을 파싱한다.

    찾지 못하면 None(호출자가 구조 이상으로 처리해야 함 — 조용히 False로
    간주하지 않는다, 계약 §3 "노트 헤더의 verified 필드를 파싱해 재확인").
    """
    match = _VERIFIED_HEADER_RE.search(note_text)
    if match is None:
        return None
    return match.group(1) == "True"


def infer_source_kind(fetch_payload: dict) -> str:
    """Infer the same effective source kind that note_pipe uses."""
    transcript = fetch_payload.get("transcript")
    if isinstance(transcript, str) and transcript.strip():
        return "transcript"
    body_text = fetch_payload.get("body_text")
    if isinstance(body_text, str) and body_text.strip():
        return "text_post"
    raise BatchConfigError(
        "fetch payload에 유효한 transcript 또는 body_text가 없어 source_kind를 정할 수 없습니다."
    )


def resolve_item_source_kind(item: QueueItem, fetch_payload: dict) -> str:
    """Validate an optional queue declaration against the fetched payload."""
    inferred = infer_source_kind(fetch_payload)
    if item.source_kind is not None and item.source_kind != inferred:
        raise BatchConfigError(
            f"queue source_kind={item.source_kind!r}와 fetch payload 실제값={inferred!r}가 다릅니다."
        )
    return inferred


# ---------------------------------------------------------------------------
# 항목 단위 실행 — fetch(+retry) → note_pipe(+retry) → 분류
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """배치 실행 1회의 구성(CLI 인자에서 도출)."""

    out_dir: Path | None
    vault: bool
    commit_vault: bool
    max_vault_writes: int
    allow_unverified_vault: bool
    fetch_cmd_template: list[str]
    retry_count: int
    batch_out_dir: Path
    #: round-07 계약 §scope 1 — `--providers` 파싱 결과([(provider, model), ...]).
    #: None/빈 리스트면 가로분산 미사용(기존 동작 보존 — note_pipe.py 기본값 사용).
    providers: list[tuple[str, str]] | None = None


def _pipe_out_dir_for_staging(config: RunnerConfig) -> Path:
    """note_pipe.py에 항상 넘기는 `--out` staging 디렉터리(round-06 G5 P0-A —
    stage-then-promote).

    **vault 모드에서도** note_pipe.py는 이 staging 디렉터리로만 쓴다 —
    `--vault`를 note_pipe.py에 넘기지 않는다. note_pipe.py는 verified 판정
    **이전**에 파일을 쓰므로(round-05 시맨틱), `--vault`로 직접 호출하면
    unverified 노트가 `--allow-unverified-vault` 없이도 실볼트에 유입되고
    max-vault-writes 캡을 우회한다(G5 P0-A). 배치는 항상 staging에 먼저 쓰고,
    outcome 판정 후 캡 여유가 있을 때만 실볼트로 원자적 승격한다
    (`promote_to_vault`).

    `config.out_dir`(사용자 `--out`) 대신 `batch_out_dir/notes`로 고정하는
    이유는 기존과 동일(배치 산출물을 batch_out_dir 하나에 모은다).
    `_expected_note_path`가 이 함수와 반드시 동일 디렉터리를 참조해야
    배치의 "예상 경로 존재 여부" 판정이 note_pipe.py 실쓰기 위치와 어긋나지
    않는다(단일 진실 원천).
    """
    return config.batch_out_dir / "notes"


def _expected_note_path(config: RunnerConfig, *, name: str) -> Path:
    """note_pipe.py가 이 항목에 대해 **staging에** 만들 예상 산출 경로를 배치가
    결정적으로 계산한다(계약 §3 — exit code만으로 unverified/hard failure를
    구분할 수 없으므로 파일 존재 여부가 1차 판정 근거).

    round-06 G5 P0-A: vault 모드에서도 staging 경로를 반환한다 — note_pipe.py는
    이제 항상 `--out staging`으로 호출되기 때문이다. 실볼트 최종 경로는
    `_vault_dest_path`가 별도로 계산한다(승격 대상). 두 경로가 서로 다른
    sanitizer를 쓰는 이유: staging은 machine path이므로 ASCII-safe
    `sanitize_for_filename`, 실볼트 표시명은 Unicode 보존
    `sanitize_vault_note_name`(T4).
    """
    stem = sanitize_for_filename(name)
    return _pipe_out_dir_for_staging(config) / f"{stem}.md"


def _vault_dest_path(name: str) -> Path:
    """staged 노트를 승격할 실볼트(`0-inbox`) 최종 경로(round-06 G5 P0-A).

    Unicode 보존 sanitizer(`sanitize_vault_note_name`, T4)를 써서 한글 노트명을
    보존한다. 이 함수는 `DEFAULT_VAULT_INBOX`를 모듈 전역으로 참조하므로
    테스트는 그 상수를 monkeypatch해 실볼트를 절대 건드리지 않는다.
    """
    stem = sanitize_vault_note_name(name)
    return DEFAULT_VAULT_INBOX / f"{stem}.md"


class VaultPromotionError(RuntimeError):
    """staged 노트를 실볼트로 승격하는 중 실패(대상 이미 존재 등, G5 P0-A P1)."""


def promote_to_vault(staged_note: Path, dest: Path) -> None:
    """staged 노트 파일을 실볼트 대상 경로로 **원자적 create-or-fail** 승격한다.

    round-06 G5 P0-A(+ Antigravity P1): note_pipe.py의 덮어쓰기 방어는 이제
    staging에만 걸리므로, 실볼트 덮어쓰기는 배치가 직접 막아야 한다.
    `os.rename`은 cross-drive(예: C: staging → D: 볼트)에서 실패하므로
    `os.open(dest, O_CREAT|O_EXCL|O_WRONLY)`로 대상 파일을 원자적으로
    "새로 만들 수 있을 때만" 열고, 열리면 staged 바이트를 복사한다. 대상이
    이미 존재하면 `os.open`이 `FileExistsError`를 던진다 — TOCTOU 창 없이
    덮어쓰기를 원천 차단한다(존재 검사와 생성이 단일 원자 연산).

    Raises:
        VaultPromotionError: 대상이 이미 존재하거나(덮어쓰기 금지) 그 외
            OS 오류로 승격에 실패한 경우.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = staged_note.read_bytes()
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise VaultPromotionError(
            f"승격 대상이 이미 실볼트에 존재합니다(덮어쓰기 절대 금지): {dest}"
        ) from exc
    except OSError as exc:
        raise VaultPromotionError(f"실볼트 승격 실패({dest}): {exc}") from exc
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except OSError as exc:  # 쓰기 도중 실패 — 부분 파일을 남기지 않도록 정리 시도.
        try:
            os.unlink(dest)
        except OSError:
            pass
        raise VaultPromotionError(f"실볼트 승격 중 쓰기 실패({dest}): {exc}") from exc


def process_item(
    item: QueueItem,
    *,
    config: RunnerConfig,
    fetch_fn: Callable[[list[str]], dict] | None = None,
    pipe_fn: Callable[[list[str]], PipeInvocationResult] | None = None,
    vault_write_count: list[int],
    assigned_provider: str | None = None,
    assigned_model: str | None = None,
    providers: list[tuple[str, str]] | None = None,
) -> ItemResult:
    """항목 1개를 처리한다(fetch → note_pipe → 분류), retry 정책 적용.

    Args:
        fetch_fn: `run_fetch`를 대체할 테스트용 fake(인자: argv, 반환: 8-key
            dict). None이면 `run_fetch`를 실제 subprocess로 호출한다.
        pipe_fn: `run_note_pipe`를 대체할 테스트용 fake. None이면 실제
            subprocess.
        vault_write_count: 배치 전체에서 공유하는 [count] 1-원소 리스트 —
            실볼트 신규 쓰기 성공 시에만 증가시킨다(계약 §4 "카운터는
            확인된 신규 파일 실쓰기 성공 시에만 증가"). list로 넘기는 이유는
            여러 항목 호출 사이에서 mutable 공유 상태가 필요하기 때문이다
            (모듈 최상위 전역 대신 호출자가 명시적으로 생성/전달).
        assigned_provider/assigned_model: round-07 계약 §scope 2 — 라운드로빈
            배정 결과. 둘 다 None이면 `--providers` 미지정(기존 동작 보존):
            note_pipe.py에 --provider/--model을 넘기지 않는다.
        providers: 재배정 대상 계산에 쓰이는 --providers 전체 목록(round-07
            계약 §scope 3 [수렴 fold A-P2-6] "다음 순번(순환)"). assigned_*가
            None이면 사용되지 않는다.
    """
    item_id = f"{item.index:04d}-{sanitize_for_filename(item.name or _stable_id_from_url(item.url, item.index))}"
    item_dir = config.batch_out_dir / "items" / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    name = item.name or _stable_id_from_url(item.url, item.index)

    fetch_argv = build_fetch_argv(config.fetch_cmd_template, item.url)
    _fetch_fn = fetch_fn or run_fetch

    records: list[ManifestRecord] = []
    fetch_payload: dict | None = None
    fetch_error: str | None = None

    max_fetch_attempts = 1 + config.retry_count
    for attempt in range(1, max_fetch_attempts + 1):
        started_at = _now_iso()
        try:
            fetch_payload = _fetch_fn(fetch_argv)
            fetch_status = "ok"
            fetch_error = None
        except FetchError as exc:
            fetch_status = "failed"
            fetch_error = str(exc)
            fetch_payload = None

        if fetch_payload is not None:
            break

        # fetch 실패 — retry 대상(모듈 docstring). 마지막 attempt가 아니면 계속.
        finished_at = _now_iso()
        records.append(
            ManifestRecord(
                item_id=item_id,
                url=item.url,
                name=item.name,
                profile=item.profile,
                attempt=attempt,
                fetch_status=fetch_status,
                pipe_exit_code=None,
                verified=None,
                note_path=None,
                error_kind="fetch_failed",
                error_message=fetch_error,
                usage_measured=False,
                usage_estimated=False,
                started_at=started_at,
                finished_at=finished_at,
                assigned_provider=assigned_provider,
                assigned_model=assigned_model,
                source_kind=item.source_kind,
            )
        )
        if attempt >= max_fetch_attempts:
            return ItemResult(
                item=item, item_id=item_id, outcome=_OUTCOME_HARD_FAILURE, note_path=None, records=records
            )

    assert fetch_payload is not None
    source_kind = resolve_item_source_kind(item, fetch_payload)
    # 앞선 transient fetch 실패 attempt도 동일 항목의 실제 payload 종류가
    # 뒤늦게 확인되면 backfill한다. manifest의 재시도 행이 null로 남아
    # 최종 pipe 입력과 어긋나는 일을 막는다.
    for record in records:
        if record.source_kind is None:
            record.source_kind = source_kind
    input_json_path = item_dir / "fetch_output.json"
    input_json_path.write_text(
        json.dumps(fetch_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # round-06 G5 P0-A(stage-then-promote): note_pipe.py는 vault 모드여도
    # 항상 staging(`--out`)으로만 쓴다. 캡 검사·실볼트 쓰기는 pipe 실행
    # 이후로 미룬다 — verified는 실행 후에만 알 수 있어 사전 가드가 원리상
    # 불가능하기 때문이다(기존 코드의 "사전 차단" 주석은 사실이 아니었다).
    staging_note_path = _expected_note_path(config, name=name)
    pipe_out_dir = _pipe_out_dir_for_staging(config)
    pipe_out_dir.mkdir(parents=True, exist_ok=True)

    pipe_argv = build_pipe_argv(
        input_json_path=input_json_path,
        profile=item.profile,
        out_dir=pipe_out_dir,
        name=name,
        provider=assigned_provider,
        model=assigned_model,
        source_kind=source_kind,
    )
    _pipe_fn = pipe_fn or run_note_pipe

    started_at = _now_iso()
    pipe_result = _pipe_fn(pipe_argv)  # note_pipe.py exit 1/2는 retry 비대상(계약 §4).

    note_exists = staging_note_path.exists()
    verified: bool | None = None
    note_text = ""
    if note_exists:
        note_text = staging_note_path.read_text(encoding="utf-8", errors="replace")
        verified = parse_verified_header(note_text)

    usage_measured, usage_estimated = _extract_usage_flags(note_text)

    # ---- 1) staged 노트 기준 outcome 판정(실볼트 승격과 무관) ----
    error_kind: str | None
    error_message: str | None
    if pipe_result.exit_code == 0:
        outcome = _OUTCOME_VERIFIED
        note_path_result = staging_note_path if note_exists else None
        error_kind = None
        error_message = None
    elif pipe_result.exit_code == 1 and note_exists:
        outcome = _OUTCOME_UNVERIFIED
        note_path_result = staging_note_path
        error_kind = "unverified"
        error_message = "verified=False(수리 예산 소진 후에도 finding 잔존) — staging에만 격리"
    elif pipe_result.exit_code == 1:
        outcome = _OUTCOME_HARD_FAILURE
        note_path_result = None
        error_kind = "pipe_hard_failure"
        error_message = pipe_result.stderr[-2000:] if pipe_result.stderr else "note_pipe.py exit 1(노트 미기록)"
    else:
        outcome = _OUTCOME_HARD_FAILURE
        note_path_result = None
        error_kind = "pipe_usage_error"
        error_message = pipe_result.stderr[-2000:] if pipe_result.stderr else f"note_pipe.py exit {pipe_result.exit_code}"

    # ---- 1.5) 가로분산 재배정(round-07 계약 §scope 3 [수렴 fold A-P0-1/C-P1-1]) ----
    # 트리거는 pipe_hard_failure(exit 1 + 노트 부재)뿐이다 — unverified(exit 1 +
    # 노트 존재)와 exit 2(usage 오류)는 provider를 바꿔도 무의미하므로 재배정
    # 하지 않는다(round-06 "결정적 실패는 재시도하지 않는다" 유지). 항목당
    # 최대 1회만 수행하고, fetch payload(input_json_path)를 재사용해 pipe만
    # 재실행한다(fetch는 다시 호출하지 않음, [수렴 fold F-P1-3]).
    reassigned_provider: str | None = None
    reassigned_model: str | None = None
    initial_pipe_exit_code: int | None = None
    initial_error_message: str | None = None
    if (
        error_kind == "pipe_hard_failure"
        and assigned_provider is not None
        and assigned_model is not None
        and providers
    ):
        next_pm = next_provider_in_cycle(providers, (assigned_provider, assigned_model))
        if next_pm is not None:
            # round-07 G5 리뷰 [P1] 수정: 재배정이 실제로 일어나는 이 분기에서만
            # 최초 시도의 실패 정보를 별도 필드에 보존한다 — 아래에서
            # pipe_result/error_kind/error_message를 두 번째 시도 값으로
            # 덮어쓰기 전에 스냅샷을 떠 둔다(계약 §scope 3 재배정 기록 요건).
            initial_pipe_exit_code = pipe_result.exit_code
            initial_error_message = error_message
            reassigned_provider, reassigned_model = next_pm
            retry_pipe_argv = build_pipe_argv(
                input_json_path=input_json_path,
                profile=item.profile,
                out_dir=pipe_out_dir,
                name=name,
                provider=reassigned_provider,
                model=reassigned_model,
                source_kind=source_kind,
            )
            pipe_result = _pipe_fn(retry_pipe_argv)

            note_exists = staging_note_path.exists()
            verified = None
            note_text = ""
            if note_exists:
                note_text = staging_note_path.read_text(encoding="utf-8", errors="replace")
                verified = parse_verified_header(note_text)
            usage_measured, usage_estimated = _extract_usage_flags(note_text)

            if pipe_result.exit_code == 0:
                outcome = _OUTCOME_VERIFIED
                note_path_result = staging_note_path if note_exists else None
                error_kind = None
                error_message = None
            elif pipe_result.exit_code == 1 and note_exists:
                outcome = _OUTCOME_UNVERIFIED
                note_path_result = staging_note_path
                error_kind = "unverified"
                error_message = "verified=False(수리 예산 소진 후에도 finding 잔존) — staging에만 격리"
            elif pipe_result.exit_code == 1:
                outcome = _OUTCOME_HARD_FAILURE
                note_path_result = None
                error_kind = "pipe_hard_failure"
                error_message = (
                    pipe_result.stderr[-2000:] if pipe_result.stderr else "note_pipe.py exit 1(노트 미기록, 재배정 후에도 실패)"
                )
            else:
                outcome = _OUTCOME_HARD_FAILURE
                note_path_result = None
                error_kind = "pipe_usage_error"
                error_message = (
                    pipe_result.stderr[-2000:] if pipe_result.stderr else f"note_pipe.py exit {pipe_result.exit_code}(재배정 후)"
                )

    # ---- 2) 실볼트 승격(stage-then-promote) ----
    # 승격 자격: vault 모드 + verified(또는 unverified+allow_unverified_vault).
    # unverified+not allow는 승격 자체를 하지 않으므로 실볼트에 절대 쓰이지
    # 않고 staging에만 남는다(계약 §4 규칙3 준수). 캡 검사는 승격 직전에
    # 하고, 카운터는 승격 성공 시에만 증가한다(수렴 iter1 P1-4).
    vault_written_this_item = False
    promote_eligible = (
        config.vault
        and config.commit_vault
        and note_exists
        and (
            outcome == _OUTCOME_VERIFIED
            or (outcome == _OUTCOME_UNVERIFIED and config.allow_unverified_vault)
        )
    )
    if promote_eligible:
        if vault_write_count[0] >= config.max_vault_writes:
            # 캡 도달 — 실볼트에 쓰지 않고 staging에만 남긴다(항목은 실패로
            # 표기하지 않는다: staged 노트는 정상 산출됐고 단지 이번 배치의
            # 실볼트 쿼터를 초과했을 뿐이다. 이후 항목은 계속 진행).
            error_kind = error_kind or "vault_write_cap_reached"
            error_message = (
                f"--max-vault-writes({config.max_vault_writes}) 도달 — "
                "이 항목은 실볼트로 승격하지 않고 staging에만 남깁니다."
            )
        else:
            dest = _vault_dest_path(name)
            try:
                promote_to_vault(staging_note_path, dest)
                vault_write_count[0] += 1
                vault_written_this_item = True
                note_path_result = dest  # 최종 위치는 실볼트.
            except VaultPromotionError as exc:
                # 승격 실패(대상 이미 존재 등) — 이 항목만 hard failure로
                # 표기하고 다른 항목은 계속 진행한다(Antigravity P1).
                outcome = _OUTCOME_HARD_FAILURE
                error_kind = "vault_promotion_failed"
                error_message = str(exc)
                note_path_result = staging_note_path  # staged 원본은 보존.

    finished_at = _now_iso()
    records.append(
        ManifestRecord(
            item_id=item_id,
            url=item.url,
            name=item.name,
            profile=item.profile,
            attempt=1,
            fetch_status="ok",
            pipe_exit_code=pipe_result.exit_code,
            verified=verified,
            note_path=str(note_path_result) if note_path_result else None,
            error_kind=error_kind,
            error_message=error_message,
            usage_measured=usage_measured,
            usage_estimated=usage_estimated,
            started_at=started_at,
            finished_at=finished_at,
            vault_written=vault_written_this_item,
            assigned_provider=assigned_provider,
            assigned_model=assigned_model,
            reassigned_provider=reassigned_provider,
            reassigned_model=reassigned_model,
            initial_pipe_exit_code=initial_pipe_exit_code,
            initial_error_message=initial_error_message,
            source_kind=source_kind,
        )
    )

    return ItemResult(
        item=item,
        item_id=item_id,
        outcome=outcome,
        note_path=note_path_result,
        records=records,
        vault_written=vault_written_this_item,
    )


_HEADER_ANY_ESTIMATED_RE = re.compile(r"^tokens_any_estimated:\s*(True|False)\s*$", re.MULTILINE)
_HEADER_TOTAL_KNOWN_RE = re.compile(r"^tokens_total_known:\s*(\d+)\s*$", re.MULTILINE)


def _extract_usage_flags(note_text: str) -> tuple[bool, bool]:
    """노트 헤더에서 usage measured/estimated 플래그를 정직하게 분리 추출한다
    (round-06 G5 P1-4 — 계약 §9 measured/estimated 구분).

    note_pipe.py가 헤더에 emit하는 두 필드를 단일 진실원천으로 파싱한다:
      - `tokens_total_known: N` — 집계된 known 토큰 총량(실측 또는 추정 합산).
      - `tokens_any_estimated: True|False` — 패스 중 하나라도 chars/4 추정치로
        채워졌는가(measured가 아니라 추정인 패스가 존재하는가).

    반환 `(usage_measured, usage_estimated)`:
      - `usage_measured`: known 총량이 0보다 크고 **어떤 패스도 추정이 아닐 때**
        True — 즉 순수 실측일 때만.
      - `usage_estimated`: 하나 이상의 패스가 추정치로 채워졌을 때 True.

    이렇게 하면 두 플래그가 항상 같은 값으로 붕괴하지 않는다(기존 버그):
    순수 실측 노트는 `(True, False)`, 추정 섞인 노트는 `(False, True)`,
    usage 정보가 아예 없으면 `(False, False)`. estimated를 measured로
    위장하지 않는 것이 계약 §9의 핵심이다.

    헤더에 `tokens_any_estimated` 필드가 없으면(구버전 노트 등) 추정 여부를
    알 수 없으므로 measured로 단정하지 않고 보수적으로 estimated=True로
    처리한다 — "모르는 것을 measured로 위장"하지 않기 위함.
    """
    if not note_text:
        return False, False

    total_match = _HEADER_TOTAL_KNOWN_RE.search(note_text)
    has_known_tokens = bool(total_match and int(total_match.group(1)) > 0)

    any_estimated_match = _HEADER_ANY_ESTIMATED_RE.search(note_text)
    if any_estimated_match is None:
        # 필드 부재 — 추정 여부 불명. measured로 단정하지 않고 보수적 처리.
        return False, has_known_tokens

    any_estimated = any_estimated_match.group(1) == "True"
    usage_measured = has_known_tokens and not any_estimated
    usage_estimated = any_estimated
    return usage_measured, usage_estimated


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# manifest/summary 기록
# ---------------------------------------------------------------------------


def write_manifest(records: list[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_summary(results: list[ItemResult], path: Path) -> dict[str, object]:
    verified_count = sum(1 for r in results if r.outcome == _OUTCOME_VERIFIED)
    unverified_count = sum(1 for r in results if r.outcome == _OUTCOME_UNVERIFIED)
    hard_failure_count = sum(1 for r in results if r.outcome == _OUTCOME_HARD_FAILURE)

    summary = {
        "total": len(results),
        "verified": verified_count,
        "unverified": unverified_count,
        "hard_failure": hard_failure_count,
        "items": [
            {"item_id": r.item_id, "url": r.item.url, "outcome": r.outcome, "note_path": str(r.note_path) if r.note_path else None}
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="note_batch.py",
        description=(
            "URL 목록 또는 JSONL 큐를 sipher fetch -> note_pipe.py로 순회 실행하는 "
            "안전한 배치 러너. 기본은 --out staging이며 실볼트 쓰기는 명시 opt-in이 필요합니다."
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--urls", help="URL 목록 파일(한 줄에 URL 하나).")
    input_group.add_argument(
        "--queue", help="JSONL 큐 파일(url/name/profile/source_kind 필드)."
    )

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--out", help="배치 산출물 루트 디렉토리(manifest/summary/items/notes).")
    output_group.add_argument(
        "--vault",
        action="store_true",
        help=(
            f"실볼트({DEFAULT_VAULT_INBOX}) 대상 모드. --commit-vault 없이는 dry-run(preview)만 "
            "수행하고 실제로 쓰지 않습니다."
        ),
    )

    parser.add_argument("--profile", choices=("default", "light"), default="default", help="URL 목록 입력의 기본 프로파일.")
    parser.add_argument("--fetch-cmd", default=None, help="fetch command 템플릿({url} 토큰 포함, 공백 분리 argv).")
    parser.add_argument(
        "--retry", type=int, default=DEFAULT_RETRY_COUNT, help=f"fetch 실패 시 재시도 횟수(기본 {DEFAULT_RETRY_COUNT})."
    )
    parser.add_argument("--commit-vault", action="store_true", help="실볼트에 실제로 쓴다(명시 opt-in, --vault와 함께 사용).")
    parser.add_argument(
        "--max-vault-writes", type=int, default=DEFAULT_MAX_VAULT_WRITES, help=f"실볼트 신규 쓰기 상한(기본 {DEFAULT_MAX_VAULT_WRITES})."
    )
    parser.add_argument(
        "--allow-unverified-vault",
        action="store_true",
        help="verified=False 항목도 실볼트에 쓰는 것을 허용한다(기본 거부).",
    )
    parser.add_argument("--batch-out-log-dir", default=None, help="manifest/summary용 별도 로그 디렉토리(기본: --out 또는 <batch temp>).")
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "provider 가로분산(round-07 계약 §scope P0-1). 형식: "
            "'provider:model,provider:model'(모델 ID의 ':'는 첫 ':'만 구분자로 처리 — "
            "예: openrouter:google/gemma-4-31b-it:free). 입력 순서 기준 "
            "라운드로빈으로 항목별 배정한다. 미지정 시 기존 동작(단일 note_pipe.py 기본 "
            "provider/model) 보존. Gemini는 통합 dispatcher의 streaming 경로로 지원한다. "
            f"저RPM 백업({', '.join(sorted(_LOW_RPM_WARNING_PROVIDERS))} 등)을 넣으면 "
            "배치 전체 처리 속도가 해당 RPM으로 하향 평준화된다."
        ),
    )

    return parser


def _resolve_fetch_cmd_template(args: argparse.Namespace) -> list[str]:
    if args.fetch_cmd:
        return args.fetch_cmd.split()
    return list(DEFAULT_FETCH_CMD_TEMPLATE)


def _load_items(args: argparse.Namespace) -> list[QueueItem]:
    if args.urls:
        items = parse_url_list(Path(args.urls))
        # URL 목록은 --profile을 일괄 적용(계약 §3 — 큐 JSONL만 항목별 profile 지원).
        items = [
            QueueItem(
                index=i.index,
                url=i.url,
                name=i.name,
                profile=args.profile,
                source_kind=i.source_kind,
            ) for i in items
        ]
        return items
    return parse_queue_jsonl(Path(args.queue))


def _validate_vault_args(args: argparse.Namespace, items: list[QueueItem], parser: argparse.ArgumentParser) -> None:
    if not args.vault:
        return
    if args.commit_vault:
        if args.max_vault_writes < 1:
            parser.error("--max-vault-writes는 1 이상이어야 합니다.")
        missing_name_items = [i for i in items if not i.name]
        if missing_name_items:
            parser.error(
                "--vault --commit-vault 모드는 모든 큐 항목에 'name'이 필요합니다 "
                f"(name 누락 {len(missing_name_items)}건, 예: item #{missing_name_items[0].index})."
            )


def main(argv: list[str] | None = None) -> None:
    _reconfigure_utf8_streams()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        items = _load_items(args)
    except (BatchConfigError, QueueParseError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    providers: list[tuple[str, str]] | None = None
    # round-07 G5 리뷰 [P2-1] 수정: `is not None`으로 빈 문자열도 파싱 단계로
    # 보낸다 — 기존 `if args.providers:`는 `--providers ""`(falsy)를 조용히
    # "미지정"으로 취급해 우회시켰다. 빈 문자열은 parse_providers_spec의
    # 명확한 ProviderSpecError를 타야 한다.
    if args.providers is not None:
        try:
            providers = parse_providers_spec(args.providers)
        except ProviderSpecError as exc:
            logger.error("%s", exc)
            raise SystemExit(2) from exc
        warning = low_rpm_warning_message(providers)
        if warning:
            logger.warning("%s", warning)

    _validate_vault_args(args, items, parser)

    if args.vault and not args.commit_vault:
        # dry-run/preview 모드(계약 §4) — 예상 대상 파일명만 manifest로 보여주고
        # 실제 fetch/pipe는 수행하지 않는다(비용 없는 미리보기).
        _run_vault_dry_run(items, args)
        return

    batch_out_dir = Path(args.batch_out_log_dir) if args.batch_out_log_dir else (
        Path(args.out) if args.out else Path(".out") / "batch"
    )

    config = RunnerConfig(
        out_dir=Path(args.out) if args.out else None,
        vault=args.vault,
        commit_vault=args.commit_vault,
        max_vault_writes=args.max_vault_writes,
        allow_unverified_vault=args.allow_unverified_vault,
        fetch_cmd_template=_resolve_fetch_cmd_template(args),
        retry_count=args.retry,
        batch_out_dir=batch_out_dir,
        providers=providers,
    )

    results = run_batch(items, config=config)

    all_records = [record for result in results for record in result.records]
    write_manifest(all_records, batch_out_dir / "manifest.jsonl")
    summary = write_summary(results, batch_out_dir / "summary.json")

    logger.info(
        "batch 완료: total=%s verified=%s unverified=%s hard_failure=%s",
        summary["total"],
        summary["verified"],
        summary["unverified"],
        summary["hard_failure"],
    )

    raise SystemExit(_aggregate_exit_code(results))


def run_batch(
    items: list[QueueItem],
    *,
    config: RunnerConfig,
    fetch_fn: Callable[[list[str]], dict] | None = None,
    pipe_fn: Callable[[list[str]], PipeInvocationResult] | None = None,
) -> list[ItemResult]:
    """항목 리스트를 순회 실행한다(한 항목 실패가 나머지를 막지 않는다, 계약 §3)."""
    vault_write_count = [0]
    results: list[ItemResult] = []
    # round-07 계약 §scope 2 [수렴 fold A-P0-2]: 입력 순서 기준 라운드로빈으로
    # 배치 시작 시 1회 결정적으로 배정한다(같은 큐+같은 --providers -> 같은 배정).
    assignment_map = assign_providers_round_robin(items, config.providers or [])
    for item in items:
        assigned = assignment_map.get(item.index)
        assigned_provider, assigned_model = assigned if assigned is not None else (None, None)
        try:
            result = process_item(
                item,
                config=config,
                fetch_fn=fetch_fn,
                pipe_fn=pipe_fn,
                vault_write_count=vault_write_count,
                assigned_provider=assigned_provider,
                assigned_model=assigned_model,
                providers=config.providers,
            )
        except Exception as exc:  # noqa: BLE001 — 항목 단위 격리(계약 §3 "한 항목 실패가 전체를 중단하지 않는다").
            logger.exception("item #%s 처리 중 예기치 않은 예외: %s", item.index, exc)
            item_id = f"{item.index:04d}-{sanitize_for_filename(item.name or _stable_id_from_url(item.url, item.index))}"
            now = _now_iso()
            result = ItemResult(
                item=item,
                item_id=item_id,
                outcome=_OUTCOME_HARD_FAILURE,
                note_path=None,
                records=[
                    ManifestRecord(
                        item_id=item_id,
                        url=item.url,
                        name=item.name,
                        profile=item.profile,
                        attempt=1,
                        fetch_status="unknown",
                        pipe_exit_code=None,
                        verified=None,
                        note_path=None,
                        error_kind="unexpected_exception",
                        error_message=str(exc),
                        usage_measured=False,
                        usage_estimated=False,
                        started_at=now,
                        finished_at=now,
                        # round-07 G5 리뷰 [P2-2] 수정: 이미 계산돼 있는
                        # 배정 정보를 폴백 레코드에도 전달해 어느 항목이
                        # 어느 provider/model에 배정된 채로 unexpected
                        # exception을 만났는지 manifest에서 추적 가능하게 한다.
                        assigned_provider=assigned_provider,
                        assigned_model=assigned_model,
                        source_kind=item.source_kind,
                    )
                ],
            )
        results.append(result)
    return results


def _aggregate_exit_code(results: list[ItemResult]) -> int:
    """계약 §3 — 전부 verified: 0 / hard failure 없이 unverified만 있음: 1 /
    hard failure 하나 이상: 2."""
    if any(r.outcome == _OUTCOME_HARD_FAILURE for r in results):
        return 2
    if any(r.outcome == _OUTCOME_UNVERIFIED for r in results):
        return 1
    return 0


def _run_vault_dry_run(items: list[QueueItem], args: argparse.Namespace) -> None:
    """`--vault`만 있고 `--commit-vault`가 없을 때 — 실제 fetch/pipe 없이
    대상 파일명만 미리 계산해 stdout에 출력한다(계약 §4 "dry-run/preview
    모드에서 대상 파일명을 먼저 manifest로 확인 가능")."""
    print("[dry-run] --commit-vault 없음 — 아래 대상 파일명만 미리보기하고 실제로 쓰지 않습니다.")
    for item in items:
        name = item.name or "(NAME 필수 — 실볼트 모드는 큐 항목에 name이 있어야 합니다)"
        try:
            target = DEFAULT_VAULT_INBOX / f"{sanitize_vault_note_name(item.name) if item.name else 'MISSING-NAME'}.md"
        except VaultNameError as exc:
            target = f"<VaultNameError: {exc}>"  # type: ignore[assignment]
        print(f"  #{item.index:04d} url={item.url} name={name!r} -> {target}")


def _reconfigure_utf8_streams() -> None:
    """Windows 기본 콘솔(cp949)에서 한국어 로그/에러가 mojibake 되는 것을 막는다
    (note_pipe.py `_reconfigure_utf8_streams` 선례 그대로)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


if __name__ == "__main__":
    main()
