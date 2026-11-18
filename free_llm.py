"""NoteFactory generation adapter.

Every model call goes through second-opinion's dispatcher.  HTTP providers,
NVIDIA NIM included, use the unified ``--request-json`` generation path;
subscription vendors use the dispatcher CLI path.  This module contains no
vendor HTTP adapter, same-route retry loop, or per-vendor rate limiter:
second-opinion owns transport and retries within one requested route.  Route
selection, ordering, and cross-route attempt history remain NoteFactory-owned.

Security invariants:
- provider credentials are never placed in argv, request JSON, logs, or results;
- system and user inputs stay in separate request fields;
- subprocess timeout kills the complete process tree;
- incomplete or failed dispatcher output never becomes a successful response.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
GITHUB_MODELS_API_BASE = "https://models.github.ai/inference"
ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"

DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_COMPLETION_TOKENS = 32768

#: 모델별 출력 토큰 예산 하한. `_call_generation_dispatcher`가 호출자 policy 값과
#: 비교해 **큰 쪽**을 쓴다 — 호출자가 더 주면 그대로 두고, 모자라면 여기까지 올린다.
#:
#: 왜 모델별인가: round-11이 넣은 단일 cap 4096은 그때 대상(Qwen3.5)에 맞춘 값인데,
#: 지금 후보들은 원 개발사 스펙상 출력 상한이 128K~974K다(2026-08-24 조사). 모두에게
#: 4096을 주는 것은 스펙의 30분의 1을 주는 셈이고, 실제로 절단을 만들고 있었다.
#:
#: 왜 thinking 모델에 더 크게 주나: 사고 토큰이 **같은 출력 예산을 잠식한다.**
#: `gemini-2.5-flash`는 thinking이 기본 켜짐이고, 예산의 대부분을 사고에 쓰고 본문이
#: 잘리는 사례가 다수 보고됐다(googleapis/python-genai#2062, ha-llmvision#609).
#: `thinkingBudget`으로 제어해도 무시된다는 보고가 있어(#1795) **예산을 넉넉히 주는 것
#: 외에 방어 수단이 없다.** 우리 실측에서도 4096은 `MAX_TOKENS`로 끊기고 16384는
#: `STOP`으로 정상 종료했다(같은 요청이 317자~3,027자로 널뜀).
#:
#: **출력 토큰은 어느 provider에서도 한도에 안 잡힌다** — NIM은 40 RPM(횟수), Gemini는
#: RPM + TPM + RPD인데 그 TPM이 *"Tokens per minute (input)"*으로 **입력만** 센다
#: (ai.google.dev/gemini-api/docs/rate-limits). 따라서 넉넉히 주는 것이 손해가 아니다.
#:
#: 값의 근거 — 2026-08-24 실측(cap 32768, source_A 2,267자, 모델당 2회):
#:
#:     muse-glimmer-30b   completion 3,703 / 4,324   ← 최대 관측
#:     gpt-oss-20b        completion 3,059 / 2,558
#:     gpt-oss-120b       completion 2,016 / 2,228
#:     minimax-m3         completion 1,529 / 1,586
#:     gemini-2.5-flash   completion 1,475
#:     gemma-4-31b-it     completion   923 /   773
#:
#: **최대 관측이 4,324다 — 옛 cap 4096을 넘는다.** `muse-glimmer`·`gpt-oss-20b`가
#: 4096 아래에서 구조적으로 잘리고 있었고, 그것이 벤치의 truncated 미달로 나타났다.
#: 같은 자수를 쓰는 데 드는 토큰이 모델마다 세 배까지 차이 난다(1,768자/923토큰 vs
#: 1,313자/3,059토큰) — 사고 토큰이 잡히는 정도의 차이로 보인다.
#:
#: **값은 131072로 통일한다** — 후보들의 원 개발사 스펙 출력 상한 중 최소치
#: (`gpt-oss` 131K · `glm-5.2` 128~131K)와 같고, 나머지는 262K~974K라 여유가 있다.
#:
#: 왜 실측 최대(4,324)보다 훨씬 크게 주나: `max_tokens`는 **상한이지 목표가 아니다.**
#: cap 32768로 재보니 모델들이 923~4,324만 쓰고 `stop`으로 끝냈다 — cap을 키워도
#: 사용량이 늘지 않는다. 그리고 출력 토큰은 어느 provider에서도 한도에 안 잡히므로
#: 크게 주는 비용이 없다. 반면 작게 주면 **긴 소스에서 조용히 잘린다** — 4~5시간
#: 유튜브 자막(6만~10만 자)이 실제 입력으로 들어오는데, 그 밴드는 아직 측정되지
#: 않았다. 상한을 스펙까지 열어두면 그 미지의 밴드에서 절단을 겪지 않는다.
#:
#: 폭주 방어는 cap이 아니라 `timeout_seconds`가 맡는다.
MODEL_MAX_COMPLETION_TOKENS: dict[str, int] = {
    # thinking 기본 켜짐 + `thinkingBudget` 설정이 무시된다는 보고 다수
    # (googleapis/python-genai#1795, #2062). 예산을 넉넉히 주는 것 외에 방어 수단이 없다.
    "gemini-3.6-flash": 131072,
    "gemini-3.5-flash": 131072,
    # 2026-09-04: 같은 Gemini Flash 계열인데 표에 없어 예산이 기본값 32K로
    # 떨어져 있었다. thinking이 켜진 채 예산이 모자라면 본문 0자에 finish=
    # MAX_TOKENS로 끝난다 — 3.6만 방어받고 나머지는 무방비였다.
    #
    # 값은 모델의 `outputTokenLimit`(Flash 65,536 / gemma 32,768)보다 크게 잡는다.
    # 2026-09-04 실측: 상한을 넘겨 요청해도 Gemini·NIM 모두 finish=STOP으로 정상
    # 응답한다 — 거부도 잘림도 없다. 무료 한도는 토큰이 아니라 요청 횟수(RPD)라
    # 예산을 아낄 이유가 없고, 상한에 딱 맞추면 구글이 상향했을 때 우리만 낮은
    # 값에 묶인다. 넉넉히 주는 쪽이 thinking 방어에도 안전하다.
    "gemini-3.8-flash": 131072,
    "gemini-3.7-flash": 131072,
    "gemini-3.5-flash-lite": 131072,
    "gemini-3.1-flash-lite": 131072,
    # Gemini API 직결 경로의 gemma. NIM 경유와 같은 모델이므로 같은 예산을 준다.
    "gemma-4-31b-it": 131072,
    "gemma-4-26b-a4b-it": 131072,
    # Think Max 추론 모드 보유(원 개발사 모델 카드). 스펙 출력 상한 384K
    "deepseek-ai/deepseek-v4-flash-0731": 131072,
    # 실측 최대 4,324 — 옛 cap 4096을 넘겨 잘리던 모델
    "meta/muse-glimmer-30b": 131072,
    # reasoning 3단계 조절 가능(공식). 실측 최대 3,059로 토큰 효율이 낮다
    "openai/gpt-oss-20b": 131072,
    "openai/gpt-oss-120b": 131072,
    # 실측 최대 1,586. 여유 충분하나 사고량 변동을 감안
    "minimaxai/minimax-m3": 131072,
    "z-ai/glm-5.2": 131072,
    "z-ai/glm-5.2:free": 131072,
    "moonshotai/kimi-k3": 131072,
    "google/gemma-4-31b-it": 131072,
    "google/gemma-4-31b-it:free": 131072,
    # 2026-09-04 critic 체인에 편입된 NIM 생존 모델. 사라진 gpt-oss-120b·
    # nemotron-super-49b·inkling 자리를 대신한다.
    "nvidia/nemotron-3-super-120b-a12b": 131072,
    "nvidia/nemotron-3.5-lightning-30b-a3b": 131072,
}


def model_completion_budget(model: str, requested: int) -> int:
    """모델별 하한과 호출자 요청 중 큰 값. 미등재 모델은 요청값을 그대로 쓴다."""
    return max(requested, MODEL_MAX_COMPLETION_TOKENS.get(model, 0))


# Import-only compatibility for frozen callers.  These values are not stored in
# RetryPolicy and are never serialized into the 0.9.8 request.
DEFAULT_STREAM_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_STREAM_READ_TIMEOUT_SECONDS = 120.0

FAILURE_CLASSES_098 = frozenset({
    "bad-invocation",
    "unknown-vendor",
    "vendor-discovery-unavailable",
    "vendor-unknown",
    "vendor-ambiguous",
    "unsupported-capability",
    "model-unknown",
    "model-ambiguous",
    "auth-failed",
    "executable-not-found",
    "vendor-state-corrupt",
    "rate-limited",
    "vendor-error",
    "no-output-timeout",
    "vendor-internal-timeout",
    "oversized-response",
    "invalid-response",
    "usage-unavailable",
    "unclassified",
})
FAILURE_ACTORS_098 = frozenset({"vendor", "user", "caller", "dispatcher"})
_FALLBACK_ACTORS = frozenset({"vendor", "user"})

#: second-opinion의 구독형(로그인 CLI) 벤더. 이들은 `--request-json` 경로가
#: effort를 codex에만 실어 보내므로, NoteFactory는 벤더 구분 없이 CLI 경로로
#: 호출하고 `model@effort` 표기를 그대로 분해해 넘긴다.
SUBSCRIPTION_VENDORS = frozenset({"codex", "agy", "claude", "grok"})
#: `--effort` 인자를 받는 벤더. agy는 dispatch CLI가 이 인자를 거부하고,
#: effort를 모델 슬러그에 붙여 쓴다(`gemini-3.8-flash-high`).
_EFFORT_CAPABLE_VENDORS = frozenset({"codex", "claude", "grok"})
#: effort를 모델 슬러그 접미사로 받는 벤더.
_EFFORT_IN_MODEL_SLUG_VENDORS = frozenset({"agy"})
#: effort를 명시하지 않은 구독형 writer 호출의 기본값.
#:
#: docs/04의 "high 필수"(round-25·27)는 **critic/arbiter 역할** 실측이다 —
#: 노트에 숨은 창작을 찾아내는 탐지 태스크라 effort에 민감했다. writer는
#: 원문에서 뽑아 정리하는 추출 작업이고 프롬프트가 구조를 이미 고정하므로
#: 같은 근거를 그대로 적용할 수 없다. 측정 없이 high를 기본으로 두면 매
#: 노트마다 추론 토큰을 낭비한다. critic 경로는 자기 effort를 명시로
#: 넘기므로 이 기본값에 영향받지 않는다.
DEFAULT_SUBSCRIPTION_EFFORT = "medium"


class StreamingUnavailableError(RuntimeError):
    """Compatibility name; streaming capability is now enforced by the dispatcher."""


class NoActiveProviderError(RuntimeError):
    """No requested route is available."""


class AllProvidersFailedError(RuntimeError):
    """NoteFactory exhausted or stopped its caller-owned route plan."""

    def __init__(self, message: str, fallback_chain: list[dict]) -> None:
        super().__init__(message)
        self.fallback_chain = fallback_chain


class ProviderCallError(RuntimeError):
    """A classified final failure from one second-opinion route dispatch."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        is_retryable: bool = False,
        failure_class: str | None = None,
        failure_actor: str | None = None,
        remedy: str | None = None,
        attempts: object = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.failure_class = failure_class
        self.failure_actor = failure_actor
        self.remedy = remedy
        self.attempts = attempts
        self.is_fallback_trigger = failure_actor in _FALLBACK_ACTORS
        self.is_retryable = is_retryable


@dataclass
class ProviderConfig:
    """Requested provider/model identity.

    api_key/base_url/kind are retained only for compatibility with existing
    configuration builders.  The dispatcher reads credentials and adapter
    metadata itself; these values are never sent across the process boundary.
    """

    name: str
    api_key: str
    model: str
    base_url: str = ""
    kind: str = "dispatcher"

    @property
    def is_active(self) -> bool:
        if self.kind in {"dispatch_cli", "dispatcher"}:
            return bool(self.model)
        return bool(self.api_key) and bool(self.model)


@dataclass(init=False)
class RetryPolicy:
    """Live request limits forwarded to one second-opinion route dispatch."""

    max_retries: int
    timeout_seconds: float
    max_completion_tokens: int

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        **_ignored_legacy_limits: object,
    ) -> None:
        # Several frozen PoC callers still pass pre-0.9.8 transport knobs.  Keep
        # their construction shape working without retaining or forwarding any
        # of those removed policy fields.
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.max_completion_tokens = max_completion_tokens


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a small dotenv-compatible KEY=VALUE file without exposing values."""
    result: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.is_file():
        return result
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            marker = value.find(" #")
            if marker >= 0:
                value = value[:marker].rstrip()
        if key:
            result[key] = value
    return result


def build_provider_configs(env: dict[str, str]) -> list[ProviderConfig]:
    """Build compatibility route records; this function does not select or call one."""
    return [
        ProviderConfig("openrouter", env.get("OPENROUTER_API_KEY", ""), env.get("OPENROUTER_MODEL", ""), env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), "openai_chat"),
        ProviderConfig("nvidia_nim", env.get("NVIDIA_NIM_API_KEY", ""), env.get("NVIDIA_NIM_MODEL", ""), env.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"), "openai_chat"),
        ProviderConfig("gemini", env.get("GEMINI_API_KEY", ""), env.get("GEMINI_MODEL", ""), env.get("GEMINI_BASE_URL", GEMINI_API_BASE), "gemini"),
        ProviderConfig("mistral", env.get("MISTRAL_API_KEY", ""), env.get("MISTRAL_MODEL", ""), env.get("MISTRAL_BASE_URL", MISTRAL_API_BASE), "openai_chat"),
        ProviderConfig("github_models", env.get("GITHUB_MODELS_API_KEY", ""), env.get("GITHUB_MODELS_MODEL", ""), env.get("GITHUB_MODELS_BASE_URL", GITHUB_MODELS_API_BASE), "openai_chat"),
        ProviderConfig("zhipu", env.get("ZHIPU_API_KEY", ""), env.get("ZHIPU_MODEL", ""), env.get("ZHIPU_BASE_URL", ZHIPU_API_BASE), "openai_chat"),
        ProviderConfig("codex", "", env.get("CODEX_MODEL", ""), kind="dispatcher"),
        ProviderConfig("agy", "", env.get("AGY_MODEL", ""), kind="dispatcher"),
        # Subscription-backed providers are reached only through second-opinion's
        # generation dispatcher.  The requested model replaces these optional
        # defaults in ``from_env_file``; activation is controlled by NoteFactory's
        # explicit paid-fallback policy, not by a secret in .env.local.
        ProviderConfig("claude", "", env.get("CLAUDE_MODEL", ""), kind="dispatcher"),
        ProviderConfig("grok", "", env.get("GROK_MODEL", ""), kind="dispatcher"),
    ]


_SO_CACHE_PARTS = ("plugins", "cache", "second-opinion", "second-opinion")
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _dispatch_cache_root(home: Path | None = None) -> Path:
    """Codex 플러그인 캐시 아래 second-opinion 설치 루트. CODEX_HOME이 없으면 홈의 `.codex`."""
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    base = Path(codex_home) if codex_home else (home or Path.home()) / ".codex"
    return base.joinpath(*_SO_CACHE_PARTS)


def _highest_cached_dispatch(root: Path) -> Path | None:
    """`<root>/<X.Y.Z>/scripts/dispatch.mjs` 중 숫자 판이 가장 높은 것. 숫자 판이 아닌 폴더는 건너뛴다."""
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    try:
        children = list(root.iterdir()) if root.is_dir() else []
    except OSError:
        return None
    for child in children:
        match = _SEMVER_RE.match(child.name)
        dispatch = child / "scripts" / "dispatch.mjs"
        if match and dispatch.is_file():
            candidates.append(((int(match[1]), int(match[2]), int(match[3])), dispatch))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def find_dispatch(*, home: Path | None = None) -> Path | None:
    """second-opinion `dispatch.mjs`를 찾는 유일한 함수(free_llm·note_pipe·model_health 공용).

    1) 환경변수 `SECOND_OPINION_DISPATCH` — 있으면 존재 여부와 무관하게 그대로(틀린 지정은 호출부가 크게 실패).
       상대 경로는 지금 프로세스 cwd 기준 절대경로로 고정한다 — 디스패처를 띄우는 자식은 cwd가 다르다.
    2) `CODEX_HOME`(없으면 `~/.codex`)의 플러그인 캐시 중 최고 판.
    3) `~/.claude`의 플러그인 캐시 중 최고 판, 4) `~/.claude` marketplace 사본.
    5) 없으면 None. plugin 파일은 읽기 전용 외부 의존성이다 — 수정·복사하지 않는다.
    """
    explicit = os.environ.get("SECOND_OPINION_DISPATCH", "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    home = home or Path.home()
    for root in (_dispatch_cache_root(home), (home / ".claude").joinpath(*_SO_CACHE_PARTS)):
        found = _highest_cached_dispatch(root)
        if found is not None:
            return found
    marketplace = (
        home / ".claude" / "plugins" / "marketplaces" / "second-opinion"
        / "plugins" / "second-opinion" / "scripts" / "dispatch.mjs"
    )
    return marketplace if marketplace.is_file() else None


def _resolve_dispatch_path() -> str:
    found = find_dispatch()
    if found is not None:
        return str(found)
    # 설치가 없으면 찾아본 자리를 그대로 돌려준다. 파일이 아니므로 호출부가
    # 「디스패처를 찾을 수 없다」로 멈추고, 메시지에 어디를 봤는지 남는다.
    return str(_dispatch_cache_root() / "*" / "scripts" / "dispatch.mjs")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=20,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def split_model_effort(model: str) -> tuple[str, str | None]:
    """`model@effort` 표기를 모델과 effort로 나눈다.

    second-opinion CLI가 쓰는 규칙과 같다. NoteFactory는 이 한 가지 표기만
    쓰고 벤더별 분기를 두지 않는다.
    """
    separator = model.rfind("@")
    if separator <= 0:
        return model, None
    candidate = model[separator + 1:].strip().lower()
    if candidate not in {"low", "medium", "high", "xhigh", "max", "ultra"}:
        return model, None
    return model[:separator], candidate


def _parse_subscription_output(raw: str, *, vendor: str) -> tuple[str, str | None]:
    """벤더별 out.txt를 (본문, 관측 모델)로 정규화한다.

    claude는 result JSON 한 덩어리, codex/grok/agy는 chunk 줄을 낸다.
    파싱 실패를 조용히 성공으로 만들지 않는다.
    """
    stripped = raw.strip()
    if not stripped:
        raise ProviderCallError("구독형 디스패처 출력이 비어 있다", failure_actor="vendor")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    # claude는 결과 본문을 `result`에, grok/agy는 `text`에 담는다. 어느 쪽이든
    # 최상위 JSON 한 덩어리이므로 필드 이름만 다르게 읽는다.
    if isinstance(payload, dict):
        for field in ("result", "text"):
            value = payload.get(field)
            if isinstance(value, str):
                usage = payload.get("modelUsage")
                reported = None
                if isinstance(usage, dict):
                    best: tuple[str, float] | None = None
                    for name, row in usage.items():
                        tokens = row.get("outputTokens") if isinstance(row, dict) else None
                        if isinstance(tokens, (int, float)) and (best is None or tokens > best[1]):
                            best = (name, float(tokens))
                    reported = best[0] if best else None
                return value, reported

    chunks: list[str] = []
    reported = None
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("text"), str):
            chunks.append(row["text"])
        if isinstance(row.get("model"), str) and reported is None:
            reported = row["model"]
    if chunks:
        return "".join(chunks), reported
    # agy는 본문을 평문 그대로 낸다. JSON 구조가 없다는 이유로 성공한 응답을
    # 버리면 안 되지만, JSON처럼 보이는데 본문 필드가 없는 출력은 실패다.
    if payload is None:
        return stripped, None
    raise ProviderCallError(f"구독형 디스패처 출력을 해석할 수 없다(vendor={vendor})", failure_actor="vendor")


def _call_subscription_dispatcher(
    config: ProviderConfig,
    prompt: str,
    system_prompt: str | None,
    *,
    policy: RetryPolicy,
) -> dict:
    """구독형 벤더(codex/agy/claude/grok)를 second-opinion CLI 경로로 호출한다.

    `--request-json` 경로는 effort를 codex에만 실어 보내서 claude가 "requires
    model, effort"로 죽고 grok/agy도 effort를 지정할 수 없다. CLI 경로는 네
    벤더를 같은 인자 규약으로 받으므로 여기서는 벤더를 구분하지 않는다.
    """
    dispatch = _resolve_dispatch_path()
    if not Path(dispatch).is_file():
        raise ProviderCallError(
            f"second-opinion 디스패처를 찾을 수 없다: {dispatch}. SECOND_OPINION_DISPATCH로 지정하라."
        )
    model, effort = split_model_effort(config.model)
    if effort is None and config.name in (_EFFORT_CAPABLE_VENDORS | _EFFORT_IN_MODEL_SLUG_VENDORS):
        effort = DEFAULT_SUBSCRIPTION_EFFORT
    if config.name in _EFFORT_IN_MODEL_SLUG_VENDORS and effort is not None:
        # agy 카탈로그는 effort까지 포함한 슬러그 하나로 모델을 고른다.
        # 이미 접미사가 붙어 있으면 두 번 붙이지 않는다.
        if not model.endswith(f"-{effort}"):
            model = f"{model}-{effort}"

    with tempfile.TemporaryDirectory(prefix="free-llm-subscription-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        brief_path = root / "brief.md"
        out_path = root / "out.txt"
        err_path = root / "err.txt"
        brief_path.write_text(
            "<system>\n" + (system_prompt or "") + "\n</system>\n\n<user>\n" + prompt + "\n</user>\n",
            encoding="utf-8",
        )
        command = [
            "node", dispatch,
            "--vendor", config.name,
            "--operation", "text",
            "--brief", str(brief_path),
            "--cwd", str(root),
            "--model", model,
            "--out", str(out_path),
            "--err", str(err_path),
            "--timeout", str(max(1, min(3600, int(policy.timeout_seconds)))),
        ]
        if effort is not None and config.name in _EFFORT_CAPABLE_VENDORS:
            command += ["--effort", effort]

        options: dict = {"cwd": root, "text": True, "encoding": "utf-8", "errors": "replace"}
        if os.name == "nt":
            options["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            options["start_new_session"] = True
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        outer_timeout = max(policy.timeout_seconds + 30.0, 60.0) * 8
        try:
            proc.wait(timeout=outer_timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(proc)
            raise ProviderCallError(f"구독형 디스패처 타임아웃({outer_timeout:.0f}s)") from exc

        raw = out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""
        if proc.returncode != 0:
            detail = err_path.read_text(encoding="utf-8", errors="replace")[-300:] if err_path.exists() else ""
            raise ProviderCallError(
                f"구독형 디스패처 실패(vendor={config.name} exit={proc.returncode}): {detail}",
                failure_actor="vendor",
            )
        text_value, reported = _parse_subscription_output(raw, vendor=config.name)

    if not text_value.strip():
        raise ProviderCallError("구독형 디스패처가 빈 본문을 돌려줬다", failure_actor="vendor")
    return {
        "text": text_value,
        "provider": config.name,
        "model": reported or model,
        "model_reported": "observed" if reported else "none",
        "requested_provider": config.name,
        "requested_model": config.model,
        "finish_reason": None,
        "truncated_suspected": False,
        "usage": None,
        "attempts": None,
    }



def _call_generation_dispatcher(
    config: ProviderConfig,
    prompt: str,
    system_prompt: str | None,
    *,
    stream: bool,
    policy: RetryPolicy,
    env_file: str | None,
) -> dict:
    """Call the unified dispatcher once with separate system/user payload fields."""
    if not isinstance(policy.max_retries, int) or not 0 <= policy.max_retries <= 16:
        raise ProviderCallError("max_retries는 0~16 범위의 정수여야 한다")
    dispatch = _resolve_dispatch_path()
    if not Path(dispatch).is_file():
        raise ProviderCallError(
            f"second-opinion 디스패처를 찾을 수 없다: {dispatch}. SECOND_OPINION_DISPATCH로 지정하라."
        )
    with tempfile.TemporaryDirectory(prefix="free-llm-dispatch-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        request_path = root / "request.json"
        response_path = root / "response.json"
        stdout_path = root / "dispatch.stdout"
        stderr_path = root / "dispatch.stderr"
        request = {
            "schema_version": 1,
            "operation": "generate",
            "provider": config.name,
            "model": config.model,
            "system": system_prompt or "",
            "user": prompt,
            "stream": bool(stream),
            "env_file": env_file or str((Path(__file__).resolve().parent / ".env.local").resolve()),
            "timeout_seconds": max(1, int(policy.timeout_seconds)),
            "max_completion_tokens": model_completion_budget(
                config.model, policy.max_completion_tokens
            ),
            "max_retries": policy.max_retries,
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        command = ["node", dispatch, "--request-json", str(request_path), "--response-json", str(response_path)]
        options: dict = {"cwd": root, "text": True, "encoding": "utf-8", "errors": "replace"}
        if os.name == "nt":
            options["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            options["start_new_session"] = True
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            proc = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, **options)
            outer_timeout = max(policy.timeout_seconds + 30.0, 60.0) * 8
            try:
                proc.wait(timeout=outer_timeout)
            except subprocess.TimeoutExpired as exc:
                _kill_process_tree(proc)
                raise ProviderCallError(f"통합 디스패처 타임아웃({outer_timeout:.0f}s)") from exc
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        payload: dict = {}
        if response_path.exists():
            try:
                loaded = json.loads(response_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                pass
    if proc.returncode != 0:
        raise ProviderCallError(
            f"통합 디스패처 실패(exit={proc.returncode}): {payload.get('message') or stderr[-300:]}",
            failure_class=(
                payload.get("failureClass")
                if isinstance(payload.get("failureClass"), str)
                else None
            ),
            failure_actor=(
                payload.get("failureActor")
                if isinstance(payload.get("failureActor"), str)
                else None
            ),
            remedy=payload.get("remedy") if isinstance(payload.get("remedy"), str) else None,
            attempts=payload.get("attempts"),
        )
    required = {"text", "provider", "model", "model_reported"}
    if (
        not required.issubset(payload)
        or not isinstance(payload["text"], str)
        or not payload["text"].strip()
        or not isinstance(payload["provider"], str)
        or not payload["provider"]
        or not isinstance(payload["model"], str)
        or not payload["model"]
    ):
        raise ProviderCallError("통합 디스패처 응답 schema가 불완전하거나 본문이 비어 있다")
    if payload["model_reported"] not in {"observed", "none"}:
        raise ProviderCallError(f"통합 디스패처 model_reported가 무효하다: {payload['model_reported']!r}")
    return {
        "text": payload["text"],
        "provider": payload["provider"],
        "model": payload["model"],
        "model_reported": payload["model_reported"],
        # Caller-owned route identity is a trust boundary.  A dispatcher/vendor
        # payload may report its observed provider/model above, but it cannot
        # rewrite what NoteFactory requested or the billing classification that
        # derives from that request.
        "requested_provider": config.name,
        "requested_model": config.model,
        "finish_reason": payload.get("finish_reason"),
        "truncated_suspected": bool(payload.get("truncated_suspected", False)),
        "usage": payload.get("usage"),
        "attempts": payload.get("attempts"),
    }


@dataclass
class FreeLLMClient:
    """Caller-owned ordered routes backed by one dispatcher call per route.

    second-opinion owns retries within each requested provider/model pair.
    NoteFactory owns cross-route order, attempt count, and attempt history.
    """

    providers: list[ProviderConfig]
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    env_file: str | None = None
    max_route_attempts: int | None = None

    @classmethod
    def from_env_file(
        cls,
        path: str | Path,
        *,
        provider: str,
        model: str,
        retry_policy: RetryPolicy | None = None,
    ) -> "FreeLLMClient":
        """Build one caller-requested route from a multi-provider environment.

        The environment is a credential/configuration catalog, not an ordered
        execution plan.  This adapter therefore selects only the exact
        provider/model pair requested by its caller.  Any fallback after that
        request stays a single route.  Callers may append qualified routes to
        the returned client's ordered provider list.
        """
        requested_provider = provider.strip()
        requested_model = model.strip()
        if not requested_provider or not requested_model:
            raise NoActiveProviderError(
                "요청 provider/model이 비어 있습니다. route 한 쌍을 명시하십시오."
            )

        configs = build_provider_configs(parse_env_file(path))
        matching = [config for config in configs if config.name == requested_provider]
        if not matching:
            raise NoActiveProviderError(
                f"요청 provider='{requested_provider}'를 인식하지 못했습니다."
            )

        requested = replace(matching[0], model=requested_model)
        if not requested.is_active:
            raise NoActiveProviderError(
                f"요청 provider='{requested_provider}'가 환경에서 비활성입니다. "
                "해당 provider의 인증 설정을 확인하십시오."
            )
        return cls(
            [requested],
            retry_policy or RetryPolicy(),
            str(Path(path).resolve()),
        )

    def generate(self, prompt: str, *, system_prompt: str | None = None, stream: bool = True) -> dict:
        """기본값이 `stream=True`인 이유 — 프로덕션이 스트리밍을 계약으로 고정했기 때문이다.

        `note_harness.py:1923`·`:1967`이 `stream=True`를 명시하므로, 기본값이 False면
        **명시하지 않은 호출자만 프로덕션과 다른 전송 경로로 돈다.** 2026-08-24에 그 함정이
        실제로 터졌다 — `tools/model_bench.py`가 stream을 명시하지 않아 non-stream으로
        후보를 재고 있었고, 같은 gemini 호출이 stream에서는 2,479자를 완주하는데
        non-stream에서는 705바이트에서 잘렸다. 그 절단이 벤치의 「자격 통과 0」으로
        나타나 모델 탓으로 읽혔다.

        전송 방식으로 갈리는 모델이 실재한다(같은 날 deepseek은 반대로 stream에서만 504).
        따라서 **측정과 생산은 같은 전송 경로여야 한다.**
        """
        if not self.providers:
            raise NoActiveProviderError("요청 provider가 없습니다.")

        route_limit = len(self.providers)
        if self.max_route_attempts is not None:
            route_limit = max(0, min(self.max_route_attempts, route_limit))

        attempt_history: list[dict] = []
        visited: set[tuple[str, str]] = set()
        last_error: ProviderCallError | None = None
        last_truncated: dict | None = None

        for requested in self.providers:
            identity = (requested.name, requested.model)
            if identity in visited:
                continue
            if len(visited) >= route_limit:
                break
            visited.add(identity)
            logger.info(
                "dispatching requested provider=%s model=%s stream=%s",
                requested.name,
                requested.model,
                stream,
            )
            try:
                if requested.name in SUBSCRIPTION_VENDORS:
                    result = _call_subscription_dispatcher(
                        requested,
                        prompt,
                        system_prompt,
                        policy=self.retry_policy,
                    )
                else:
                    result = _call_generation_dispatcher(
                        requested,
                        prompt,
                        system_prompt,
                        stream=stream,
                        policy=self.retry_policy,
                        env_file=self.env_file,
                    )
            except ProviderCallError as exc:
                last_error = exc
                record = {
                    "provider": requested.name,
                    "model": requested.model,
                    "status": "failed",
                    "error": str(exc),
                    "failureClass": exc.failure_class,
                    "failureActor": exc.failure_actor,
                    "remedy": exc.remedy,
                    "attempts": exc.attempts,
                }
                attempt_history.append(record)
                logger.warning(
                    "route failed provider=%s model=%s actor=%s class=%s",
                    requested.name,
                    requested.model,
                    exc.failure_actor,
                    exc.failure_class,
                )
                if exc.failure_actor not in _FALLBACK_ACTORS:
                    break
                continue

            # A transport can return text yet explicitly say it ended without a
            # completion signal.  This is not a usable critic/generation result:
            # handing it to the harness makes the first route a single point of
            # failure and prevents the already-configured free suffix from running.
            # Keep the last receipt so a single-route caller still receives the
            # exact truncation signal and can fail closed.
            if result.get("truncated_suspected"):
                last_truncated = result
                attempt_history.append({
                    "provider": requested.name,
                    "model": requested.model,
                    "status": "truncated",
                    "finish_reason": result.get("finish_reason"),
                    "attempts": result.get("attempts"),
                })
                continue

            attempt_history.append({
                "provider": requested.name,
                "model": requested.model,
                "status": "success",
                "actual_provider": result["provider"],
                "actual_model": result["model"],
                "attempts": result.get("attempts"),
            })
            result["fallback_chain"] = attempt_history
            return result

        if last_truncated is not None:
            last_truncated["fallback_chain"] = attempt_history
            return last_truncated

        detail = f": {last_error}" if last_error is not None else ""
        raise AllProvidersFailedError(
            f"모든 허용 route 호출이 실패하거나 중단되었습니다 (시도: {len(attempt_history)}개){detail}",
            fallback_chain=attempt_history,
        ) from last_error
