"""Standalone vendor-usage snapshot and paid writer/critic role selection.

The bundled Node adapters read only the local Claude/Codex/Grok auth-backed
usage sources and Antigravity's verified loopback quota RPC.  They are copied
into NoteFactory so this module never imports or executes Madi at runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_CLI = ROOT / "tools" / "vendor-usage" / "snapshot-all.mjs"

PAID_VENDOR_MODELS = {
    "claude": "claude-sonnet-5",
    "codex": "gpt-5.6-luna",
    "agy": "gemini-3.7-flash",
    "grok": "grok-4.6",
}
_VENDOR_ALIASES = {
    "antigravity": "agy",
}
ABSOLUTE_FLOOR = 0.10


def canonical_paid_vendor(provider: str) -> str:
    """Normalize a caller/provider alias before paid-route decisions."""
    return _VENDOR_ALIASES.get(provider, provider)


def is_paid_route(provider: str, model: str) -> bool:
    """Return whether a caller-requested route can consume paid capacity."""
    canonical_provider = canonical_paid_vendor(provider)
    if canonical_provider in PAID_VENDOR_MODELS:
        return True
    return provider == "openrouter" and not model.endswith(":free")


def _terminate_snapshot_process(proc: subprocess.Popen[Any]) -> None:
    """Terminate the Node child tree when the outer Python timeout fires."""
    try:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        # `taskkill /T` handles descendants on Windows, but it can be absent,
        # denied, or time out. A direct process kill is the bounded fallback.
        if getattr(proc, "poll", lambda: None)() is None:
            kill = getattr(proc, "kill", None)
            if callable(kill):
                kill()
    except (OSError, subprocess.TimeoutExpired):
        pass


def snapshot_usage(*, timeout_seconds: float = 25.0) -> dict[str, Any]:
    """Read usage without Madi dependency; callers decide the failure policy."""
    try:
        run_options: dict[str, Any] = {}
        if os.name == "nt":
            run_options["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            run_options["start_new_session"] = True
        proc = subprocess.Popen(
            ["node", str(SNAPSHOT_CLI)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **run_options,
        )
        try:
            stdout, _stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_snapshot_process(proc)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_snapshot_process(proc)
                try:
                    proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            raise
        payload = json.loads(stdout)
        if isinstance(payload, dict) and isinstance(payload.get("providers"), dict):
            return payload
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return {"ts": None, "providers": {}}


def _vendor_entry(snapshot: dict[str, Any], vendor: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return a fully trustworthy usage entry, or its concrete unavailable reason.

    AGY's Gemini group has an independent availability state.  A partially
    parsed/degraded group is not safe to use for an automatic paid choice even
    if one individual window happens to be numeric.
    """
    providers = snapshot.get("providers") if isinstance(snapshot, dict) else None
    if not isinstance(providers, dict):
        return None, "usage snapshot 형식이 없습니다"
    if vendor == "agy":
        provider = providers.get("antigravity")
        if not isinstance(provider, dict) or provider.get("ok") is not True:
            reason = provider.get("reason") if isinstance(provider, dict) else None
            return None, f"AGY usage source unavailable ({reason or 'unknown'})"
        groups = provider.get("groups")
        group = groups.get("gemini") if isinstance(groups, dict) else None
        if not isinstance(group, dict):
            return None, "AGY Gemini usage group이 없습니다"
        if group.get("availability") != "available":
            return None, f"AGY Gemini usage 상태가 {group.get('availability')!r}입니다 ({group.get('reason') or 'unknown'})"
        return group, None
    entry = providers.get(vendor)
    if not isinstance(entry, dict) or entry.get("ok") is not True:
        reason = entry.get("reason") if isinstance(entry, dict) else None
        return None, f"{vendor} usage source unavailable ({reason or 'unknown'})"
    return entry, None


def _valid_windows(windows: dict[str, Any] | None) -> tuple[list[dict[str, float]], str | None]:
    """Validate every reported window instead of silently ignoring malformed data."""
    if not isinstance(windows, dict):
        return [], "usage windows가 없습니다"
    values: list[dict[str, float]] = []
    for key in ("five", "seven"):
        window = windows.get(key)
        if window is None:
            continue
        if not isinstance(window, dict):
            return [], f"{key} usage window 형식이 무효합니다"
        left = window.get("left")
        time_left_ratio = window.get("timeLeftRatio")
        if not (
            isinstance(left, (int, float)) and 0 <= left <= 1
            and isinstance(time_left_ratio, (int, float)) and 0 <= time_left_ratio <= 1
        ):
            return [], f"{key} usage window 측정값이 무효합니다"
        values.append({"left": float(left), "timeLeftRatio": float(time_left_ratio)})
    return values, None if values else "측정된 usage window가 없습니다"


def reset_relative_headroom(windows: dict[str, Any] | None) -> float | None:
    """Madi checkpoint와 같은 pace 기준 여유: ``left - timeLeftRatio``.

    값이 낮을수록 현재 잔여량이 리셋까지 남은 시간에 비해 부족하다. 단순 남은
    퍼센트가 아니라 리셋 임박성을 함께 보는 이유다.
    """
    entries, error = _valid_windows(windows)
    if error is not None:
        return None
    values = [entry["left"] - entry["timeLeftRatio"] for entry in entries]
    return min(values) if values else None


def paid_vendor_usage(snapshot: dict[str, Any], vendor: str) -> float | None:
    """Return reset-relative headroom; higher headroom means more usable capacity."""
    entry, error = _vendor_entry(snapshot, vendor)
    if error is not None or entry is None:
        return None
    return reset_relative_headroom(entry.get("windows"))


def paid_vendor_selectable(snapshot: dict[str, Any], vendor: str) -> bool:
    """잔여량이 실제로 소진된 vendor만 역할 선택에서 제외한다.

    Madi의 pace guard는 큰 라운드를 유예하는 scheduler 기준이다. NoteFactory의
    fallback에서는 pace를 hard block으로 쓰면 주간 창 초반에 모든 vendor를
    동시에 배제할 수 있으므로, absolute floor만 선택 불가 기준으로 둔다.
    """
    entry, error = _vendor_entry(snapshot, vendor)
    if error is not None or entry is None:
        return False
    windows, window_error = _valid_windows(entry.get("windows"))
    if window_error is not None:
        return False
    for window in windows:
        if window["left"] < ABSOLUTE_FLOOR:
            return False
    return True


def paid_vendor_unavailable_reason(snapshot: dict[str, Any], vendor: str) -> str | None:
    """Explain non-selection without falsely calling a measurement failure exhaustion."""
    entry, error = _vendor_entry(snapshot, vendor)
    if error is not None or entry is None:
        return error or "usage source unavailable"
    _windows, error = _valid_windows(entry.get("windows"))
    return error


def select_paid_roles(vendors: list[str], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Choose paid writer/critic according to the user-confirmed role rule."""
    requested = list(dict.fromkeys(vendors))
    unsupported = [vendor for vendor in requested if vendor not in PAID_VENDOR_MODELS]
    unique = [vendor for vendor in requested if vendor in PAID_VENDOR_MODELS]
    if not unique:
        detail = "; ".join(f"지원하지 않는 유료 vendor={vendor!r}" for vendor in unsupported)
        raise ValueError(detail or "지원되는 유료 vendor가 없습니다")
    usage = {vendor: paid_vendor_usage(snapshot, vendor) for vendor in unique}
    unavailable = [vendor for vendor, value in usage.items() if value is None]
    exhausted = [vendor for vendor in unique if vendor not in unavailable and not paid_vendor_selectable(snapshot, vendor)]
    available = [vendor for vendor in unique if vendor not in unavailable and vendor not in exhausted]
    notices = [f"지원하지 않는 유료 vendor={vendor!r}는 선택하지 않습니다." for vendor in unsupported]
    for vendor in unavailable:
        reason = paid_vendor_unavailable_reason(snapshot, vendor) or "unknown"
        if vendor in {"agy", "grok"}:
            notices.append(
                f"{vendor} usage를 읽을 수 없습니다 ({reason}). 해당 앱/CLI를 실행해 usage source를 켜거나 다른 vendor를 사용합니다."
            )
        else:
            notices.append(f"{vendor} usage를 읽을 수 없습니다 ({reason}). 다른 vendor를 사용합니다.")
    notices.extend(
        f"{vendor} usage 잔여량이 {ABSOLUTE_FLOOR:.0%} 미만이라 이번 유료 호출에서는 선택하지 않습니다."
        for vendor in exhausted
    )
    if not available:
        detail = "\n".join(notices) or "현재 잔여량이 있는 유료 vendor가 없습니다"
        raise ValueError(detail)
    if len(available) == 1:
        vendor = available[0]
        return {
            "writer": {"vendor": vendor, "model": PAID_VENDOR_MODELS[vendor]},
            "critic": {"vendor": vendor, "model": PAID_VENDOR_MODELS[vendor]},
            "usage": usage,
            "notices": notices,
            "mode": "single_vendor_same_vendor_allowed",
        }
    # 작가는 호출량이 크므로 리셋 대비 사용 여유(headroom)가 가장 큰 vendor를 쓴다.
    writer_vendor = max(available, key=lambda vendor: usage[vendor])
    # critic은 writer와 다른 다음 후보를 사용한다.
    critic_vendor = max((vendor for vendor in available if vendor != writer_vendor), key=lambda vendor: usage[vendor])
    return {
        "writer": {"vendor": writer_vendor, "model": PAID_VENDOR_MODELS[writer_vendor]},
        "critic": {"vendor": critic_vendor, "model": PAID_VENDOR_MODELS[critic_vendor]},
        "usage": usage,
        "notices": notices,
        "mode": "multi_vendor_split_by_higher_headroom",
    }
