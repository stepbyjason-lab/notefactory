"""장문 비영상 게시물의 원문 보전 준비본.

Sipher는 원문과 원저자 게시 순서만 제공한다. 이 모듈은 장문 게시물을
게시물별로 한 번만 분석해, 모델이 제안한 anchor를 실제 원문 범위로 검증하고
source hash 기반 cache에 기록한다. 원문 문자열은 어느 단계에서도 모델 출력으로
대체하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


LONG_MATERIAL_THRESHOLD_CHARS = 5_000
MIN_MAPPABLE_POST_CHARS = 1_000
#: 게시물 하나가 이 크기를 넘으면 전체 합계와 무관하게 원문 보전 대상으로 본다.
#:
#: 전체 합계 기준(5,000자)은 34k 원문이 writer→critic→repair를 돌며 전송량이
#: 폭발하는 것을 막으려고 둔 것이다. 그런데 사례 8(전체 3,207자)은 그 문턱을
#: 넘지 못해 보호를 못 받았고, 그 안의 프롬프트 2,495자가 writer 요약에 흡수돼
#: 복사 가능한 실물이 노트에서 사라졌다. 잡담이 길게 붙은 게시물은 보호받고
#: 짧고 밀도 높은 프롬프트 공유 게시물은 못 받는 역전이었다.
#:
#: 실측 분포 — 잡아야 하는 자료: 9,380·8,267·7,108·2,495자.
#: 잡으면 안 되는 일반 서술: 764·716·256자. 그 사이에서 2,000자로 둔다.
SINGLE_MATERIAL_THRESHOLD_CHARS = 2_000
#: 노트에 남으면 안 되는 처리 용어. 원문 자료는 렌더 후 노트에 그대로 실리므로,
#: "마커"·"전문이 없다" 류 문장은 언제나 파이프라인 내부 상태가 새어 나온 것이다.
_PROCESS_DISCLOSURE_TERMS = ("PRESERVE", "마커", "marker", "플레이스홀더", "placeholder", "렌더러")
#: 자료 전문은 렌더 후 노트에 실린다. "전문이 없다"는 서술은 항상 거짓이다.
_ABSENCE_CLAIM_RE = re.compile(r"전문[^.!?]*?(제시되지\s*않|포함되(어|지)\s*있지\s*않|담겨\s*있지\s*않|확인할\s*수\s*없|없다|없으며|없습니다)")
_MARKER_RE = re.compile(r"<!--\s*PRESERVE\s+(?P<payload>\{.*?\})\s*-->", re.DOTALL)


class LongMaterialPlanError(ValueError):
    pass


@dataclass(frozen=True)
class SourceItem:
    source_id: str
    text: str


@dataclass(frozen=True)
class PreservedRange:
    artifact_id: int
    source_id: str
    label: str
    start: int
    end: int
    text: str
    card: dict[str, Any] | None = None

    @property
    def marker(self) -> str:
        payload = {
            "id": self.artifact_id,
            "source_id": self.source_id,
            "label": self.label,
            "start_anchor": self.text[:80],
            "end_anchor": self.text[-80:],
        }
        return "<!--PRESERVE " + json.dumps(payload, ensure_ascii=False) + "-->"


@dataclass(frozen=True)
class LongMaterialPreparation:
    source_hash: str
    items: tuple[SourceItem, ...]
    ranges: tuple[PreservedRange, ...]
    cache_hit: bool


def is_long_material_candidate(
    *,
    source_kind: str,
    content_chars: int,
    items: tuple[SourceItem, ...] = (),
) -> bool:
    """원문 보전 준비가 필요한 입력인지 판정한다.

    두 조건은 서로 다른 위험을 막으므로 OR로 묶는다 — 전체 합계는 전송량
    폭발을, 단일 게시물 크기는 복사용 자료가 요약에 흡수되는 것을 막는다.
    영상 전사는 어느 쪽에도 해당하지 않는다.
    """
    if source_kind != "text_post":
        return False
    if content_chars >= LONG_MATERIAL_THRESHOLD_CHARS:
        return True
    return any(len(item.text) >= SINGLE_MATERIAL_THRESHOLD_CHARS for item in items)


def source_items_from_sipher(data: dict[str, Any]) -> tuple[SourceItem, ...]:
    items: list[SourceItem] = []
    body = data.get("body_text")
    if isinstance(body, str) and body.strip():
        items.append(SourceItem("body", body))
    thread = data.get("author_thread")
    if isinstance(thread, list):
        for index, post in enumerate(thread, 1):
            if not isinstance(post, dict):
                continue
            text = post.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            code = post.get("code")
            identifier = f"thread:{code}" if isinstance(code, str) and code else f"thread:{index}"
            items.append(SourceItem(identifier, text))
    return tuple(items)


def source_hash(items: tuple[SourceItem, ...]) -> str:
    canonical = json.dumps(
        [{"source_id": item.source_id, "text": item.text} for item in items],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _find_anchor(text: str, anchor: str, *, name: str, occurrence: int | None = None) -> int:
    value = anchor.strip()
    if len(value) < 16:
        raise LongMaterialPlanError(f"{name} anchor가 너무 짧습니다")
    positions: list[int] = []
    cursor = 0
    while True:
        found = text.find(value, cursor)
        if found < 0:
            break
        positions.append(found)
        cursor = found + 1
    if not positions:
        raise LongMaterialPlanError(f"{name} anchor를 원문에서 찾지 못했습니다")
    if occurrence is not None:
        if type(occurrence) is not int or occurrence < 1 or occurrence > len(positions):
            raise LongMaterialPlanError(f"{name} anchor occurrence가 원문 범위를 벗어났습니다")
        return positions[occurrence - 1]
    if len(positions) > 1:
        raise LongMaterialPlanError(f"{name} anchor가 원문에 둘 이상 있어 모호합니다")
    return positions[0]


def _resolve_item_ranges(item: SourceItem, raw_ranges: list[dict[str, Any]], offset: int) -> list[PreservedRange]:
    result: list[PreservedRange] = []
    occupied: list[tuple[int, int]] = []
    for local_index, raw in enumerate(raw_ranges, 1):
        if not isinstance(raw, dict):
            raise LongMaterialPlanError("보전 범위 항목이 객체가 아닙니다")
        label = raw.get("label")
        start_anchor = raw.get("start_anchor")
        end_anchor = raw.get("end_anchor")
        start_occurrence = raw.get("start_occurrence")
        card = raw.get("writer_card")
        if card is not None and not isinstance(card, dict):
            raise LongMaterialPlanError("writer_card가 객체가 아닙니다")
        if not isinstance(label, str) or not label.strip():
            raise LongMaterialPlanError("보전 범위 label이 없습니다")
        if not isinstance(start_anchor, str) or not isinstance(end_anchor, str):
            raise LongMaterialPlanError("보전 범위 anchor가 문자열이 아닙니다")
        start = _find_anchor(item.text, start_anchor, name="start", occurrence=start_occurrence)
        end_start = item.text.find(end_anchor.strip(), start)
        if end_start < 0:
            raise LongMaterialPlanError("end anchor를 start 뒤에서 찾지 못했습니다")
        if end_start < start:
            raise LongMaterialPlanError("end anchor가 start anchor보다 앞에 있습니다")
        end = end_start + len(end_anchor.strip())
        if end - start < 120:
            raise LongMaterialPlanError("보전 범위가 너무 짧습니다")
        if any(start < previous_end and previous_start < end for previous_start, previous_end in occupied):
            raise LongMaterialPlanError("보전 범위가 겹칩니다")
        occupied.append((start, end))
        result.append(
            PreservedRange(
                offset + local_index,
                item.source_id,
                label.strip(),
                start,
                end,
                item.text[start:end],
                card,
            )
        )
    return result


def _serialize(preparation: LongMaterialPreparation) -> dict[str, Any]:
    return {
        "source_hash": preparation.source_hash,
        "ranges": [
            {
                "source_id": item.source_id,
                "label": item.label,
                "start": item.start,
                "end": item.end,
                "text_hash": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                **({"writer_card": item.card} if item.card else {}),
            }
            for item in preparation.ranges
        ],
    }


def _from_cache(items: tuple[SourceItem, ...], payload: dict[str, Any]) -> LongMaterialPreparation:
    by_id = {item.source_id: item for item in items}
    ranges: list[PreservedRange] = []
    for index, raw in enumerate(payload.get("ranges", []), 1):
        if not isinstance(raw, dict):
            raise LongMaterialPlanError("cache 범위 항목이 잘못되었습니다")
        item = by_id.get(raw.get("source_id"))
        start, end, label = raw.get("start"), raw.get("end"), raw.get("label")
        if item is None or not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            raise LongMaterialPlanError("cache 범위가 현재 원문과 맞지 않습니다")
        text = item.text[start:end]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != raw.get("text_hash"):
            raise LongMaterialPlanError("cache 원문 hash가 일치하지 않습니다")
        card = raw.get("writer_card")
        if card is not None and not isinstance(card, dict):
            raise LongMaterialPlanError("cache writer_card가 잘못되었습니다")
        ranges.append(PreservedRange(index, item.source_id, label, start, end, text, card))
    if not ranges:
        raise LongMaterialPlanError("cache에 보전 범위가 없습니다")
    return LongMaterialPreparation(source_hash(items), items, tuple(ranges), True)


def load_cached(items: tuple[SourceItem, ...], cache_path: Path) -> LongMaterialPreparation | None:
    """현재 원문 hash와 일치하는 검증된 준비본만 읽는다."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("source_hash") != source_hash(items):
            return None
        return _from_cache(items, payload)
    except (OSError, json.JSONDecodeError, LongMaterialPlanError):
        return None


def prepare(
    items: tuple[SourceItem, ...],
    *,
    cache_path: Path,
    map_item: Callable[[SourceItem], list[dict[str, Any]]],
) -> LongMaterialPreparation:
    """게시물별 map을 수행하고, 검증된 결과를 source hash cache에 남긴다."""
    digest = source_hash(items)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("source_hash") == digest:
                return _from_cache(items, cached)
        except (OSError, json.JSONDecodeError, LongMaterialPlanError):
            pass

    ranges: list[PreservedRange] = []
    for item in items:
        if len(item.text) < MIN_MAPPABLE_POST_CHARS:
            continue
        try:
            ranges.extend(_resolve_item_ranges(item, map_item(item), len(ranges)))
        except LongMaterialPlanError:
            # 한 게시물의 경계가 불명확하면 그 게시물은 writer 원문에 남긴다.
            continue
    if not ranges:
        raise LongMaterialPlanError("검증된 장문 원문 보전 범위가 없습니다")
    preparation = LongMaterialPreparation(digest, items, tuple(ranges), False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_serialize(preparation), ensure_ascii=False, indent=2), encoding="utf-8")
    return preparation


def _card_block(item: PreservedRange) -> str:
    """writer가 자료를 다시 읽지 않고도 설명할 수 있는 구조 요약."""
    if not item.card:
        return ""
    lines = [f"[원문 자료 {item.artifact_id} — {item.label}]"]
    fmt = item.card.get("format")
    if isinstance(fmt, str) and fmt.strip():
        lines.append(f"형식: {fmt.strip()}")
    for key, title in (("key_structure", "구성 요소"), ("source_grounding", "원문 소제목")):
        values = item.card.get(key)
        if isinstance(values, list):
            cleaned = [str(value).strip() for value in values if str(value).strip()]
            if cleaned:
                lines.append(f"{title}: " + " / ".join(cleaned))
    return "\n".join(lines) + "\n"


def writer_source_items(preparation: LongMaterialPreparation) -> dict[str, str]:
    """writer가 다시 쓰지 않을 보전 범위를 inline marker로 대체한다."""
    by_item: dict[str, list[PreservedRange]] = {}
    for item in preparation.ranges:
        by_item.setdefault(item.source_id, []).append(item)
    output: dict[str, str] = {}
    for source in preparation.items:
        text = source.text
        for artifact in sorted(by_item.get(source.source_id, []), key=lambda value: value.start, reverse=True):
            text = text[:artifact.start] + _card_block(artifact) + artifact.marker + text[artifact.end:]
        output[source.source_id] = text
    return output


def copy_with_writer_source(data: dict[str, Any], source_items: dict[str, str]) -> dict[str, Any]:
    """writer/critic용 JSON 복사본에만 marker를 넣고 수집 원본은 보존한다."""
    copied = deepcopy(data)
    if "body" in source_items and isinstance(copied.get("body_text"), str):
        copied["body_text"] = source_items["body"]
    thread = copied.get("author_thread")
    if isinstance(thread, list):
        for index, post in enumerate(thread, 1):
            if not isinstance(post, dict):
                continue
            code = post.get("code")
            source_id = f"thread:{code}" if isinstance(code, str) and code else f"thread:{index}"
            if source_id in source_items:
                post["text"] = source_items[source_id]
    return copied


def render_markers(note: str, preparation: LongMaterialPreparation) -> str:
    # 같은 게시물 안의 빈 템플릿과 값이 채워진 예시처럼, 앞뒤 80자가 같은 독립
    # 자료가 실재한다(사례 3의 3D 템플릿/판교 예시). anchor만으로는 둘을 구분할
    # 수 없으므로 artifact_id를 1차 키로 쓰고, anchor는 검증용으로만 본다.
    by_id = {item.artifact_id: item for item in preparation.ranges}
    by_signature: dict[tuple[str, str, str], PreservedRange] = {}
    for item in preparation.ranges:
        by_signature.setdefault((item.source_id, item.text[:80], item.text[-80:]), item)
    seen: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        try:
            raw = json.loads(match.group("payload"))
            identifier = raw.get("id")
            if isinstance(identifier, int) and identifier in by_id:
                item = by_id[identifier]
                if (item.source_id, item.text[:80], item.text[-80:]) != (
                    raw["source_id"], raw["start_anchor"], raw["end_anchor"],
                ):
                    raise KeyError("marker anchor가 해당 자료와 맞지 않습니다")
            else:
                item = by_signature[(raw["source_id"], raw["start_anchor"], raw["end_anchor"])]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LongMaterialPlanError("writer marker가 검증된 원문 범위를 가리키지 않습니다") from exc
        if item.artifact_id in seen:
            raise LongMaterialPlanError("같은 원문 보전 marker가 둘 이상입니다")
        seen.add(item.artifact_id)
        longest = max((len(run) for run in re.findall(r"`+", item.text)), default=0)
        fence = "`" * max(3, longest + 1)
        return f"{fence}text\n{item.text}\n{fence}"

    rendered = _MARKER_RE.sub(replace, note)
    expected = {item.artifact_id for item in preparation.ranges}
    if seen != expected:
        raise LongMaterialPlanError("writer가 모든 원문 보전 marker를 배치하지 않았습니다")
    return _strip_process_disclosure(rendered)


def _strip_process_disclosure(note: str) -> str:
    """writer가 흘린 처리 용어 문장을 노트에서 제거한다.

    자료 전문은 이 시점에 이미 노트 안에 있다. "마커 위치에 보존되어 있다"거나
    "전문이 제시되지 않았다"는 서술은 독자에게 거짓이므로, 코드 블록 밖의 해당
    문장만 잘라낸다. 약한 모델은 프롬프트 금지만으로 막히지 않는다.
    """
    output: list[str] = []
    in_fence = False
    for line in note.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence or not any(term in line for term in _PROCESS_DISCLOSURE_TERMS):
            if in_fence or not _ABSENCE_CLAIM_RE.search(line):
                output.append(line)
                continue
        kept = [
            part
            for part in _split_sentences(line)
            if not any(term in part for term in _PROCESS_DISCLOSURE_TERMS)
            and not _ABSENCE_CLAIM_RE.search(part)
        ]
        cleaned = " ".join(part.strip() for part in kept).strip()
        if cleaned and cleaned not in {"-", "*", "#"}:
            output.append(cleaned)
    return "\n".join(output)


def _split_sentences(line: str) -> list[str]:
    parts: list[str] = []
    buffer = ""
    for char in line:
        buffer += char
        if char in ".!?" and buffer.strip():
            parts.append(buffer)
            buffer = ""
    if buffer.strip():
        parts.append(buffer)
    return parts
