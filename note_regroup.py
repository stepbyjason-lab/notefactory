"""구간별 전사 노트를 주제 기준으로 다시 묶는다(round-33-H1 재개 2).

긴 전사는 구간마다 따로 쓰고 이어 붙였다. 그러자 최상위 제목이 시각 구간이 되고 같은
주제가 여러 곳에 흩어졌다. 여기서는 모델에게 절 제목과 요지만 주어 주제 묶음·순서·
첫머리 요약을 받고, 본문 재조립은 코드가 한다. 구간별 본문 문장은 다시 쓰지 않는다.
병합 흔적(같은 뜻 줄 반복, 개요 필드 누출)도 여기서 코드로 정리한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from note_harness import (
    DEFAULT_PROMPTS_DIR,
    _record_usage,
    _summarize_tokens,
    extract_json_object,
    load_prompt,
)
from note_validate import _SOURCE_URL_RE, _canonical_url, _source_urls

REGROUP_PROMPT_NAME = "harness/regroup.md"
OPERATIONS_TITLE = "운영·안내"
LINKS_TITLE = "링크와 원문 설명"
LIMITS_TITLE = "한계 및 미해결 질문"
_GIST_CHARS = 300
_GIST_SENTENCES = 3
#: 같은 입력으로 몇 번 물어 운영 분류를 투표하나(round-33-H1 재개 3). 모두 운영이라고 한 절만 옮긴다.
REGROUP_VOTES = 3
#: 읽을 수 있는 답 `REGROUP_VOTES`개를 얻으려고 부르는 최대 횟수.
REGROUP_MAX_CALLS = 5
#: 구간 개요 문장의 글자 2개 묶음이 본문에 이 비율보다 적게 있으면 본문에 없는 내용으로 본다.
#: 실측(b-full-g1·g2 도입 20문장): 본문을 요약한 도입 문장은 0.56~0.89였다.
_COVERED_RATIO = 0.5
#: 시각·주소 없는 줄끼리 글자 2개 묶음이 이 비율 이상 겹치면 같은 줄로 본다. 실측(round-33-H1
#: 노트 끝부분 줄 쌍): 뜻이 다른 줄도 0.38까지 겹쳐서, 글자 겹침은 거의 같은 줄만 잡게 둔다.
_SAME_LINE_RATIO = 0.5
#: 수집 상태 구역의 항목 이름(note_pipe `COLLECTION_STATUS_MESSAGES`의 「- 댓글:」·「- OCR:」).
#: 구간마다 같은 상태 블록을 받아 저마다 다른 말로 한 줄씩 적으므로, 항목마다 첫 줄만 남긴다.
_STATUS_SUBJECTS = ("댓글", "OCR")

_TS_BLOCK_RE = re.compile(r"\[t=[^\]]*\]")
_NOTE_TIME_RE = re.compile(r"t=(\d{1,2}):(\d{2}):(\d{2})")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_URL_RE = re.compile(r"https?://[^\s)>\]]+?(?=[.,:;]?(?:[\s)>\]]|$))")
#: 링크 줄에서 주소를 뺀 나머지가 이것뿐이면 설명이 없는 줄이다.
_NO_DESCRIPTION_RE = re.compile(r"^(?:명시된\s*설명\s*없음\.?)?$")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
_TIME_SPAN_RE = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[–~-]\s*\d{1,2}:\d{2}(?::\d{2})?)?(?:\s*구간)?"
)
_PLAN_LEAK_RE = re.compile(r"^\s*[-*]\s*(?:\*\*)?(?:관련\s*)?고유명사(?:\*\*)?\s*[:：]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_WORD_CHARS_RE = re.compile(r"[^0-9a-z가-힣]")


class RegroupError(RuntimeError):
    """모델 응답으로 주제 묶음을 만들 수 없다. 이미 쓴 토큰은 `tokens`에 담는다."""

    tokens: dict[str, object] = {}


@dataclass
class Section:
    title: str
    body: list[str]
    chunk: int


@dataclass
class ChunkNote:
    head: list[str] = field(default_factory=list)
    intro: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)


@dataclass
class RegroupOutcome:
    note: str
    diagnostics: dict[str, object]
    tokens: dict[str, object]
    model_responses: list[dict]


# ---------------------------------------------------------------------------
# 구간 노트 읽기와 병합 흔적 정리
# ---------------------------------------------------------------------------


def strip_plan_leaks(lines: list[str]) -> tuple[list[str], int]:
    """개요 패스의 필드(「고유명사: …」 나열)가 노트에 새어 나온 줄을 뺀다."""
    kept = [line for line in lines if not _PLAN_LEAK_RE.match(line)]
    return kept, len(lines) - len(kept)


#: 구간 작가 소스 맨 앞에 코드가 넣는 안내(note_pipe `chunk_part_notice`)의 표지와 고정 문구(round-33-H1
#: 리뷰 수리 2 — 기준 문서 「조건 3을 다시 낮춘다」). 표지 또는 안내 문장 전체를 받고, 숫자 자리만 숫자로
#: 받는다. 문장 일부(「N개 구간으로 나눈 것 가운데」)는 강의 내용과 겹칠 수 있어 받지 않는다. 이 글자가 든
#: 문장만 지운다. 「N번째 구간」·「이 구간」·「전사 중 후반부」처럼 낱말로 되받은 문장은 강의 내용 문장과
#: 가를 수 없어 지우지 않는다(알려진 한계). 안내 문구를 바꾸면 여기와 시험을 같이 바꾼다.
_CHUNK_NOTICE_RE = re.compile(
    r"\[구간 안내\]"
    r"|이 전사는 한 강의를 \d+개 구간으로 나눈 것 가운데 \d+번째 구간이다"
    r"|앞뒤 구간은 따로 정리되므로, 이 구간에 없는 내용이 강의에 없다는 뜻은 아니다"
)
#: 자리표시 주소의 말줄임(`...`·`…`).
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_URL_TAIL_PUNCT = ".,;:!?，。`…"
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+\.)?\s*")


def _is_placeholder_link_line(line: str, source_urls: set[str]) -> bool:
    """줄에 든 주소가 모두 말줄임이 든 자리표시 주소이면 참(`watch?v=...`).

    다른 주소가 하나라도 있으면(입력 밖 주소도) 거짓이다. 입력 주소 끝에 말줄임이 붙은 것은
    자리표시가 아니라 입력 주소로 본다.
    """
    urls = _SOURCE_URL_RE.findall(line)
    return bool(urls) and all(
        _ELLIPSIS_RE.search(url) and _canonical_url(url.rstrip(_URL_TAIL_PUNCT)) not in source_urls
        for url in urls
    )


def _strip_notice_sentences(line: str) -> tuple[str | None, list[str]]:
    """줄에서 구간 안내 표지·문구가 든 문장만 뺀다. 남는 문장이 없으면 줄째 없앤다(None)."""
    if line.lstrip().startswith("#") or not _CHUNK_NOTICE_RE.search(line):
        return line, []
    prefix = _BULLET_PREFIX_RE.match(line).group(0)  # type: ignore[union-attr]
    sentences = [s for s in _SENTENCE_RE.split(line[len(prefix):].strip()) if s]
    removed = [s for s in sentences if _CHUNK_NOTICE_RE.search(s)]
    rest = [s for s in sentences if s not in removed]
    return (prefix + " ".join(rest) if rest else None), removed


def strip_process_traces(note: str, source_text: str) -> tuple[str, dict[str, list[str]]]:
    """다 만든 구간 병합 노트에서 틀릴 수 없는 만든 과정의 표지 두 가지만 지운다.

    round-33-H1 리뷰 수리 2: 낱말·정규식으로 「만든 과정의 문장인가」를 가르던 규칙은 고칠 때마다
    본문 문장과 입력 주소를 지웠다. 그래서 기준 문서 「조건 3을 다시 낮춘다」대로 표지만 본다.
    - 구간 안내 표지: 코드가 넣은 「[구간 안내]」나 안내 문구(`_CHUNK_NOTICE_RE`)가 글자 그대로 든 문장.
      절과 상관없이 문장 단위로 지우고, 같은 줄의 다른 문장은 남긴다.
    - 링크 절에서 말줄임이 든 자리표시 주소밖에 없는 줄. 다른 주소가 함께 든 줄(입력 주소든 입력 밖
      주소든), 입력 주소 끝에 말줄임이 붙은 줄은 남긴다.
    표지 없이 구간 안내를 되받은 문장과 입력에 없는 실제 형식의 주소는 지우지 않는다(알려진 한계).
    반환: (노트, {"chunk_echo": 지운 문장, "unsourced_links": 지운 자리표시 링크 줄}).
    """
    source_urls = {_canonical_url(u) for u in _source_urls(source_text)}
    removed: dict[str, list[str]] = {"chunk_echo": [], "unsourced_links": []}
    out: list[str] = []
    section = ""
    for line in note.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            out.append(line)
            continue
        if section == LINKS_TITLE and _is_placeholder_link_line(line, source_urls):
            removed["unsourced_links"].append(line)
            continue
        kept, echoed = _strip_notice_sentences(line)
        removed["chunk_echo"].extend(echoed)
        if kept is not None:
            out.append(kept)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    # 내용이 모두 빠진 링크·한계 절은 제목만 남기지 않는다.
    text = re.sub(rf"(?m)^## (?:{LINKS_TITLE}|{LIMITS_TITLE})\s*(?=^## |\Z)", "", text)
    return text.strip() + "\n", removed


def parse_chunk_note(note: str, chunk: int) -> ChunkNote:
    """구간 노트 하나를 제목·도입·절·링크·한계로 나눈다."""
    parsed = ChunkNote()
    target = parsed.intro
    for line in note.strip().splitlines():
        if line.startswith("# ") or line.startswith("**출처:**"):
            parsed.head.append(line)
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            if title == LINKS_TITLE:
                target = parsed.links
            elif title.startswith("한계"):
                target = parsed.limits
            else:
                # 절 제목 끝의 시각 표기(`### 제목 [t=…]`)는 지운다 — 제목은 주제를 말한다.
                title = _TS_BLOCK_RE.sub("", title).strip()
                parsed.sections.append(Section(title=title, body=[], chunk=chunk))
                target = parsed.sections[-1].body
            continue
        if target is parsed.links or target is parsed.limits:
            if line.strip():
                target.append(line)
        else:
            target.append(line)
    return parsed


def _bigrams(text: str) -> set[str]:
    plain = _WORD_CHARS_RE.sub("", _BRACKET_RE.sub("", text.replace("`", "")).lower())
    return {plain[i : i + 2] for i in range(len(plain) - 1)}


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def dedupe_lines(lines: list[str], *, links: bool = False) -> list[str]:
    """여러 구간이 각자 쓴 링크·한계 줄에서 같은 줄과 같은 뜻 줄을 한 번만 남긴다.

    시각 표기가 있는 줄은 구간마다 다른 내용이라 글자가 같을 때만 뺀다. 주소가 있는
    줄은 이미 나온 주소만 담았으면 뺀다. 둘 다 없는 줄은 이렇게 본다.
    - 링크 절(`links=True`): 주소 없는 줄은 「링크 없음」류라 첫 줄만 남긴다.
    - 수집 상태 항목(댓글·OCR)을 말하는 줄은 항목마다 첫 줄만 남긴다.
    - 그 밖에는 앞서 남긴 줄과 거의 같을 때만 뺀다.
    """
    kept: list[str] = []
    seen: set[str] = set()
    seen_urls: set[str] = set()
    seen_subjects: set[str] = set()
    plain_grams: list[set[str]] = []
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        urls = set(_URL_RE.findall(line))
        if urls:
            if urls <= seen_urls:
                continue
            seen_urls |= urls
            rest = re.sub(r"[`*\-—–:·()\s]+", " ", _URL_RE.sub("", line)).strip()
            if links and _NO_DESCRIPTION_RE.match(rest):
                # 「명시된 설명 없음」뿐인 줄은 주소만 남긴다.
                line = "- " + " ".join(sorted(urls))
                key = line
        elif not _TS_BLOCK_RE.search(line) and not _TIME_RE.search(line):
            subject = "링크" if links else next((s for s in _STATUS_SUBJECTS if s in line), None)
            if subject is not None:
                if subject in seen_subjects:
                    continue
                seen_subjects.add(subject)
            grams = _bigrams(line)
            if any(_overlap(grams, other) >= _SAME_LINE_RATIO for other in plain_grams):
                continue
            plain_grams.append(grams)
        seen.add(key)
        kept.append(line)
    return kept


def _has_text(lines: list[str]) -> bool:
    return any(line.strip() for line in lines)


def _sentences(lines: list[str]) -> list[str]:
    text = " ".join(line.strip() for line in lines if line.strip())
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _gist(section: Section) -> str:
    texts = [
        _TS_BLOCK_RE.sub("", line).strip().lstrip("-*").replace("**", "").strip()
        for line in section.body
        if line.strip() and not line.lstrip().startswith("#")
    ]
    gist = " ".join(_sentences(texts)[:_GIST_SENTENCES])
    return gist[:_GIST_CHARS]


def _attach_uncovered_intro(chunks: list[ChunkNote]) -> int:
    """구간 도입 문단에만 있고 본문에 없는 문장을 그 구간의 첫 절 끝에 붙인다.

    도입 문단은 재배열 뒤 자리가 없어 빠지므로, 본문에 없는 내용이 함께 사라지지 않게 한다.
    """
    body_grams = _bigrams(
        "\n".join(line for c in chunks for s in c.sections for line in s.body)
    )
    attached = 0
    for chunk in chunks:
        if not chunk.sections:
            continue
        uncovered = [
            sentence
            for sentence in _sentences(chunk.intro)
            if (grams := _bigrams(_TS_BLOCK_RE.sub("", sentence)))
            and len(grams & body_grams) / len(grams) < _COVERED_RATIO
        ]
        if uncovered:
            first = chunk.sections[0]
            first.body.extend(["", " ".join(uncovered)])
            attached += len(uncovered)
    return attached


# ---------------------------------------------------------------------------
# 모델 입력과 응답
# ---------------------------------------------------------------------------


def build_regroup_message(sections: list[Section], intros: list[str]) -> str:
    lines = ["[절 목록]"]
    for number, section in enumerate(sections, start=1):
        lines.append(f"{number}. 제목: {section.title}")
        lines.append(f"   요지: {_gist(section) or '(본문 없음)'}")
    if intros:
        lines.extend(["", "[구간 개요 — 요약 참고용, 묶음 대상이 아니다]"])
        lines.extend(f"- {intro}" for intro in intros)
    return "\n".join(lines)


def clean_topic_title(title: str, fallback: str) -> str:
    """주제 제목에서 시각·구간 표기를 지운다. 남는 것이 없으면 대체 제목을 쓴다."""
    cleaned = _TIME_SPAN_RE.sub("", title).strip(" -–~·:,()[]")
    return cleaned if cleaned and not _TIME_RE.search(cleaned) else fallback


def parse_regroup_response(
    text: str, sections: list[Section]
) -> tuple[list[str], list[tuple[str, list[int]]], list[int], dict[str, int]]:
    """모델 응답을 (요약, 주제 묶음, 운영 절, 진단)으로 읽는다. 번호는 0부터로 바꾼다.

    빠진 절은 제 제목으로 따로 묶어 뒤에 붙인다 — 어떤 절도 버리지 않는다.
    """
    try:
        data = json.loads(extract_json_object(text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RegroupError(f"재배열 응답이 JSON이 아니다: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        raise RegroupError("재배열 응답에 topics 목록이 없다")
    summary_raw = data.get("summary", [])
    summary = [summary_raw] if isinstance(summary_raw, str) else summary_raw
    if not isinstance(summary, list):
        raise RegroupError("재배열 응답의 summary가 목록이 아니다")

    count = len(sections)
    seen: set[int] = set()
    diag = {"unknown_ids": 0, "duplicate_ids": 0, "missing_ids": 0}

    def take(raw: object) -> list[int]:
        picked: list[int] = []
        for value in raw if isinstance(raw, list) else []:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= count:
                diag["unknown_ids"] += 1
                continue
            if value - 1 in seen:
                diag["duplicate_ids"] += 1
                continue
            seen.add(value - 1)
            picked.append(value - 1)
        return picked

    topics: list[tuple[str, list[int]]] = []
    for topic in data["topics"]:
        if not isinstance(topic, dict):
            continue
        ids = take(topic.get("sections"))
        if ids:
            title = clean_topic_title(str(topic.get("title") or ""), sections[ids[0]].title)
            topics.append((title, ids))
    operations = take(data.get("operations", []))
    for index in range(count):
        if index not in seen:
            diag["missing_ids"] += 1
            topics.append((clean_topic_title(sections[index].title, "기타"), [index]))
    if not topics:
        raise RegroupError("재배열 응답에 본문 주제가 하나도 없다")
    return [str(s).strip() for s in summary if str(s).strip()], topics, operations, diag


Answer = tuple[list[str], list[tuple[str, list[int]]], list[int], dict[str, int]]


def _labels(answer: Answer, count: int) -> list[object]:
    labels: list[object] = [None] * count
    for position, (_title, ids) in enumerate(answer[1]):
        for index in ids:
            labels[index] = position
    for index in answer[2]:
        labels[index] = "ops"
    return labels


def _agreement(a: list[object], b: list[object]) -> float:
    """두 답이 절 쌍마다 「같이 묶었나」를 같게 판단한 비율."""
    count = len(a)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if not pairs:
        return 1.0
    same = sum((a[i] == a[j]) == (b[i] == b[j]) for i, j in pairs)
    return same / len(pairs)


def source_order(sections: list[Section]) -> list[tuple[int, int]]:
    """절마다 원문에서 나오는 자리를 (가장 이른 시각, 절 번호)로 낸다.

    구간 작가가 구간 안 절을 시간 순서로 쓰지 않기도 해서 절 번호만으로는 원문 순서와
    어긋난다(b-full-t1). 시각이 없는 절은 앞 절의 시각을 이어 받고, 시각이 하나도 없는
    전사는 절 번호 순서가 된다.
    """
    keys: list[tuple[int, int]] = []
    last = 0
    for index, section in enumerate(sections):
        times = [
            int(h) * 3600 + int(m) * 60 + int(s)
            for line in section.body
            for h, m, s in _NOTE_TIME_RE.findall(line)
        ]
        if times:
            last = min(times)
        keys.append((last, index))
    return keys


def combine_answers(
    answers: list[Answer], count: int, order: list[tuple[int, int]] | None = None
) -> tuple[list[str], list[tuple[str, list[int]]], list[int], dict[str, object]]:
    """여러 번 받은 답을 하나로 합친다.

    - 운영 절: 모든 답이 운영이라고 한 절만 옮긴다(답이 둘 이상일 때). 잘못 옮기면
      주제가 흩어지지만, 운영 멘트가 본문에 남는 것은 가벼운 결함이라 기본값은 내용이다.
    - 묶음·요약: 다른 답들과 묶음이 가장 많이 일치하는 답을 쓴다(같으면 앞의 답).
    - 그 답이 운영으로 보냈지만 만장일치가 아니어서 남은 절은, 원문 순서로 가장 가까운
      절이 든 묶음에 넣는다.
    - 순서: 묶음은 그 안의 절이 원문에 처음 나오는 순서로, 묶음 안 절도 원문 순서로 놓는다.
      원문 순서는 `order`(절마다 정렬 열쇠, 없으면 절 번호)다.
    """
    rank = order or [(0, i) for i in range(count)]
    labels = [_labels(answer, count) for answer in answers]
    scores = [
        sum(_agreement(labels[k], labels[m]) for m in range(len(answers)) if m != k)
        for k in range(len(answers))
    ]
    chosen = max(range(len(answers)), key=lambda k: (scores[k], -k))
    summary, chosen_topics, chosen_ops, _diag = answers[chosen]
    votes = [sum(index in answer[2] for answer in answers) for index in range(count)]
    unanimous = len(answers) >= 2
    operations = sorted(
        (i for i in range(count) if unanimous and votes[i] == len(answers)), key=lambda i: rank[i]
    )
    topics = [(title, [i for i in ids]) for title, ids in chosen_topics]
    topic_of = {i: position for position, (_t, ids) in enumerate(topics) for i in ids}
    # ponytail: 남은 절은 원문에서 가장 가까운 절의 묶음에 넣는다. 강의는 이웃 절이 같은 주제인 경우가 많아서다.
    for index in sorted(set(chosen_ops) - set(operations)):
        neighbor = min(
            (j for j in topic_of if j != index), key=lambda j: (abs(j - index), j > index), default=None
        )
        if neighbor is None:
            topics.append((clean_topic_title("", "기타"), [index]))
            topic_of[index] = len(topics) - 1
        else:
            topics[topic_of[neighbor]][1].append(index)
            topic_of[index] = topic_of[neighbor]
    topics = sorted(
        ((title, sorted(ids, key=lambda i: rank[i])) for title, ids in topics if ids),
        key=lambda item: rank[item[1][0]],
    )
    pairwise = [
        round(_agreement(labels[k], labels[m]), 3)
        for k in range(len(answers))
        for m in range(k + 1, len(answers))
    ]
    diag: dict[str, object] = {
        "answers": len(answers),
        "chosen_answer": chosen + 1,
        "topic_agreement": pairwise,
        "ops_votes_split": sum(1 for v in votes if 0 < v < len(answers)),
        "ops_kept_as_content": len(set(chosen_ops) - set(operations)),
    }
    for key in ("unknown_ids", "duplicate_ids", "missing_ids"):
        diag[key] = sum(answer[3][key] for answer in answers)
    return summary, topics, operations, diag


# ---------------------------------------------------------------------------
# 재조립
# ---------------------------------------------------------------------------


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(_THOUSANDS_RE.sub("", _TS_BLOCK_RE.sub("", text))))


def _render_section(section: Section, *, heading: bool) -> list[str]:
    body = [
        f"#{_TS_BLOCK_RE.sub('', line).rstrip()}" if line.startswith("#") else line
        for line in section.body
    ]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return ([f"### {section.title}", ""] if heading else []) + body + [""]


def assemble_note(
    chunks: list[ChunkNote],
    sections: list[Section],
    summary: list[str],
    topics: list[tuple[str, list[int]]],
    operations: list[int],
) -> tuple[str, int]:
    """묶음대로 본문을 다시 놓는다. 반환: (노트, 본문에 없는 숫자가 있어 뺀 요약 문장 수)."""
    body: list[str] = []
    for title, ids in topics:
        body.extend([f"## {title}", ""])
        single = len(ids) == 1 and sections[ids[0]].title == title
        for index in ids:
            body.extend(_render_section(sections[index], heading=not single))
    if operations:
        body.extend([f"## {OPERATIONS_TITLE}", ""])
        for index in operations:
            body.extend(_render_section(sections[index], heading=True))

    body_numbers = _numbers("\n".join(body))
    kept_summary = [s for s in summary if _numbers(s) <= body_numbers]
    head = chunks[0].head if chunks and chunks[0].head else []
    links = dedupe_lines([line for c in chunks for line in c.links], links=True)
    limits = dedupe_lines([line for c in chunks for line in c.limits])
    out = [*head, ""]
    if kept_summary:
        out.extend([" ".join(kept_summary), ""])
    out.extend(body)
    if links:
        out.extend([f"## {LINKS_TITLE}", *links, ""])
    if limits:
        out.extend([f"## {LIMITS_TITLE}", *limits, ""])
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    return text, len(summary) - len(kept_summary)


def regroup_chunk_notes(
    notes: list[str],
    generate_fn: Callable[..., dict],
    *,
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> RegroupOutcome:
    """구간 노트들을 주제 기준 노트 하나로 다시 묶는다. 읽을 수 있는 답이 하나도 없으면 RegroupError.

    모델 호출 도중 어떤 예외로 끝나도 그때까지 쓴 토큰을 그 예외의 `tokens`에 싣고 다시 올린다.
    """
    chunks = [parse_chunk_note(note, index) for index, note in enumerate(notes)]
    leaks = 0
    for chunk in chunks:
        chunk.intro, removed = strip_plan_leaks(chunk.intro)
        leaks += removed
        for section in chunk.sections:
            section.body, removed = strip_plan_leaks(section.body)
            leaks += removed
    attached = _attach_uncovered_intro(chunks)
    all_sections = [s for c in chunks for s in c.sections]
    sections = [s for s in all_sections if _has_text(s.body)]
    if not sections:
        raise RegroupError("다시 묶을 절이 없다")
    intros = [" ".join(_sentences(c.intro)) for c in chunks if _has_text(c.intro)]
    system_prompt = load_prompt(REGROUP_PROMPT_NAME, prompts_dir=prompts_dir)
    message = build_regroup_message(sections, intros)

    responses: list[dict] = []
    usages = []
    errors: list[str] = []
    answers: list[Answer] = []
    try:
        while len(answers) < REGROUP_VOTES and len(responses) < REGROUP_MAX_CALLS:
            response = generate_fn(message, system_prompt=system_prompt)
            responses.append(response)
            raw = str(response.get("text", ""))
            usages.append(
                _record_usage(
                    "regroup", usage=response.get("usage"), prompt_text=message + system_prompt,
                    output_text=raw, response=response,
                )
            )
            try:
                answers.append(parse_regroup_response(raw, sections))
            except RegroupError as exc:
                errors.append(str(exc))
    except Exception as exc:
        # 리뷰 F4 — 도중에 끝나도(제공자 예외 등) 이미 쓴 호출의 토큰은 호출자가 헤더에 더한다.
        try:
            exc.tokens = _summarize_tokens(usages)  # type: ignore[attr-defined]
        except AttributeError:  # 속성을 못 붙이는 예외는 그대로 올린다.
            pass
        raise
    if not answers:
        error = RegroupError(" / ".join(errors))
        error.tokens = _summarize_tokens(usages)
        raise error
    summary, topics, operations, diag = combine_answers(answers, len(sections), source_order(sections))
    note, summary_dropped = assemble_note(chunks, sections, summary, topics, operations)
    diag.update(
        {
            "sections": len(sections),
            "topics": len(topics),
            "operations": len(operations),
            "operations_titles": [sections[i].title for i in operations],
            "summary_sentences": len(summary) - summary_dropped,
            "summary_dropped": summary_dropped,
            "empty_sections_removed": len(all_sections) - len(sections),
            "intro_sentences_attached": attached,
            "plan_leaks_removed": leaks,
            "calls": len(responses),
        }
    )
    return RegroupOutcome(note, diag, _summarize_tokens(usages), responses)
