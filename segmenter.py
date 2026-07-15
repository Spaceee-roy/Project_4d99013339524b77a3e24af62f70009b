import json
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import ollama


TIMESTAMPS_PATH = Path(".aai_cache/text.json")
OUTPUT_PATH = Path("output.txt")
MODEL = "qwen3.5:9b"
MODEL_SEED = 42
CLIP_COUNT = 10
MODEL_CANDIDATE_COUNT = 24
TARGET_DURATION_S = 30
MIN_DURATION_S = 27
MAX_DURATION_S = 33
ANCHORS_PER_CLIP = 3


client = ollama.Client(host="http://127.0.0.1:11434")


def load_words(path=TIMESTAMPS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words")
    if not words:
        raise ValueError(f"No timestamped words found in {path}")
    return words


def make_units(words):
    """Create semantic sentence units, repairing obvious false ASR stops."""
    raw_units = []
    first = 0
    for index, word in enumerate(words):
        token = word["text"].rstrip('"\'')
        if token.endswith((".", "?", "!")):
            raw_units.append(_make_unit(words, first, index))
            first = index + 1

    if first < len(words):
        raw_units.append(_make_unit(words, first, len(words) - 1))

    units = []
    index = 0
    while index < len(raw_units):
        unit = raw_units[index]
        while index + 1 < len(raw_units) and _looks_incomplete(unit["text"]):
            index += 1
            unit = _make_unit(
                words, unit["start_word"], raw_units[index]["end_word"]
            )
        units.append(unit)
        index += 1
    return units


def _looks_incomplete(text):
    """Detect punctuation inserted after a word that cannot naturally finish a thought."""
    tokens = _words(text)
    if not tokens:
        return True
    return tokens[-1] in {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "because",
        "but",
        "cause",
        "for",
        "from",
        "her",
        "his",
        "in",
        "is",
        "its",
        "just",
        "my",
        "of",
        "on",
        "or",
        "our",
        "really",
        "the",
        "their",
        "to",
        "was",
        "were",
        "with",
        "your",
    }


def _make_unit(words, first, last):
    return {
        "start_word": first,
        "end_word": last,
        "start_s": words[first]["start_s"],
        "end_s": words[last]["end_s"],
        "text": " ".join(word["text"] for word in words[first : last + 1]),
    }


def format_timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def transcript_for_model(units):
    return "\n".join(
        f"U{unit_id:04d}|{unit['start_s']:.2f}-{unit['end_s']:.2f}|{unit['text']}"
        for unit_id, unit in enumerate(units, start=1)
    )


def response_schema():
    return {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "minItems": MODEL_CANDIDATE_COUNT,
                "maxItems": MODEL_CANDIDATE_COUNT,
                "items": {
                    "type": "object",
                    "properties": {
                        "hook_unit": {"type": "integer"},
                        "development_unit": {"type": "integer"},
                        "payoff_unit": {"type": "integer"},
                    },
                    "required": [
                        "hook_unit",
                        "development_unit",
                        "payoff_unit",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clips"],
        "additionalProperties": False,
    }


def find_viral_segments(units):
    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a ruthless professional short-form video editor. "
                    "Choose exact transcript units and never rewrite dialogue or invent IDs."
                ),
            },
            {
                "role": "user",
                "content": f"""
Rank the {MODEL_CANDIDATE_COUNT} strongest DISTINCT clip candidates in this transcript. Python will time them and keep the best {CLIP_COUNT}.

For each candidate choose exactly three chronological sentence anchors:
- hook_unit: the opening idea or line. It must work cold, immediately create curiosity, and avoid acknowledgements or unresolved references.
- development_unit: the essential context, escalation, evidence, or mechanism needed to understand the hook.
- payoff_unit: the strongest answer, reveal, consequence, punchline, or memorable conclusion. It must feel finished.

Editorial rules:
- The three anchors must form one self-contained story when stitched with ellipses: setup, development, payoff.
- Preserve cause and effect. Never skip a sentence that makes the next excerpt confusing.
- Keep all three anchors within 90 source seconds so they come from the same local discussion.
- Remove filler and repetition, not essential logic.
- Do not submit alternate versions of the same concept. Every candidate needs a different central promise.
- Spread candidates across the full transcript instead of clustering near the beginning.
- Favor surprising facts, emotional stakes, humor, useful explanations, contrarian insights, and quotable conclusions.
- Unit IDs must exist and hook_unit < development_unit < payoff_unit.
- Return candidates best-first.

TIMESTAMPED SENTENCES:
{transcript_for_model(units)}
""",
            },
        ],
        think=False,
        format=response_schema(),
        options={
            "num_ctx": 16384,
            "num_predict": 3072,
            "temperature": 0.15,
            "seed": MODEL_SEED,
        },
    )
    return json.loads(response["message"]["content"])


def _parse_unit_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(?:U)?(\d{1,5})\s*", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_anchors(clip, unit_count):
    preferred = [
        _parse_unit_id(clip.get("hook_unit")),
        _parse_unit_id(clip.get("development_unit")),
        _parse_unit_id(clip.get("payoff_unit")),
    ]
    if all(value is not None for value in preferred):
        anchors = preferred
    else:
        anchors = []

        def collect(value):
            parsed = _parse_unit_id(value)
            if parsed is not None:
                anchors.append(parsed)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key != "title":
                        collect(item)

        collect({key: value for key, value in clip.items() if key != "title"})

    anchors = sorted(set(value for value in anchors if 1 <= value <= unit_count))
    if len(anchors) >= ANCHORS_PER_CLIP:
        return [anchors[0], anchors[len(anchors) // 2], anchors[-1]]
    if len(anchors) == 2 and anchors[1] - anchors[0] >= 2:
        return [anchors[0], (anchors[0] + anchors[1]) // 2, anchors[1]]
    if len(anchors) == 1 and unit_count >= 3:
        spacing = min(6, (unit_count - 1) // 2)
        center = min(max(anchors[0], spacing + 1), unit_count - spacing)
        return [center - spacing, center, center + spacing]
    return None


def _words(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _weak_boundary_penalty(text, is_start=False, is_payoff=False):
    tokens = _words(text)
    if not tokens:
        return 10

    penalty = 0
    generic_fillers = {
        "yeah",
        "right",
        "okay",
        "ok",
        "correct",
        "well",
        "um",
        "uh",
        "mm",
        "hmm",
        "really",
        "no",
        "yes",
    }
    if len(tokens) <= 3 and tokens[0] in generic_fillers:
        penalty += 4
    if is_start and tokens[0] in generic_fillers:
        penalty += 5
    if is_start and text[:1].islower():
        penalty += 5
    if is_start and len(tokens) <= 2:
        penalty += 2
    if is_start and tokens[0] in {
        "and",
        "but",
        "because",
        "then",
        "it",
        "it's",
        "that",
        "that's",
        "they",
        "this",
        "those",
        "these",
        "oh",
        "in",
        "on",
        "at",
        "from",
        "with",
        "of",
    }:
        penalty += 4
    if is_start and len(tokens) >= 3 and tokens[:3] in (
        ["ever", "since", "it"],
        ["ever", "since", "they"],
    ):
        penalty += 6
    if is_payoff and text.rstrip().endswith("?"):
        penalty += 3
    if is_payoff and len(tokens) <= 3 and tokens[0] in generic_fillers:
        penalty += 6
    return penalty


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _bad_opening(text):
    sentences = _sentences(text)
    if not sentences:
        return True
    first = sentences[0]
    tokens = _words(first)
    if not tokens:
        return True

    banned_starts = {
        "and",
        "at",
        "because",
        "but",
        "cause",
        "correct",
        "from",
        "gotcha",
        "however",
        "in",
        "into",
        "mm",
        "no",
        "of",
        "oh",
        "okay",
        "on",
        "or",
        "really",
        "right",
        "so",
        "then",
        "well",
        "with",
        "yeah",
        "yes",
    }
    if tokens[0] in banned_starts:
        return True
    if tokens[:3] in (["ever", "since", "it"], ["ever", "since", "they"]):
        return True
    if " ".join(tokens[:3]) in {"i don't know", "i dont know"}:
        return True
    if tokens[:3] == ["i", "got", "you"]:
        return True
    if tokens[:3] == ["i", "like", "that"]:
        return True
    if tokens[:2] == ["one", "is"]:
        return True
    if tokens[0] == "not" and len(tokens) <= 5:
        return True
    if tokens[0] in {"he", "i", "it", "she", "they", "we"} and "that" in tokens and len(tokens) <= 10:
        return True

    finite_markers = {
        "am",
        "are",
        "can",
        "could",
        "did",
        "does",
        "has",
        "have",
        "is",
        "might",
        "must",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
    if len(tokens) <= 4 and tokens[0] in {"a", "an", "the"} and not finite_markers.intersection(tokens):
        return True
    return False


def _bad_payoff(text):
    sentences = _sentences(text)
    if not sentences:
        return True
    last = sentences[-1]
    tokens = _words(last)
    if not tokens or last.endswith("?") or _looks_incomplete(last):
        return True
    if len(tokens) <= 3 and tokens[0] in {
        "correct",
        "damn",
        "gotcha",
        "hmm",
        "mm",
        "no",
        "okay",
        "really",
        "right",
        "wow",
        "yeah",
        "yes",
    }:
        return True

    lowered = text.lower()
    outro_markers = {
        "always a pleasure",
        "good stuff",
        "good to have you",
        "see you next",
        "subscribe",
        "thanks for listening",
        "thanks for watching",
        "there's a lot in that",
        "until next time",
        "so there it is",
    }
    return any(marker in lowered for marker in outro_markers)


def _anchor_window_candidates(anchor_id, role, units):
    options = []
    search_radius = 18
    for start_id in range(max(1, anchor_id - search_radius), anchor_id + 1):
        for end_id in range(anchor_id, min(len(units), anchor_id + search_radius) + 1):
            duration = units[end_id - 1]["end_s"] - units[start_id - 1]["start_s"]
            if duration < 4.0 or duration > 24.0:
                continue

            before_anchor = units[anchor_id - 1]["start_s"] - units[start_id - 1]["start_s"]
            after_anchor = units[end_id - 1]["end_s"] - units[anchor_id - 1]["end_s"]
            role_penalty = 0
            if role == "hook":
                role_penalty = before_anchor * 0.8
            elif role == "payoff":
                role_penalty = after_anchor * 0.8
            else:
                role_penalty = abs(before_anchor - after_anchor) * 0.15

            boundary_penalty = _weak_boundary_penalty(
                units[start_id - 1]["text"], is_start=role == "hook"
            ) + _weak_boundary_penalty(
                units[end_id - 1]["text"], is_payoff=role == "payoff"
            )
            local_score = abs(duration - TARGET_DURATION_S / 3) + role_penalty + boundary_penalty
            options.append((local_score, start_id, end_id, duration))

    if not options:
        unit = units[anchor_id - 1]
        duration = unit["end_s"] - unit["start_s"]
        return [(0, anchor_id, anchor_id, duration)]
    return sorted(options)[:32]


def _best_anchor_windows(anchors, units):
    role_options = [
        _anchor_window_candidates(anchors[0], "hook", units),
        _anchor_window_candidates(anchors[1], "development", units),
        _anchor_window_candidates(anchors[2], "payoff", units),
    ]
    best = None
    for windows in product(*role_options):
        # Leave at least one complete sentence out at each ellipsis.
        if windows[0][2] + 1 >= windows[1][1] or windows[1][2] + 1 >= windows[2][1]:
            continue
        hook_text = " ".join(
            unit["text"] for unit in units[windows[0][1] - 1 : windows[0][2]]
        )
        payoff_text = " ".join(
            unit["text"] for unit in units[windows[2][1] - 1 : windows[2][2]]
        )
        if _bad_opening(hook_text) or _bad_payoff(payoff_text):
            continue
        total = sum(window[3] for window in windows)
        duration_penalty = abs(total - TARGET_DURATION_S) * 6
        if total < MIN_DURATION_S or total > MAX_DURATION_S:
            duration_penalty += 30
        score = duration_penalty + sum(window[0] for window in windows)
        if best is None or score < best[0]:
            best = (score, windows)

    if best is None:
        return None
    return [(window[1], window[2]) for window in best[1]]


def resolve_candidate(clip, rank, units, words):
    anchors = extract_anchors(clip, len(units))
    if anchors is None:
        return None
    anchor_span = units[anchors[-1] - 1]["end_s"] - units[anchors[0] - 1]["start_s"]
    if anchor_span > 90:
        anchors = _nearby_anchors(anchors[1], len(units))
    windows = _best_anchor_windows(anchors, units)
    if windows is None:
        return None

    ranges = []
    for start_id, end_id in windows:
        start_unit = units[start_id - 1]
        end_unit = units[end_id - 1]
        ranges.append(
            {
                "start_unit": start_id,
                "end_unit": end_id,
                "start_s": start_unit["start_s"],
                "end_s": end_unit["end_s"],
                "text": " ".join(
                    word["text"]
                    for word in words[start_unit["start_word"] : end_unit["end_word"] + 1]
                ),
            }
        )

    combined = " ".join(part["text"] for part in ranges)
    if not _parts_are_coherent(ranges):
        anchors = _nearby_anchors(anchors[1], len(units))
        windows = _best_anchor_windows(anchors, units)
        if windows is None:
            return None
        ranges = _resolve_windows(windows, units, words)
        combined = " ".join(part["text"] for part in ranges)
        if not _parts_are_coherent(ranges):
            return None

    duration = sum(part["end_s"] - part["start_s"] for part in ranges)
    if (
        duration < MIN_DURATION_S
        or _bad_opening(ranges[0]["text"])
        or _bad_payoff(ranges[-1]["text"])
    ):
        return None

    return {
        "rank": rank,
        "anchors": anchors,
        "center": anchors[1],
        "ranges": ranges,
        "duration": duration,
        "tokens": _meaningful_tokens(combined),
    }


def _nearby_anchors(center, unit_count):
    spacing = min(6, (unit_count - 1) // 2)
    center = min(max(center, spacing + 1), unit_count - spacing)
    return [center - spacing, center, center + spacing]


def _resolve_windows(windows, units, words):
    ranges = []
    for start_id, end_id in windows:
        start_unit = units[start_id - 1]
        end_unit = units[end_id - 1]
        ranges.append(
            {
                "start_unit": start_id,
                "end_unit": end_id,
                "start_s": start_unit["start_s"],
                "end_s": end_unit["end_s"],
                "text": " ".join(
                    word["text"]
                    for word in words[start_unit["start_word"] : end_unit["end_word"] + 1]
                ),
            }
        )
    return ranges


def _meaningful_tokens(text):
    common = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "what",
        "when",
        "where",
        "there",
        "they",
        "your",
        "about",
        "just",
        "like",
        "move",
        "moved",
        "moves",
        "moving",
        "because",
        "into",
        "then",
        "right",
        "okay",
        "yeah",
    }
    return {_stem(token) for token in _words(text) if len(token) >= 4 and token not in common}


def _stem(token):
    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _parts_are_coherent(ranges):
    """Require all three excerpts to connect by topic words or nearby context."""
    token_sets = [_meaningful_tokens(part["text"]) for part in ranges]
    connected = {0}
    changed = True
    while changed:
        changed = False
        for left in tuple(connected):
            for right in range(len(ranges)):
                if right in connected:
                    continue
                source_gap = max(
                    0.0,
                    ranges[right]["start_s"] - ranges[left]["end_s"],
                    ranges[left]["start_s"] - ranges[right]["end_s"],
                )
                if token_sets[left] & token_sets[right] or source_gap <= 2:
                    connected.add(right)
                    changed = True
    return len(connected) == len(ranges)


def _source_overlap(left, right):
    overlap = 0.0
    for first in left["ranges"]:
        for second in right["ranges"]:
            overlap += max(0.0, min(first["end_s"], second["end_s"]) - max(first["start_s"], second["start_s"]))
    return overlap / min(left["duration"], right["duration"])


def _text_similarity(left, right):
    union = left["tokens"] | right["tokens"]
    return len(left["tokens"] & right["tokens"]) / len(union) if union else 0


def _is_duplicate(candidate, selected):
    for existing in selected:
        candidate_topic = candidate.get("topic_tokens", set())
        existing_topic = existing.get("topic_tokens", set())
        topic_union = candidate_topic | existing_topic
        topic_similarity = (
            len(candidate_topic & existing_topic) / len(topic_union)
            if topic_union
            else 0
        )
        if (
            _source_overlap(candidate, existing) > 0.20
            or _text_similarity(candidate, existing) > 0.45
            or topic_similarity > 0.50
        ):
            return True
    return False


def _is_relaxed_duplicate(candidate, selected):
    """Last-slot check: allow related subjects, never the same clip or promise."""
    for existing in selected:
        candidate_topic = candidate.get("topic_tokens", set())
        existing_topic = existing.get("topic_tokens", set())
        topic_union = candidate_topic | existing_topic
        topic_similarity = (
            len(candidate_topic & existing_topic) / len(topic_union)
            if topic_union
            else 0
        )
        if (
            _source_overlap(candidate, existing) > 0.20
            or _text_similarity(candidate, existing) > 0.65
            or topic_similarity > 0.80
        ):
            return True
    return False


def _fallback_candidates(units, words, existing):
    candidates = []
    occupied = [candidate["center"] for candidate in existing]
    for index in range(CLIP_COUNT * 4):
        center = round((index + 0.5) * len(units) / (CLIP_COUNT * 4))
        center = min(max(center, 7), len(units) - 6)
        if any(abs(center - used) < 10 for used in occupied):
            continue
        anchors = [center - 6, center, center + 6]
        clip = {
            "hook_unit": anchors[0],
            "development_unit": anchors[1],
            "payoff_unit": anchors[2],
        }
        candidate = resolve_candidate(clip, MODEL_CANDIDATE_COUNT + index, units, words)
        if candidate and not _is_duplicate(candidate, existing + candidates):
            candidates.append(candidate)
            occupied.append(center)
    return candidates


def review_schema(candidate_count):
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "topic_key": {"type": "string"},
                        "hook": {"type": "integer", "minimum": 1, "maximum": 10},
                        "continuity": {"type": "integer", "minimum": 1, "maximum": 10},
                        "payoff": {"type": "integer", "minimum": 1, "maximum": 10},
                        "standalone": {"type": "integer", "minimum": 1, "maximum": 10},
                        "novelty": {"type": "integer", "minimum": 1, "maximum": 10},
                        "shareability": {"type": "integer", "minimum": 1, "maximum": 10},
                        "reject": {"type": "boolean"},
                    },
                    "required": [
                        "candidate_id",
                        "topic_key",
                        "hook",
                        "continuity",
                        "payoff",
                        "standalone",
                        "novelty",
                        "shareability",
                        "reject",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def review_candidates(candidates):
    """Judge the finished edits, where missing context and weak payoffs are visible."""
    if len(candidates) <= CLIP_COUNT:
        return candidates

    previews = []
    for candidate_id, candidate in enumerate(candidates, start=1):
        parts = "\n".join(
            f"  PART {part_number}: {part['text']}"
            for part_number, part in enumerate(candidate["ranges"], start=1)
        )
        previews.append(
            f"C{candidate_id:02d} | {candidate['duration']:.2f}s | "
            f"source center {candidate['center']}\n{parts}"
        )

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the final editorial gate for short-form clips. Judge only "
                    "the assembled dialogue shown to you and return candidate IDs."
                ),
            },
            {
                "role": "user",
                "content": f"""
Score every assembled candidate below. Be strict: these scores directly determine publication.

Reject or rank down any clip that:
- opens with a reaction, pronoun, conjunction, or reference that needs missing context;
- changes subject between parts or skips required cause-and-effect logic;
- ends on setup, a transition, an unanswered question, filler, or an unfinished thought;
- repeats the central promise or source material of a stronger candidate;
- is under {MIN_DURATION_S} seconds.

High-scoring clips must:
- work cold and immediately create curiosity;
- form a complete setup, development, and payoff;
- preserve the claim, mechanism, and consequence when the subject is explanatory;
- make PART 1 lead naturally to PART 2 and PART 2 lead naturally to PART 3;
- use ellipses only to remove filler while preserving meaning;
- end on a reveal, answer, consequence, punchline, or quotable conclusion;
- have a distinct central promise rather than another version of a nearby candidate.

Set reject=true for mixed subjects, missing bridge sentences, context-dependent openings, weak/outro endings, incomplete explanations, or clips that would confuse a viewer who has not seen the source. Give each candidate a short topic_key describing its central promise; candidates with the same promise must use the same topic_key.

About duration: approximately {TARGET_DURATION_S} seconds is ideal, but a clip may run over {MAX_DURATION_S} seconds when needed to finish a complete sentence or thought. Never prefer a cut-off thought merely because it is shorter.

Return one review for every candidate. Scores are integers from 1 to 10. Do not omit candidates.

CANDIDATES:
{chr(10).join(previews)}
""",
            },
        ],
        think=False,
        format=review_schema(len(candidates)),
        options={
            "num_ctx": 16384,
            "num_predict": 4096,
            "temperature": 0.1,
            "seed": MODEL_SEED,
        },
    )

    try:
        result = json.loads(response["message"]["content"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return candidates

    seen = set()
    for review in result.get("reviews", []):
        candidate_id = _parse_unit_id(review.get("candidate_id"))
        if candidate_id is None or not 1 <= candidate_id <= len(candidates) or candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidate = candidates[candidate_id - 1]
        scores = {
            key: max(1, min(10, int(review.get(key, 1))))
            for key in ("hook", "continuity", "payoff", "standalone", "novelty", "shareability")
        }
        essentials = (scores["hook"], scores["continuity"], scores["payoff"], scores["standalone"])
        candidate["review_pass"] = not bool(review.get("reject", True)) and min(essentials) >= 7
        candidate["review_score"] = (
            scores["hook"] * 1.4
            + scores["continuity"] * 1.7
            + scores["payoff"] * 1.7
            + scores["standalone"] * 1.4
            + scores["novelty"]
            + scores["shareability"]
        )
        topic_key = str(review.get("topic_key", "")).strip()
        candidate["topic_key"] = topic_key
        candidate["topic_tokens"] = _meaningful_tokens(topic_key)

    for candidate in candidates:
        candidate.setdefault("review_pass", False)
        candidate.setdefault("review_score", 0.0)
        candidate.setdefault("topic_key", "")
        candidate.setdefault("topic_tokens", set())
    return sorted(candidates, key=lambda candidate: (candidate["review_pass"], candidate["review_score"]), reverse=True)


def select_candidates(result, units, words):
    resolved = [
        candidate
        for rank, clip in enumerate(result.get("clips", []))
        if (candidate := resolve_candidate(clip, rank, units, words)) is not None
    ]

    # Let the reviewer compare fallback alternatives against resolved candidates;
    # do not suppress them merely because an unselected model candidate is nearby.
    pool = resolved + _fallback_candidates(units, words, [])
    ranked = review_candidates(pool[:24])

    selected = []
    bucket_counts = Counter()

    def quality(candidate):
        reviewed_score = candidate.get("review_score", 0.0)
        if not reviewed_score:
            reviewed_score = max(0.0, 50.0 - candidate.get("rank", 50))
        bucket = min(3, int(candidate["center"] * 4 / max(1, len(units))))
        coverage_bonus = 3.0 if bucket_counts[bucket] == 0 else 1.0 if bucket_counts[bucket] == 1 else 0.0
        return reviewed_score + coverage_bonus

    for require_review_pass in (True, False):
        while len(selected) < CLIP_COUNT:
            eligible = [
                candidate
                for candidate in ranked
                if candidate not in selected
                and (not require_review_pass or candidate.get("review_pass", False))
                and not _is_duplicate(candidate, selected)
            ]
            if not eligible:
                break
            winner = max(eligible, key=quality)
            selected.append(winner)
            bucket = min(3, int(winner["center"] * 4 / max(1, len(units))))
            bucket_counts[bucket] += 1

    if len(selected) < CLIP_COUNT:
        for candidate in ranked:
            if candidate in selected or _is_relaxed_duplicate(candidate, selected):
                continue
            selected.append(candidate)
            if len(selected) == CLIP_COUNT:
                break

    return sorted(selected, key=lambda candidate: candidate.get("review_score", 0.0), reverse=True)


def format_results(result, units, words):
    sections = []
    for number, clip in enumerate(select_candidates(result, units, words), start=1):
        mini_clips = "\n".join(
            f"- [{format_timestamp(part['start_s'])} - {format_timestamp(part['end_s'])}] "
            f"({part['end_s'] - part['start_s']:.2f}s) {part['text']}"
            for part in clip["ranges"]
        )
        combined = " ... ".join(part["text"] for part in clip["ranges"])
        sections.append(
            f"### Clip {number}\n"
            f"**Selected duration:** {clip['duration']:.2f}s\n\n"
            f"**Mini clips:**\n{mini_clips}\n\n"
            f"**Combined clip:** {combined}"
        )
    return "\n\n".join(sections)


def main():
    words = load_words()
    units = make_units(words)
    result = find_viral_segments(units)
    output = format_results(result, units, words)
    OUTPUT_PATH.write_text(output + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({output.count('### Clip ')} clips)", file=sys.stderr)


if __name__ == "__main__":
    main()
