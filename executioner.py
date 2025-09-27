#!/usr/bin/env python3
"""
Full-production hybrid VideoSegmenter:
- SRT parsing
- Sentence chunking & boundary detection via embeddings
- Candidate clip generation with timestamps/duration
- BERTopic run once (per transcript) -> topic ids for each candidate
- Emotion classification (batched) via HF pipeline
- Keyword extraction (KeyBERT) and NER (spaCy) in main
- Multiprocess scoring (lightweight) using precomputed features
- Redundancy filtering via embeddings
- Top-k non-overlapping selection and CSV output
"""

import re
import logging
import random
from pathlib import Path
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

import pysrt
import numpy as np
import pandas as pd
from tqdm import tqdm

import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Config
# -------------------------
class Cfg:
    # clip durations
    MAX_DURATION_SECONDS = 45
    MIN_DURATION_SECONDS = 10

    # segmentation
    WINDOW_SIZE = 2
    DYNAMIC_STD_FACTOR = 0.5
    RELATIVE_DROP = 0.15
    DYNAMIC_THRESHOLD_PERCENTILE = 90.0

    # redundancy / diversity
    REDUNDANCY_SIMILARITY_THRESHOLD = 0.90

    # scoring weights (tune as needed)
    WEIGHTS = {
        'topic_score': 0.8,
        'contextual_topic': 0.7,
        'hook': 2.0,
        'named_entity': 0.3,
        'brevity': 1.0,
        'virality_prior': 1.0,
        # juicy features
        'emotion': 0.35,
        'surprise': 0.15,
        'cliffhanger': 0.15,
        'narrative_arc': 0.15,
        'speech_delivery': 0.1
    }

    VIRALITY_PRIORS = {'secret': 1.3, 'biggest': 1.2, 'hidden': 1.25, 'money': 1.5, 'danger': 1.4}
    BAD_WORDS = {'sponsor', 'advertisement', 'subscribe'}

    HOOK_PATTERNS = [
        re.compile(r"^(what|why|how|who|where|when)\b", re.I),
        re.compile(r"\b(no one|nobody|everyone|never|always)\b", re.I),
        re.compile(r"\b(the (craziest|weirdest|biggest|most))\b", re.I),
        re.compile(r"\b(you won'?t believe|mind[- ]?blow)\b", re.I),
        re.compile(r"\b(\d+[\,\d]*\s*(million|billion|trillion|%)?)\b", re.I),
    ]

    SURPRISE_PATTERNS = re.compile(r"\b(actually|turns out|but|however|in fact)\b", re.I)
    NARRATIVE_PATTERNS = re.compile(r"(this means|the truth is|nobody tells you|what nobody knows|here's why)", re.I)

# -------------------------
# Utilities
# -------------------------
def safe_time_to_dt(t: time) -> datetime:
    return datetime.combine(datetime.min, t)

def duration_seconds(start: time, end: time) -> float:
    return (safe_time_to_dt(end) - safe_time_to_dt(start)).total_seconds()

def format_time(t: time) -> str:
    return t.strftime('%H:%M:%S.%f')[:-3]

# -------------------------
# Model loading (main process)
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("Loading models (this may take a bit)...")
nlp = spacy.load("en_core_web_sm")
EMBED_MODEL_NAME = "all-mpnet-base-v2"  # quality for boundaries + redundancy
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
kw_model = KeyBERT(model=embed_model)

# BERTopic will be instantiated later (after we have candidates)
# Emotion pipeline (we will use batched calls)
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)
import assemblyai as aai

def get_pauses_from_assemblyai(audio_path: str, api_key= "4d99013339524b77a3e24af62f70009b" ) -> List[Dict]:
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(audio_start_from=0, audio_end_at=None, speaker_labels=False)
    
    transcript = transcriber.transcribe(audio_path, config=config)

    pauses = []
    if transcript.status == "completed":
        for i, word in enumerate(transcript.words[:-1]):
            gap = transcript.words[i+1].start - word.end
            if gap >= 500:  # 500ms silence
                pauses.append({"start": word.end, "end": transcript.words[i+1].start})
    else:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return pauses

# -------------------------
# SRT parsing -> sentence entries with time mapping
# -------------------------
def load_subtitles(srt_path: str) -> List[Dict]:
    subs = pysrt.open(srt_path, encoding='utf-8')
    # Build mapping char -> time (approximate)
    full_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
    doc = nlp(full_text)

    char_to_time = {}
    cursor = 0
    for sub in subs:
        text = sub.text_without_tags.replace("\n", " ") + " "
        for i in range(len(text)):
            char_to_time[cursor + i] = (sub.start.to_time(), sub.end.to_time())
        cursor += len(text)

    sentence_entries = []
    for sent in doc.sents:
        start_char = sent.start_char
        end_char = sent.end_char - 1
        if start_char in char_to_time and end_char in char_to_time:
            s_time = char_to_time[start_char][0]
            e_time = char_to_time[end_char][1]
            sentence_entries.append({
                "text": sent.text.strip(),
                "start": s_time,
                "end": e_time
            })
    logging.info(f"Loaded {len(sentence_entries)} sentence entries from subtitles.")
    return sentence_entries

# -------------------------
# Similarity-based boundary detection
# -------------------------
def compute_similarity(sentences: List[str], cfg: Cfg) -> np.ndarray:
    if len(sentences) <= cfg.WINDOW_SIZE:
        return np.array([])
    window_texts = [" ".join(sentences[i:i + cfg.WINDOW_SIZE]) for i in range(len(sentences) - cfg.WINDOW_SIZE + 1)]
    embeddings = embed_model.encode(window_texts, show_progress_bar=False, convert_to_numpy=True)
    if embeddings.shape[0] < 2:
        return np.array([])
    sims = np.diag(cosine_similarity(embeddings[:-1], embeddings[1:]))
    return sims

def detect_boundaries(sentence_entries: List[Dict], cfg: Cfg) -> List[int]:
    sentences = [e['text'] for e in sentence_entries]
    similarities = compute_similarity(sentences, cfg)
    if similarities.size == 0:
        return [0, len(sentences)]

    sim_mean = float(np.mean(similarities))
    sim_std = float(np.std(similarities))
    threshold = sim_mean - cfg.DYNAMIC_STD_FACTOR * sim_std
    boundaries = {0}

    for i in range(len(similarities)):
        is_break = (similarities[i] < threshold) or (
            similarities[i] / max(similarities[i-1], 1e-6) < (1 - cfg.RELATIVE_DROP)
        )
        if is_break:
            # lookahead average check
            look_ahead = [similarities[j] for j in range(i+1, min(i+3, len(similarities)))]
            if look_ahead and float(np.mean(look_ahead)) < threshold:
                boundaries.add(i+1)
                sentence_entries[i+1]["is_boundary"] = True
            else:
                boundaries.add(i+1)
                sentence_entries[i+1]["is_boundary"] = True

    boundaries.add(len(sentences))
    b = sorted(list(boundaries))
    logging.info(f"Detected {len(b)-1} coarse segments from {len(sentences)} sentences.")
    return b


# -------------------------
# Generate candidate clips (with timestamps)
# -------------------------
def generate_candidates(
    sentence_entries: List[Dict],
    boundaries: List[int],
    pauses: Optional[List[Dict]],
    cfg: Cfg
) -> List[Dict]:
    candidates = []
    start_idx = 0
    start_t = sentence_entries[0]['start']
    last_pause = None
    flagged = False

    for i, sent in enumerate(sentence_entries):
        curr_end = sent['end']
        dur = duration_seconds(start_t, curr_end)

        # check pause alignment (within 0.5s tolerance) if pauses exist
        is_pause = False
        if pauses:
            is_pause = any(abs(curr_end.second + curr_end.microsecond/1e6 - (p['start']/1000.0)) < 0.5 for p in pauses)
        if is_pause:
            last_pause = curr_end

        # semantic boundary flagged
        if i in boundaries:
            flagged = True

        # cut if semantic flagged + pause found + dur >= MIN
        if flagged and is_pause and dur >= cfg.MIN_DURATION_SECONDS:
            text = " ".join(e['text'] for e in sentence_entries[start_idx:i+1]).strip()
            candidates.append({
                "Preview": text,
                "Start": start_t,
                "End": curr_end,
                "Duration": dur
            })
            start_idx = i+1
            start_t = curr_end
            flagged = False
            last_pause = None

        # cut if hitting max duration
        elif dur >= cfg.MAX_DURATION_SECONDS:
            cut_t = last_pause if last_pause else curr_end
            cut_idx = i if last_pause else i
            text = " ".join(e['text'] for e in sentence_entries[start_idx:cut_idx+1]).strip()
            dur_cut = duration_seconds(start_t, cut_t)
            if dur_cut >= cfg.MIN_DURATION_SECONDS:  # ensure valid clip
                candidates.append({
                    "Preview": text,
                    "Start": start_t,
                    "End": cut_t,
                    "Duration": dur_cut
                })
            start_idx = cut_idx+1
            start_t = cut_t
            flagged = False
            last_pause = None

    # flush last segment
    if start_idx < len(sentence_entries):
        end_t = sentence_entries[-1]['end']
        dur = duration_seconds(start_t, end_t)
        text = " ".join(e['text'] for e in sentence_entries[start_idx:]).strip()
        if dur >= cfg.MIN_DURATION_SECONDS:
            candidates.append({
                "Preview": text,
                "Start": start_t,
                "End": end_t,
                "Duration": dur
            })
        elif candidates:  # too short: merge into previous
            candidates[-1]["Preview"] += " " + text
            candidates[-1]["End"] = end_t
            candidates[-1]["Duration"] = duration_seconds(candidates[-1]["Start"], end_t)
        else:  # single short clip, keep anyway
            candidates.append({
                "Preview": text,
                "Start": start_t,
                "End": end_t,
                "Duration": dur
            })

    logging.info(f"Generated {len(candidates)} candidate clips (min duration enforced).")
    return candidates

# -------------------------
# Lightweight scoring worker (safe for multiprocessing)
# -------------------------
def score_worker(arg_tuple):
    """
    arg_tuple: (index, preview, duration, topic_score, keywords, topic_id, emotion_score, entity_count, global_topics)
    This function does not load heavy models; it's safe to run in worker processes.
    """
    (idx, preview, duration, topic_score, keywords, topic_id, emotion_score, entity_count, global_topics) = arg_tuple
    cfg = Cfg()
    text = preview
    words = set(text.lower().split())

    # topic_score from KeyBERT (precomputed)
    topic_score_val = float(topic_score)

    # contextual topic score: count overlapping keywords with global topics
    contextual_topic_score = sum(1 for kw in keywords if kw in global_topics) if keywords else 0

    # hook score
    hook_score = sum(1 for pat in cfg.HOOK_PATTERNS if pat.search(text))

    # named entity score
    entity_score = int(entity_count)

    # brevity
    brevity_bonus = 1.0 if cfg.MIN_DURATION_SECONDS <= duration <= 15 else 0.5

    # virality priors
    virality_boost = sum(cfg.VIRALITY_PRIORS.get(w, 0) for w in words)

    # juicy features (precomputed emotion used)
    emotion = float(emotion_score)
    surprise = int(bool(cfg.SURPRISE_PATTERNS.search(text)))
    cliffhanger = int(text.strip().endswith(("?", "...")))
    narrative = int(bool(cfg.NARRATIVE_PATTERNS.search(text)))
    # placeholder speech delivery (random-ish)
    speech_delivery = random.uniform(0.3, 0.8)

    w = cfg.WEIGHTS
    final_score = (
        w['topic_score'] * topic_score_val +
        w['contextual_topic'] * contextual_topic_score +
        w['hook'] * hook_score +
        w['named_entity'] * entity_score +
        w['brevity'] * brevity_bonus +
        w['virality_prior'] * virality_boost +
        w['emotion'] * emotion +
        w['surprise'] * surprise +
        w['cliffhanger'] * cliffhanger +
        w['narrative_arc'] * narrative +
        w['speech_delivery'] * speech_delivery
    )

    # penalize if bad words present
    if any(b in words for b in Cfg.BAD_WORDS):
        final_score *= 0.2

    return {
        'Idx': idx,
        'Score': float(final_score),
        'Start': None,    # fill later
        'End': None,
        'Duration': float(duration),
        'Preview': preview,
        'Keywords': keywords,
        'TopicID': int(topic_id),
        'EmotionScore': float(emotion),
        'EntityCount': int(entity_count),
        'HookScore': int(hook_score),
        'Surprise': int(surprise),
        'Cliffhanger': int(cliffhanger),
        'Narrative': int(narrative)
    }

# -------------------------
# Redundancy filter (embedding-based)
# -------------------------
def filter_redundant(clips: List[Dict], embeddings: np.ndarray, threshold: float) -> List[Dict]:
    if not clips:
        return []
    # order by score desc
    clips_sorted = sorted(clips, key=lambda x: x['Score'], reverse=True)
    final = []
    emb_index_lookup = {c['Idx']: i for i, c in enumerate(clips_sorted)}  # mapping for embeddings order
    # compute embeddings for the sorted previews in the same order as embeddings input
    # we already have embeddings aligned to candidates order; need to map
    # build array where row i corresponds to sorted clip i in original embedding index space
    orig_idx_to_embedding_index = {i: i for i in range(embeddings.shape[0])}
    used = []
    for clip in clips_sorted:
        i_orig = clip['Idx']
        emb_vec = embeddings[i_orig]
        is_redundant = False
        for kept in final:
            kept_vec = embeddings[kept['Idx']]
            sim = float(cosine_similarity([emb_vec], [kept_vec])[0][0])
            if sim > threshold:
                is_redundant = True
                break
        if not is_redundant:
            final.append(clip)
    return final

# -------------------------
# Non-overlapping selection
# -------------------------
def select_non_overlapping(clips: List[Dict], k: int) -> List[Dict]:
    selected = []
    for clip in sorted(clips, key=lambda x: x['Score'], reverse=True):
        # clip's Start/End should be datetime.time objects set previously
        if any((clip['Start'] < c['End']) and (c['Start'] < clip['End']) for c in selected):
            continue
        selected.append(clip)
        if len(selected) >= k:
            break
    return selected
# -------------------------
# Precompute per-candidate features
# -------------------------
def precompute_features(candidates: List[Dict], cfg: Cfg) -> Tuple[List[Dict], np.ndarray, List[str]]:
    texts = [c['Preview'] for c in candidates]

    # --- Embeddings (for redundancy + BERTopic + scoring) ---
    embeddings = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # --- BERTopic (topic clustering) ---
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(texts)
    for i, c in enumerate(candidates):
        c["TopicID"] = int(topics[i])
        c["TopicScore"] = float(np.max(probs[i])) if probs[i] is not None else 0.0

    # --- Emotion classification (batched) ---
    emo_results = emotion_pipeline(texts, batch_size=16, truncation=True)
    for i, emo_scores in enumerate(emo_results):
        # take max probability as "emotion intensity"
        c = candidates[i]
        c["EmotionScore"] = max(s["score"] for s in emo_scores)

    # --- Keywords (KeyBERT) ---
    for i, text in enumerate(texts):
        kw = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
        candidates[i]["Keywords"] = [w for w, s in kw]

    # --- Named Entities (spaCy) ---
    for i, text in enumerate(texts):
        doc = nlp(text)
        candidates[i]["EntityCount"] = sum(1 for ent in doc.ents if ent.label_ not in ("CARDINAL", "ORDINAL"))

    # --- Global topic keywords (for contextual overlap scoring) ---
    global_keywords = kw_model.extract_keywords(" ".join(texts), keyphrase_ngram_range=(1,2), stop_words='english', top_n=25)
    global_topics = [w for w, s in global_keywords]

    return candidates, embeddings, global_topics

# -------------------------
# Main pipeline
# -------------------------
def process_file(srt_path: str,audio_path: Optional[str] = None, top_k: int = 10) -> Optional[pd.DataFrame]:
    cfg = Cfg()
    logging.info(f"Processing {srt_path}")

    sentence_entries = load_subtitles(srt_path)
    if not sentence_entries:
        logging.warning("No sentence entries extracted.")
        return None

    boundaries = detect_boundaries(sentence_entries, cfg)

    # --- Add pause detection here ---
    pauses = None
    if audio_path:
        pauses = get_pauses_from_assemblyai(audio_path)
        logging.info(f"Detected {len(pauses)} pauses from audio.")
    # -------------------------------

    candidates = generate_candidates(sentence_entries, boundaries, pauses, cfg)
    if not candidates:
        logging.warning("No candidate clips produced.")
        return None


    # precompute heavy features
    candidates, embeddings, global_topics = precompute_features(candidates, cfg)

    # Prepare worker args (primitives only)
    worker_args = []
    for i, c in enumerate(candidates):
        worker_args.append((
            i,
            c['Preview'],
            c['Duration'],
            c.get('TopicScore', 0.0),
            c.get('Keywords', []),
            c.get('TopicID', -1),
            c.get('EmotionScore', 0.0),
            c.get('EntityCount', 0),
            global_topics
        ))

    # Multiprocess scoring (workers are lightweight)
    logging.info("Scoring candidates with multiprocessing...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(score_worker, worker_args), total=len(worker_args), desc="Scoring"))

    # attach timestamps back (Start/End)
    for r in results:
        idx = r['Idx']
        r['Start'] = candidates[idx]['Start']
        r['End'] = candidates[idx]['End']

    # Convert to list for redundancy filtering
    # redundancy filter uses original embeddings (index aligned to candidate index)
    non_redundant = filter_redundant(results, embeddings, cfg.REDUNDANCY_SIMILARITY_THRESHOLD)

    # Convert Start/End to comparable datetimes for overlap checking
    for c in non_redundant:
        # already datetime.time objects from pysrt -> keep them
        pass

    top_clips = select_non_overlapping(non_redundant, top_k)

    # Format DataFrame
    df_rows = []
    for rank, clip in enumerate(top_clips, start=1):
        df_rows.append({
            'Rank': rank+1,
            'Score': f"{clip['Score']:.4f}",
            'Start': format_time(clip['Start']),
            'End': format_time(clip['End']),
            'Duration': f"{clip['Duration']:.2f}s",
            'Preview': clip['Preview'],
            'Keywords': ", ".join(clip.get('Keywords', [])),
            'TopicID': clip.get('TopicID', -1),
            'EmotionScore': f"{clip.get('EmotionScore',0.0):.3f}",
            'EntityCount': clip.get('EntityCount', 0)
        })

    df = pd.DataFrame(df_rows)
    out_csv = Path("viral_clips.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved top {len(df)} clips to {out_csv.resolve()}")
    return df

