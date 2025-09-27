#!/usr/bin/env python3
"""
Full-production Video Segmenter (copy-paste ready)
- SRT parsing
- Sentence chunking & boundary detection via embeddings
- Candidate clip generation with timestamps/duration
- BERTopic run once (per transcript) -> topic ids for each candidate
- Emotion classification (batched) via HF pipeline
- Audio parsing (AssemblyAI) for pause detection
- Pause, Pace, Silence Ratio, and Filler Word Detection (approx)
- Keyword extraction (KeyBERT) and NER (spaCy)
- Multiprocess scoring (lightweight) using precomputed features
- Redundancy filtering via embeddings
- Top-k non-overlapping selection and CSV output

INSTRUCTIONS:
- Install required packages (sentence-transformers, bertopic, keybert, spacy, pysrt, assemblyai, transformers, sklearn, tqdm, pandas, numpy)
- Set environment variable ASSEMBLYAI_API_KEY if you want AssemblyAI pause detection.
- Run: python video_segmenter_full.py <file.srt> [file.mp3] [top_k]

This file is designed so you can drop it into your project and run. Swap HF models or tune weights as desired.
"""

import os
import re
import sys
import logging
import random
import string
from pathlib import Path
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

import pysrt
import numpy as np
import pandas as pd
from tqdm import tqdm

import assemblyai as aai
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline as hf_pipeline
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

    # base micro priors (fallback in worker)
    VIRALITY_PRIORS = {
        'secret': 1.3, 'biggest': 1.2, 'hidden': 1.25, 'money': 1.5, 'danger': 1.4
    }

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


def time_to_seconds(t: time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

# -------------------------
# Model loading (main process)
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("Loading models (this may take a bit)...")
# spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # attempt to provide clearer error
    raise RuntimeError("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")

# embedding model
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'all-mpnet-base-v2')
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# KeyBERT (uses embedding model)
kw_model = KeyBERT(model=embed_model)

# emotion pipeline
emotion_pipeline = hf_pipeline(
    "text-classification",
    model=os.getenv('EMOTION_MODEL', 'j-hartmann/emotion-english-distilroberta-base'),
    top_k=None
)

# hook classifier (placeholder — replace with your finetuned model for better results)
from transformers import pipeline as hf_pipeline

# Replace with your hook-finetuned model once ready.
hook_classifier = hf_pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=1
)

FILLERS = {"uh", "um", "erm", "uhh", "umm", "like", "you know", "ah", "eh"}

# -------------------------
# AssemblyAI pauses: word-level timestamps
# -------------------------

def get_pauses_from_assemblyai(audio_path: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    Use AssemblyAI to transcribe and return pause windows in seconds as list of dicts {"start_s":.., "end_s":..}
    If API key not provided, reads ASSEMBLYAI_API_KEY env var. Returns [] on failure.
    """
    key = api_key or os.getenv('ASSEMBLYAI_API_KEY')
    if not key:
        logging.warning('No AssemblyAI API key supplied; skipping audio pause detection.')
        return []

    aai.settings.api_key = key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(audio_start_from=0, audio_end_at=None, speaker_labels=False)

    logging.info('Submitting audio to AssemblyAI for pause detection (this may take a moment)...')
    transcript = transcriber.transcribe(audio_path, config=config)

    pauses = []
    if getattr(transcript, 'status', None) == 'completed':
        words = getattr(transcript, 'words', []) or []
        for i in range(len(words) - 1):
            w = words[i]
            w_next = words[i + 1]
            gap_ms = w_next.start - w.end
            gap_s = gap_ms / 1000.0
            if gap_s >= 0.5:  # 500ms silence
                pauses.append({
                    'start_s': w.end / 1000.0,
                    'end_s': w_next.start / 1000.0,
                    'duration_s': gap_s
                })
    else:
        logging.warning('AssemblyAI transcription failed or incomplete; using no pauses.')

    logging.info(f'Detected {len(pauses)} pauses from AssemblyAI.')
    return pauses

# -------------------------
# SRT parsing -> sentence entries with time mapping
# -------------------------

def load_subtitles(srt_path: str) -> List[Dict]:
    subs = pysrt.open(srt_path, encoding='utf-8')
    full_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
    doc = nlp(full_text)

    # map characters to subtitle time windows (approx)
    char_to_time = {}
    cursor = 0
    for sub in subs:
        text = sub.text_without_tags.replace('\n', ' ') + ' '
        for i in range(len(text)):
            char_to_time[cursor + i] = (sub.start.to_time(), sub.end.to_time())
        cursor += len(text)

    sentence_entries = []
    for sent in doc.sents:
        start_char = sent.start_char
        end_char = max(sent.end_char - 1, sent.start_char)
        if start_char in char_to_time and end_char in char_to_time:
            s_time = char_to_time[start_char][0]
            e_time = char_to_time[end_char][1]
            sentence_entries.append({
                'text': sent.text.strip(),
                'start': s_time,
                'end': e_time
            })

    logging.info(f'Loaded {len(sentence_entries)} sentence entries from subtitles.')
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
        prev_sim = similarities[i - 1] if i > 0 else similarities[i]
        is_break = (similarities[i] < threshold) or (
            similarities[i] / max(prev_sim, 1e-6) < (1 - cfg.RELATIVE_DROP)
        )
        if is_break:
            boundaries.add(i + 1)
            sentence_entries[i + 1]['is_boundary'] = True

    boundaries.add(len(sentences))
    b = sorted(list(boundaries))
    logging.info(f'Detected {len(b)-1} coarse segments from {len(sentences)} sentences.')
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
    if not sentence_entries:
        return candidates

    start_idx = 0
    start_t = sentence_entries[0]['start']
    last_pause_time = None
    flagged = False

    for i, sent in enumerate(sentence_entries):
        curr_end = sent['end']
        dur = duration_seconds(start_t, curr_end)

        # check pause alignment (within 0.5s tolerance) if pauses exist
        is_pause = False
        if pauses:
            curr_end_s = time_to_seconds(curr_end)
            # consider a pause matching if curr_end near pause boundary
            for p in pauses:
                # p has start_s in seconds
                if abs(curr_end_s - p['start_s']) < 0.5 or (p['start_s'] <= curr_end_s <= p['end_s']):
                    is_pause = True
                    last_pause_time = curr_end
                    break

        # semantic boundary flagged
        if i in boundaries:
            flagged = True

        # cut if semantic flagged + pause found + dur >= MIN
        if flagged and is_pause and dur >= cfg.MIN_DURATION_SECONDS:
            text = " ".join(e['text'] for e in sentence_entries[start_idx:i+1]).strip()
            candidates.append({
                'Preview': text,
                'Start': start_t,
                'End': curr_end,
                'Duration': dur
            })
            start_idx = i + 1
            start_t = curr_end
            flagged = False
            last_pause_time = None

        # cut if hitting max duration
        elif dur >= cfg.MAX_DURATION_SECONDS:
            cut_t = last_pause_time if last_pause_time else curr_end
            cut_idx = i if last_pause_time else i
            text = " ".join(e['text'] for e in sentence_entries[start_idx:cut_idx+1]).strip()
            dur_cut = duration_seconds(start_t, cut_t)
            if dur_cut >= cfg.MIN_DURATION_SECONDS:
                candidates.append({
                    'Preview': text,
                    'Start': start_t,
                    'End': cut_t,
                    'Duration': dur_cut
                })
            start_idx = cut_idx + 1
            start_t = cut_t
            flagged = False
            last_pause_time = None

    # flush last segment
    if start_idx < len(sentence_entries):
        end_t = sentence_entries[-1]['end']
        dur = duration_seconds(start_t, end_t)
        text = " ".join(e['text'] for e in sentence_entries[start_idx:]).strip()
        if dur >= cfg.MIN_DURATION_SECONDS:
            candidates.append({
                'Preview': text,
                'Start': start_t,
                'End': end_t,
                'Duration': dur
            })
        elif candidates:  # too short: merge into previous
            candidates[-1]['Preview'] += ' ' + text
            candidates[-1]['End'] = end_t
            candidates[-1]['Duration'] = duration_seconds(candidates[-1]['Start'], end_t)
        else:  # single short clip, keep anyway
            candidates.append({
                'Preview': text,
                'Start': start_t,
                'End': end_t,
                'Duration': dur
            })

    logging.info(f'Generated {len(candidates)} candidate clips (min duration enforced).')
    return candidates

# -------------------------
# Precompute per-candidate features (main process heavy work)
# -------------------------

def get_hook_score_ml(text: str) -> float:
    if not text or text.strip() == "":
        return 0.0
    try:
        res = hook_classifier(text[:250])
        if isinstance(res, list) and res:
            label = res[0].get('label', '').lower()
            score = float(res[0].get('score', 0.0))
            if 'pos' in label or 'hook' in label or 'yes' in label:
                return score
            return score if score >= 0.6 else 0.0
    except Exception:
        return 0.0
    return 0.0


def compute_delivery_features_from_text(text: str, duration_seconds_val: float) -> Dict[str, float]:
    if not text or duration_seconds_val <= 0:
        return {"pace": 0.0, "pause_ratio": 0.0, "filler_density": 0.0, "delivery_score": 0.0}

    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]
    word_count = len(words)
    pace = word_count / max(0.001, duration_seconds_val)

    # punctuation-based pause heuristic
    pause_tokens = sum(1 for ch in text if ch in {',', ';', '—', '-', '...'} )
    pause_time_est = pause_tokens * 0.5
    pause_ratio = min(0.9, pause_time_est / (duration_seconds_val + 1e-6))

    filler_count = sum(1 for w in words if w.lower() in FILLERS)
    filler_density = filler_count / max(1, word_count)

    delivery_score = (
        (min(pace / 3.0, 1.0) * 0.5) +
        (1.0 - min(pause_ratio * 2.0, 1.0)) * 0.3 +
        (1.0 - min(filler_density * 5.0, 1.0)) * 0.2
    )
    return {
        'pace': float(pace),
        'pause_ratio': float(pause_ratio),
        'filler_density': float(filler_density),
        'delivery_score': float(delivery_score)
    }


def precompute_features(candidates: List[Dict], cfg: Cfg) -> Tuple[List[Dict], np.ndarray, List[str]]:
    texts = [c['Preview'] for c in candidates]

    # --- Embeddings (for redundancy + BERTopic + scoring) ---
    embeddings = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # --- BERTopic (topic clustering) ---
    topic_model = BERTopic(verbose=False)
    try:
        topics, probs = topic_model.fit_transform(texts, embeddings)
    except Exception:
        # fallback to empty topics
        topics = [-1] * len(texts)
        probs = [None] * len(texts)

    for i, c in enumerate(candidates):
        c['TopicID'] = int(topics[i]) if topics is not None else -1
        try:
            c['TopicScore'] = float(np.max(probs[i])) if probs is not None and probs[i] is not None else 0.0
        except Exception:
            c['TopicScore'] = 0.0

    # --- Emotion classification (batched) ---
    try:
        emo_results = emotion_pipeline(texts, truncation=True, batch_size=16)
    except Exception:
        emo_results = [ [{'label':'neutral','score':0.0}] for _ in texts ]

    for i, emo_scores in enumerate(emo_results):
        try:
            if isinstance(emo_scores, list) and emo_scores:
                candidates[i]['EmotionScore'] = float(max(s.get('score', 0.0) for s in emo_scores))
            elif isinstance(emo_scores, dict):
                candidates[i]['EmotionScore'] = float(emo_scores.get('score', 0.0))
            else:
                candidates[i]['EmotionScore'] = 0.0
        except Exception:
            candidates[i]['EmotionScore'] = 0.0

    # --- Keywords (KeyBERT) ---
    for i, text in enumerate(texts):
        try:
            kw = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=6)
            candidates[i]['Keywords'] = [w for w, s in kw]
        except Exception:
            candidates[i]['Keywords'] = []

    # --- Named Entities (spaCy) ---
    for i, text in enumerate(texts):
        try:
            doc = nlp(text)
            candidates[i]['EntityCount'] = sum(1 for ent in doc.ents if ent.label_ not in ("CARDINAL", "ORDINAL"))
            candidates[i]['Entities'] = [ent.text for ent in doc.ents if ent.label_ not in ("CARDINAL", "ORDINAL")]
        except Exception:
            candidates[i]['EntityCount'] = 0
            candidates[i]['Entities'] = []

    # --- Global topic keywords (for contextual overlap scoring) ---
    try:
        global_kw = kw_model.extract_keywords(" ".join(texts), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=30)
        global_topics = [w for w, s in global_kw]
    except Exception:
        global_topics = []

    # --- Hook ML score (main-process only) ---
    for i, c in enumerate(candidates):
        try:
            c['HookScoreML'] = float(get_hook_score_ml(c['Preview']))
        except Exception:
            c['HookScoreML'] = 0.0

    # --- Delivery / Prosody features (approximate) ---
    for i, c in enumerate(candidates):
        dur = float(c.get('Duration', 0.0))
        delivery = compute_delivery_features_from_text(c.get('Preview', ''), dur)
        c['DeliveryPace'] = delivery['pace']
        c['DeliveryPauseRatio'] = delivery['pause_ratio']
        c['DeliveryFillerDensity'] = delivery['filler_density']
        c['DeliveryScore'] = delivery['delivery_score']

    # --- Virality priors (expanded, richer dictionary) ---
    VIRALITY_PRIORS_EXPANDED = {
        "secret": 1.3, "hidden": 1.25, "unknown": 1.2, "revealed": 1.15, "exposed": 1.25,
        "money": 1.5, "wealth": 1.3, "rich": 1.25, "millionaire": 1.4, "billion": 1.4, "pay": 1.2,
        "danger": 1.4, "deadly": 1.3, "risk": 1.2, "worst": 1.15, "scary": 1.25, "collapse": 1.3,
        "everyone": 1.2, "nobody": 1.2, "people": 1.1, "society": 1.1, "your": 1.05,
        "weird": 1.25, "crazy": 1.2, "insane": 1.2, "unbelievable": 1.3, "wild": 1.2,
        "success": 1.2, "famous": 1.25, "genius": 1.3, "smartest": 1.25,
        "top": 1.1, "most": 1.15, "best": 1.15, "biggest": 1.2, "first": 1.15
    }

    table = str.maketrans('', '', string.punctuation)
    for i, c in enumerate(candidates):
        words = [w.lower().translate(table) for w in c.get('Preview', '').split()]
        virality_boost = 0.0
        for w in words:
            if w in VIRALITY_PRIORS_EXPANDED:
                virality_boost += float(VIRALITY_PRIORS_EXPANDED[w])
        c['ViralityBoost'] = float(virality_boost)

    logging.info(f'Precomputed features for {len(candidates)} candidates.')
    return candidates, embeddings, global_topics

# -------------------------
# Lightweight scoring worker (safe for multiprocessing)
# -------------------------

def score_worker(arg_tuple):
    (idx, preview, duration, topic_score, keywords, topic_id, emotion_score, entity_count, global_topics) = arg_tuple
    cfg = Cfg()
    text = preview or ''
    table = str.maketrans('', '', string.punctuation)
    words = [w.lower().translate(table) for w in text.split()]

    topic_score_val = float(topic_score)

    contextual_topic_score = 0
    if keywords and global_topics:
        contextual_topic_score = sum(1 for kw in keywords if kw in global_topics)

    # parse ML-provided hook token if present
    hook_score_ml = 0.0
    if keywords:
        for kw in keywords:
            if isinstance(kw, str) and kw.startswith('HOOK_SCORE:'):
                try:
                    hook_score_ml = max(hook_score_ml, float(kw.split(':', 1)[1]))
                except Exception:
                    pass

    hook_score_regex = sum(1 for pat in cfg.HOOK_PATTERNS if pat.search(text))
    hook_score = max(hook_score_ml, float(hook_score_regex))

    entity_score = int(entity_count)
    brevity_bonus = 1.0 if cfg.MIN_DURATION_SECONDS <= duration <= 15 else 0.6

    # virality micro-priors fallback
    virality_boost = 0.0
    micro_priors = cfg.VIRALITY_PRIORS if hasattr(cfg, 'VIRALITY_PRIORS') else {}
    for w in words:
        virality_boost += float(micro_priors.get(w, 0.0))

    emotion = float(emotion_score)
    surprise = int(bool(cfg.SURPRISE_PATTERNS.search(text)))
    cliffhanger = int(text.strip().endswith(('?', '...')))
    narrative = int(bool(cfg.NARRATIVE_PATTERNS.search(text)))

    speech_delivery = 0.5
    if keywords:
        for kw in keywords:
            if isinstance(kw, str) and kw.startswith('DELIVERY:'):
                try:
                    speech_delivery = max(speech_delivery, float(kw.split(':', 1)[1]))
                except Exception:
                    pass

    w = cfg.WEIGHTS
    final_score = (
        w.get('topic_score', 1.0) * topic_score_val +
        w.get('contextual_topic', 1.0) * contextual_topic_score +
        w.get('hook', 1.0) * hook_score +
        w.get('named_entity', 0.1) * entity_score +
        w.get('brevity', 1.0) * brevity_bonus +
        w.get('virality_prior', 1.0) * virality_boost +
        w.get('emotion', 0.0) * emotion +
        w.get('surprise', 0.0) * surprise +
        w.get('cliffhanger', 0.0) * cliffhanger +
        w.get('narrative_arc', 0.0) * narrative +
        w.get('speech_delivery', 0.0) * speech_delivery
    )

    if any((b in words) for b in Cfg.BAD_WORDS):
        final_score *= 0.2

    return {
        'Idx': int(idx),
        'Score': float(final_score),
        'Start': None,
        'End': None,
        'Duration': float(duration),
        'Preview': text,
        'Keywords': keywords,
        'TopicID': int(topic_id) if topic_id is not None else -1,
        'EmotionScore': float(emotion),
        'EntityCount': int(entity_count),
        'HookScore': float(hook_score),
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
    clips_sorted = sorted(clips, key=lambda x: x['Score'], reverse=True)
    final = []
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
        if any((clip['Start'] < c['End']) and (c['Start'] < clip['End']) for c in selected):
            continue
        selected.append(clip)
        if len(selected) >= k:
            break
    return selected

# -------------------------
# Main pipeline
# -------------------------

def process_file(srt_path: str, audio_path: Optional[str] = None, top_k: int = 10) -> Optional[pd.DataFrame]:
    cfg = Cfg()
    logging.info(f"Processing {srt_path}")

    sentence_entries = load_subtitles(srt_path)
    if not sentence_entries:
        logging.warning("No sentence entries extracted.")
        return None

    boundaries = detect_boundaries(sentence_entries, cfg)

    pauses = []
    if audio_path:
        pauses = get_pauses_from_assemblyai(audio_path)

    candidates = generate_candidates(sentence_entries, boundaries, pauses, cfg)
    if not candidates:
        logging.warning("No candidate clips produced.")
        return None

    candidates, embeddings, global_topics = precompute_features(candidates, cfg)

    # Prepare worker args (primitives only) — embed DELIVERY/HOOK/VIRAL into Keywords tokens
    worker_args = []
    for i, c in enumerate(candidates):
        kw = list(c.get('Keywords', []))
        kw.append(f"DELIVERY:{c.get('DeliveryScore', 0.5):.3f}")
        kw.append(f"HOOK_SCORE:{c.get('HookScoreML', 0.0):.3f}")
        kw.append(f"VIRAL:{c.get('ViralityBoost', 0.0):.3f}")

        worker_args.append((
            i,
            c['Preview'],
            c['Duration'],
            c.get('TopicScore', 0.0),
            kw,
            c.get('TopicID', -1),
            c.get('EmotionScore', 0.0),
            c.get('EntityCount', 0),
            global_topics
        ))

    # Multiprocess scoring (workers are lightweight)
    logging.info('Scoring candidates with multiprocessing...')
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(score_worker, worker_args), total=len(worker_args), desc='Scoring'))

    # attach timestamps back (Start/End)
    for r in results:
        idx = r['Idx']
        r['Start'] = candidates[idx]['Start']
        r['End'] = candidates[idx]['End']

    # redundancy filter uses embeddings aligned to candidate index
    non_redundant = filter_redundant(results, embeddings, cfg.REDUNDANCY_SIMILARITY_THRESHOLD)

    # Convert Start/End to comparable datetimes (they already are time objects)
    top_clips = select_non_overlapping(non_redundant, top_k)

    # Format DataFrame
    df_rows = []
    for rank, clip in enumerate(top_clips, start=1):
        df_rows.append({
            'Rank': rank,
            'Score': f"{clip['Score']:.4f}",
            'Start': format_time(clip['Start']),
            'End': format_time(clip['End']),
            'Duration': f"{clip['Duration']:.2f}s",
            'Preview': clip['Preview'],
            'Keywords': ", ".join([k for k in (clip.get('Keywords') or []) if not k.startswith(('DELIVERY:', 'HOOK_SCORE:', 'VIRAL:'))]),
            'TopicID': clip.get('TopicID', -1),
            'EmotionScore': f"{clip.get('EmotionScore',0.0):.3f}",
            'EntityCount': clip.get('EntityCount', 0)
        })

    df = pd.DataFrame(df_rows)
    out_csv = Path('viral_clips.csv')
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved top {len(df)} clips to {out_csv.resolve()}")
    return df

# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python video_segmenter_full.py <file.srt> [file.mp3] [top_k]')
        sys.exit(1)

    srt_file = sys.argv[1]
    audio = sys.argv[2] if len(sys.argv) > 2 else None
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    df_result = process_file(srt_file, audio, top_k=top_k)
    if df_result is not None:
        print(df_result.to_string(index=False))
