#!/usr/bin/env python3
"""
executioner.py — Windows-safe, single-AssemblyAI-call, semantic-aware video segmenter

Key behaviors:
- Loads heavy ML models exactly once (global lazy loader).
- Calls AssemblyAI at most once per unique audio file, caches transcript locally.
- Semantic-aware segmentation (idea units) using sentence embeddings + discourse cues.
- Endpoint refinement using word timestamps, prosody & silence detection.
- Threaded scoring (ThreadPoolExecutor) so worker threads reuse the single global models
  and do NOT reload them per worker — avoids Windows spawn/fork issues.
- No CLI wrapper, import-safe. Call process_file(...) from your code.
- Requires: sentence-transformers, bertopic, keybert, spacy(en_core_web_trf), pysrt,
  assemblyai, transformers, sklearn, tqdm, pandas, numpy, librosa, python-dotenv, moviepy (optional)
"""

# Standard libs
import os
import re
import sys
import logging
import string
import tempfile
import subprocess
import math
import json
import hashlib
from pathlib import Path
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Third-party libs
import pysrt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML + NLP
import assemblyai as aai
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline as hf_pipeline
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# Audio
import librosa
from dotenv import load_dotenv
load_dotenv()

# Optional: moviepy for fps detection
try:
    from moviepy import VideoFileClip
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
class Cfg:
    MAX_DURATION_SECONDS = 45
    MIN_DURATION_SECONDS = 15
    WINDOW_SIZE = 2
    DYNAMIC_STD_FACTOR = 0.5
    RELATIVE_DROP = 0.15
    REDUNDANCY_SIMILARITY_THRESHOLD = 0.90

    WEIGHTS = {
        'topic_score': 0.8,
        'contextual_topic': 0.7,
        'hook': 2.0,
        'named_entity': 0.3,
        'brevity': 1.0,
        'virality_prior': 1.0,
        'emotion': 0.35,
        'surprise': 0.15,
        'cliffhanger': 0.15,
        'narrative_arc': 0.15,
        'speech_delivery': 0.1,
        'prosody': 0.25
    }

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

FILLERS = {"uh", "um", "erm", "uhh", "umm", "like", "you know", "ah", "eh"}

# ---------------------------------------------------------------------
# Globals: models and transcript cache
# ---------------------------------------------------------------------
_GLOBAL_MODELS: Optional[Dict[str, Any]] = None
TRANSCRIPT_CACHE_DIR = Path(".aai_cache")
TRANSCRIPT_CACHE_DIR.mkdir(exist_ok=True)

def get_models():
    """
    Lazy-load heavy models once and return dict of models.
    Thread-safe for reuse by thread workers.
    """
    global _GLOBAL_MODELS
    if _GLOBAL_MODELS is not None:
        return _GLOBAL_MODELS

    logging.info("[MODELS] Loading heavy models (this happens once)...")

    EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'all-mpnet-base-v2')
    EMOTION_MODEL = os.getenv('EMOTION_MODEL', 'j-hartmann/emotion-english-distilroberta-base')
    HOOK_MODEL = os.getenv('HOOK_MODEL', None)

    # SentenceTransformer
    st_model = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info(f"[MODELS] SentenceTransformer loaded: {EMBED_MODEL_NAME}")

    # spaCy
    try:
        nlp = spacy.load("en_core_web_trf")
        logging.info("[MODELS] spaCy loaded: en_core_web_trf")
    except Exception as e:
        logging.error("spaCy en_core_web_trf not found. Please install it: python -m spacy download en_core_web_trf")
        raise

    # KeyBERT
    try:
        kw_model = KeyBERT(model=st_model)
        logging.info("[MODELS] KeyBERT loaded")
    except Exception as e:
        logging.warning(f"[MODELS] KeyBERT load failed: {e}")
        kw_model = None

    # Emotion pipeline
    try:
        emotion_pipe = hf_pipeline("text-classification", model=EMOTION_MODEL, top_k=None)
        logging.info(f"[MODELS] Emotion pipeline loaded: {EMOTION_MODEL}")
    except Exception as e:
        logging.warning(f"[MODELS] Emotion pipeline failed: {e}")
        emotion_pipe = None

    # Hook classifier
    try:
        hook_name = HOOK_MODEL or "distilbert-base-uncased-finetuned-sst-2-english"
        hook_pipe = hf_pipeline("text-classification", model=hook_name, top_k=1)
        logging.info(f"[MODELS] Hook classifier loaded: {hook_name}")
    except Exception as e:
        logging.warning(f"[MODELS] Hook classifier failed: {e}")
        hook_pipe = None

    # BERTopic
    try:
        topic_model = BERTopic(embedding_model=st_model, verbose=False)
        logging.info("[MODELS] BERTopic loaded")
    except Exception as e:
        logging.warning(f"[MODELS] BERTopic failed: {e}")
        topic_model = None

    _GLOBAL_MODELS = {
        "st_model": st_model,
        "spacy": nlp,
        "keybert": kw_model,
        "emotion": emotion_pipe,
        "hook": hook_pipe,
        "bertopic": topic_model
    }

    logging.info("[MODELS] Done loading models.")
    return _GLOBAL_MODELS

def _audio_path_to_cache_key(audio_path: str) -> str:
    p = Path(audio_path)
    mtime = p.stat().st_mtime if p.exists() else 0.0
    key_raw = f"{str(p.resolve())}|{mtime}"
    return hashlib.sha256(key_raw.encode('utf-8')).hexdigest()

def _get_cached_transcript_json_path(audio_path: str) -> Path:
    key = _audio_path_to_cache_key(audio_path)
    return TRANSCRIPT_CACHE_DIR / f"{key}.json"

def transcribe_and_cache_with_assemblyai(audio_path: str, api_key: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Single-call AssemblyAI transcription with disk caching.
    Returns {'transcript': str|None, 'words':[], 'pauses':[], 'speakers':[]}
    """
    cache_path = _get_cached_transcript_json_path(audio_path)
    if cache_path.exists() and not force:
        try:
            with open(cache_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            logging.info(f"[AssemblyAI] Loaded cached transcript from {cache_path}")
            return data
        except Exception as e:
            logging.warning(f"[AssemblyAI] Failed to read cache {cache_path}: {e}")

    key = api_key or os.getenv('ASSEMBLYAI_API_KEY')
    if not key:
        logging.info("[AssemblyAI] No API key found; skipping AssemblyAI features.")
        return {'transcript': None, 'words': [], 'pauses': [], 'speakers': []}

    aai.settings.api_key = key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True, audio_start_from=0, audio_end_at=None)

    logging.info("[AssemblyAI] Submitting audio for transcription (once)...")
    try:
        transcript_obj = transcriber.transcribe(audio_path, config=config)
    except Exception as e:
        logging.warning(f"[AssemblyAI] Transcription call failed: {e}")
        return {'transcript': None, 'words': [], 'pauses': [], 'speakers': []}

    words = []
    pauses = []
    speakers = []

    if getattr(transcript_obj, 'status', None) == 'completed':
        words_raw = getattr(transcript_obj, 'words', []) or []
        for w in words_raw:
            try:
                words.append({
                    'text': getattr(w, 'text', ''),
                    'start_s': getattr(w, 'start', 0) / 1000.0,
                    'end_s': getattr(w, 'end', 0) / 1000.0
                })
            except Exception:
                pass
        for i in range(len(words_raw) - 1):
            w = words_raw[i]
            w_next = words_raw[i + 1]
            gap_ms = w_next.start - w.end
            gap_s = gap_ms / 1000.0
            if gap_s >= 0.5:
                pauses.append({
                    'start_s': w.end / 1000.0,
                    'end_s': w_next.start / 1000.0,
                    'duration_s': gap_s
                })
        for utter in getattr(transcript_obj, 'utterances', []):
            speakers.append({
                'speaker': getattr(utter, 'speaker', 'spk_unknown'),
                'start_s': getattr(utter, 'start', 0) / 1000.0,
                'end_s': getattr(utter, 'end', 0) / 1000.0
            })
        logging.info(f"[AssemblyAI] Completed: {len(words)} words, {len(pauses)} pauses, {len(speakers)} speakers.")
    else:
        logging.warning("[AssemblyAI] Transcription incomplete; returning minimal features.")

    serialized = {
        'transcript': getattr(transcript_obj, 'text', None) if transcript_obj else None,
        'words': words,
        'pauses': pauses,
        'speakers': speakers
    }

    try:
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(serialized, fh)
        logging.info(f"[AssemblyAI] Cached transcript to {cache_path}")
    except Exception as e:
        logging.warning(f"[AssemblyAI] Failed to write cache {cache_path}: {e}")

    return serialized

# ---------------------------------------------------------------------
# Utilities: time, ffmpeg, fps
# ---------------------------------------------------------------------
def safe_time_to_dt(t: time) -> datetime:
    return datetime.combine(datetime.min, t)

def duration_seconds(start: time, end: time) -> float:
    return (safe_time_to_dt(end) - safe_time_to_dt(start)).total_seconds()

def format_time(t: time) -> str:
    return t.strftime('%H:%M:%S.%f')[:-3]

def time_to_seconds(t: time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm', '.avi'}

def extract_audio_ffmpeg(input_path: str, out_wav: str) -> bool:
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-vn', '-ac', '1', '-ar', '16000', '-f', 'wav', str(out_wav)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logging.warning(f"[FFMPEG] audio extraction failed: {e}")
        return False

def get_video_fps(media_path: Optional[str]) -> float:
    default_fps = 30.0
    if media_path is None:
        return default_fps
    try:
        if _HAS_MOVIEPY:
            clip = VideoFileClip(media_path)
            fps_val = float(getattr(clip, 'fps', default_fps) or default_fps)
            clip.reader.close()
            if hasattr(clip, 'audio') and clip.audio is not None:
                try:
                    clip.audio.reader.close_proc()
                except Exception:
                    pass
            return fps_val
    except Exception:
        pass
    try:
        cmd = [
            'ffprobe', '-v', '0', '-of', 'json', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate', str(media_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = json.loads(proc.stdout)
        if 'streams' in out and out['streams']:
            r_frame_rate = out['streams'][0].get('r_frame_rate', '30/1')
            if '/' in r_frame_rate:
                a, b = r_frame_rate.split('/')
                fps_val = float(a) / float(b) if float(b) != 0 else default_fps
                return fps_val
    except Exception:
        pass
    return default_fps

# ---------------------------------------------------------------------
# SRT parsing -> sentence entries
# ---------------------------------------------------------------------
def load_subtitles(srt_path: str) -> List[Dict]:
    models = get_models()
    nlp = models.get('spacy')
    subs = pysrt.open(srt_path, encoding='utf-8')
    full_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
    doc = nlp(full_text)

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

    logging.info(f"[SRT] Loaded {len(sentence_entries)} sentence entries from subtitles.")
    return sentence_entries

# ---------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------
def compute_sentence_embeddings(sentence_entries: List[Dict]) -> np.ndarray:
    models = get_models()
    st_model = models.get('st_model')
    sentences = [s['text'] for s in sentence_entries]
    if not sentences:
        return np.zeros((0, st_model.get_sentence_embedding_dimension() if hasattr(st_model, 'get_sentence_embedding_dimension') else 768))
    embeddings = st_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def semantic_drift_score(centroid: np.ndarray, emb: np.ndarray) -> float:
    if centroid is None or emb is None:
        return 0.0
    try:
        sim = float(cosine_similarity([centroid], [emb])[0][0])
        return 1.0 - sim
    except Exception:
        return 0.0

def update_centroid(centroid: Optional[np.ndarray], new_emb: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if centroid is None:
        return new_emb.copy()
    return (1.0 - alpha) * centroid + alpha * new_emb

# ---------------------------------------------------------------------
# Semantic-aware boundary detection
# ---------------------------------------------------------------------
def detect_semantic_boundaries(sentence_entries: List[Dict], cfg: Cfg,
                               min_coherence_time: float = 2.0,
                               drift_threshold: float = 0.35,
                               require_drift_count: int = 1) -> List[int]:
    if not sentence_entries:
        return [0, 0]

    embeddings = compute_sentence_embeddings(sentence_entries)
    if embeddings.shape[0] == 0:
        return detect_boundaries(sentence_entries, cfg)

    n = len(sentence_entries)
    boundaries = {0}
    centroid = None
    current_start_idx = 0
    drift_counts = 0

    discourse_flags = [bool(re.search(r"(\b(in conclusion|to conclude|to sum up|so that|therefore|thus|in short|basically|that means)\b)|[.?!]\s*$", s['text'], re.I))
                       or s['text'].strip().endswith(('?', '...')) for s in sentence_entries]

    for i in range(n):
        emb = embeddings[i]
        if centroid is None:
            centroid = emb.copy()
            drift = 0.0
        else:
            drift = semantic_drift_score(centroid, emb)

        seg_start = sentence_entries[current_start_idx]['start']
        seg_end = sentence_entries[i]['end']
        seg_length = duration_seconds(seg_start, seg_end)

        discourse = discourse_flags[i]

        if drift >= drift_threshold:
            drift_counts += 1
        else:
            drift_counts = max(0, drift_counts - 1)

        strong_drift = drift_counts >= require_drift_count
        min_size_ok = seg_length >= min_coherence_time

        if strong_drift and (discourse or min_size_ok):
            boundaries.add(i + 1)
            centroid = None
            current_start_idx = i + 1
            drift_counts = 0
            if current_start_idx < n:
                centroid = embeddings[current_start_idx].copy()
        else:
            centroid = update_centroid(centroid, emb, alpha=0.18)

    boundaries.add(n)
    b = sorted(list(boundaries))

    # Post-process merging
    final_boundaries = [b[0]]
    for k in range(1, len(b)):
        prev_idx = final_boundaries[-1]
        this_idx = b[k]
        segA_embs = embeddings[prev_idx:this_idx]
        if segA_embs.size == 0:
            final_boundaries.append(this_idx)
            continue
        centroidA = np.mean(segA_embs, axis=0)
        next_start = this_idx
        next_end = b[k + 1] if k + 1 < len(b) else n
        segB_embs = embeddings[next_start:next_end] if next_start < next_end else np.empty((0, embeddings.shape[1]))
        if segB_embs.size == 0:
            final_boundaries.append(this_idx)
            continue
        centroidB = np.mean(segB_embs, axis=0)
        sim = float(cosine_similarity([centroidA], [centroidB])[0][0])
        if sim >= 0.60:
            logging.debug(f"[MERGE] Merging boundary at {this_idx} (sim {sim:.3f})")
            continue
        else:
            final_boundaries.append(this_idx)

    if final_boundaries[-1] != n:
        final_boundaries.append(n)

    logging.info(f"[SEMANTIC] Detected {len(final_boundaries)-1} semantic segments.")
    return sorted(list(set(final_boundaries)))

# ---------------------------------------------------------------------
# Fallback similarity-window detection
# ---------------------------------------------------------------------
def compute_similarity(sentences: List[str], cfg: Cfg) -> np.ndarray:
    models = get_models()
    st_model = models['st_model']
    if len(sentences) <= cfg.WINDOW_SIZE:
        return np.array([])
    window_texts = [" ".join(sentences[i:i + cfg.WINDOW_SIZE]) for i in range(len(sentences) - cfg.WINDOW_SIZE + 1)]
    embeddings = st_model.encode(window_texts, show_progress_bar=False, convert_to_numpy=True)
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
    logging.info(f"[BOUNDARIES] Detected {len(b)-1} coarse segments from {len(sentences)} sentences.")
    return b

# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------
def generate_candidates(
    sentence_entries: List[Dict],
    boundaries: List[int],
    pauses: Optional[List[Dict]],
    speaker_segments: Optional[List[Dict]],
    cfg: Cfg
) -> List[Dict]:
    candidates = []
    if not sentence_entries:
        return candidates

    start_idx = 0
    start_t = sentence_entries[0]['start']
    flagged = False

    BACKWARD_LEEWAY = 5
    FORWARD_LEEWAY = 10

    def find_backward_cut(idx, max_end):
        max_end_s = time_to_seconds(max_end)
        for j in range(idx, -1, -1):
            end_time = sentence_entries[j]['end']
            end_s = time_to_seconds(end_time)
            if max_end_s - end_s > BACKWARD_LEEWAY:
                break
            text = sentence_entries[j]['text'].strip()
            if text.endswith(('.', '?', '!')):
                return j, end_time
            if pauses:
                for p in pauses:
                    if abs(end_s - p['start_s']) < 0.5 or (p['start_s'] <= end_s <= p['end_s']):
                        return j, end_time
        return None, max_end

    def find_forward_cut(idx, max_end):
        max_end_s = time_to_seconds(max_end)
        for j in range(idx + 1, len(sentence_entries)):
            end_time = sentence_entries[j]['end']
            end_s = time_to_seconds(end_time)
            if end_s - max_end_s > FORWARD_LEEWAY:
                break
            text = sentence_entries[j]['text'].strip()
            if text.endswith(('.', '?', '!')):
                return j, end_time
        return None, max_end

    for i, sent in enumerate(sentence_entries):
        curr_end = sent['end']
        dur = duration_seconds(start_t, curr_end)

        if speaker_segments:
            curr_start_s = time_to_seconds(start_t)
            curr_end_s = time_to_seconds(curr_end)
            for sp in speaker_segments:
                if sp['start_s'] > curr_start_s and sp['start_s'] < curr_end_s:
                    flagged = True
                    break

        if i in boundaries:
            flagged = True

        if flagged and dur >= cfg.MIN_DURATION_SECONDS:
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

        elif dur >= cfg.MAX_DURATION_SECONDS:
            cut_idx, cut_t = find_backward_cut(i, curr_end)
            if cut_idx is None:
                cut_idx = i
                cut_t = curr_end

            if cut_idx == i:
                f_idx, f_t = find_forward_cut(i, curr_end)
                if f_idx is not None:
                    cut_idx, cut_t = f_idx, f_t

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

    logging.info(f'[CANDIDATES] Generated {len(candidates)} candidate clips.')
    return candidates

# ---------------------------------------------------------------------
# Prosody and endpoint refinement utilities
# ---------------------------------------------------------------------
def compute_audio_prosody_features(audio_path: str, start_s: float, end_s: float, hop_length: int = 512) -> Dict[str, Any]:
    try:
        duration = max(0.001, end_s - start_s)
        y, sr = librosa.load(audio_path, sr=None, offset=start_s, duration=duration)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
        pitch_vals = []
        for col in range(pitches.shape[1]):
            col_p = pitches[:, col]
            col_mag = magnitudes[:, col]
            if np.any(col_mag > 0):
                idx = int(np.argmax(col_mag))
                pitch_vals.append(float(col_p[idx]))
            else:
                pitch_vals.append(0.0)
        pitch_arr = np.array(pitch_vals)
        mean_pitch = float(np.mean(pitch_arr[pitch_arr > 0])) if np.any(pitch_arr > 0) else 0.0
        pitch_var = float(np.var(pitch_arr[pitch_arr > 0])) if np.any(pitch_arr > 0) else 0.0
        mean_rms = float(np.mean(rms)) if rms.size > 0 else 0.0
        rms_var = float(np.var(rms)) if rms.size > 0 else 0.0
        return {
            'rms': rms,
            'pitch_arr': pitch_arr,
            'sr': sr,
            'hop_length': hop_length,
            'mean_pitch': mean_pitch,
            'pitch_var': pitch_var,
            'mean_rms': mean_rms,
            'rms_var': rms_var
        }
    except Exception as e:
        logging.debug(f"[PROSODY] analysis failed: {e}")
        return {'rms': np.array([]), 'pitch_arr': np.array([]), 'sr': None, 'hop_length': hop_length, 'mean_pitch': 0.0, 'pitch_var': 0.0, 'mean_rms': 0.0, 'rms_var': 0.0}

def find_last_word_end_for_candidate(candidate: Dict, word_timestamps: List[Dict]) -> Optional[float]:
    if not word_timestamps:
        return None
    cand_start = time_to_seconds(candidate['Start'])
    cand_end = time_to_seconds(candidate['End'])
    last_word_end = None
    for w in word_timestamps:
        ws = float(w.get('start_s', 0.0))
        we = float(w.get('end_s', 0.0))
        if ws >= cand_start - 0.01 and we <= cand_end + 0.5:
            if last_word_end is None or we > last_word_end:
                last_word_end = we
    return last_word_end

def detect_silence_after_point(audio_path: str, point_s: float, max_search: float = 1.0, silence_threshold_ratio: float = 0.06, min_silence_duration: float = 0.18, hop_length:int=512) -> Optional[float]:
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=max(0.0, point_s - 0.2), duration=max_search + 0.6)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        n_frames = rms.shape[0]
        frame_dur = (len(y) / sr) / max(1, n_frames)
        times = np.arange(n_frames) * frame_dur + max(0.0, point_s - 0.2)
        median_rms = float(np.median(rms)) if rms.size > 0 else 0.0
        threshold = median_rms * silence_threshold_ratio
        min_frames = max(1, int(math.ceil(min_silence_duration / frame_dur)))
        low = rms < threshold
        run_start = None
        for i in range(len(low)):
            if low[i]:
                if run_start is None:
                    run_start = i
                if i - run_start + 1 >= min_frames:
                    return float(times[run_start])
            else:
                run_start = None
        return None
    except Exception as e:
        logging.debug(f"[SILENCE] detection failed: {e}")
        return None

def find_prosody_drop(audio_path: str, anchor_end: float, look_ahead: float = 1.0, rms_drop_ratio: float = 0.35, hop_length:int=512) -> Optional[float]:
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=max(0.0, anchor_end - 0.05), duration=look_ahead + 0.1)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if rms.size == 0:
            return None
        baseline = np.median(rms)
        if baseline <= 0:
            return None
        threshold = baseline * (1 - rms_drop_ratio)
        n_frames = rms.shape[0]
        frame_dur = (len(y) / sr) / max(1, n_frames)
        for i in range(n_frames):
            if rms[i] <= threshold:
                t_rel = i * frame_dur - 0.05
                t_abs = anchor_end + t_rel
                return max(anchor_end, float(t_abs))
        return None
    except Exception as e:
        logging.debug(f"[PROSODY_DROP] failed: {e}")
        return None

def snap_time_to_frame(t: float, fps: float) -> float:
    if fps <= 0:
        return t
    frame = round(t * fps)
    return float(frame) / float(fps)

def refine_candidate_endpoints(
    candidates: List[Dict],
    word_timestamps: List[Dict],
    audio_path: Optional[str],
    media_path: Optional[str],
    cfg: Cfg,
    padding: float = 0.08,
    max_lookahead: float = 1.0,
    require_signals: int = 2,
    fps: Optional[float] = None,
    fade_out: float = 0.12
) -> List[Dict]:
    fps_val = fps or get_video_fps(media_path)
    logging.info(f"[REFINE] fps for snapping: {fps_val:.2f}")

    for idx, c in enumerate(candidates):
        meta = {
            'anchor_used': None,
            'anchor_time': None,
            'prosody_time': None,
            'silence_time': None,
            'semantic_signal': None,
            'final_end': None,
            'snapped_end': None,
            'padding_applied': None,
            'fade_out': fade_out,
            'signals_count': 0
        }

        cand_start_s = time_to_seconds(c['Start'])
        cand_end_s = time_to_seconds(c['End'])

        last_word_end = find_last_word_end_for_candidate(c, word_timestamps) if word_timestamps else None
        if last_word_end is not None:
            meta['anchor_used'] = 'word_ts'
            meta['anchor_time'] = float(last_word_end)
            meta['signals_count'] += 1
        else:
            meta['anchor_used'] = 'subtitle_end'
            meta['anchor_time'] = float(cand_end_s)

        prosody_time = None
        if audio_path and os.path.exists(audio_path):
            prosody_time = find_prosody_drop(audio_path, meta['anchor_time'], look_ahead=max_lookahead)
            if prosody_time is not None:
                meta['prosody_time'] = float(prosody_time)
                meta['signals_count'] += 1

        silence_time = None
        if audio_path and os.path.exists(audio_path):
            silence_time = detect_silence_after_point(audio_path, meta['anchor_time'], max_search=max_lookahead)
            if silence_time is not None:
                meta['silence_time'] = float(silence_time)
                meta['signals_count'] += 1

        meta['semantic_signal'] = None
        try:
            models = get_models()
            st_model = models.get('st_model')
            preview = c.get('Preview', '')
            next_preview = c.get('NextPreview', '')
            if preview and next_preview:
                emb_a = st_model.encode([preview], show_progress_bar=False, convert_to_numpy=True)
                emb_b = st_model.encode([next_preview], show_progress_bar=False, convert_to_numpy=True)
                sim = float(cosine_similarity(emb_a, emb_b)[0][0])
                boundary = sim < 0.55
                meta['semantic_signal'] = {'sim': sim, 'boundary': bool(boundary)}
                if boundary:
                    meta['signals_count'] += 1
        except Exception as e:
            logging.debug(f"[REFINE] semantic check failed: {e}")

        suggested_times = []
        if meta['anchor_time'] is not None:
            suggested_times.append(('anchor', meta['anchor_time']))
        if meta['prosody_time'] is not None:
            suggested_times.append(('prosody', meta['prosody_time']))
        if meta['silence_time'] is not None:
            suggested_times.append(('silence', meta['silence_time']))
        if meta.get('semantic_signal') and meta['semantic_signal'].get('boundary'):
            suggested_times.append(('semantic', meta['anchor_time']))

        final_time = None
        if suggested_times:
            times_sorted = sorted([t for (_, t) in suggested_times])
            clusters = []
            cluster = [times_sorted[0]]
            for t in times_sorted[1:]:
                if abs(t - cluster[-1]) <= 0.30:
                    cluster.append(t)
                else:
                    clusters.append(cluster)
                    cluster = [t]
            clusters.append(cluster)
            best_cluster = max(clusters, key=lambda c: len(c))
            if len(best_cluster) >= require_signals:
                final_time = float(min(best_cluster))
                meta['reason'] = f'{len(best_cluster)} signals agree (tight)'
            else:
                chosen = float(max(times_sorted))
                final_time = float(chosen + padding)
                meta['reason'] = 'not enough signals for tight cut; applied safe padding'
        else:
            final_time = float(cand_end_s + padding)
            meta['reason'] = 'no signals; subtitle end + padding'

        if final_time > cand_end_s + max_lookahead:
            final_time = cand_end_s + max_lookahead
            meta['reason'] += '; clamped to max_lookahead'

        snapped = snap_time_to_frame(final_time, fps_val)
        meta['final_end'] = float(final_time)
        meta['snapped_end'] = float(snapped)
        meta['padding_applied'] = float(max(0.0, snapped - meta['final_end'])) if snapped > meta['final_end'] else 0.0
        meta['signals_count'] = meta.get('signals_count', 0)
        c['_end_refinement'] = meta

        sec = snapped
        hrs = int(sec // 3600) % 24
        mins = int((sec % 3600) // 60)
        secs = int(sec % 60)
        micros = int((sec - math.floor(sec)) * 1e6)
        new_time = time(hour=hrs, minute=mins, second=secs, microsecond=micros)
        c['End'] = new_time
        c['Duration'] = duration_seconds(c['Start'], c['End'])

    logging.info(f'[REFINE] Completed endpoint refinement for {len(candidates)} candidates.')
    return candidates

# ---------------------------------------------------------------------
# Precompute features
# ---------------------------------------------------------------------
def get_hook_score_ml(text: str) -> float:
    if not text or text.strip() == "":
        return 0.0
    models = get_models()
    hook_classifier = models.get('hook')
    if hook_classifier is None:
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

def precompute_features(candidates: List[Dict], cfg: Cfg, audio_path: Optional[str]) -> Tuple[List[Dict], np.ndarray, List[str]]:
    models = get_models()
    st_model = models.get('st_model')
    topic_model = models.get('bertopic')
    kw_model = models.get('keybert')
    nlp = models.get('spacy')
    emotion_pipeline = models.get('emotion')

    texts = [c['Preview'] for c in candidates]
    embeddings = st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    topics, probs = None, None
    if topic_model is not None:
        try:
            topics, probs = topic_model.fit_transform(texts, embeddings)
        except Exception as e:
            logging.warning(f"[BERTopic] fit_transform failed: {e}")
            try:
                topic_model_local = BERTopic(embedding_model=st_model, verbose=False)
                topics, probs = topic_model_local.fit_transform(texts, embeddings)
            except Exception as e2:
                logging.warning(f"[BERTopic] fallback failed: {e2}")
                topics = [-1] * len(texts)
                probs = [None] * len(texts)
    else:
        topics = [-1] * len(texts)
        probs = [None] * len(texts)

    for i, c in enumerate(candidates):
        c['TopicID'] = int(topics[i]) if topics is not None else -1
        try:
            c['TopicScore'] = float(np.max(probs[i])) if probs is not None and probs[i] is not None else 0.0
        except Exception:
            c['TopicScore'] = 0.0

    try:
        if emotion_pipeline is not None:
            emo_results = emotion_pipeline(texts, truncation=True, batch_size=16)
        else:
            emo_results = [ [{'label':'neutral','score':0.0}] for _ in texts ]
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

    for i, text in enumerate(texts):
        try:
            if kw_model is not None:
                kw = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=6)
                candidates[i]['Keywords'] = [w for w, s in kw]
            else:
                candidates[i]['Keywords'] = []
        except Exception:
            candidates[i]['Keywords'] = []

    for i, text in enumerate(texts):
        try:
            doc = nlp(text)
            candidates[i]['EntityCount'] = sum(1 for ent in doc.ents if ent.label_ not in ("CARDINAL", "ORDINAL"))
            candidates[i]['Entities'] = [ent.text for ent in doc.ents if ent.label_ not in ("CARDINAL", "ORDINAL")]
        except Exception:
            candidates[i]['EntityCount'] = 0
            candidates[i]['Entities'] = []

    try:
        if kw_model is not None:
            global_kw = kw_model.extract_keywords(" ".join(texts), keyphrase_ngram_range=(1, 2), stop_words='english', top_n=30)
            global_topics = [w for w, s in global_kw]
        else:
            global_topics = []
    except Exception:
        global_topics = []

    for i, c in enumerate(candidates):
        try:
            c['HookScoreML'] = float(get_hook_score_ml(c['Preview']))
        except Exception:
            c['HookScoreML'] = 0.0

    for i, c in enumerate(candidates):
        dur = float(c.get('Duration', 0.0))
        delivery = compute_delivery_features_from_text(c.get('Preview', ''), dur)
        c['DeliveryPace'] = delivery['pace']
        c['DeliveryPauseRatio'] = delivery['pause_ratio']
        c['DeliveryFillerDensity'] = delivery['filler_density']
        c['DeliveryScore'] = delivery['delivery_score']

    for i, c in enumerate(candidates):
        try:
            if audio_path and Path(audio_path).exists():
                start_s = time_to_seconds(c['Start'])
                end_s = time_to_seconds(c['End'])
                pros = compute_audio_prosody_features(audio_path, start_s, end_s)
                c.update({
                    'mean_pitch': pros.get('mean_pitch', 0.0),
                    'pitch_var': pros.get('pitch_var', 0.0),
                    'loudness_mean': pros.get('mean_rms', 0.0),
                    'loudness_var': pros.get('rms_var', 0.0),
                    'ProsodyScore': float(pros.get('mean_rms', 0.0) * 0.4 + pros.get('pitch_var', 0.0) * 0.6)
                })
            else:
                c['ProsodyScore'] = 0.0
                c['mean_pitch'] = 0.0
                c['pitch_var'] = 0.0
                c['loudness_mean'] = 0.0
                c['loudness_var'] = 0.0
        except Exception:
            c['ProsodyScore'] = 0.0

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

    logging.info(f'[PRECOMPUTE] Precomputed features for {len(candidates)} candidates.')
    return candidates, embeddings, global_topics

# ---------------------------------------------------------------------
# Scoring worker (thread-safe: uses global models but only local data)
# ---------------------------------------------------------------------
def score_worker(arg_tuple):
    (idx, preview, duration, topic_score, keywords, topic_id, emotion_score, entity_count, global_topics, prosody_score, delivery_score) = arg_tuple
    cfg = Cfg()
    text = preview or ''
    table = str.maketrans('', '', string.punctuation)
    words = [w.lower().translate(table) for w in text.split()]

    topic_score_val = float(topic_score)

    contextual_topic_score = 0
    if keywords and global_topics:
        contextual_topic_score = sum(1 for kw in keywords if kw in global_topics)

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

    virality_boost = 0.0
    micro_priors = cfg.VIRALITY_PRIORS if hasattr(cfg, 'VIRALITY_PRIORS') else {}
    for w in words:
        virality_boost += float(micro_priors.get(w, 0.0))

    emotion = float(emotion_score)
    surprise = int(bool(cfg.SURPRISE_PATTERNS.search(text)))
    cliffhanger = int(text.strip().endswith(('?', '...')))
    narrative = int(bool(cfg.NARRATIVE_PATTERNS.search(text)))

    speech_delivery = float(delivery_score or 0.5)
    prosody_val = float(prosody_score or 0.0)

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
        w.get('speech_delivery', 0.0) * speech_delivery +
        w.get('prosody', 0.0) * prosody_val
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

# ---------------------------------------------------------------------
# Redundancy filtering & selection
# ---------------------------------------------------------------------
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

def select_non_overlapping(clips: List[Dict], k: int) -> List[Dict]:
    selected = []
    for clip in sorted(clips, key=lambda x: x['Score'], reverse=True):
        if any((clip['Start'] < c['End']) and (c['Start'] < clip['End']) for c in selected):
            continue
        selected.append(clip)
        if len(selected) >= k:
            break
    return selected

# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------
def generate_html_report(df: pd.DataFrame, csv_path: Path, out_path: str = 'viral_clips_report.html'):
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Viral Clips Report</title>",
        "<style>body{font-family:Inter,Arial,Helvetica,sans-serif;background:#0b0b0b;color:#f5f5f5;padding:24px;} table{width:100%;border-collapse:collapse;margin-top:12px;} th,td{padding:8px;border-bottom:1px solid #222;text-align:left;} th{background:#111;} tr:hover{background:#111111}</style>",
        "</head><body>",
        f"<h1>Viral Clips Report</h1>",
        f"<p>Generated: {datetime.utcnow().isoformat()} UTC</p>",
        "<p>Download CSV: <a href='" + str(csv_path.name) + "'>" + str(csv_path.name) + "</a></p>",
        "<table><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    ]
    for _, row in df.iterrows():
        html.append("<tr>" + "".join(f"<td>{row[c]}</td>" for c in df.columns) + "</tr>")
    html.append("</table></body></html>")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html))
    logging.info(f"[REPORT] Visual report saved to {Path(out_path).resolve()}")

# ---------------------------------------------------------------------
# Main pipeline: process_file — uses threaded scoring so models aren't reloaded
# ---------------------------------------------------------------------
def process_file(
    srt_path: str,
    media_path: Optional[str] = None,
    top_k: int = 10,
    num_workers: Optional[int] = None,
    refine_endings: bool = True,
    padding: float = 0.08,
    fade_out: float = 0.12,
    require_signals: int = 2,
    fps: Optional[float] = None,
    semantic_min_coherence: float = 2.0,
    semantic_drift_threshold: float = 0.35,
    semantic_require_drift: int = 1,
    assemblyai_force_refresh: bool = False
) -> Optional[pd.DataFrame]:
    """
    Processes srt and media, returns DataFrame of top clips.
    Uses threading for scoring so heavy models are loaded only once globally.
    """
    cfg = Cfg()
    srt_path = str(srt_path)
    media_path = str(media_path) if media_path else None
    logging.info(f"[PROCESS] Running on {srt_path} with media {media_path}")

    # prepare audio
    temp_audio = None
    audio_for_prosody = None
    if media_path:
        if is_video_file(media_path):
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            ok = extract_audio_ffmpeg(media_path, temp_audio.name)
            if ok:
                audio_for_prosody = temp_audio.name
            else:
                audio_for_prosody = None
        else:
            audio_for_prosody = media_path
    else:
        audio_for_prosody = None

    # ensure models loaded once
    get_models()

    # subtitles -> sentences
    sentence_entries = load_subtitles(srt_path)
    if not sentence_entries:
        logging.warning("[PROCESS] No sentences extracted.")
        return None

    # AssemblyAI: single call & cache
    audio_words = []
    audio_pauses = []
    audio_speakers = []
    if audio_for_prosody:
        transcript_data = transcribe_and_cache_with_assemblyai(audio_for_prosody, force=assemblyai_force_refresh)
        audio_words = transcript_data.get('words', [])
        audio_pauses = transcript_data.get('pauses', [])
        audio_speakers = transcript_data.get('speakers', [])
    else:
        if os.getenv('ASSEMBLYAI_API_KEY'):
            logging.warning("[PROCESS] ASSEMBLYAI_API_KEY set but no audio provided; skipping AssemblyAI.")
        audio_words = []
        audio_pauses = []
        audio_speakers = []

    # semantic boundaries
    boundaries = detect_semantic_boundaries(
        sentence_entries,
        cfg,
        min_coherence_time=semantic_min_coherence,
        drift_threshold=semantic_drift_threshold,
        require_drift_count=semantic_require_drift
    )

    # candidate generation
    candidates = generate_candidates(sentence_entries, boundaries, audio_pauses, audio_speakers, cfg)
    if not candidates:
        logging.warning("[PROCESS] No candidates.")
        return None

    # annotate for refinement
    for i, c in enumerate(candidates):
        c['NextPreview'] = candidates[i + 1].get('Preview', '') if i + 1 < len(candidates) else ''

    # precompute features
    candidates, embeddings, global_topics = precompute_features(candidates, cfg, audio_for_prosody)

    # refine endings
    if refine_endings:
        candidates = refine_candidate_endpoints(
            candidates,
            word_timestamps=audio_words,
            audio_path=audio_for_prosody,
            media_path=media_path,
            cfg=cfg,
            padding=padding,
            max_lookahead=1.0,
            require_signals=require_signals,
            fps=fps,
            fade_out=fade_out
        )

    # prepare thread worker args
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
            global_topics,
            c.get('ProsodyScore', 0.0),
            c.get('DeliveryScore', 0.0)
        ))

    # threaded scoring (threads reuse global models)
    logging.info('[PROCESS] Scoring candidates with ThreadPoolExecutor...')
    max_workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(score_worker, arg) for arg in worker_args]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Scoring'):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                logging.warning(f"[SCORE] Worker failed: {e}")

    # attach timestamps
    for r in results:
        idx = r['Idx']
        r['Start'] = candidates[idx]['Start']
        r['End'] = candidates[idx]['End']

    non_redundant = filter_redundant(results, embeddings, cfg.REDUNDANCY_SIMILARITY_THRESHOLD)
    top_clips = select_non_overlapping(non_redundant, top_k)

    # format DataFrame
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
            'EntityCount': clip.get('EntityCount', 0),
            'EndRefinement': json.dumps(candidates[clip['Idx']].get('_end_refinement', {}))
        })

    df = pd.DataFrame(df_rows)
    out_csv = Path('viral_clips.csv')
    df.to_csv(out_csv, index=False)
    logging.info(f"[PROCESS] Saved top {len(df)} clips to {out_csv.resolve()}")

    # report
    generate_html_report(df, out_csv, out_path='viral_clips_report.html')

    # cleanup
    if temp_audio:
        try:
            os.unlink(temp_audio.name)
        except Exception:
            pass

    return df

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "process_file",
    "get_models",
    "transcribe_and_cache_with_assemblyai",
    "detect_semantic_boundaries",
    "precompute_features",
    "generate_html_report"
]
