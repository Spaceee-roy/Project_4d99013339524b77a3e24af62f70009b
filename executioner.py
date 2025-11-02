"""
executioner.py — Self-contained segmenter with verbose SRT lifecycle logging.

Core features:
- Accepts media path; if .srt missing, extracts audio -> AssemblyAI (single call) -> writes SRT.
- Semantic segmentation with trailing-sentence validator (extend/backtrack).
- Endpoint refinement (silence/prosody/frame snapping).
- Global model loading (spaCy, SentenceTransformer, KeyBERT, BERTopic, HF pipelines).
- Threaded scoring to avoid Windows spawn model reloads.
- Detailed logs around SRT creation: audio extraction, upload, job polling, transcript summary,
  SRT write confirmation (existence, size, lines), and parsing summary.

Usage:
- Ensure dependencies installed (see earlier messages).
- Export ASSEMBLYAI_API_KEY to use AssemblyAI features.
- Call process_file(media_path) from your main script.

"""

# Standard libs
import os
import re
import json
import math
import hashlib
import logging
import tempfile
import subprocess
import time as time_module
import string
from pathlib import Path
from datetime import datetime, time
from typing import List, Dict, Optional, Any, Tuple
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

# Logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class Cfg:
    MIN_DURATION_SECONDS = 10.0
    MAX_DURATION_SECONDS = 60.0
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

    PAUSE_THRESHOLD_SECONDS = 0.5
    SILENCE_MIN_DURATION = 0.18

    TRAILING_CONJUNCTIONS = {
        'and','but','or','so','because','then','however','although','yet','still','though'
    }
    TRAILING_SUBORDINATORS = {
        'if','when','while','although','because','since','unless','until','whereas','after','before'
    }
    UNCERTAIN_ENDING_WORDS = {
        'that','which','who','whom','whose','where','when','how','why','because','if'
    }

    TRAILING_PATTERN = re.compile(
        r'(?:(?:\b(?:' + r'|'.join(list(TRAILING_CONJUNCTIONS | TRAILING_SUBORDINATORS)) + r')\b)\s*$)|[,;:]$',
        re.I
    )

    CLOSURE_MARKERS = {'therefore','thus','in conclusion','to conclude','finally','so','hence'}

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

# ------------------------------------------------------------------------------
# Globals and caches
# ------------------------------------------------------------------------------
_TRANSCRIPT_CACHE_DIR = Path(".aai_cache")
_TRANSCRIPT_CACHE_DIR.mkdir(exist_ok=True)

_GLOBAL_MODELS: Optional[Dict[str, Any]] = None

def _audio_cache_key(audio_path: str) -> str:
    p = Path(audio_path)
    mtime = p.stat().st_mtime if p.exists() else 0.0
    raw = f"{str(p.resolve())}|{mtime}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def _audio_cache_path(audio_path: str) -> Path:
    return _TRANSCRIPT_CACHE_DIR / f"{_audio_cache_key(audio_path)}.json"

# ------------------------------------------------------------------------------
# Model loading (lazy, global)
# ------------------------------------------------------------------------------
def get_models():
    global _GLOBAL_MODELS
    if _GLOBAL_MODELS is not None:
        return _GLOBAL_MODELS

    logging.info("[MODELS] Loading heavy models... (this happens once)")
    EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'all-mpnet-base-v2')
    EMOTION_MODEL = os.getenv('EMOTION_MODEL', 'j-hartmann/emotion-english-distilroberta-base')
    HOOK_MODEL = os.getenv('HOOK_MODEL', None)

    st_model = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info(f"[MODELS] SentenceTransformer loaded: {EMBED_MODEL_NAME}")

    try:
        nlp = spacy.load("en_core_web_trf")
        logging.info("[MODELS] spaCy loaded: en_core_web_trf")
    except Exception as e:
        logging.error("[MODELS] spaCy en_core_web_trf not found. Install: python -m spacy download en_core_web_trf")
        raise

    try:
        kw_model = KeyBERT(model=st_model)
        logging.info("[MODELS] KeyBERT loaded")
    except Exception as e:
        logging.warning(f"[MODELS] KeyBERT load failed: {e}")
        kw_model = None

    try:
        emotion_pipe = hf_pipeline("text-classification", model=EMOTION_MODEL, top_k=None)
        logging.info(f"[MODELS] Emotion pipeline loaded: {EMOTION_MODEL}")
    except Exception as e:
        logging.warning(f"[MODELS] Emotion pipeline load failed: {e}")
        emotion_pipe = None

    try:
        hook_name = HOOK_MODEL or "distilbert-base-uncased-finetuned-sst-2-english"
        hook_pipe = hf_pipeline("text-classification", model=hook_name, top_k=1)
        logging.info(f"[MODELS] Hook classifier loaded: {hook_name}")
    except Exception as e:
        logging.warning(f"[MODELS] Hook classifier failed: {e}")
        hook_pipe = None

    try:
        topic_model = BERTopic(embedding_model=st_model, verbose=False)
        logging.info("[MODELS] BERTopic loaded")
    except Exception as e:
        logging.warning(f"[MODELS] BERTopic load failed: {e}")
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

# ------------------------------------------------------------------------------
# Helpers: time, ffmpeg, fps
# ------------------------------------------------------------------------------
def format_time(t: time) -> str:
    return t.strftime('%H:%M:%S.%f')[:-3]

def time_to_seconds(t: time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def seconds_to_time(s: float) -> time:
    if s < 0:
        s = 0.0
    hrs = int(s // 3600) % 24
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    micros = int((s - math.floor(s)) * 1e6)
    return time(hour=hrs, minute=mins, second=secs, microsecond=micros)

def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm', '.avi'}

def extract_audio_ffmpeg(input_path: str, out_wav: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Extract audio; return (success, meta). meta includes file size and duration if obtainable.
    """
    try:
        cmd = ['ffmpeg', '-y', '-i', str(input_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(out_wav)]
        logging.info(f"[AUDIO] Running ffmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size = Path(out_wav).stat().st_size if Path(out_wav).exists() else 0
        duration = 0.0
        try:
            y, sr = librosa.load(out_wav, sr=None, duration=0.1)  # quick probe to ensure file readable
            # full duration expensive; skip
        except Exception:
            pass
        meta = {"size_bytes": size, "duration_est_s": duration}
        logging.info(f"[AUDIO] Extracted audio to {out_wav} (size={size} bytes)")
        return True, meta
    except Exception as e:
        logging.warning(f"[AUDIO] ffmpeg extraction failed: {e}")
        return False, {}

# ------------------------------------------------------------------------------
# AssemblyAI transcription (single-call per audio + disk cache) with logging
# ------------------------------------------------------------------------------
def transcribe_and_cache(audio_path: str, force: bool = False) -> Dict[str, Any]:
    """
    Call AssemblyAI once per audio file and cache result to disk. Log extensively.
    """
    cache_path = _audio_cache_path(audio_path)
    try:
        if cache_path.exists() and not force:
            with open(cache_path, 'r', encoding='utf-8') as fh:
                cached = json.load(fh)
            logging.info(f"[AAI] Loaded cached transcript ({cache_path.name}): words={len(cached.get('words', []))}")
            return cached
    except Exception as e:
        logging.warning(f"[AAI] Failed reading cache {cache_path}: {e}")

    api_key = os.getenv('ASSEMBLYAI_API_KEY')
    if not api_key:
        logging.warning("[AAI] No AssemblyAI API key; returning empty transcript.")
        return {'text': None, 'words': [], 'utterances': [], 'pauses': []}

    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True, audio_start_from=0, audio_end_at=None)

    # Log audio upload metadata
    try:
        size = Path(audio_path).stat().st_size if Path(audio_path).exists() else -1
    except Exception:
        size = -1
    logging.info(f"[AAI] Uploading audio: {audio_path} (size={size} bytes)")

    try:
        transcript_obj = transcriber.transcribe(audio_path, config=config)
    except Exception as e:
        logging.error(f"[AAI] Transcription request failed: {e}")
        return {'text': None, 'words': [], 'utterances': [], 'pauses': []}

    # Polling and status check is handled inside transcribe(); we assume returned object contains final status
    if getattr(transcript_obj, 'status', None) != 'completed':
        logging.warning(f"[AAI] Transcript status: {getattr(transcript_obj, 'status', None)}")
    else:
        logging.info("[AAI] Transcript completed successfully.")

    words = []
    pauses = []
    utterances = []
    try:
        raw_words = getattr(transcript_obj, 'words', []) or []
        for w in raw_words:
            words.append({'text': getattr(w, 'text', ''), 'start_s': getattr(w, 'start', 0) / 1000.0, 'end_s': getattr(w, 'end', 0) / 1000.0})
        for i in range(len(words)-1):
            gap = words[i+1]['start_s'] - words[i]['end_s']
            if gap >= Cfg.PAUSE_THRESHOLD_SECONDS:
                pauses.append({'start_s': words[i]['end_s'], 'end_s': words[i+1]['start_s'], 'duration_s': gap})
        for u in getattr(transcript_obj, 'utterances', []) or []:
            utterances.append({'speaker': getattr(u, 'speaker','spk_0'), 'start_s': getattr(u, 'start',0)/1000.0, 'end_s': getattr(u, 'end',0)/1000.0, 'text': getattr(u, 'text','')})
    except Exception as e:
        logging.warning(f"[AAI] Failed to parse transcript object: {e}")

    serialized = {'text': getattr(transcript_obj, 'text', None), 'words': words, 'utterances': utterances, 'pauses': pauses}

    try:
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(serialized, fh)
        logging.info(f"[AAI] Cached transcript to {cache_path} (words={len(words)}, pauses={len(pauses)}, utterances={len(utterances)})")
    except Exception as e:
        logging.warning(f"[AAI] Failed to write cache: {e}")

    # Log a short transcript snippet for sanity
    txt = serialized.get('text') or ''
    sample = txt[:300].replace('\n',' ') if txt else ''
    logging.info(f"[AAI] Transcript sample: {sample!r} (len={len(txt)})")

    return serialized

# ------------------------------------------------------------------------------
# Write SRT from AssemblyAI transcript — with lifecycle logging
# ------------------------------------------------------------------------------
def write_srt_from_transcript_with_logging(transcript_data: Dict[str, Any], srt_out: str, max_chars_per_line: int = 80) -> str:
    """
    Write SRT from transcript_data (words list). Log before, during, and after file creation.
    """
    srt_path = Path(srt_out)
    logging.info(f"[SRT] Preparing to write SRT to: {srt_path.resolve()}")

    words = transcript_data.get('words', [])
    if not words:
        logging.warning("[SRT] No words in transcript_data. Creating a minimal empty SRT.")
        subs = pysrt.SubRipFile()
        subs.append(pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(0,0,0,0), end=pysrt.SubRipTime(0,0,0,500), text=""))
        subs.save(str(srt_path), encoding='utf-8')
        # Post-write checks
        exists = srt_path.exists()
        size = srt_path.stat().st_size if exists else 0
        logging.info(f"[SRT] Wrote minimal SRT. exists={exists}, size={size} bytes")
        return str(srt_path)

    # Log words count and a sample
    logging.info(f"[SRT] Transcript contains {len(words)} words. Sample: {words[:5]}")

    # grouping into subtitle chunks
    subs_chunks = []
    current_chunk = []
    chunk_start = words[0]['start_s']
    chunk_end = words[0]['end_s']
    char_count = 0
    max_gap = 1.0

    for w in words:
        w_text = w['text'].strip()
        gap = w['start_s'] - chunk_end
        if (char_count + len(w_text) + 1 > max_chars_per_line) or (gap > max_gap and current_chunk):
            subs_chunks.append((chunk_start, chunk_end, " ".join(current_chunk).strip()))
            current_chunk = []
            char_count = 0
            chunk_start = w['start_s']
        current_chunk.append(w_text)
        char_count += len(w_text) + 1
        chunk_end = w['end_s']
    if current_chunk:
        subs_chunks.append((chunk_start, chunk_end, " ".join(current_chunk).strip()))

    logging.info(f"[SRT] Built {len(subs_chunks)} subtitle chunks from words.")

    # Write via pysrt
    try:
        srt_obj = pysrt.SubRipFile()
        for i, (st, et, txt) in enumerate(subs_chunks, start=1):
            start_time = pysrt.SubRipTime(0,0,0,0)
            end_time = pysrt.SubRipTime(0,0,0,0)
            # convert floats to h:m:s,ms
            st_t = seconds_to_time(st)
            et_t = seconds_to_time(et + 0.3)
            srt_obj.append(pysrt.SubRipItem(index=i,
                                            start=pysrt.SubRipTime(st_t.hour, st_t.minute, st_t.second, st_t.microsecond//1000),
                                            end=pysrt.SubRipTime(et_t.hour, et_t.minute, et_t.second, et_t.microsecond//1000),
                                            text=txt))
        srt_obj.save(str(srt_path), encoding='utf-8')
        logging.info(f"[SRT] Successfully wrote SRT to {srt_path}")
    except Exception as e:
        logging.error(f"[SRT] Failed to write SRT: {e}")
        raise

    # Post-write validations and logs
    try:
        exists = srt_path.exists()
        size = srt_path.stat().st_size if exists else 0
        lines = 0
        if exists:
            with open(srt_path, 'r', encoding='utf-8') as fh:
                raw = fh.read()
                lines = raw.count('\n')
        logging.info(f"[SRT] Post-write: exists={exists}, size={size} bytes, approx_lines={lines}")
        if size < 200:
            logging.warning(f"[SRT] SRT file size small ({size} bytes); transcript may be incomplete.")
    except Exception as e:
        logging.warning(f"[SRT] Post-write validation failed: {e}")

    return str(srt_path)

# ------------------------------------------------------------------------------
# SRT parsing to sentence entries with safety checks and logging
# ------------------------------------------------------------------------------
def srt_to_sentence_entries_with_logging(srt_path: str) -> List[Dict[str, Any]]:
    logging.info(f"[PARSE] Parsing SRT for sentence mapping: {srt_path}")
    srt_file = Path(srt_path)
    if not srt_file.exists():
        logging.error(f"[PARSE] SRT path does not exist: {srt_path}")
        return []

    try:
        subs = pysrt.open(srt_path, encoding='utf-8')
    except Exception as e:
        logging.error(f"[PARSE] Failed to open SRT: {e}")
        return []

    # Quick sanity: if empty or tiny, log and return
    overall_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
    if not overall_text or len(overall_text.strip()) < 3:
        logging.warning(f"[PARSE] SRT appears empty or too short ({len(overall_text)} chars).")
        return []

    logging.info(f"[PARSE] SRT contains {len(subs)} subtitle entries; total_chars={len(overall_text)}")

    # Build char -> time mapping
    char_to_time = {}
    cursor = 0
    for sub in subs:
        txt = sub.text_without_tags.replace('\n', ' ') + ' '
        for i in range(len(txt)):
            char_to_time[cursor + i] = (sub.start.to_time(), sub.end.to_time())
        cursor += len(txt)

    models = get_models()
    nlp = models['spacy']

    # Pass through spaCy — guard against empty input
    try:
        doc = nlp(overall_text)
    except Exception as e:
        logging.error(f"[PARSE] spaCy failed on SRT text: {e}")
        return []

    sentence_entries = []
    for sent in doc.sents:
        start_char = sent.start_char
        end_char = max(sent.end_char - 1, sent.start_char)
        if start_char in char_to_time and end_char in char_to_time:
            s_time = char_to_time[start_char][0]
            e_time = char_to_time[end_char][1]
            sentence_entries.append({'text': sent.text.strip(), 'start': s_time, 'end': e_time})
    logging.info(f"[PARSE] Extracted {len(sentence_entries)} sentence entries from SRT.")
    return sentence_entries

# ------------------------------------------------------------------------------
# Semantic segmentation (centroid + drift) and fallback
# ------------------------------------------------------------------------------
def compute_embeddings_for_sentences(sentence_entries: List[Dict[str, Any]]) -> np.ndarray:
    st = get_models()['st_model']
    texts = [s['text'] for s in sentence_entries]
    if not texts:
        return np.zeros((0, st.get_sentence_embedding_dimension() if hasattr(st, 'get_sentence_embedding_dimension') else 768))
    return st.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def detect_semantic_boundaries(sentence_entries: List[Dict[str, Any]],
                               min_coherence_time: float = 2.0,
                               drift_threshold: float = 0.35,
                               require_drift_count: int = 1) -> List[int]:
    if not sentence_entries:
        return [0, 0]
    embeddings = compute_embeddings_for_sentences(sentence_entries)
    if embeddings.shape[0] == 0:
        logging.warning("[SEM] Embeddings empty; falling back to naive boundaries.")
        return [0, len(sentence_entries)]

    n = len(sentence_entries)
    boundaries = {0}
    centroid = None
    current_start_idx = 0
    drift_counts = 0

    closure_re = re.compile(r'(in conclusion|to conclude|therefore|thus|finally|so that|that means)', re.I)
    discourse_flags = [bool(closure_re.search(s['text']) or s['text'].strip().endswith(('.', '?', '...'))) for s in sentence_entries]

    for i in range(n):
        emb = embeddings[i]
        if centroid is None:
            centroid = emb.copy()
            drift = 0.0
        else:
            try:
                sim = float(cosine_similarity([centroid], [emb])[0][0])
                drift = 1.0 - sim
            except Exception:
                drift = 0.0

        seg_start = sentence_entries[current_start_idx]['start']
        seg_end = sentence_entries[i]['end']
        seg_length = time_to_seconds(seg_end) - time_to_seconds(seg_start)

        if drift >= drift_threshold:
            drift_counts += 1
        else:
            drift_counts = max(0, drift_counts - 1)

        strong_drift = drift_counts >= require_drift_count
        min_size_ok = seg_length >= min_coherence_time

        if strong_drift and (discourse_flags[i] or min_size_ok):
            boundaries.add(i + 1)
            centroid = None
            current_start_idx = i + 1
            drift_counts = 0
            if current_start_idx < n:
                centroid = embeddings[current_start_idx].copy()
        else:
            centroid = (0.82 * centroid + 0.18 * emb) if centroid is not None else emb.copy()

    boundaries.add(n)
    b = sorted(list(boundaries))

    logging.info(f"[SEM] Detected {len(b)-1} segments.")
    return b

# ------------------------------------------------------------------------------
# Candidate generation honoring boundaries + speaker segments + pauses
# ------------------------------------------------------------------------------
def generate_candidates(sentence_entries: List[Dict[str, Any]],
                        boundaries: List[int],
                        pauses: List[Dict[str, Any]],
                        speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cfg = Cfg()
    candidates = []
    if not sentence_entries:
        return candidates

    n = len(sentence_entries)
    i = 0
    while i < n:
        # find next boundary greater than i
        j = next((b for b in boundaries if b > i), n)
        seg_start_idx = i
        seg_end_idx = j - 1 if j - 1 < n else n - 1

        seg_start = sentence_entries[seg_start_idx]['start']
        seg_end = sentence_entries[seg_end_idx]['end']
        seg_dur = time_to_seconds(seg_end) - time_to_seconds(seg_start)

        # expand if too short
        while seg_dur < cfg.MIN_DURATION_SECONDS and seg_end_idx + 1 < n:
            seg_end_idx += 1
            seg_end = sentence_entries[seg_end_idx]['end']
            seg_dur = time_to_seconds(seg_end) - time_to_seconds(seg_start)

        # clamp to MAX_DURATION by splitting inside if needed
        if seg_dur > cfg.MAX_DURATION_SECONDS:
            # greedily step forward until near max
            k = seg_start_idx
            acc_end = sentence_entries[k]['end']
            while k < n and (time_to_seconds(acc_end) - time_to_seconds(seg_start)) < cfg.MAX_DURATION_SECONDS:
                k += 1
                if k < n:
                    acc_end = sentence_entries[k]['end']
            seg_end_idx = min(seg_end_idx, k)

        preview = " ".join(sentence_entries[s]['text'] for s in range(seg_start_idx, seg_end_idx+1)).strip()
        candidates.append({
            'Preview': preview,
            'Start': sentence_entries[seg_start_idx]['start'],
            'End': sentence_entries[seg_end_idx]['end'],
            'Duration': time_to_seconds(sentence_entries[seg_end_idx]['end']) - time_to_seconds(sentence_entries[seg_start_idx]['start']),
            'SentIdxSpan': (seg_start_idx, seg_end_idx)
        })
        i = seg_end_idx + 1

    logging.info(f"[CAND] Generated {len(candidates)} candidate segments.")
    return candidates

# ------------------------------------------------------------------------------
# Trailing-sentence validator (extend/backtrack logic) with logging
# ------------------------------------------------------------------------------
def is_sentence_trailing(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if s.endswith('...'):
        return True
    if s.endswith(',') or s.endswith(';') or s.endswith(':'):
        return True
    if s.endswith('?') or s.endswith('!'):
        return False
    words = [w.strip(string.punctuation).lower() for w in s.split() if w.strip(string.punctuation)]
    if not words:
        return False
    last = words[-1]
    if last in Cfg.TRAILING_CONJUNCTIONS or last in Cfg.TRAILING_SUBORDINATORS or last in Cfg.UNCERTAIN_ENDING_WORDS:
        return True
    if Cfg.TRAILING_PATTERN.search(s):
        return True
    return False

def validate_segment_ending(candidate: Dict[str, Any], sentence_entries: List[Dict[str, Any]],
                            pauses: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = {'action': 'none', 'reason': None}
    idx_a, idx_b = candidate['SentIdxSpan']
    last_text = sentence_entries[idx_b]['text']
    trailing = is_sentence_trailing(last_text)
    logging.debug(f"[VALIDATE] Candidate spans {idx_a}-{idx_b}. Last sentence trailing={trailing}")

    if not trailing:
        meta['reason'] = 'not trailing'
        candidate['_validate_meta'] = meta
        return candidate

    # if trailing and next sentence exists, examine semantic sim and pause
    if idx_b + 1 >= len(sentence_entries):
        # backtrack if possible
        if idx_b - 1 >= idx_a:
            new_b = idx_b - 1
            candidate['SentIdxSpan'] = (idx_a, new_b)
            candidate['End'] = sentence_entries[new_b]['end']
            candidate['Duration'] = time_to_seconds(candidate['End']) - time_to_seconds(candidate['Start'])
            meta['action'] = 'backtrack'
            meta['reason'] = 'trailing and no next sentence; backtracked 1'
            candidate['_validate_meta'] = meta
            logging.info(f"[VALIDATE] Backtracked candidate to end at sentence {new_b}")
            return candidate
        meta['reason'] = 'trailing but no alt; kept'
        candidate['_validate_meta'] = meta
        return candidate

    next_text = sentence_entries[idx_b + 1]['text']
    try:
        st = get_models()['st_model']
        embs = st.encode([last_text, next_text], show_progress_bar=False, convert_to_numpy=True)
        sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
    except Exception as e:
        logging.debug(f"[VALIDATE] Embedding similarity failed: {e}")
        sim = 0.0

    last_end_s = time_to_seconds(sentence_entries[idx_b]['end'])
    pause_after = None
    for p in pauses:
        if p['start_s'] >= last_end_s - 0.05 and p['start_s'] <= last_end_s + 1.0:
            pause_after = p
            break

    logging.debug(f"[VALIDATE] sim={sim:.3f}, pause_after={pause_after}")

    if sim >= 0.55:
        # extend to next
        new_b = idx_b + 1
        candidate['SentIdxSpan'] = (idx_a, new_b)
        candidate['End'] = sentence_entries[new_b]['end']
        candidate['Duration'] = time_to_seconds(candidate['End']) - time_to_seconds(candidate['Start'])
        meta['action'] = 'extend'
        meta['reason'] = f'semantic sim {sim:.2f} => extended'
        candidate['_validate_meta'] = meta
        logging.info(f"[VALIDATE] Extended candidate to include sentence {new_b} (sim={sim:.2f})")
        return candidate

    if pause_after and pause_after.get('duration_s', 0.0) >= 0.6:
        meta['action'] = 'keep'
        meta['reason'] = f'long pause ({pause_after.get("duration_s"):.2f}s) => keep end'
        candidate['_validate_meta'] = meta
        logging.info("[VALIDATE] Keeping candidate because of long pause after trailing sentence.")
        return candidate

    # else backtrack if possible
    if idx_b - 1 >= idx_a:
        new_b = idx_b - 1
        candidate['SentIdxSpan'] = (idx_a, new_b)
        candidate['End'] = sentence_entries[new_b]['end']
        candidate['Duration'] = time_to_seconds(candidate['End']) - time_to_seconds(candidate['Start'])
        meta['action'] = 'backtrack'
        meta['reason'] = f'trailing & low sim ({sim:.2f}) & no long pause => backtracked'
        candidate['_validate_meta'] = meta
        logging.info(f"[VALIDATE] Backtracked candidate to end at sentence {new_b}")
        return candidate

    meta['action'] = 'keep'
    meta['reason'] = 'fallback keep'
    candidate['_validate_meta'] = meta
    return candidate

# ------------------------------------------------------------------------------
# Endpoint refinement: audio silence/prosody and frame snapping
# ------------------------------------------------------------------------------
def snap_time_to_frame(t: float, fps: float) -> float:
    if fps <= 0:
        return t
    frame = round(t * fps)
    return frame / fps

def detect_silence(audio_path: str, start_search: float, max_search: float = 1.0, silence_thresh_ratio: float = 0.06, min_silence_duration: float = 0.15) -> Optional[float]:
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=max(0.0, start_search - 0.2), duration=max_search + 0.5)
        rms = librosa.feature.rms(y=y)[0]
        if rms.size == 0:
            return None
        median = float(np.median(rms))
        thresh = median * silence_thresh_ratio
        duration = len(y) / sr
        frame_dur = duration / max(1, rms.size)
        times = np.arange(rms.size) * frame_dur + max(0.0, start_search - 0.2)
        low = rms < thresh
        min_frames = max(1, int(math.ceil(min_silence_duration / frame_dur)))
        run = 0
        for i, v in enumerate(low):
            if v:
                run += 1
                if run >= min_frames:
                    return float(times[i - run + 1])
            else:
                run = 0
        return None
    except Exception as e:
        logging.debug(f"[SILENCE] detection failed: {e}")
        return None

def refine_endpoint(candidate: Dict[str, Any], audio_path: Optional[str], fps: float = 30.0, padding: float = 0.06) -> Dict[str, Any]:
    end_s = time_to_seconds(candidate['End'])
    chosen = end_s + padding
    if audio_path and Path(audio_path).exists():
        silence = detect_silence(audio_path, end_s, max_search=1.0)
        if silence is not None:
            chosen = min(end_s + 1.0, silence + 0.02)
            logging.debug(f"[REFINE] Silence at {silence:.3f}s, choosing {chosen:.3f}s")
    snapped = snap_time_to_frame(chosen, fps)
    candidate['_final_end_s'] = float(snapped)
    candidate['End'] = seconds_to_time(snapped)
    candidate['Duration'] = time_to_seconds(candidate['End']) - time_to_seconds(candidate['Start'])
    return candidate

# ------------------------------------------------------------------------------
# Precompute features and scoring (threaded)
# ------------------------------------------------------------------------------
def precompute_and_score(candidates: List[Dict[str, Any]], audio_path: Optional[str], top_k: int = 6, num_workers: int = 4) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    models = get_models()
    st = models['st_model']
    nlp = models['spacy']
    kw_model = models.get('keybert')
    emotion_pipe = models.get('emotion')
    topic_model = models.get('bertopic')

    texts = [c['Preview'] for c in candidates]
    embeddings = st.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    topics = [-1] * len(texts)
    probs = [None] * len(texts)
    if topic_model is not None:
        try:
            topics, probs = topic_model.fit_transform(texts, embeddings)
        except Exception as e:
            logging.debug(f"[BERTopic] fit_transform failed: {e}")

    for i, c in enumerate(candidates):
        c['TopicID'] = int(topics[i]) if topics is not None else -1
        try:
            c['TopicScore'] = float(np.max(probs[i])) if probs and probs[i] is not None else 0.0
        except Exception:
            c['TopicScore'] = 0.0

    try:
        emo_results = emotion_pipe(texts, truncation=True, batch_size=16) if emotion_pipe is not None else [ [{'label':'neutral','score':0.0}] for _ in texts ]
    except Exception:
        emo_results = [ [{'label':'neutral','score':0.0}] for _ in texts ]

    for i, emo in enumerate(emo_results):
        try:
            if isinstance(emo, list):
                candidates[i]['EmotionScore'] = float(max(x.get('score',0.0) for x in emo))
            else:
                candidates[i]['EmotionScore'] = float(emo.get('score',0.0))
        except Exception:
            candidates[i]['EmotionScore'] = 0.0

    # keywords & entities
    for i, text in enumerate(texts):
        try:
            if kw_model is not None:
                kw = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=6)
                candidates[i]['Keywords'] = [w for w, s in kw]
            else:
                candidates[i]['Keywords'] = []
        except Exception:
            candidates[i]['Keywords'] = []
        try:
            doc = nlp(text)
            candidates[i]['EntityCount'] = sum(1 for ent in doc.ents if ent.label_ not in ("CARDINAL","ORDINAL"))
        except Exception:
            candidates[i]['EntityCount'] = 0

    # scoring worker
    def score_single(arg):
        idx, preview, duration, topic_score, keywords, topic_id, emotion_score, entity_count = arg
        cfg = Cfg()
        table = str.maketrans('', '', string.punctuation)
        words = [w.lower().translate(table) for w in preview.split()]
        topical_overlap = sum(1 for kw in keywords if isinstance(kw, str) and kw.lower() in preview.lower())
        hook_score = sum(1 for pat in cfg.HOOK_PATTERNS if pat.search(preview))
        brevity = 1.0 if cfg.MIN_DURATION_SECONDS <= duration <= 30 else 0.6
        score = float(topic_score) * 1.0 + float(emotion_score) * 0.3 + brevity * 0.5 + hook_score * 0.2 + topical_overlap * 0.1
        return {'Idx': idx, 'Score': float(score)}

    args = []
    for i, c in enumerate(candidates):
        args.append((i, c['Preview'], c.get('Duration',0.0), c.get('TopicScore',0.0), c.get('Keywords',[]), c.get('TopicID',-1), c.get('EmotionScore',0.0), c.get('EntityCount',0)))

    scores = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = [ex.submit(score_single, a) for a in args]
        for fut in as_completed(futures):
            try:
                scores.append(fut.result())
            except Exception as e:
                logging.warning(f"[SCORE] Worker failed: {e}")

    for s in scores:
        idx = s['Idx']
        candidates[idx]['Score'] = s['Score']

    return candidates, embeddings

# ------------------------------------------------------------------------------
# Redundancy filtering and selection
# ------------------------------------------------------------------------------
def filter_redundant_and_select(candidates: List[Dict[str, Any]], embeddings: np.ndarray, top_k: int = 6) -> List[Dict[str, Any]]:
    items = sorted(candidates, key=lambda x: x.get('Score', 0.0), reverse=True)
    final = []
    for it in items:
        emb_idx = it.get('SentIdxSpan', (0,0))[0]
        if not (0 <= emb_idx < embeddings.shape[0]):
            final.append(it)
            if len(final) >= top_k:
                break
            continue
        emb_vec = embeddings[emb_idx]
        redundant = False
        for kept in final:
            kidx = kept.get('SentIdxSpan', (0,0))[0]
            if 0 <= kidx < embeddings.shape[0]:
                sim = float(cosine_similarity([emb_vec], [embeddings[kidx]])[0][0])
                if sim > Cfg.REDUNDANCY_SIMILARITY_THRESHOLD:
                    redundant = True
                    break
        if not redundant:
            final.append(it)
        if len(final) >= top_k:
            break
    logging.info(f"[SELECT] Selected {len(final)} non-redundant clips.")
    return final

# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def generate_html_report(clips: List[Dict[str, Any]], csv_path: Path, out_path: str = 'viral_clips_report.html'):
    rows = []
    for i, c in enumerate(clips):
        rows.append({
            'Rank': i+1,
            'Score': f"{c.get('Score',0.0):.4f}",
            'Start': format_time(c['Start']),
            'End': format_time(c['End']),
            'Duration': f"{c['Duration']:.2f}s",
            'Preview': c['Preview'],
            'ValidateMeta': json.dumps(c.get('_validate_meta', {}))
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    html_rows = []
    for _, r in df.iterrows():
        html_rows.append("<tr>" + "".join(f"<td>{r[col]}</td>" for col in df.columns) + "</tr>")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Viral Clips</title></head><body>
    <h1>Viral Clips</h1><p>Generated: {datetime.utcnow().isoformat()} UTC</p>
    <p>CSV: {csv_path.name}</p><table border='1'><tr>{''.join(f'<th>{c}</th>' for c in df.columns)}</tr>{''.join(html_rows)}</table></body></html>"""
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    logging.info(f"[REPORT] Saved HTML report to {out_path}")

# ------------------------------------------------------------------------------
# Main pipeline: process_file with detailed logging around SRT lifecycle
# ------------------------------------------------------------------------------
def process_file(media_path: str,
                 srt_path: Optional[str] = None,
                 top_k: int = 6,
                 make_srt_if_missing: bool = True,
                 assemblyai_force_refresh: bool = False,
                 refine_endings: bool = True,
                 num_workers: int = 4,
                 fps: float = 30.0) -> Optional[pd.DataFrame]:
    logging.info(f"[PROCESS] Started processing media: {media_path}")
    models = get_models()  # ensure models loaded once
    logging.info("[PROCESS] SRT and audio files are being generated.")
    # determine srt path to use or create
    if srt_path:
        srt_target = Path(srt_path)
    else:
        srt_target = Path(media_path).with_suffix('.srt')

    logging.info(f"[PROCESS] Target SRT path: {srt_target.resolve()}")

    temp_audio = None
    audio_for_transcript = None

    # If SRT not present and allowed, create via AssemblyAI
    if not srt_target.exists() and make_srt_if_missing:
        logging.info("[PROCESS] SRT missing; creating from media via AssemblyAI.")
        # extract audio
        if is_video_file(media_path):
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            ok, meta = extract_audio_ffmpeg(media_path, temp_audio.name)
            if not ok:
                logging.error("[PROCESS] Audio extraction failed; aborting SRT creation.")
                if temp_audio:
                    try: os.unlink(temp_audio.name)
                    except Exception: pass
                return None
            audio_for_transcript = temp_audio.name
            logging.info(f"[PROCESS] Audio extraction meta: {meta}")
        else:
            audio_for_transcript = media_path

        # log audio existence
        if audio_for_transcript and Path(audio_for_transcript).exists():
            size = Path(audio_for_transcript).stat().st_size
            logging.info(f"[PROCESS] Audio ready for transcription: {audio_for_transcript} (size={size} bytes)")
        else:
            logging.error("[PROCESS] No audio available for transcription; aborting.")
            return None

        # run AssemblyAI transcription (single call)
        transcript = transcribe_and_cache(audio_for_transcript, force=assemblyai_force_refresh)

        # log transcript summary
        txt = transcript.get('text') or ''
        words = transcript.get('words', [])
        pauses = transcript.get('pauses', [])
        uttrs = transcript.get('utterances', [])
        logging.info(f"[PROCESS] Transcription summary: words={len(words)}, pauses={len(pauses)}, utterances={len(uttrs)}")
        if txt:
            logging.info(f"[PROCESS] Transcript preview: {txt[:300]}")
        else:
            logging.warning("[PROCESS] Transcript text is empty. The resulting SRT may be minimal or empty.")

        # write srt with logging
        try:
            srt_written = write_srt_from_transcript_with_logging(transcript, str(srt_target))
            # verify file immediately
            if Path(srt_written).exists():
                size = Path(srt_written).stat().st_size
                logging.info(f"[PROCESS] SRT written at {srt_written} (size={size} bytes)")
                if size < 200:
                    logging.warning(f"[PROCESS] SRT file small ({size} bytes) — verify transcript content.")
            else:
                logging.error("[PROCESS] SRT write reported success but file not found.")
                return None
        except Exception as e:
            logging.error(f"[PROCESS] Failed to write SRT: {e}")
            return None

    elif not srt_target.exists():
        logging.error("[PROCESS] SRT not found and make_srt_if_missing is False. Aborting.")
        return None
    else:
        logging.info(f"[PROCESS] Using existing SRT at {srt_target}")

    # small pause to let filesystem settle (avoid race)
    time_module.sleep(0.04)

    # parse srt to sentence entries with logging
    sentence_entries = srt_to_sentence_entries_with_logging(str(srt_target))
    if not sentence_entries:
        logging.error("[PROCESS] No sentence entries parsed from SRT — aborting pipeline.")
        # if temporary audio created, remove
        if temp_audio:
            try: os.unlink(temp_audio.name)
            except Exception: pass
        return None

    # get transcript audio features if not available
    if 'transcript' not in locals():
        # try to retrieve cached transcript if exists
        if audio_for_transcript:
            transcript = transcribe_and_cache(audio_for_transcript, force=assemblyai_force_refresh)
        else:
            transcript = {'words': [], 'pauses': [], 'utterances': []}

    audio_words = transcript.get('words', [])
    audio_pauses = transcript.get('pauses', [])
    audio_utterances = transcript.get('utterances', [])

    # semantic boundary detection
    boundaries = detect_semantic_boundaries(sentence_entries)
    logging.info(f"[PROCESS] Semantic boundaries indices: {boundaries}")

    # candidate generation
    candidates = generate_candidates(sentence_entries, boundaries, audio_pauses, audio_utterances)
    if not candidates:
        logging.error("[PROCESS] No candidate segments generated; aborting.")
        if temp_audio:
            try: os.unlink(temp_audio.name)
            except Exception: pass
        return None

    # annotate next previews for refinement if desired
    for idx, c in enumerate(candidates):
        c['NextPreview'] = candidates[idx+1]['Preview'] if idx+1 < len(candidates) else ''

    # validate endings
    validated = []
    for c in candidates:
        v = validate_segment_ending(c, sentence_entries, audio_pauses, audio_words)
        validated.append(v)

    # refine endpoints with audio silence & snapping
    if refine_endings:
        for i, c in enumerate(validated):
            try:
                validated[i] = refine_endpoint(c, audio_for_transcript or (media_path if not is_video_file(media_path) else None), fps=fps)
            except Exception as e:
                logging.warning(f"[PROCESS] Endpoint refinement failed for candidate {i}: {e}")

    # precompute features and score
    scored, embeddings = precompute_and_score(validated, audio_for_transcript or None, top_k=top_k, num_workers=num_workers)

    # redundancy filtering & selection
    selected = filter_redundant_and_select(scored, embeddings, top_k)

    # write CSV & HTML report
    csv_path = Path('viral_clips.csv')
    generate_html_report(selected, csv_path, out_path='viral_clips_report.html')

    # produce DataFrame to return
    rows = []
    for rank, c in enumerate(selected, start=1):
        rows.append({
            'Rank': rank,
            'Score': f"{c.get('Score',0.0):.4f}",
            'Start': format_time(c['Start']),
            'End': format_time(c['End']),
            'Duration': f"{c['Duration']:.2f}s",
            'Preview': c['Preview'],
            'ValidateMeta': c.get('_validate_meta', {})
        })
    df = pd.DataFrame(rows)

    # cleanup temp audio
    if temp_audio:
        try:
            os.unlink(temp_audio.name)
            logging.info(f"[CLEANUP] Removed temp audio {temp_audio.name}")
        except Exception:
            pass

    logging.info("[PROCESS] Completed processing pipeline successfully.")
    return df

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
__all__ = [
    'process_file',
    'get_models',
    'transcribe_and_cache',
    'write_srt_from_transcript_with_logging'
]

# End of executioner.py
