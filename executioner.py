#!/usr/bin/env python3
"""
executioner.py — Full-production Video Segmenter
- SRT parsing
- Speaker-aware segmentation (AssemblyAI diarization if API key supplied)
- Sentence chunking & boundary detection via embeddings
- Candidate clip generation with timestamps/duration (respects speaker changes)
- BERTopic run once (per transcript) -> topic ids for each candidate
- Emotion classification (batched) via HF pipeline
- Audio parsing (AssemblyAI) for pause detection (optional)
- Prosody extraction (librosa) for pitch/loudness
- Pause, Pace, Silence Ratio, and Filler Word Detection (approx)
- Keyword extraction (KeyBERT) and NER (spaCy)
- Multiprocess scoring (lightweight) using precomputed features
- Redundancy filtering via embeddings
- Top-k non-overlapping selection and CSV + HTML visual report output

USAGE:
- Install required packages:
  pip install sentence-transformers bertopic keybert spacy pysrt assemblyai transformers sklearn tqdm pandas numpy librosa soundfile
  python -m spacy download en_core_web_trf
- Export ASSEMBLYAI_API_KEY if you want diarization/pause detection
- Run: python executioner.py <file.srt> [file.mp3|file.mp4] [top_k]

Notes:
- If you pass a video file (e.g., .mp4) and no audio file, ffmpeg will be used to extract a temporary WAV for prosody analysis.
- Hook model: keeps your sentiment placeholder but supports overriding via HOOK_MODEL env var later.
"""

import os
import re
import sys
import logging
import string
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

import pysrt
import numpy as np
import pandas as pd
from tqdm import tqdm

# external libs for audio + NLP + ML
import assemblyai as aai
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# audio analysis
import librosa
from dotenv import load_dotenv
load_dotenv()
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
        'speech_delivery': 0.1,
        'prosody': 0.25
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
# Utils
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# -------------------------
# Model Loading (one-time init)
# -------------------------
def load_models():
    global nlp, embed_model, kw_model, emotion_pipeline, hook_classifier

    print("🔄 Loading NLP and ML models (once at startup)...")

    try:
        nlp = spacy.load('en_core_web_trf')
    except Exception:
        raise RuntimeError("spaCy model en_core_web_trf not found. Run: python -m spacy download en_core_web_trf")

    # embeddings
    EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'all-mpnet-base-v2')
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # keybert
    kw_model = KeyBERT(model=embed_model)

    # emotion pipeline
    emotion_pipeline = hf_pipeline(
        "text-classification",
        model=os.getenv('EMOTION_MODEL', 'j-hartmann/emotion-english-distilroberta-base'),
        top_k=None
    )

    # hook model or fallback
    HOOK_MODEL = os.getenv('HOOK_MODEL', None)
    if HOOK_MODEL:
        try:
            hook_tokenizer = AutoTokenizer.from_pretrained(HOOK_MODEL)
            hook_model = AutoModelForSequenceClassification.from_pretrained(HOOK_MODEL)
            hook_classifier = hf_pipeline('text-classification', model=hook_model, tokenizer=hook_tokenizer, top_k=None)
            logging.info(f"✅ Loaded hook model from {HOOK_MODEL}")
        except Exception as e:
            logging.warning(f"Failed to load HOOK_MODEL {HOOK_MODEL}: {e}. Using fallback sentiment model.")
            hook_classifier = hf_pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=1
            )
    else:
        hook_classifier = hf_pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=1
        )

    print("✅ Models loaded successfully and ready for inference.\n")

def extract_audio_ffmpeg(input_path: str, out_wav: str) -> bool:
    """
    Extracts audio to wav using ffmpeg. Returns True on success.
    Requires ffmpeg in PATH.
    """
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-vn', '-ac', '1', '-ar', '16000', '-f', 'wav', str(out_wav)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logging.warning(f"ffmpeg audio extraction failed: {e}")
        return False

# -------------------------
# Model loading (main process)
# -------------------------
print("Loading models (this may take a bit)...")

# spaCy
try:
    nlp = spacy.load('en_core_web_trf')
except Exception:
    raise RuntimeError("spaCy model en_core_web_trf not found. Run: python -m spacy download en_core_web_trf")

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

# Hook classifier: KEEP your sentiment placeholder but allow override via HOOK_MODEL env var
HOOK_MODEL = os.getenv('HOOK_MODEL', None)
if HOOK_MODEL:
    try:
        hook_tokenizer = AutoTokenizer.from_pretrained(HOOK_MODEL)
        hook_model = AutoModelForSequenceClassification.from_pretrained(HOOK_MODEL)
        hook_classifier = hf_pipeline('text-classification', model=hook_model, tokenizer=hook_tokenizer, top_k=None)
        logging.info(f"Loaded hook model from {HOOK_MODEL}")
    except Exception as e:
        logging.warning(f"Failed to load HOOK_MODEL {HOOK_MODEL}: {e}. Falling back to default sentiment classifier.")
        hook_classifier = hf_pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=1
        )
else:
    hook_classifier = hf_pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        top_k=1
    )

FILLERS = {"uh", "um", "erm", "uhh", "umm", "like", "you know", "ah", "eh"}

# -------------------------
# AssemblyAI helpers (pauses + diarization)
# -------------------------
def get_audio_features_from_assemblyai(
    audio_path: str, 
    api_key: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """
    Uses AssemblyAI to transcribe and return both pause windows and speaker segments.
    
    Returns:
        Dict[str, List[Dict]]: {'pauses': List, 'speakers': List}
    """
    
    # 1. API Key Check (using the correct env var name)
    key = api_key or os.getenv('ASSEMBLYAI_API_KEY')
    if not key:
        logging.warning('No AssemblyAI API key supplied; skipping audio feature detection.')
        return {'pauses': [], 'speakers': []}

    aai.settings.api_key = key
    transcriber = aai.Transcriber()
    
    # 2. Merged Configuration
    config = aai.TranscriptionConfig(
        speaker_labels=True,  # Enables speaker diarization
        audio_start_from=0, 
        audio_end_at=None
    )
    
    logging.info('Submitting audio to AssemblyAI for pause and speaker detection (this may take a moment)...')
    transcript = transcriber.transcribe(audio_path, config=config)

    # Initialize results
    pauses: List[Dict] = []
    speakers: List[Dict] = []

    if getattr(transcript, 'status', None) == 'completed':
        
        # --- Pause Detection (Pauses between words) ---
        words = getattr(transcript, 'words', []) or []
        for i in range(len(words) - 1):
            w = words[i]
            w_next = words[i + 1]
            gap_ms = w_next.start - w.end
            gap_s = gap_ms / 1000.0
            
            # Pause threshold remains at 500ms
            if gap_s >= 0.5:
                pauses.append({
                    'start_s': w.end / 1000.0,
                    'end_s': w_next.start / 1000.0,
                    'duration_s': gap_s
                })

        # --- Speaker Segment Detection (Diarization) ---
        for utter in getattr(transcript, 'utterances', []):
            speakers.append({
                'speaker': getattr(utter, 'speaker', 'spk_unknown'),
                'start_s': getattr(utter, 'start', 0) / 1000.0,
                'end_s': getattr(utter, 'end', 0) / 1000.0
            })
            
        logging.info(f'Detected {len(pauses)} pauses and {len(speakers)} speaker segments from AssemblyAI.')
    else:
        logging.warning('AssemblyAI transcription failed or incomplete; using no audio features.')

    return {
        'pauses': pauses, 
        'speakers': speakers
    }


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
# Candidate generation (respect speaker segments)
# -------------------------
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

    BACKWARD_LEEWAY = 5   # seconds before cap to search for a natural stop
    FORWARD_LEEWAY = 10   # seconds after cap to allow finishing a sentence

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

        # if any speaker boundary inside current running window, force a flag to cut at boundary
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

    logging.info(f'Generated {len(candidates)} candidate clips (backward + forward leeway enforced).')
    return candidates

# -------------------------
# Prosody extraction (librosa)
# -------------------------
def compute_audio_prosody(audio_path: str, start_s: float, end_s: float) -> Dict[str, float]:
    try:
        duration = max(0.001, end_s - start_s)
        y, sr = librosa.load(audio_path, sr=None, offset=start_s, duration=duration)
        # pitch via piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]
        mean_pitch = float(np.mean(pitch_vals)) if pitch_vals.size > 0 else 0.0
        pitch_var = float(np.var(pitch_vals)) if pitch_vals.size > 0 else 0.0
        rms = librosa.feature.rms(y=y)[0]
        loud_mean = float(np.mean(rms))
        loud_var = float(np.var(rms))
        return {
            'mean_pitch': mean_pitch,
            'pitch_var': pitch_var,
            'loudness_mean': loud_mean,
            'loudness_var': loud_var
        }
    except Exception as e:
        logging.debug(f"Prosody analysis failed for {audio_path} {start_s}-{end_s}: {e}")
        return {'mean_pitch': 0.0, 'pitch_var': 0.0, 'loudness_mean': 0.0, 'loudness_var': 0.0}

# -------------------------
# Precompute per-candidate features
# -------------------------
def get_hook_score_ml(text: str) -> float:
    if not text or text.strip() == "":
        return 0.0
    try:
        res = hook_classifier(text[:250])
        if isinstance(res, list) and res:
            # try best guess: if labels like POS/NEG or LABEL_1 exist, use score for positive/hooky label
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
    texts = [c['Preview'] for c in candidates]

    # --- Embeddings (for redundancy + BERTopic + scoring) ---
    embeddings = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # --- BERTopic (topic clustering) ---
    topic_model = BERTopic(verbose=False)
    try:
        topics, probs = topic_model.fit_transform(texts, embeddings)
    except Exception:
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

    # --- Delivery / Prosody features (text approx) ---
    for i, c in enumerate(candidates):
        dur = float(c.get('Duration', 0.0))
        delivery = compute_delivery_features_from_text(c.get('Preview', ''), dur)
        c['DeliveryPace'] = delivery['pace']
        c['DeliveryPauseRatio'] = delivery['pause_ratio']
        c['DeliveryFillerDensity'] = delivery['filler_density']
        c['DeliveryScore'] = delivery['delivery_score']

    # --- Audio-based prosody if audio provided ---
    for i, c in enumerate(candidates):
        try:
            if audio_path and Path(audio_path).exists():
                start_s = time_to_seconds(c['Start'])
                end_s = time_to_seconds(c['End'])
                pros = compute_audio_prosody(audio_path, start_s, end_s)
                c.update(pros)
                # simple prosody scalar combining loudness & pitch variance (tunable)
                c['ProsodyScore'] = float(pros.get('loudness_mean', 0.0) * 0.4 + pros.get('pitch_var', 0.0) * 0.6)
            else:
                c['ProsodyScore'] = 0.0
                c['mean_pitch'] = 0.0
                c['pitch_var'] = 0.0
                c['loudness_mean'] = 0.0
                c['loudness_var'] = 0.0
        except Exception:
            c['ProsodyScore'] = 0.0

    # --- Virality priors (expanded) ---
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
# Scoring worker
# -------------------------
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

# -------------------------
# Redundancy / selection
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
# Visual HTML report
# -------------------------
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
    logging.info(f"Visual report saved to {Path(out_path).resolve()}")

# -------------------------
# Main pipeline
# -------------------------
def process_file(srt_path: str, media_path: Optional[str] = None, top_k: int = 10) -> Optional[pd.DataFrame]:
    load_models()
    cfg = Cfg()
    srt_path = str(srt_path)
    media_path = str(media_path) if media_path else None
    logging.info(f"Processing {srt_path}")

    # prepare audio: if media_path is video and not audio, extract wav to temp
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
            # assume it's audio already
            audio_for_prosody = media_path
    else:
        audio_for_prosody = None

    sentence_entries = load_subtitles(srt_path)
    if not sentence_entries:
        logging.warning("No sentence entries extracted.")
        return None

    boundaries = detect_boundaries(sentence_entries, cfg)

    pauses = []
    speakers = []
    if audio_for_prosody:
        audio_features = get_audio_features_from_assemblyai(audio_for_prosody)
        all_pauses = audio_features['pauses']
        all_speakers = audio_features['speakers']
    else:
        # still attempt if ASSEMBLYAI_API_KEY exists but no audio provided — warn and skip
        if os.getenv('ASSEMBLYAI_API_KEY'):
            logging.warning("ASSEMBLYAI_API_KEY is set but no audio provided; skipping AssemblyAI calls.")
        pauses = []
        speakers = []

    candidates = generate_candidates(sentence_entries, boundaries, pauses, speakers, cfg)
    if not candidates:
        logging.warning("No candidate clips produced.")
        return None

    candidates, embeddings, global_topics = precompute_features(candidates, cfg, audio_for_prosody)

    # Prepare worker args (primitives only)
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

    # Multiprocess scoring
    logging.info('Scoring candidates with multiprocessing...')
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(score_worker, worker_args), total=len(worker_args), desc='Scoring'))

    # attach timestamps back
    for r in results:
        idx = r['Idx']
        r['Start'] = candidates[idx]['Start']
        r['End'] = candidates[idx]['End']

    non_redundant = filter_redundant(results, embeddings, cfg.REDUNDANCY_SIMILARITY_THRESHOLD)
    top_clips = select_non_overlapping(non_redundant, top_k)

    # Format DataFrame & CSV
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

    # Visual report
    generate_html_report(df, out_csv, out_path='viral_clips_report.html')

    # cleanup temp audio
    if temp_audio:
        try:
            os.unlink(temp_audio.name)
        except Exception:
            pass

    return df

# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python executioner.py <file.srt> [file.mp3|file.mp4] [top_k]')
        sys.exit(1)

    srt_file = sys.argv[1]
    media = sys.argv[2] if len(sys.argv) > 2 else None
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    df_result = process_file(srt_file, media, top_k=top_k)
    if df_result is not None:
        print(df_result)
