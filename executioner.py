import re
import pysrt
import spacy
import numpy as np
import pandas as pd
import torch
import sys
import logging
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

# --- Configuration ---
class Cfg:
    MAX_DURATION_SECONDS: int = 45
    MIN_DURATION_SECONDS: int = 7

    DYNAMIC_STD_FACTOR: float = 0.5
    WINDOW_SIZE: int = 2
    RELATIVE_DROP: float = 0.15
    DYNAMIC_THRESHOLD_PERCENTILE: float = 90.0
    REDUNDANCY_SIMILARITY_THRESHOLD: float = 0.90

    WEIGHTS = {
        'topic_score': 0.8,
        'named_entity': 0.3,
        'hook': 2.0,
        'brevity': 1.0,
        'contextual_topic': 0.7,
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

# Global dictionary for models loaded in worker processes
models = {}

def safe_time_to_dt(t: time) -> datetime:
    return datetime.combine(datetime.min, t)

def init_worker():
    """Load models in each multiprocessing worker."""
    global models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models['nlp'] = spacy.load("en_core_web_sm")
    models['embed_model'] = SentenceTransformer("all-mpnet-base-v2", device=device)
    models['kw_model'] = KeyBERT(model=models['embed_model'])

def score_clip_worker(args) -> Tuple[float, str]:
    """Compute score for a clip (parallel worker)."""
    global models
    text, duration, cfg_dict, global_topics, start_t, end_t = args

    cfg = Cfg()
    cfg.__dict__.update(cfg_dict)

    words = set(text.lower().split())
    doc = models['nlp'](text)

    keywords = models['kw_model'].extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
    topic_scores = [score for _, score in keywords]
    topic_score = np.mean(topic_scores) if topic_scores else 0
    topics = ", ".join([kw for kw, _ in keywords]) if keywords else "N/A"

    contextual_topic_score = sum(1 for kw, _ in keywords if kw in global_topics)
    hook_score = sum(1 for pat in cfg.HOOK_PATTERNS if pat.search(text))
    entity_score = len(doc.ents)
    brevity_bonus = 1.0 if cfg.MIN_DURATION_SECONDS <= duration <= 15 else 0.5
    virality_boost = sum(cfg.VIRALITY_PRIORS.get(w, 0) for w in words)

    w = cfg.WEIGHTS
    final_score = (
        w['topic_score'] * topic_score +
        w['contextual_topic'] * contextual_topic_score +
        w['hook'] * hook_score +
        w['named_entity'] * entity_score +
        w['brevity'] * brevity_bonus +
        virality_boost
    )

    if any(word in words for word in cfg.BAD_WORDS):
        final_score *= 0.2

    return final_score, topics

class VideoSegmenter:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.cfg = Cfg()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nlp = spacy.load("en_core_web_sm")
        self.embed_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.kw_model = KeyBERT(model=self.embed_model)

    def _load_subtitles(self, file_path: str) -> List[Dict]:
        subs = pysrt.open(file_path, encoding='utf-8')
        full_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
        doc = self.nlp(full_text)

        char_to_time_map = {}
        char_counter = 0
        for sub in subs:
            text = sub.text_without_tags.replace("\n", " ") + " "
            for i in range(len(text)):
                char_to_time_map[char_counter + i] = (sub.start.to_time(), sub.end.to_time())
            char_counter += len(text)

        sentence_entries = []
        for sent in doc.sents:
            start_char, end_char = sent.start_char, sent.end_char - 1
            if start_char in char_to_time_map and end_char in char_to_time_map:
                sentence_entries.append({
                    "text": sent.text.strip(),
                    "start": char_to_time_map[start_char][0],
                    "end": char_to_time_map[end_char][1]
                })
        return sentence_entries

    def _compute_similarity(self, sentences: List[str]) -> np.ndarray:
        if len(sentences) <= self.cfg.WINDOW_SIZE:
            return np.array([])
        window_texts = [" ".join(sentences[i:i + self.cfg.WINDOW_SIZE]) for i in range(len(sentences)-self.cfg.WINDOW_SIZE+1)]
        embeddings = self.embed_model.encode(window_texts, device=self.device, show_progress_bar=False)
        sim = np.diag(cosine_similarity(embeddings[:-1], embeddings[1:]))
        return sim if sim.size > 0 else np.array([])

    def _detect_boundaries(self, sentence_entries: List[Dict]) -> List[int]:
        sentences = [e['text'] for e in sentence_entries]
        similarities = self._compute_similarity(sentences)
        if similarities.size == 0:
            return [0, len(sentences)]

        sim_mean, sim_std = float(np.mean(similarities)), float(np.std(similarities))
        abs_threshold = sim_mean - self.cfg.DYNAMIC_STD_FACTOR * sim_std
        boundaries = {0}

        for i in range(len(similarities)):
            is_break = (similarities[i] < abs_threshold) or (similarities[i] / max(similarities[i-1], 1e-6) < (1 - self.cfg.RELATIVE_DROP))
            look_ahead_indices = range(i + 1, min(i + 3, len(similarities)))
            if is_break and look_ahead_indices:
                look_ahead_avg = np.mean([similarities[j] for j in look_ahead_indices])
                if look_ahead_avg < abs_threshold:
                    boundaries.add(i + 1)
            elif is_break:
                boundaries.add(i + 1)

        boundaries.add(len(sentences))
        return sorted(list(boundaries))

    def _filter_redundant_clips(self, clips: List[Dict]) -> List[Dict]:
        if not clips:
            return []
        clips = sorted(clips, key=lambda x: x['Score'], reverse=True)
        final_clips = []
        embeddings = self.embed_model.encode([c['Preview'] for c in clips], show_progress_bar=False)
        for i, clip in enumerate(clips):
            is_redundant = False
            for j, final_clip in enumerate(final_clips):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > self.cfg.REDUNDANCY_SIMILARITY_THRESHOLD:
                    is_redundant = True
                    break
            if not is_redundant:
                final_clips.append(clip)
        return final_clips

    def _select_non_overlapping(self, clips: List[Dict], k: int) -> List[Dict]:
        """Select up to k non-overlapping clips using original non-overlap logic."""
        selected = []
        for clip in clips:
            overlap = any(clip['Start'] < c['End'] and c['Start'] < clip['End'] for c in selected)
            if not overlap:
                selected.append(clip)
            if len(selected) >= k:
                break
        return selected

    def process_file(self, srt_path: str, top_k: int = 10) -> Optional[pd.DataFrame]:
        logging.info(f"🚀 Starting processing for {srt_path}")
        sentence_entries = self._load_subtitles(srt_path)
        if not sentence_entries:
            return None

        boundaries = self._detect_boundaries(sentence_entries)
        full_text = " ".join(e['text'] for e in sentence_entries)
        global_keywords = self.kw_model.extract_keywords(full_text, top_n=20)
        global_topics = [kw for kw, _ in global_keywords]

        candidates = []
        for i in range(len(boundaries) - 1):
            segment = sentence_entries[boundaries[i]:boundaries[i+1]]
            for start_idx in range(len(segment)):
                for end_idx in range(start_idx, len(segment)):
                    clip_entries = segment[start_idx:end_idx+1]
                    start_t, end_t = clip_entries[0]['start'], clip_entries[-1]['end']
                    duration = (safe_time_to_dt(end_t) - safe_time_to_dt(start_t)).total_seconds()
                    if self.cfg.MIN_DURATION_SECONDS <= duration <= self.cfg.MAX_DURATION_SECONDS:
                        text = " ".join(e['text'] for e in clip_entries)
                        candidates.append((text, duration, self.cfg.__dict__, global_topics, start_t, end_t))

        if not candidates:
            logging.warning("No valid clips generated. Adjust duration settings.")
            return None

        with Pool(initializer=init_worker, processes=cpu_count()) as pool:
            results = list(tqdm(pool.map(score_clip_worker, candidates), total=len(candidates), desc="Scoring clips"))

        all_clips = []
        for idx, (score, topics) in enumerate(results):
            candidate = candidates[idx]
            all_clips.append({
                'Score': score,
                'Start': candidate[4],
                'End': candidate[5],
                'Duration': candidate[1],
                'Topics': topics,
                'Preview': candidate[0]  # needed for redundancy check
            })

        scores = [c['Score'] for c in all_clips]
        threshold = np.percentile(scores, self.cfg.DYNAMIC_THRESHOLD_PERCENTILE)
        strong_clips = [c for c in all_clips if c['Score'] >= threshold]
        strong_clips = sorted(strong_clips, key=lambda x: x['Score'], reverse=True)
        strong_clips = self._filter_redundant_clips(strong_clips)
        top_clips = self._select_non_overlapping(strong_clips, top_k)

        df_data = []
        for i, clip in enumerate(top_clips):
            df_data.append({
                'Rank': i+1,
                'Score': f"{clip['Score']:.2f}",
                'Start': clip['Start'].strftime('%H:%M:%S,%f')[:-3],
                'End': clip['End'].strftime('%H:%M:%S,%f')[:-3],
                'Duration': f"{clip['Duration']:.2f}s",
                'Topics': clip['Topics']
            })

        df = pd.DataFrame(df_data)
        df.to_csv("viral_clips.csv", index=False)
        logging.info(f"✅ Saved {len(df)} top non-overlapping clips to viral_clips.csv")
        return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_segmenter.py <yourfile.srt>")
    else:
        srt_path = sys.argv[1]
        d = VideoSegmenter()
        df = d.process_file(srt_path)
        print(df)
