import re
import pysrt
import spacy
import numpy as np
import pandas as pd
import torch
from datetime import datetime, time
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from transformers import pipeline
import logging
from bertopic import BERTopic


# --- Configuration ---
class Cfg:
    MAX_DURATION_SECONDS: int = 45
    MIN_DURATION_SECONDS: int = 7

    DYNAMIC_STD_FACTOR: float = 0.5
    TIME_GAP_THRESHOLD: float = 1.5
    WINDOW_SIZE: int = 2
    RELATIVE_DROP: float = 0.15

    DYNAMIC_THRESHOLD_PERCENTILE: float = 90.0
    TOPIC_KEYWORD_THRESHOLD: float = 0.35
    REDUNDANCY_SIMILARITY_THRESHOLD: float = 0.90
    MERGE_SIMILARITY_THRESHOLD: float = 0.65

    WEIGHTS = {
        'topic_score': 0.8,
        'sentiment': 0.5,
        'named_entity': 0.3,
        'emotion_intensity': 0.75,
        'hook': 2.0,
        'surprise': 1.5,
        'brevity': 1.0,
        'narrative_arc': 1.2,
        'explainer': 1.1,
        'cliffhanger': 1.3,
        'controversy': 0.8,
        'contextual_topic': 0.7,
        'speech_rate_boost': 0.5,
        'vocal_intensity_boost': 0.6
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
    SURPRISE_PATTERNS = [re.compile(r"\b(actually|in fact|turns out|counterintuitive|paradox|but|however)\b", re.I)]
    CLIFFHANGER_END_PATTERNS = [re.compile(r"\?$", re.M), re.compile(r"\.\.\.$", re.M)]

    MAX_TITLE_LEN = 72
    MAX_THUMB_LEN = 28


def safe_time_to_dt(t: time) -> datetime:
    return datetime.combine(datetime.min, t)


class VideoSegmenter:
    def __init__(self):
        logging.info("Initializing VideoSegmenter and loading models...")
        self._load_models()
        self.cfg = Cfg()

    def _load_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.embed_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
            self.kw_model = KeyBERT(model=self.embed_model)
            self.topic_model = BERTopic(embedding_model=self.embed_model)
            self.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None, device=0 if self.device == "cuda" else -1
            )
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            logging.error(f"Failed to load a model: {e}")
            raise

    def _load_subtitles(self, file_path: str) -> List[Dict]:
        subs = pysrt.open(file_path, encoding='utf-8')
        full_text = " ".join(sub.text_without_tags.strip().replace("\n", " ") for sub in subs)
        doc = self.nlp(full_text)

        char_to_time_map = {}
        char_counter = 0
        for sub in subs:
            text = sub.text_without_tags.strip().replace("\n", " ") + " "
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
        if len(sentences) <= self.cfg.WINDOW_SIZE: return np.array([])
        window_texts = [" ".join(sentences[i:i + self.cfg.WINDOW_SIZE]) for i in range(len(sentences) - self.cfg.WINDOW_SIZE + 1)]
        embeddings = self.embed_model.encode(window_texts, device=self.device, show_progress_bar=False)
        return np.diag(cosine_similarity(embeddings[:-1], embeddings[1:]))

    def _detect_boundaries(self, sentence_entries: List[Dict]) -> List[int]:
        sentences = [e['text'] for e in sentence_entries]
        similarities = self._compute_similarity(sentences)
        if similarities.size == 0: return [0, len(sentences)]

        sim_mean, sim_std = float(np.mean(similarities)), float(np.std(similarities))
        abs_threshold = sim_mean - self.cfg.DYNAMIC_STD_FACTOR * sim_std

        boundaries = {0, len(sentences)}
        for i in range(1, len(similarities)):
            drop = similarities[i-1] - similarities[i]
            rel_drop = drop / max(similarities[i-1], 1e-6)
            is_break = (similarities[i] < abs_threshold) or (rel_drop > self.cfg.RELATIVE_DROP)

            if is_break:
                boundaries.add(i + 1)
        return sorted(list(boundaries))

    def _score_segment(self, segment_text: str, duration: float, all_keywords: List[str], global_topics: List[str]) -> Tuple[float, Dict]:
        words = set(segment_text.lower().split())
        doc = self.nlp(segment_text)
        if all_keywords:
            topic_score = np.mean([score for _, score in all_keywords])
            topics = ", ".join([kw for kw, _ in all_keywords])
        else:
            topic_score, topics = 0.0, "N/A"

        contextual_topic_score = sum(1 for kw, _ in all_keywords if kw in global_topics)
        hook_score = sum(1 for pat in self.cfg.HOOK_PATTERNS if pat.search(segment_text))
        entity_score = len(doc.ents)
        brevity_bonus = 1.0 if self.cfg.MIN_DURATION_SECONDS <= duration <= 15 else 0.5
        virality_boost = sum(self.cfg.VIRALITY_PRIORS.get(w, 0) for w in words)

        w = self.cfg.WEIGHTS
        final_score = (
            w['topic_score'] * topic_score +
            w['contextual_topic'] * contextual_topic_score +
            w['hook'] * hook_score +
            w['named_entity'] * entity_score +
            w['brevity'] * brevity_bonus +
            virality_boost
        )
        if any(word in words for word in self.cfg.BAD_WORDS):
            final_score *= 0.2
        details = {'Topics': topics, 'HookScore': hook_score}
        return final_score, details

    def _gen_titles(self, topics: str) -> List[str]:
        base = topics.split(',')[0].strip().title() if topics and topics != 'N/A' else 'This Concept'
        return [
            f"The Truth About {base}",
            f"Why {base} Is NOT What You Think",
            f"{base} Explained in {self.cfg.MIN_DURATION_SECONDS+5} Seconds",
            f"This Will Change How You See {base}"
        ][:3]

    def _filter_redundant_clips(self, clips: List[Dict]) -> List[Dict]:
        if not clips: return []
        embeddings = self.embed_model.encode([c['Preview'] for c in clips], show_progress_bar=False)
        final_indices = []
        for i in range(len(clips)):
            is_redundant = False
            for j in final_indices:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > self.cfg.REDUNDANCY_SIMILARITY_THRESHOLD:
                    is_redundant = True
                    break
            if not is_redundant:
                final_indices.append(i)
        return [clips[i] for i in final_indices]

    def process_file(self, srt_path: str, top_k: int = 10) -> Optional[pd.DataFrame]:
        logging.info(f"🚀 Starting processing for {srt_path}")
        sentence_entries = self._load_subtitles(srt_path)
        if not sentence_entries: return None
        boundaries = self._detect_boundaries(sentence_entries)

        full_transcript = " ".join(e['text'] for e in sentence_entries)
        global_keywords = self.kw_model.extract_keywords(full_transcript, top_n=20)
        global_topics = [kw for kw, _ in global_keywords]

        all_clips = []
        for i in range(len(boundaries) - 1):
            coarse_segment = sentence_entries[boundaries[i]:boundaries[i+1]]
            for start_idx in range(len(coarse_segment)):
                for end_idx in range(start_idx, len(coarse_segment)):
                    clip_entries = coarse_segment[start_idx:end_idx+1]
                    start_t, end_t = clip_entries[0]['start'], clip_entries[-1]['end']
                    duration = (safe_time_to_dt(end_t) - safe_time_to_dt(start_t)).total_seconds()
                    if self.cfg.MIN_DURATION_SECONDS <= duration <= self.cfg.MAX_DURATION_SECONDS:
                        text = " ".join(e['text'] for e in clip_entries)
                        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
                        score, details = self._score_segment(text, duration, keywords, global_topics)
                        all_clips.append({
                            'Score': score, 'Start': start_t, 'End': end_t, 'Duration': duration,
                            'Preview': text, **details
                        })

        if not all_clips:
            logging.warning("No valid micro-clips could be generated. Try adjusting duration settings.")
            return None

        scores = [c['Score'] for c in all_clips]
        dynamic_threshold = np.percentile(scores, self.cfg.DYNAMIC_THRESHOLD_PERCENTILE)
        strong_clips = [c for c in all_clips if c['Score'] >= dynamic_threshold]
        strong_clips = sorted(strong_clips, key=lambda x: x['Score'], reverse=True)
        diverse_clips = self._filter_redundant_clips(strong_clips)
        top_clips = diverse_clips[:top_k]

        output_data = []
        for i, clip in enumerate(top_clips):
            titles = self._gen_titles(clip['Topics'])
            output_data.append({
                'Rank': i + 1,
                'Score': f"{clip['Score']:.2f}",
                'Start': clip['Start'].strftime('%H:%M:%S,%f')[:-3],
                'End': clip['End'].strftime('%H:%M:%S,%f')[:-3],
                'Duration': f"{clip['Duration']:.2f}s",
                'Titles': " | ".join(titles),
                'ThumbText': titles[1].upper(),
                'Hashtags': f"#{clip['Topics'].replace(' ','')} #science #explained #viral",
                'Preview': clip['Preview'][:250] + "..." if len(clip['Preview']) > 250 else clip['Preview']
            })

        df = pd.DataFrame(output_data)
        df.to_csv("viral_clips.csv", index=False)
        logging.info(f"✅ Identified {len(df)} high-potential viral clips. Saved to viral_clips.csv")
        return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_segmenter.py <yourfile.srt>")
    else:
        srt_path = sys.argv[1]
        d = VideoSegmenter()
        df = d.process_file(srt_path)
        print(df)
