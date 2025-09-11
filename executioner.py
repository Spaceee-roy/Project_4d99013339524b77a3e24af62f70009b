import re, pysrt, math, spacy, numpy as np, pandas as pd, torch, sys, csv, logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from transformers import pipeline
from bertopic import BERTopic
from tqdm import tqdm

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
    REDUNDANCY_SIMILARITY_THRESHOLD: float = 0.85
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
            logging.info("⏳ Loading SpaCy model...")
            self.nlp = spacy.load("en_core_web_sm")

            logging.info("⏳ Loading SentenceTransformer...")
            self.embed_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)

            logging.info("⏳ Loading KeyBERT...")
            self.kw_model = KeyBERT(model=self.embed_model)

            logging.info("⏳ Loading BERTopic (optional)...")
            self.topic_model = BERTopic(embedding_model=self.embed_model)

            logging.info("⏳ Loading Emotion Classifier...")
            self.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None, device=0 if self.device == "cuda" else -1
            )

            logging.info("⏳ Loading Summarizer...")
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

            logging.info("✅ All models loaded successfully!")
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

    def _detect_boundaries(self, sentence_entries: List[Dict]) -> List[int]:
        sentences = [e['text'] for e in sentence_entries]
        if len(sentences) <= self.cfg.WINDOW_SIZE:
            return [0, len(sentences)]

        embeddings = self.embed_model.encode(
            [" ".join(sentences[i:i+self.cfg.WINDOW_SIZE]) for i in range(len(sentences)-self.cfg.WINDOW_SIZE+1)],
            device=self.device,
            show_progress_bar=False
        )
        similarities = np.diag(cosine_similarity(embeddings[:-1], embeddings[1:]))

        sim_mean, sim_std = float(np.mean(similarities)), float(np.std(similarities))
        abs_threshold = sim_mean - self.cfg.DYNAMIC_STD_FACTOR * sim_std
        boundaries = {0}

        for i in range(len(similarities)):
            is_break = (similarities[i] < abs_threshold) or \
                       (similarities[i] / max(similarities[i-1], 1e-6) < (1 - self.cfg.RELATIVE_DROP))
            if is_break:
                boundaries.add(i+1)

        boundaries.add(len(sentences))
        return sorted(list(boundaries))

    def _score_segment(self, text: str, duration: float, doc, global_topics: List[str]) -> float:
        words = set(text.lower().split())
        hook_score = sum(1 for pat in self.cfg.HOOK_PATTERNS if pat.search(text))
        entity_score = len(doc.ents)
        brevity_bonus = 1.0 if self.cfg.MIN_DURATION_SECONDS <= duration <= 15 else 0.5
        virality_boost = sum(self.cfg.VIRALITY_PRIORS.get(w, 0) for w in words)

        w = self.cfg.WEIGHTS
        score = (
            w['hook'] * hook_score +
            w['named_entity'] * entity_score +
            w['brevity'] * brevity_bonus +
            virality_boost
        )
        if any(word in words for word in self.cfg.BAD_WORDS):
            score *= 0.2
        return score

    def _gen_titles(self, topics: str) -> List[str]:
        base = topics.split(',')[0].strip().title() if topics and topics != 'N/A' else 'This Concept'
        return [
            f"The Truth About {base}",
            f"Why {base} Is NOT What You Think",
            f"{base} Explained in {self.cfg.MIN_DURATION_SECONDS+5} Seconds",
        ]

    def _filter_redundant_clips(self, clips: List[Dict]) -> List[Dict]:
        if not clips:
            return []
        clips = sorted(clips, key=lambda x: x['Score'], reverse=True)
        final_clips = []
        clip_embeddings = self.embed_model.encode([c['Preview'] for c in clips], show_progress_bar=False)
        for i, clip in enumerate(clips):
            is_redundant = False
            for j, final_clip in enumerate(final_clips):
                sim = cosine_similarity([clip_embeddings[i]], [clip_embeddings[j]])[0][0]
                if sim > self.cfg.REDUNDANCY_SIMILARITY_THRESHOLD:
                    is_redundant = True
                    break
            if not is_redundant:
                final_clips.append(clip)
        return final_clips

    def process_file(self, srt_path: str, top_k: int = 10) -> Optional[pd.DataFrame]:
        logging.info(f"🚀 Processing {srt_path}")
        sentence_entries = self._load_subtitles(srt_path)
        if not sentence_entries:
            return None

        # Precompute embeddings & SpaCy docs
        sent_texts = [e['text'] for e in sentence_entries]
        sent_embeddings = self.embed_model.encode(sent_texts, batch_size=32, convert_to_tensor=True)
        docs = list(self.nlp.pipe(sent_texts, batch_size=32))

        # Global keywords once
        full_transcript = " ".join(sent_texts)
        global_keywords = self.kw_model.extract_keywords(full_transcript, top_n=20)
        global_topics = [kw for kw, _ in global_keywords]

        # Boundaries + candidates
        boundaries = self._detect_boundaries(sentence_entries)
        all_clips = []

        for i in tqdm(range(len(boundaries)-1), desc="Generating Clips"):
            seg = sentence_entries[boundaries[i]:boundaries[i+1]]
            for start in range(len(seg)):
                for window in [2,3,4,5]:
                    end = start + window
                    if end >= len(seg): break
                    start_t, end_t = seg[start]['start'], seg[end]['end']
                    duration = (safe_time_to_dt(end_t) - safe_time_to_dt(start_t)).total_seconds()
                    if self.cfg.MIN_DURATION_SECONDS <= duration <= self.cfg.MAX_DURATION_SECONDS:
                        text = " ".join(e['text'] for e in seg[start:end+1])
                        doc = self.nlp(text)  # lightweight parse for entities
                        score = self._score_segment(text, duration, doc, global_topics)
                        all_clips.append({
                            'Score': score, 'Start': start_t, 'End': end_t,
                            'Duration': duration, 'Preview': text, 'Topics': "N/A"
                        })

        if not all_clips:
            return None

        # Shortlist before KeyBERT
        scores = [c['Score'] for c in all_clips]
        threshold = np.percentile(scores, self.cfg.DYNAMIC_THRESHOLD_PERCENTILE)
        shortlist = [c for c in all_clips if c['Score'] >= threshold][:top_k*3]

        # KeyBERT enrichment
        for c in shortlist:
            keywords = self.kw_model.extract_keywords(c['Preview'], keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
            topic_score = np.mean([score for _, score in keywords]) if keywords else 0
            c['Topics'] = ", ".join([kw for kw, _ in keywords]) if keywords else "N/A"
            c['Score'] += self.cfg.WEIGHTS['topic_score'] * topic_score

        # Redundancy filter
        final_clips = self._filter_redundant_clips(shortlist)[:top_k]

        # Output
        output_data = []
        for i, clip in enumerate(final_clips):
            titles = self._gen_titles(clip['Topics'])
            start_t, end_t = clip['Start'], clip['End']
            new_end_dt = datetime.combine(datetime.min, end_t) + timedelta(seconds=1)
            output_data.append({
                'Rank': i+1,
                'Score': f"{clip['Score']:.2f}",
                'Start': start_t.strftime('%H:%M:%S'),
                'End': new_end_dt.strftime('%H:%M:%S'),
                'Duration': f"{clip['Duration']:.2f}s",
                'Titles': " | ".join(titles),
                'ThumbText': titles[1].upper(),
                'Hashtags': f"#{clip['Topics'].replace(' ','')} #science #explained #viral",
                'Preview': clip['Preview'][:250] + "..." if len(clip['Preview'])>250 else clip['Preview']
            })

        df = pd.DataFrame(output_data)
        df.to_csv("viral_clips.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        logging.info(f"✅ Saved {len(df)} clips to viral_clips.csv")
        return df
