import pysrt
import spacy
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from transformers import pipeline
import logging

# Assumes all constants (thresholds, weights) and keyword sets are in config.py
import config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoSegmenter:
    """
    A class to encapsulate the video subtitle segmentation and scoring pipeline.
    """
    def __init__(self):
        """Initializes the segmenter and loads all necessary models."""
        logging.info("Initializing VideoSegmenter and loading models...")
        self._load_models()
        logging.info("Models loaded successfully.")

    def _load_models(self):
        """Loads and assigns all NLP models to instance attributes."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        try:
            self.nlp = spacy.load("en_core_web_trf")
            self.embed_model = SentenceTransformer("all-mpnet-base-v2").to(self.device)
            self.kw_model = KeyBERT(model=self.embed_model)
            # Using a more accurate model for emotion classification
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                top_k=None,
                device=0 if self.device == "cuda" else -1
            )
            # Using a more accurate model for sentiment analysis
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            logging.error(f"Failed to load a model: {e}")
            raise

    def _load_subtitles(self, file_path: str) -> Tuple[List[Dict], List[int]]:
        """Reads an SRT file and maps sentences to timestamps."""
        try:
            subs = pysrt.open(file_path, encoding='utf-8')
        except Exception as e:
            logging.error(f"Could not open or parse SRT file at {file_path}: {e}")
            return [], []

        full_text = ""
        index_map = []
        current_char = 0
        for sub in subs:
            text = sub.text_without_tags.strip().replace("\n", " ")
            start, end = current_char, current_char + len(text)
            index_map.append((start, end, sub.start.to_time(), sub.end.to_time()))
            full_text += text + " "
            current_char += len(text) + 1

        doc = self.nlp(full_text.strip())
        sentence_entries = []
        sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]

        for sent, (sent_start_char, sent_end_char) in zip(doc.sents, sentence_spans):
            start_time, end_time = None, None
            # Find the timestamps that envelop the current sentence
            for idx_start, idx_end, t_start, t_end in index_map:
                if idx_start <= sent_start_char < idx_end:
                    start_time = t_start
                if idx_start < sent_end_char <= idx_end:
                    end_time = t_end
                    break # Found the end, no need to search further
            
            sentence_entries.append({
                "text": sent.text.strip(),
                "start": start_time or index_map[0][2],
                "end": end_time or index_map[-1][3]
            })

        # Map subtitle start times to sentence indices
        subtitle_start_indices = []
        for idx_start, _, _, _ in index_map:
            for i, (s_start, s_end) in enumerate(sentence_spans):
                if s_start <= idx_start < s_end:
                    subtitle_start_indices.append(i)
                    break
            else: # If a subtitle start doesn't fall in any sentence (e.g., empty subtitle)
                if subtitle_start_indices: # point to the last one
                    subtitle_start_indices.append(subtitle_start_indices[-1])
                
        return sentence_entries, subtitle_start_indices

    def _time_to_datetime(self, t: datetime.time) -> datetime:
        """Converts a time object to a datetime object for calculations."""
        return datetime.combine(datetime.min, t)

    def _compute_similarity(self, sentences: List[str], window_size: int = 2) -> np.ndarray:
        """Computes cosine similarity between embeddings of sentence windows."""
        if len(sentences) <= window_size:
            return np.array([])
        window_texts = [" ".join(sentences[i:i + window_size]) for i in range(len(sentences) - window_size + 1)]
        embeddings = self.embed_model.encode(window_texts, device=self.device, show_progress_bar=False)
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        return np.diag(similarities)

    def _detect_boundaries(self, sentences: List[str], sentence_entries: List[Dict], subtitle_start_indices: List[int]) -> List[int]:
        """Identifies topic boundaries based on similarity drops and time gaps."""
        similarities = self._compute_similarity(sentences)
        if similarities.size == 0:
            return [0, len(sentences)]
            
        sim_mean, sim_std = np.mean(similarities), np.std(similarities)
        dynamic_threshold = sim_mean - config.DYNAMIC_STD_FACTOR * sim_std
        
        boundaries = []
        for i in range(1, len(similarities)):
            is_potential_break = False
            # 1. Check for a significant drop in semantic similarity
            if similarities[i-1] - similarities[i] > dynamic_threshold:
                is_potential_break = True

            # 2. Check for a significant time gap
            start_dt = self._time_to_datetime(sentence_entries[i + 1]["start"])
            end_dt = self._time_to_datetime(sentence_entries[i]["end"])
            if (start_dt - end_dt).total_seconds() > config.TIME_GAP_THRESHOLD:
                is_potential_break = True
            
            # 3. Avoid splitting if a discourse marker suggests continuation
            if is_potential_break:
                text = sentence_entries[i + 1]["text"].lower()
                has_discourse_marker = any(marker in text.split()[:3] for marker in config.DISCOURSE_MARKERS)
                if not has_discourse_marker:
                    # Align boundary to the start of the next subtitle for cleaner cuts
                    next_sub_idx = min((idx for idx in subtitle_start_indices if idx > i), default=len(sentences))
                    if next_sub_idx not in boundaries:
                        boundaries.append(next_sub_idx)
                        
        return [0] + sorted(set(boundaries)) + [len(sentences)]

    def _extract_topics(self, segment_text: str) -> Tuple[str, float, List[str]]:
        """Extracts keyphrases using KeyBERT."""
        keywords = self.kw_model.extract_keywords(
            segment_text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10
        )
        all_keywords = [kw for kw, _ in keywords] if keywords else []
        filtered = [(kw, score) for kw, score in keywords if score >= config.TOPIC_KEYWORD_THRESHOLD]
        
        if filtered:
            return ", ".join([kw for kw, _ in filtered]), filtered[0][1], all_keywords
        return "N/A", 0.0, all_keywords

    def _score_segment(self, segment_text: str, topic_score: float, topics: str) -> Tuple[float, str, float]:
        """Calculates a final score for a text segment based on multiple factors."""
        # 1. Thematic & Keyword Scoring
        words_in_text = set(segment_text.lower().split())
        edu_score = sum(1 for word in config.EDUCATIONAL_KEYWORDS if word in words_in_text)
        story_score = sum(1 for word in config.STORY_KEYWORDS if word in words_in_text)
        qna_score = sum(1 for word in config.QNA_WORDS if word in words_in_text)
        bad_score = sum(1 for word in config.BAD_WORDS if word in words_in_text)
        
        # Thematic scoring based on structured keywords
        theme_score = 0
        for category in config.TOPIC_KEYWORDS.values():
            theme_score += sum(1 for word in category if word in words_in_text)

        edu_story_boost = (
            edu_score * config.BOOST_WEIGHTS['educational'] +
            story_score * config.BOOST_WEIGHTS['story'] +
            qna_score * config.BOOST_WEIGHTS['qna'] +
            theme_score * config.BOOST_WEIGHTS['topic_keyword'] +
            bad_score * config.BOOST_WEIGHTS['bad_word_penalty']
        )
        
        # 2. NER Scoring (replaces old TIME_WORD logic)
        doc = self.nlp(segment_text)
        informative_ents = {"PERSON", "ORG", "GPE", "DATE", "TIME", "NORP", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}
        entity_score = sum(1 for ent in doc.ents if ent.label_ in informative_ents)

        # 3. Emotion & Sentiment Scoring
        emotion_results = self.emotion_classifier(topics if topics != "N/A" else segment_text[:512])[0]
        top_emotion = max(emotion_results, key=lambda x: x['score'])
        emotion_label, emotion_intensity = top_emotion['label'], top_emotion['score']
        
        sentiment_results = self.sentiment_classifier(segment_text[:512])[0]
        # Convert sentiment label to a numeric score [-1, 1]
        if sentiment_results['label'] == 'positive':
            sentiment_score = sentiment_results['score']
        elif sentiment_results['label'] == 'negative':
            sentiment_score = -sentiment_results['score']
        else: # neutral
            sentiment_score = 0.0

        # 4. Final Weighted Score Calculation
        final_score = (
            config.WEIGHTS['topic_score'] * topic_score +
            config.WEIGHTS['sentiment'] * sentiment_score +
            config.WEIGHTS['edu_story_boost'] * edu_story_boost +
            config.WEIGHTS['named_entity'] * entity_score +
            config.WEIGHTS['emotion_intensity'] * emotion_intensity
        )
        return final_score, emotion_label, emotion_intensity

    def _break_long_segment(self, segment_info: List[Dict]) -> List[Tuple[int, int, float]]:
        """Breaks a segment into smaller chunks if it exceeds MAX_DURATION_SECONDS."""
        sub_segments = []
        start_idx = 0
        while start_idx < len(segment_info):
            start_time = self._time_to_datetime(segment_info[start_idx]["start"])
            end_idx = start_idx
            
            # Find the end index for the sub-segment
            while end_idx < len(segment_info):
                current_duration = (self._time_to_datetime(segment_info[end_idx]["end"]) - start_time).total_seconds()
                if current_duration > config.MAX_DURATION_SECONDS:
                    break
                end_idx += 1
            
            # Ensure the segment is not empty and has at least one sentence
            end_idx = max(start_idx + 1, end_idx)
            final_sub_segment = segment_info[start_idx:end_idx]
            duration = (self._time_to_datetime(final_sub_segment[-1]["end"]) - start_time).total_seconds()
            sub_segments.append((start_idx, end_idx, duration))
            start_idx = end_idx
            
        return sub_segments
        
    def process_file(self, file_path: str, output_path: str = "segments.csv") -> None:
        """
        The main pipeline to process an SRT file and save scored segments to a CSV.
        """
        logging.info(f"Starting processing for {file_path}")
        sentence_entries, subtitle_start_indices = self._load_subtitles(file_path)
        if not sentence_entries:
            logging.warning("No sentences found in subtitle file. Aborting.")
            return

        sentences = [entry["text"] for entry in sentence_entries]
        boundaries = self._detect_boundaries(sentences, sentence_entries, subtitle_start_indices)

        all_segments = []
        for i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[i], boundaries[i + 1]
            segment_info = sentence_entries[seg_start:seg_end]
            
            # Break up very long segments
            sub_segments_indices = self._break_long_segment(segment_info)

            for sub_start_local, sub_end_local, duration in sub_segments_indices:
                if sub_end_local - sub_start_local < config.MIN_SENTENCES_PER_SEGMENT or duration < config.MIN_DURATION_SECONDS:
                    continue
                
                sub_segment_info = segment_info[sub_start_local:sub_end_local]
                segment_text = " ".join(entry["text"] for entry in sub_segment_info)
                topics, top_score, all_keywords = self._extract_topics(segment_text)
                
                if topics == "N/A":
                    continue

                score, emotion, emotion_intensity = self._score_segment(segment_text, top_score, topics)
                
                if score >= config.FINAL_SCORE_THRESHOLD:
                    all_segments.append({
                        "Score": score,
                        "TopTopicScore": top_score,
                        "Emotion": emotion,
                        "EmotionIntensity": emotion_intensity,
                        "Start": sub_segment_info[0]["start"],
                        "End": sub_segment_info[-1]["end"],
                        "Duration": duration,
                        "Topics": topics,
                        "Preview": segment_text[:400] + ("..." if len(segment_text) > 400 else ""),
                        "AllKeywords": ", ".join(all_keywords),
                    })

        if not all_segments:
            logging.warning("No segments met the scoring threshold.")
            return

        top_segments_df = pd.DataFrame(all_segments).sort_values(by="Score", ascending=False)
        top_segments_df.to_csv(output_path, index=False)
        logging.info(f"✅ Top segments saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    segmenter = VideoSegmenter()
    segmenter.process_file('n.srt')
