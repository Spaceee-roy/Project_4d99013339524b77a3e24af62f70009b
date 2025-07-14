import pysrt
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import torch
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from textblob import TextBlob
from transformers import pipeline

DISCOURSE_MARKERS = {
    "however", "anyway", "so", "but", "nevertheless", "still", "though",
    "instead", "on the other hand"
}

EDUCATIONAL_KEYWORDS = {
    "explain", "reason", "because", "process", "method", "fact", "data", "research", "study"
}

STORY_KEYWORDS = {
    "once", "happened", "remember", "told", "story", "experience", "felt", "saw", "heard",
    "when i", "then", "after that"
}

def load_models():
    nlp = spacy.load("en_core_web_trf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-mpnet-base-v2").to(device)
    kw_model = KeyBERT(model=embed_model)
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return nlp, embed_model, kw_model, device, emotion_classifier

def load_subtitles(file_path: str, nlp) -> Tuple[List[Dict], List[int]]:
    subs = pysrt.open(file_path, encoding='utf-8')
    full_text = ""
    index_map = []
    current_char = 0
    for sub in subs:
        text = sub.text_without_tags.strip().replace("\n", " ")
        start = current_char
        end = current_char + len(text)
        index_map.append((start, end, sub.start.to_time(), sub.end.to_time()))
        full_text += text + " "
        current_char += len(text) + 1
    doc = nlp(full_text.strip())
    sentence_entries = []
    sentence_spans = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_start = sent.start_char
        sent_end = sent.end_char
        sentence_start_time = None
        sentence_end_time = None
        for idx_start, idx_end, t_start, t_end in index_map:
            if idx_start <= sent_start < idx_end and sentence_start_time is None:
                sentence_start_time = t_start
            if idx_start < sent_end <= idx_end:
                sentence_end_time = t_end
                break
        if sentence_start_time is None:
            sentence_start_time = index_map[0][2]
        if sentence_end_time is None:
            sentence_end_time = index_map[-1][3]
        sentence_entries.append({
            "text": sent_text,
            "start": sentence_start_time,
            "end": sentence_end_time
        })
        sentence_spans.append((sent_start, sent_end))
    subtitle_start_indices = []
    for idx_start, _, _, _ in index_map:
        for i, (s_start, s_end) in enumerate(sentence_spans):
            if s_start <= idx_start < s_end:
                subtitle_start_indices.append(i)
                break
        else:
            subtitle_start_indices.append(len(sentence_entries) - 1)
    return sentence_entries, subtitle_start_indices

def time_to_datetime(t: datetime.time) -> datetime:
    return datetime.combine(datetime.min, t)

def compute_similarity(sentences: List[str], embed_model, device, window_size: int = 2) -> np.ndarray:
    window_texts = [
        " ".join(sentences[i:i + window_size])
        for i in range(len(sentences) - window_size)
    ]
    embeddings = embed_model.encode(window_texts, device=device)
    similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
    return np.diag(similarities)

def detect_boundaries(
    sentences: List[str],
    sentence_entries: List[Dict],
    subtitle_start_indices: List[int],
    similarities: np.ndarray,
    dynamic_std_factor: float,
    time_gap_threshold: float
) -> List[int]:
    boundaries = []
    sim_mean, sim_std = np.mean(similarities), np.std(similarities)
    dynamic_threshold = sim_mean - dynamic_std_factor * sim_std
    for i in range(1, len(similarities)):
        sim_drop = similarities[i - 1] - similarities[i]
        start_dt = time_to_datetime(sentence_entries[i + 1]["start"])
        end_dt = time_to_datetime(sentence_entries[i]["end"])
        time_gap = (start_dt - end_dt).total_seconds()
        text = sentence_entries[i + 1]["text"].lower()
        has_discourse_marker = any(marker in text.split()[:3] for marker in DISCOURSE_MARKERS)
        if (sim_drop > dynamic_threshold or time_gap > time_gap_threshold) and not has_discourse_marker:
            next_sub_idx = min(
                (idx for idx in subtitle_start_indices if idx > i),
                default=len(sentences)
            )
            if next_sub_idx not in boundaries:
                boundaries.append(next_sub_idx)
    return [0] + sorted(set(boundaries)) + [len(sentences)]

def extract_topics(segment_text: str, kw_model, score_threshold: float = 0.45) -> Tuple[str, float]:
    keywords = kw_model.extract_keywords(
        segment_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=100
    )
    if keywords:
        filtered = [(kw, score) for kw, score in keywords if score >= score_threshold]
        if filtered:
            topics = ", ".join([kw for kw, _ in filtered])
            return topics, filtered[0][1]
    return "N/A", 0.0

def score_edu_story_boost(text: str) -> float:
    text = text.lower()
    edu_score = sum(1 for word in EDUCATIONAL_KEYWORDS if word in text)
    story_score = sum(1 for word in STORY_KEYWORDS if word in text)
    return edu_score * 0.5 + story_score * 0.5

def named_entity_score(nlp, text: str) -> int:
    doc = nlp(text)
    informative_types = {"PERSON", "ORG", "GPE", "DATE", "TIME", "NORP"}
    return sum(1 for ent in doc.ents if ent.label_ in informative_types)

def emotion_score(emotion_classifier, topics: str, segment_preview: str) -> Tuple[str, float]:
    # Use topics if available, else use preview (max 512 chars)
    input_text = topics if topics and topics != "N/A" else segment_preview[:512]
    emotions = emotion_classifier(input_text)
    top_emotion = max(emotions[0], key=lambda x: x['score'])
    return top_emotion['label'], top_emotion['score']

def segment_score(text: str, topic_score: float, nlp, emotion_classifier, topics: str, segment_text: str) -> Tuple[float, str, float]:
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    edu_story_boost = score_edu_story_boost(text)
    entity_score = named_entity_score(nlp, text)
    emotion, emotion_intensity = emotion_score(emotion_classifier, topics, segment_text)
    return (
        0.5 * topic_score +
        0.3 * sentiment_score +
        0.3 * subjectivity_score +
        1.0 * edu_story_boost +
        0.5 * entity_score +
        0.5 * emotion_intensity
    ), emotion, emotion_intensity

def generate_segments(
    boundaries: List[int],
    sentence_entries: List[Dict],
    kw_model,
    nlp,
    emotion_classifier,
    min_sentences: int,
    min_duration: int,
    topic_score_threshold: float = 0.45
) -> List[Dict]:
    segments = []
    for idx in range(len(boundaries) - 1):
        start_idx, end_idx = boundaries[idx], boundaries[idx + 1]
        if end_idx - start_idx < min_sentences:
            continue
        segment_info = sentence_entries[start_idx:end_idx]
        start_time = segment_info[0]["start"]
        end_time = segment_info[-1]["end"]
        duration = (time_to_datetime(end_time) - time_to_datetime(start_time)).total_seconds()
        if duration < min_duration:
            continue
        segment_text = " ".join(entry["text"] for entry in segment_info)
        topics, top_score = extract_topics(segment_text, kw_model, topic_score_threshold)
        if topics == "N/A":
            continue
        score, emotion, emotion_intensity = segment_score(segment_text, top_score, nlp, emotion_classifier, topics, segment_text)
        segments.append({
            "Segment": len(segments) + 1,
            "Start": start_time,
            "End": end_time,
            "Preview": segment_text[:400] + ("..." if len(segment_text) > 400 else ""),
            "Topics": topics,
            "TopTopicScore": top_score,
            "Score": score,
            "Emotion": emotion,
            "EmotionIntensity": emotion_intensity
        })
    top_segments = sorted(segments, key=lambda x: x["Score"], reverse=True)
    return top_segments

def save_segments_to_csv(segments: List[Dict], output_path: str):
    df = pd.DataFrame(segments)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Top segments saved to {output_path}.\n")

def segment_srt_pipeline(
    file_path: str,
    dynamic_std_factor: float = 1.5,
    time_gap_threshold: float = 4,
    min_sentences: int = 3,
    min_duration: int = 20,
    topic_score_threshold: float = 0.45
):
    nlp, embed_model, kw_model, device, emotion_classifier = load_models()
    sentence_entries, subtitle_start_indices = load_subtitles(file_path, nlp)
    sentences = [entry["text"] for entry in sentence_entries]
    if not sentences:
        print("No sentences found in subtitle file.")
        return
    similarities = compute_similarity(sentences, embed_model, device)
    boundaries = detect_boundaries(
        sentences,
        sentence_entries,
        subtitle_start_indices,
        similarities,
        dynamic_std_factor,
        time_gap_threshold
    )
    segments = generate_segments(
        boundaries,
        sentence_entries,
        kw_model,
        nlp,
        emotion_classifier,
        min_sentences,
        min_duration,
        topic_score_threshold
    )
    output_path = "segments.csv"
    save_segments_to_csv(segments, output_path)

if __name__ == "__main__":
    segment_srt_pipeline('n.srt')
