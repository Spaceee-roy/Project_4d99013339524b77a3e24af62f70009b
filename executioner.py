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

# Discourse markers commonly signaling topic shifts
DISCOURSE_MARKERS = {
    "however", "anyway", "so", "but", "nevertheless", "still", "though",
    "instead", "on the other hand"
}

def load_models() -> Tuple[spacy.language.Language, SentenceTransformer, KeyBERT, str]:
    """
    Load NLP, embedding, and keyword extraction models.

    Returns:
        nlp: Spacy language model
        embed_model: SentenceTransformer embedding model
        kw_model: KeyBERT keyword extraction model
        device: torch device ("cuda" or "cpu")
    """
    nlp = spacy.load("en_core_web_trf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-mpnet-base-v2").to(device)
    kw_model = KeyBERT(model=embed_model)
    return nlp, embed_model, kw_model, device
def load_subtitles(file_path: str, nlp) -> Tuple[List[Dict], List[int]]:
    """
    Load and clean subtitles into full, capitalized sentences with correct punctuation and timestamps.

    Returns:
        sentence_entries: List of dicts with cleaned sentence text and timestamps.
        subtitle_start_indices: List of indices marking the first sentence of each subtitle.
    """
    subs = pysrt.open(file_path, encoding='utf-8')
    sentence_entries = []
    subtitle_start_indices = []
    sentence_idx = 0

    buffer_text = ""
    buffer_start = None

    for sub in subs:
        doc = nlp(sub.text_without_tags)

        for sent in doc.sents:
            text = sent.text.strip()

            # Capitalize first word
            if text:
                text = text[0].upper() + text[1:]

            # Ensure it ends with a period
            if text and not text.endswith(('.', '!', '?')):
                text += '.'

            if buffer_text:
                # Continuation of a previous sentence — join and keep old start time
                buffer_text += " " + text
                end_time = sub.end.to_time()
                sentence_entries[-1]["text"] = buffer_text
                sentence_entries[-1]["end"] = end_time
            else:
                buffer_text = text
                buffer_start = sub.start.to_time()
                end_time = sub.end.to_time()
                sentence_entries.append({
                    "text": buffer_text,
                    "start": buffer_start,
                    "end": end_time
                })
                subtitle_start_indices.append(sentence_idx)
                sentence_idx += 1

            # Clear buffer
            buffer_text = ""

    return sentence_entries, subtitle_start_indices



def time_to_datetime(t: datetime.time) -> datetime:
    """
    Convert a datetime.time object to datetime.datetime on a fixed date.

    Args:
        t: datetime.time object

    Returns:
        datetime.datetime object with date set to minimal date.
    """
    return datetime.combine(datetime.min, t)

def compute_similarity(sentences: List[str], embed_model, device, window_size: int = 2) -> np.ndarray:
    """
    Compute semantic similarity between consecutive sentence windows.

    Args:
        sentences: List of sentence texts.
        embed_model: SentenceTransformer model.
        device: Device to run model on.
        window_size: Number of sentences in each window.

    Returns:
        Array of similarity scores (diagonal of cosine similarity matrix).
    """
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
    """
    Detect segment boundaries based on similarity drops, time gaps, and discourse markers.

    Args:
        sentences: List of sentences.
        sentence_entries: Sentence metadata with timing.
        subtitle_start_indices: Sentence indices marking subtitle starts.
        similarities: Similarity scores between windows.
        dynamic_std_factor: Controls sensitivity of similarity threshold.
        time_gap_threshold: Threshold in seconds for pause to indicate boundary.

    Returns:
        Sorted list of boundary indices for segmentation.
    """
    boundaries = []
    sim_mean, sim_std = np.mean(similarities), np.std(similarities)
    dynamic_threshold = sim_mean - dynamic_std_factor * sim_std

    for i in range(1, len(similarities)):
        sim_drop = similarities[i - 1] - similarities[i]

        # Convert times to datetime for subtraction
        start_dt = time_to_datetime(sentence_entries[i + 1]["start"])
        end_dt = time_to_datetime(sentence_entries[i]["end"])
        time_gap = (start_dt - end_dt).total_seconds()

        text = sentence_entries[i + 1]["text"].lower()
        has_discourse_marker = any(marker in text for marker in DISCOURSE_MARKERS)

        if sim_drop > dynamic_threshold or time_gap > time_gap_threshold or has_discourse_marker:
            next_sub_idx = min(
                (idx for idx in subtitle_start_indices if idx > i),
                default=len(sentences)
            )
            if next_sub_idx not in boundaries:
                boundaries.append(next_sub_idx)

    # Always include start and end boundaries
    return [0] + sorted(set(boundaries)) + [len(sentences)]

def extract_topics(segment_text: str, kw_model) -> str:
    """
    Extract keywords/topics from a segment of text using KeyBERT.

    Args:
        segment_text: Text of the segment.
        kw_model: KeyBERT model.

    Returns:
        Comma-separated string of extracted keywords or 'N/A'.
    """
    keywords = kw_model.extract_keywords(
        segment_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=5
    )
    return ", ".join([kw for kw, _ in keywords]) if keywords else "N/A"

def generate_segments(
    boundaries: List[int],
    sentence_entries: List[Dict],
    kw_model,
    nlp,
    min_sentences: int,
    min_duration: int
) -> List[Dict]:
    """
    Generate final segments from detected boundaries with metadata and topics.

    Args:
        boundaries: List of boundary indices.
        sentence_entries: Sentence metadata.
        kw_model: KeyBERT keyword extraction model.
        nlp: Spacy NLP model.
        min_sentences: Minimum sentences per segment.
        min_duration: Minimum segment duration in seconds.

    Returns:
        List of segment dicts with segment info.
    """
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
        topics = extract_topics(segment_text, kw_model)

        segments.append({
            "Segment": len(segments) + 1,
            "Start": start_time,
            "End": end_time,
            "Preview": segment_text[:200] + ("..." if len(segment_text) > 200 else ""),
            "Topics": topics
        })

    return segments

def save_segments_to_csv(segments: List[Dict], output_path: str):
    """
    Save the generated segments to a CSV file.

    Args:
        segments: List of segment dictionaries.
        output_path: CSV file path.
    """
    df = pd.DataFrame(segments)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Segments saved to {output_path}.\n")

def segment_srt_pipeline(
    file_path: str,
    dynamic_std_factor: float = 1.5,
    time_gap_threshold: float = 2.5,
    min_sentences: int = 3,
    min_duration: int = 10,
):
    """
    Main pipeline to process an SRT file and generate segmented CSV output.

    Args:
        file_path: Path to SRT subtitle file.
        dynamic_std_factor: Sensitivity multiplier for similarity drop threshold.
        time_gap_threshold: Time in seconds to detect pauses between sentences.
        min_sentences: Minimum sentences per segment.
        min_duration: Minimum segment length in seconds.
    """
    
    nlp, embed_model, kw_model, device = load_models()
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
        min_sentences,
        min_duration
    )

    output_path = "segments.csv"
    save_segments_to_csv(segments, output_path)

    print("Segments:")
    for segment in segments:
        print(f"Segment {segment['Segment']}: {segment['Preview']}")

if __name__ == "__main__":
    file_path = input("Enter SRT file path: ").strip()
    
    segment_srt_pipeline(file_path)
