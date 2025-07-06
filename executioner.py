import pysrt
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd

def segment_srt():
    # Load models
    nlp = spacy.load("en_core_web_trf")
    model = SentenceTransformer("all-mpnet-base-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load SRT and extract sentences with timestamps and subtitle indices
    subs = pysrt.open(input("Enter SRT file path: ").strip(), encoding='utf-8')
    sentence_entries = []
    subtitle_start_indices = []  # Store the index of the first sentence of each subtitle

    sentence_idx = 0
    for sub in subs:
        doc = nlp(sub.text_without_tags)
        first_sentence = True
        for sent in doc.sents:
            if first_sentence:
                subtitle_start_indices.append(sentence_idx)
                first_sentence = False
            sentence_entries.append({
                "text": sent.text.strip(),
                "start": sub.start.to_time(),
                "end": sub.end.to_time()
            })
            sentence_idx += 1

    sentences = [entry["text"] for entry in sentence_entries]

    # Encode sentences
    embeddings = model.encode(sentences, device=device)
    similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
    similarity_scores = np.diag(similarities)

    # Find boundaries: large drops in similarity
    boundaries = []
    threshold_drop = 0.35  # Tune this for sensitivity

    for i in range(1, len(similarity_scores)):
        drop = similarity_scores[i - 1] - similarity_scores[i]
        if drop > threshold_drop:
            # Find the next subtitle boundary after i
            next_sub_idx = min([idx for idx in subtitle_start_indices if idx > i], default=len(sentences))
            if next_sub_idx not in boundaries:
                boundaries.append(next_sub_idx)

    # Always include start and end
    boundaries = [0] + sorted(set(boundaries)) + [len(sentences)]

    # Prepare segments for CSV
    segments = []
    for idx in range(len(boundaries) - 1):
        start_idx = boundaries[idx]
        end_idx = boundaries[idx + 1]

        segment_info = sentence_entries[start_idx:end_idx]
        start_time = segment_info[0]["start"]
        end_time = segment_info[-1]["end"]
        segment_text = " ".join(entry["text"] for entry in segment_info)

        # Extract basic topics using noun phrases
        doc = nlp(segment_text)
        noun_phrases = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
        topics = ", ".join(list(noun_phrases)[:5])

        # Add to segments list
        segments.append({
            "Segment": idx + 1,
            "Start": start_time,
            "End": end_time,
            "Preview": segment_text[:200] + ("..." if len(segment_text) > 200 else ""),
            "Topics": topics
        })

    # Save to CSV
    df = pd.DataFrame(segments)
    csv_path = "segments.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Segments saved to {csv_path}.\n")
    print("Segments:")
    for idx, segment in enumerate(segments, start=1):
        print(f"Segment {idx}: {segment['Preview']}")
