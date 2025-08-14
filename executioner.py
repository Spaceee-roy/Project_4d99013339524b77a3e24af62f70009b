import os
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
import subprocess

# ---------------------------------
# Config: use external config if available; otherwise fall back to sane defaults
# ---------------------------------
# try:
#     import config  # type: ignore
#     HAVE_EXTERNAL_CONFIG = True
# except Exception:
HAVE_EXTERNAL_CONFIG = False

class _Cfg:
    # Core timings
    MAX_DURATION_SECONDS: int = 20               # hard cap for viral micro-clips
    MIN_DURATION_SECONDS: int = 7                # keep them punchy
    MIN_SENTENCES_PER_SEGMENT: int = 1

    # Boundary detection
    DYNAMIC_STD_FACTOR: float = 0.5             # smaller => more sensitive to drops
    TIME_GAP_THRESHOLD: float = 1.5              # seconds between subtitles considered a gap
    WINDOW_SIZE: int = 2                         # for similarity windows

    # Topic extraction / scoring
    TOPIC_KEYWORD_THRESHOLD: float = 0.35
    FINAL_SCORE_THRESHOLD: float = 3.0           # raise/lower to control strictness

    # Weights for final score
    WEIGHTS = {
        'topic_score': 0.8,
        'sentiment': 0.5,
        'edu_story_boost': 0.4,
        'named_entity': 0.25,
        'emotion_intensity': 0.75,
        'hook': 2.0,
        'surprise': 1.5,
        'brevity': 1.0,
    }

    # Thematic keyword buckets (very generic, adjust per niche)
    KEYWORDS = {
        'space': {
            'universe','cosmos','galaxy','black','hole','gravity','relativity','quantum','light','speed',
            'mars','moon','alien','aliens','exoplanet','nebula','supernova','asteroid','comet','orbit','nasa'
        },
        'mindblow': {
            'turns','out','actually','counterintuitive','paradox','blows','mind','unbelievable','craziest','weird'
        },
        'numbers': {
            'billion','trillion','million','percent','times','x','1','2','3','4','5','6','7','8','9','0'
        },
    }

    # Structured keywords for boosts
    EDUCATIONAL_KEYWORDS = {'explain','because','reason','how','why','imagine','suppose','think','fact','evidence'}
    STORY_KEYWORDS        = {'story','once','when','then','so','but','suddenly','and','remember'}
    QNA_WORDS             = {'question','ask','asked','answer','what','why','how','who','where'}
    BAD_WORDS             = {'uh','um','you know'}  # tiny penalty

    # Discourse markers to avoid splitting right after them
    DISCOURSE_MARKERS     = {'and','but','so','because','then','also','plus','however','therefore'}

    # Hook patterns: question, superlatives, counterintuitive claims, numbers
    HOOK_PATTERNS = [
        re.compile(r"^(what|why|how|who|where|when)\b", re.I),
        re.compile(r"\b(no one|nobody|everyone|never|always)\b", re.I),
        re.compile(r"\b(the (craziest|weirdest|wildest|biggest|fastest|most|least))\b", re.I),
        re.compile(r"\b(you won'?t believe|mind[- ]?blow|blow your mind)\b", re.I),
        re.compile(r"\b(aliens?|black hole|speed of light|quantum|relativity)\b", re.I),
        re.compile(r"\b(\d+[\,\d]*\s*(million|billion|trillion|%)?)\b", re.I),
        re.compile(r"\b(here'?s the thing|the reason)\b", re.I),
    ]

    SURPRISE_PATTERNS = [
        re.compile(r"\b(actually|in fact|turns out|counterintuitive|paradox)\b", re.I),
        re.compile(r"\b(but|however|except)\b", re.I),
    ]

    # Title/thumbnail helpers
    MAX_TITLE_LEN = 72
    MAX_THUMB_LEN = 28

CFG = _Cfg()

# ---------------------------------
# Utility
# ---------------------------------

def safe_time_to_dt(t: time) -> datetime:
    return datetime.combine(datetime.min, t)


def seconds_to_time(sec: float) -> time:
    sec = max(0.0, float(sec))
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    milliseconds = int((sec - int(sec)) * 1000)
    return time(hours, minutes, seconds, milliseconds*1000)


# ---------------------------------
# VideoSegmenter
# ---------------------------------
class VideoSegmenter:
    """SRT → viral micro-clip proposals (timestamps + packaging metadata)."""

    def __init__(self):
        logging.info("Initializing VideoSegmenter and loading models...")
        self._load_models()
        logging.info("Models loaded successfully.")

    def _load_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        try:
            # spaCy
            try:
                self.nlp = spacy.load("en_core_web_trf")
            except Exception:
                self.nlp = spacy.load("en_core_web_sm")  # fallback

            # Embeddings
            self.embed_model = SentenceTransformer("all-mpnet-base-v2").to(self.device)
            self.kw_model = KeyBERT(model=self.embed_model)

            # Classifiers
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=0 if self.device == "cuda" else -1,
            )
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
            )
        except Exception as e:
            logging.error(f"Failed to load a model: {e}")
            raise

    # ----------
    # SRT loading & sentence mapping
    # ----------
    def _load_subtitles(self, file_path: str) -> Tuple[List[Dict], List[int]]:
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
            for idx_start, idx_end, t_start, t_end in index_map:
                if idx_start <= sent_start_char < idx_end:
                    start_time = t_start
                if idx_start < sent_end_char <= idx_end:
                    end_time = t_end
                    break
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
            else:
                if subtitle_start_indices:
                    subtitle_start_indices.append(subtitle_start_indices[-1])

        return sentence_entries, subtitle_start_indices

    # ----------
    # Similarity & boundaries
    # ----------
    def _compute_similarity(self, sentences: List[str], window_size: Optional[int] = None) -> np.ndarray:
        if window_size is None:
            window_size = getattr(CFG, 'WINDOW_SIZE', 2)
        if len(sentences) <= window_size:
            return np.array([])
        window_texts = [" ".join(sentences[i:i + window_size]) for i in range(len(sentences) - window_size + 1)]
        embeddings = self.embed_model.encode(window_texts, device=self.device, show_progress_bar=False)
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        return np.diag(similarities)

    def _detect_boundaries(self, sentences: List[str], sentence_entries: List[Dict], subtitle_start_indices: List[int]) -> List[int]:
        similarities = self._compute_similarity(sentences)
        if similarities.size == 0:
            return [0, len(sentences)]

        sim_mean, sim_std = float(np.mean(similarities)), float(np.std(similarities))
        dynamic_threshold = sim_mean - getattr(CFG, 'DYNAMIC_STD_FACTOR', 0.35) * sim_std

        boundaries = []
        for i in range(1, len(similarities)):
            is_potential_break = False
            if similarities[i-1] - similarities[i] > dynamic_threshold:
                is_potential_break = True

            # time gap
            start_dt = safe_time_to_dt(sentence_entries[i + 1]["start"])
            end_dt = safe_time_to_dt(sentence_entries[i]["end"])
            if (start_dt - end_dt).total_seconds() > getattr(CFG, 'TIME_GAP_THRESHOLD', 1.2):
                is_potential_break = True

            if is_potential_break:
                text = sentence_entries[i + 1]["text"].lower()
                has_discourse_marker = any(m in text.split()[:3] for m in getattr(CFG, 'DISCOURSE_MARKERS', set()))
                if not has_discourse_marker:
                    next_sub_idx = min((idx for idx in subtitle_start_indices if idx > i), default=len(sentences))
                    if next_sub_idx not in boundaries:
                        boundaries.append(next_sub_idx)

        return [0] + sorted(set(boundaries)) + [len(sentences)]

    # ----------
    # Hooks & surprise scoring
    # ----------
    def _hook_strength(self, text: str) -> float:
        score = 0.0
        t = text.strip().lower()
        for pat in getattr(CFG, 'HOOK_PATTERNS', []):
            if pat.search(t):
                score += 1.0
        # shorter starts with interrogative gets a tiny bonus
        if re.match(r"^(what|why|how|who|where|when)\b", t):
            score += 0.5
        return score

    def _surprise_strength(self, text: str) -> float:
        s = 0.0
        t = text.strip().lower()
        for pat in getattr(CFG, 'SURPRISE_PATTERNS', []):
            if pat.search(t):
                s += 1.0
        # Contrastive conjunction mid-sentence
        if re.search(r"\bbut\b", t):
            s += 0.25
        return s

    # ----------
    # Topic extraction & scoring
    # ----------
    def _extract_topics(self, segment_text: str) -> Tuple[str, float, List[str]]:
        keywords = self.kw_model.extract_keywords(
            segment_text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10
        )
        all_keywords = [kw for kw, _ in keywords] if keywords else []
        filtered = [(kw, score) for kw, score in keywords if score >= getattr(CFG, 'TOPIC_KEYWORD_THRESHOLD', 0.35)]
        if filtered:
            return ", ".join([kw for kw, _ in filtered]), filtered[0][1], all_keywords
        return "N/A", 0.0, all_keywords

    def _score_segment(self, segment_text: str, topic_score: float, topics: str, duration: float) -> Tuple[float, str, float, float, float]:
        # 1) Thematic & Keyword Scoring
        words_in_text = set(segment_text.lower().split())
        edu_score = sum(1 for w in getattr(CFG, 'EDUCATIONAL_KEYWORDS', set()) if w in words_in_text)
        story_score = sum(1 for w in getattr(CFG, 'STORY_KEYWORDS', set()) if w in words_in_text)
        qna_score = sum(1 for w in getattr(CFG, 'QNA_WORDS', set()) if w in words_in_text)
        bad_score = sum(1 for w in getattr(CFG, 'BAD_WORDS', set()) if w in words_in_text)

        theme_score = 0
        for category in getattr(CFG, 'KEYWORDS', {}).values():
            theme_score += sum(1 for w in category if w in words_in_text)

        edu_story_boost = (
            edu_score * CFG.WEIGHTS.get('educational', 0.0) +
            story_score * CFG.WEIGHTS.get('story', 0.0) +
            qna_score * CFG.WEIGHTS.get('qna', 0.0) +
            theme_score * CFG.WEIGHTS.get('topic_keyword', 0.0) +
            bad_score * CFG.WEIGHTS.get('bad_word_penalty', 0.0)
        )

        # 2) NER scoring
        doc = self.nlp(segment_text)
        informative_ents = {"PERSON","ORG","GPE","DATE","TIME","NORP","EVENT","PRODUCT","WORK_OF_ART","LAW"}
        entity_score = sum(1 for ent in doc.ents if ent.label_ in informative_ents)

        # 3) Emotion & Sentiment
        emotion_results = self.emotion_classifier(topics if topics != "N/A" else segment_text[:512])[0]
        top_emotion = max(emotion_results, key=lambda x: x['score'])
        emotion_label, emotion_intensity = top_emotion['label'], float(top_emotion['score'])

        sentiment_results = self.sentiment_classifier(segment_text[:512])[0]
        if sentiment_results['label'] == 'positive':
            sentiment_score = float(sentiment_results['score'])
        elif sentiment_results['label'] == 'negative':
            sentiment_score = -float(sentiment_results['score'])
        else:
            sentiment_score = 0.0

        # 4) Hooks & Surprise
        hook_score = self._hook_strength(segment_text)
        surprise_score = self._surprise_strength(segment_text)

        # 5) Brevity bonus
        brevity_bonus = 0.0
        if CFG.MIN_DURATION_SECONDS <= duration <= 15:
            brevity_bonus = 1.0
        elif duration <= CFG.MAX_DURATION_SECONDS:
            brevity_bonus = 0.5

        final_score = (
            CFG.WEIGHTS['topic_score'] * topic_score +
            CFG.WEIGHTS['sentiment'] * sentiment_score +
            CFG.WEIGHTS['edu_story_boost'] * edu_story_boost +
            CFG.WEIGHTS['named_entity'] * entity_score +
            CFG.WEIGHTS['emotion_intensity'] * emotion_intensity +
            CFG.WEIGHTS['hook'] * hook_score +
            CFG.WEIGHTS['surprise'] * surprise_score +
            CFG.WEIGHTS['brevity'] * brevity_bonus
        )
        return final_score, emotion_label, emotion_intensity, hook_score, surprise_score

    # ----------
    # Segment chopping logic to enforce viral length
    # ----------
    def _break_to_viral_microclips(self, segment_info: List[Dict]) -> List[Tuple[int, int, float]]:
        """Slice a longer segment into 7–20s clips aligned to sentence/sub boundaries."""
        subclips = []
        n = len(segment_info)
        i = 0
        while i < n:
            # Start at i and extend until we hit MAX or a natural boundary
            start_t = safe_time_to_dt(segment_info[i]['start'])
            j = i
            while j < n:
                dur = (safe_time_to_dt(segment_info[j]['end']) - start_t).total_seconds()
                if dur >= CFG.MAX_DURATION_SECONDS:
                    break
                j += 1
            # ensure at least one sentence and >= MIN_DURATION
            if j == i:
                j = min(i + 1, n)
            # backtrack to meet MIN_DURATION if needed (extend forward when short)
            while j < n:
                dur = (safe_time_to_dt(segment_info[j-1]['end']) - start_t).total_seconds()
                if dur >= CFG.MIN_DURATION_SECONDS:
                    break
                j += 1
            j = min(j, n)
            final_dur = (safe_time_to_dt(segment_info[j-1]['end']) - start_t).total_seconds()
            if final_dur >= CFG.MIN_DURATION_SECONDS:
                subclips.append((i, j, final_dur))
            i = j
        return subclips

    # ----------
    # Packaging helpers (titles, thumbnails, captions)
    # ----------
    def _gen_title(self, topics: str, hook_score: float, surprise_score: float) -> str:
        base = topics.split(',')[0].strip() if topics and topics != 'N/A' else ''
        templates = [
            f"The {base.title()} Fact That Shocks Everyone" if base else "The Space Fact That Shocks Everyone",
            f"Why {base.title()} Works Like This" if base else "Why Space Works Like This",
            f"You Won't Believe This About {base.title()}" if base else "You Won't Believe This About Space",
            f"{base.title()} Explained in 15 Seconds" if base else "Space Explained in 15 Seconds",
        ]
        # Prefer more clicky when hooks/surprise are strong
        idx = 2 if hook_score + surprise_score >= 2.0 else 0
        title = templates[idx]
        return (title[:CFG.MAX_TITLE_LEN]).strip()

    def _gen_thumb_text(self, topics: str) -> str:
        base = topics.split(',')[0].strip().title() if topics and topics != 'N/A' else 'This Will Change You'
        candidates = [base, 'Wait… What?', 'Mind = Blown', 'The Reason']
        # pick shortest that fits
        for c in sorted(candidates, key=len):
            if len(c) <= CFG.MAX_THUMB_LEN:
                return c
        return candidates[0][:CFG.MAX_THUMB_LEN]

    def _gen_hashtags(self, topics: str) -> str:
        base_tags = ['#neildegrassetyson','#science','#space','#shorts','#fyp']
        topic_first = topics.split(',')[0].strip().replace(' ','') if topics and topics != 'N/A' else ''
        if topic_first:
            base_tags.insert(1, f"#{topic_first[:18]}")
        return ' '.join(dict.fromkeys(base_tags))

    # ----------
    # Main
    # ----------
    def process_file(self, file_path: str, output_path: str = "segments.csv", clips_path: str = "viral_clips.csv", top_k: int = 10, export_clips: bool = False, keep_all: bool = False) -> None:
        logging.info(f"Starting processing for {file_path}")
        sentence_entries, subtitle_start_indices = self._load_subtitles(file_path)
        if not sentence_entries:
            logging.warning("No sentences found in subtitle file. Aborting.")
            return

        sentences = [e['text'] for e in sentence_entries]
        boundaries = self._detect_boundaries(sentences, sentence_entries, subtitle_start_indices)

        # Collect longer semantic segments first
        coarse_segments = []
        for i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[i], boundaries[i + 1]
            coarse_segments.append(sentence_entries[seg_start:seg_end])

        all_segments_rows = []
        viral_rows = []

        for segment_info in coarse_segments:
            # break into microclips respecting configured durations
            microclips = self._break_to_viral_microclips(segment_info)
            for s, e, duration in microclips:
                sub_segment_info = segment_info[s:e]
                segment_text = " ".join(entry['text'] for entry in sub_segment_info)
                topics, top_score, all_keywords = self._extract_topics(segment_text)
                if topics == 'N/A':
                    continue
                score, emotion, emotion_intensity, hook_score, surprise_score = self._score_segment(segment_text, top_score, topics, duration)

                row_common = {
                    'Score': score,
                    'TopTopicScore': top_score,
                    'Emotion': emotion,
                    'EmotionIntensity': emotion_intensity,
                    'HookScore': hook_score,
                    'SurpriseScore': surprise_score,
                    'Start': sub_segment_info[0]['start'],
                    'End': sub_segment_info[-1]['end'],
                    'Duration': duration,
                    'Topics': topics,
                    'Preview': (segment_text[:220] + ('...' if len(segment_text) > 220 else '')),
                    'AllKeywords': ", ".join(all_keywords),
                }
                all_segments_rows.append(row_common.copy())

                # Only accept strong candidates (hard filter)
                if score >= CFG.FINAL_SCORE_THRESHOLD:
                    title = self._gen_title(topics, hook_score, surprise_score)
                    thumb = self._gen_thumb_text(topics)
                    hashtags = self._gen_hashtags(topics)
                    viral_rows.append({
                        **row_common,
                        'Title': title,
                        'ThumbText': thumb,
                        'Hashtags': hashtags,
                        'FFmpegCmd': self._ffmpeg_cmd_placeholder(file_path, row_common['Start'], row_common['End']),
                    })

        if not all_segments_rows:
            logging.warning("No micro-clips found. Try checking SRT alignment or relaxing some thresholds.")
            return

        # Save detailed segments only if requested
        seg_df = pd.DataFrame(all_segments_rows).sort_values(by='Score', ascending=False)
        if keep_all:
            seg_df.to_csv(output_path, index=False)
            logging.info(f"✅ Detailed segments saved to {output_path}")

        if not viral_rows:
            logging.warning("No segments passed the strict FINAL_SCORE_THRESHOLD. Consider lowering it or adjusting weights.")
            return

        # Keep top_k viral clips, strictly the best ones only
        viral_rows_sorted = sorted(viral_rows, key=lambda r: r['Score'], reverse=True)
        top_viral = viral_rows_sorted[:top_k]

        viral_df = pd.DataFrame(top_viral)
        viral_df.to_csv(clips_path, index=False)
        logging.info(f"🚀 Top {len(top_viral)} viral-ready clips saved to {clips_path}")

        # Optionally export the top clips using ffmpeg (one file per clip)
        if export_clips:
            exported = []
            for idx, row in enumerate(top_viral, start=1):
                out_file = self._ffmpeg_cmd_for_clip(file_path, row['Start'], row['End'], idx)
                cmd = out_file['cmd']
                outfile = out_file['outfile']
                try:
                    logging.info(f"Exporting clip {idx}/{len(top_viral)} -> {outfile}")
                    subprocess.run(cmd, check=False)
                    exported.append(outfile)
                except Exception as e:
                    logging.error(f"Failed to export clip {outfile}: {e}")
            logging.info(f"Exported {len(exported)} clip(s).")
        return


    # ----------
    # Helpers
    # ----------
    @staticmethod
    def _ffmpeg_cmd_placeholder(input_video_path: str, start_t: time, end_t: time) -> str:
        """Return a ready-to-run ffmpeg command (without audio/music/captions overlay)."""
        def t_to_seconds(t: time) -> float:
            return t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
        ss = max(0.0, t_to_seconds(start_t) - 0.20)  # small pre-roll pad
        to = max(0.0, t_to_seconds(end_t) + 0.10)    # small post-roll pad
        dur = max(0.0, to - ss)
        out = os.path.splitext(os.path.basename(input_video_path))[0]
        return (
            f"ffmpeg -y -ss {ss:.2f} -t {dur:.2f} -i \"{input_video_path}\" "
            f"-vf 'fps=30,scale=1080:-2' -r 30 -c:v libx264 -preset veryfast -c:a aac -b:a 160k "
            f"\"{out}_clip_%03d.mp4\""
        )


    def _ffmpeg_cmd_for_clip(self, input_video_path: str, start_t: time, end_t: time, idx: int = 1) -> Dict[str, str]:
        """Return a dict with a shell-friendly cmd list and an output filename for a single clip export."""
        def t_to_seconds(t: time) -> float:
            return t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
        ss = max(0.0, t_to_seconds(start_t) - 0.20)
        to = max(0.0, t_to_seconds(end_t) + 0.10)
        dur = max(0.0, to - ss)
        base = os.path.splitext(os.path.basename(input_video_path))[0]
        outfile = f"{base}_topclip_{idx:02d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ss:.2f}",
            "-t", f"{dur:.2f}",
            "-i", input_video_path,
            "-vf", "fps=30,scale=1080:-2",
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-b:a", "160k",
            outfile,
        ]
        return {"cmd": cmd, "outfile": outfile}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate viral micro-clip proposals from SRT")
    parser.add_argument('srt', help='Path to subtitle file (.srt)')
    parser.add_argument('--video', required=True, help='Path to the original full video file')
    parser.add_argument('--segments_csv', default='segments.csv', help='Detailed segments CSV output path')
    parser.add_argument('--viral_csv', default='viral_clips.csv', help='Top viral clips CSV output path')
    parser.add_argument('--top_k', type=int, default=10, help='How many top clips to keep')
    parser.add_argument('--keep_all', action='store_true', help='If set, also save the full segments CSV (segments_csv)')
    parser.add_argument('--export_clips', action='store_true', help='If set, run ffmpeg to export the top clips to files')
    args = parser.parse_args()

    seg = VideoSegmenter()
    seg.process_file(
        args.srt,
        output_path=args.segments_csv,
        clips_path=args.viral_csv,
        top_k=args.top_k
    )

    # Only export clips if flag is set
    if args.export_clips:
        import csv, subprocess, os
        with open(args.viral_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                start_time = row['Start']
                end_time = row['End']
                out_name = f"topclip_{i:02d}.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", args.video,
                    "-ss", start_time,
                    "-to", end_time,
                    "-c", "copy",
                    out_name
                ]
                subprocess.run(cmd, check=True)
                print(f"[OK] Exported {out_name}")
