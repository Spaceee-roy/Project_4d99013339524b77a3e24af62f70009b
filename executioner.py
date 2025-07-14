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

# --- Constants --- 
# These are weights which are tied in a function with this equation:  
# EDUCATIONAL_KEYWORDS * 0.5 + QNA_WORDS * 0.6 + STORY_KEYWORDS * 0.5 + KEYWORDS * 1 + TIME_WORD * 0.5 - BADWARDO * 50

# Says to the code that you are in the middle of a topic continue further.

DISCOURSE_MARKERS = {
    "however", "anyway", "so", "but", "nevertheless", "still", "though",
    "instead", "on the other hand"
}

# The code looks for educational videos

EDUCATIONAL_KEYWORDS = {
    "explain", "reason", "because", "process", "method", "fact", "data", "research", "study", "question"
}

# The code is wired to capture questions and answers

QNA_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "whom", "whose", "is", "are", "do", "does", "did", "can", "could", "should", "would"
}

# Everyone likes a good story

STORY_KEYWORDS = {
    "once", "happened", "remember", "told", "story", "experience", "felt", "saw", "heard",
    "when i", "then", "after that"
}

# Feel free to add more 😊

KEYWORDS = {
    "fission", "fusion", "space", "time", "universe", "I know", "i know", "mars", "aliens", "humans", "mind", "sky", "stars", "star",
"black hole", "quantum", "energy", "light", "darkness", "galaxy", "planets", "moon", "sun", "gravity", "science", "wormhole", "nebula", "cosmos", "dimensions",
"parallel", "reality", "simulation", "AI", "robots", "machine", "future", "technology", "deep space", "NASA", "spacex", "rocket", "astrophysics", "astronomy", "intelligence",
"brain", "thoughts", "consciousness", "dreams", "theory", "Einstein", "relativity", "expansion", "atoms", "molecules", "matter", "antimatter", "big bang", "collapse", "infinity",
"mystery", "exploration", "journey", "alien life", "terraform", "extraterrestrial", "civilization", "time travel", "teleport", "wormholes", "event horizon", "multiverse", "lightyears",
"truth", "unknown", "beyond", "curiosity", "knowledge", "awakening", "discovery", "experiment", "reality check", "dimensions", "string theory", "particles", "observer", "existence",
"eternal", "void", "creation", "beginning", "end", "evolution", "species", "lifeforms", "intelligent life", "cybernetics", "biohack", "space-time", "perception", "simulation theory",
"neural", "singularity", "nanotech", "augmented", "virtual", "metaverse", "uplift", "alien tech", "spaceship", "hover", "orb", "probe", "astro", "launch", "explosion",
"impact", "asteroid", "comet", "solar", "eclipse", "crater", "mission", "explore", "colonize", "interstellar", "galactic", "observatory", "hubble", "james webb", "andromeda",
"drake equation", "fermi paradox", "SETI", "signal", "decoded", "message", "ancient", "timeline", "hyperdrive", "warp", "hyperspace", "bioengineer", "cyberpunk", "dystopia",
"android", "spacesuit", "gravity waves", "black matter", "dark energy", "quasar", "pulsar", "magnetar", "xenon", "exo", "astrobiology", "DNA", "CRISPR", "terraforming",
"habitable", "exo planet", "milky way", "supernova", "nova", "light speed", "tachyon", "future tech", "sci-fi", "intergalactic", "astro traveler", "bio life", "machine learning",
"thought control", "brain waves", "cyber brain", "upload", "brain chip", "neuron", "synapse", "bio data", "alien DNA", "genome", "mutation", "cosmic", "planet x", "satellite",
"signals", "broadcast", "solar system", "timeline", "space walk", "zero gravity", "oxygen", "habitat", "moon base", "mars base", "explorer", "starlight", "unidentified",
"craft", "energy source", "magnetic", "quantum field", "singularity", "deep mind", "inception", "truth seeker", "AI core", "space race", "launch pad", "orbital", "celestial",
"rocket fuel", "mission control", "space junk", "microgravity", "deep thought", "space anomaly", "telescope", "explore more", "beyond limits", "infinite loop", "paradox", "emotion", "race", "racism", "racist", "war", "peace", "alien", "sentient", "origin", "invasion", "colonization", "species x", "galactic war", "first contact", "classified", "encrypted", "decoded signal",
"broadcast signal", "intercepted", "quantum leap", "space-time rip", "dark web", "nanoparticles", "cognition", "neuroplasticity", "intellect", "psychic",
"mental", "telepathy", "telekinesis", "dimensions fold", "rift", "portal", "plasma", "ion", "fusion core", "power surge",
"solar flare", "corona", "photosphere", "exoplanet", "red giant", "white dwarf", "neutron star", "gamma rays", "infrared", "radiation",
"energy burst", "cosmic dust", "dark zone", "hyperreality", "mirror dimension", "event", "cause", "effect", "time loop", "temporal shift",
"mind loop", "false memory", "dream state", "lucid", "awakening code", "bio signal", "life code", "empathy", "emotion AI", "bio tech",
"cyber system", "dream tech", "memory chip", "identity", "soul", "artificial mind", "core AI", "emotional data", "machine soul", "android emotion",
"cloning", "rebirth", "replicant", "matrix", "mainframe", "data stream", "conscious AI", "brain print", "soul transfer", "digital twin",
"mind hack", "code injection", "spirit", "ghost", "ghost in machine", "ancestors", "ancient tech", "advanced race", "lost planet", "forgotten world",
"artifact", "crashed ship", "first traveler", "alien war", "cosmic war", "destroyer", "creator", "origin myth", "simulation glitch", "time glitch",
"space fold", "quantum entanglement", "observer effect", "atomic time", "deep field", "light cone", "superintelligence", "bio circuit", "hive mind", "singular mind",
"collective", "post-human", "neo-human", "transcend", "post-Earth", "exo-society", "interplanetary", "star system", "voidwalkers", "truth code",
"transmission", "frequency", "mind frequency", "planetary AI", "ancient ruins", "time capsule", "forbidden knowledge", "infinite minds", "ultra being", "divine tech",
"beyond physics", "the unknown", "psi energy", "noosphere", "hologram", "data sphere", "universal memory", "eternity", "deep scan", "starlink",
"terraformed", "space gods", "immortality", "afterlife", "next life", "reincarnation", "timeline merge", "space station", "edge of universe", "quantum fabric",
"reality distortion", "mind expansion", "core truth", "origin code", "machine god", "neural link", "dimensional travel", "void tech", "parallel world", "alien message",
"galactic empire", "universe reset", "simulation crash", "cosmic code", "hidden reality", "multi-self", "mirror soul", "truth loop", "fate", "destiny",
"control", "resistance", "harmony", "chaos", "balance", "power", "fear", "love", "hate", "survival",
"instinct", "gene code", "epigenetics", "ancestral memory", "primal", "biotech war", "cultural collapse", "techno cult", "digital religion", "space religion",
"creator beings", "genesis", "last civilization", "digital extinction", "AI uprising", "planetfall", "terraform protocol", "AI laws", "prime directive", "dimension gate",
"hyper space", "emotional core", "conscious core", "dark signal", "omega code", "AI prophecy", "bio-robot", "planetary core", "inner universe", "spatial shift",
"void scream", "intelligent virus", "bio invasion", "quantum AI", "network mind", "edge consciousness", "interdimensional", "reality break", "alien whisper", "reality warp", "comedy", "stand-up", "storytime", "humor", "improv", "JRE", "Young Sheldon", "sitcom", "nerdy", "family", "brains", "genius", "physics", "math", "science", "education", "geek", "curiosity",
"astronomy", "astrophysics", "cosmic", "space travel", "aliens", "UFO", "extraterrestrial", "Bigfoot", "paranormal", "weird science", "quantum", "quantum physics", "relativity", "Einstein",
"black holes", "wormholes", "multiverse", "consciousness", "mind", "psychology", "neuroscience", "brain", "memory", "dreams", "lucid dreaming", "neuroplasticity", "intelligence",
"psychic", "telepathy", "conspiracy", "government secrets", "deep state", "plum island", "Lyme disease", "bioweapon theory", "biotech", "genetics", "CRISPR", "cloning", "biohacking",
"AI", "artificial intelligence", "machine learning", "robotics", "Neuralink", "digital twin", "simulation", "matrix", "virtual reality", "Augmented Reality", "metaverse", "technology",
"internet", "cybersecurity", "privacy", "surveillance", "NSA", "privacy rights", "cryptocurrency", "bitcoin", "blockchain", "crypto", "economics", "finance", "longevity", "biohacks",
"health", "wellness", "fitness", "nutrition", "ketogenic", "intermittent fasting", "supplements", "cognitive enhancement", "nootropics", "cold therapy", "blood flow restriction",
"martial arts", "MMA", "UFC", "training", "discipline", "motivation", "David Goggins", "Jocko Willink", "military", "Navy SEAL", "survival", "adventure", "exploration", "mountaineering",
"deep-sea", "travel", "Mars", "SpaceX", "NASA", "rocket", "colonization", "first contact", "ancient civilizations", "archaeology", "Atlantis", "Gobekli Tepe", "ancient tech",
"history", "culture", "politics", "elections", "Democracy", "free speech", "content moderation", "media", "journalism", "social media", "Facebook", "Twitter", "Mark Zuckerberg",
"Elon Musk", "Jordan Peterson", "Sam Harris", "Alex Jones", "Joe Rogan", "Donald Trump", "Bernie Sanders", "Kamala Harris", "policy", "healthcare", "immigration", "ICE raids",
"environment", "climate change", "wildfires", "surveillance", "government", "whistleblower", "Edward Snowden", "privacy", "mass surveillance", "free thought", "controversy",
"cancel culture", "censorship", "philosophy", "existentialism", "meaning of life", "spirituality", "meditation", "psychedelics", "ayahuasca", "psilocybin", "DMT", "mental health",
"depression", "anxiety", "PTSD", "self-help", "personal growth", "mindfulness", "resilience", "inspiration", "motivational", "biohacker", "human potential", "peak performance",
"sleep hacks", "productivity", "habits", "discipline", "storytelling", "interview", "guest", "long-form", "raw", "unfiltered", "authentic", "open conversation", "deep dive",
"controversial", "provocative", "in-depth", "analysis", "debate", "perspective", "educational", "entertaining", "insight", "callout", "rants", "opinions", "humble", "curious",
"skeptic", "critical thinking", "myth busting", "debunk", "evidence-based", "research", "science vs myth", "fact-check", "data", "statistics", "stories", "life lessons",
"relationships", "travel stories", "food", "cooking", "Guy Fieri", "outdoors", "hunting", "bow hunting", "mountain biking", "adrenaline", "extreme", "bio", "profile", "bio profile",
"gigantic", "epic", "legendary", "incredible", "shock", "wow", "mind blown", "must hear", "viral", "clip", "highlight", "snippet", "full episode", "podcast", "podcast clip",
"Spotify", "YouTube", "platform", "exclusive", "live stream", "audio", "visual", "HD", "HD audio", "guest list", "guest drop", "studio", "equipment", "mic", "headphones",
"recording", "production", "editing", "transcript", "shorts", "TikTok", "Reels", "viral clip", "share", "join", 'myth'
}

# The time_word is there to capture dates, times, data and more. 

# 1972, 73%, four years old, 12:00 pm, 9 big apples.

TIME_WORD = {
   "1", "one", "2", "two", "3", "three", "4", "four", "5", "five",
"6", "six", "7", "seven", "8", "eight", "9", "nine", "10", "ten",
"11", "eleven", "12", "twelve", "13", "thirteen", "14", "fourteen", "15", "fifteen",
"16", "sixteen", "17", "seventeen", "18", "eighteen", "19", "nineteen", "20", "twenty",
"21", "twenty-one", "22", "twenty-two", "23", "twenty-three", "24", "twenty-four", "25", "twenty-five",
"26", "twenty-six", "27", "twenty-seven", "28", "twenty-eight", "29", "twenty-nine", "30", "thirty",
"31", "thirty-one", "32", "thirty-two", "33", "thirty-three", "34", "thirty-four", "35", "thirty-five",
"36", "thirty-six", "37", "thirty-seven", "38", "thirty-eight", "39", "thirty-nine", "40", "forty",
"41", "forty-one", "42", "forty-two", "43", "forty-three", "44", "forty-four", "45", "forty-five",
"46", "forty-six", "47", "forty-seven", "48", "forty-eight", "49", "forty-nine", "50", "fifty",
"51", "fifty-one", "52", "fifty-two", "53", "fifty-three", "54", "fifty-four", "55", "fifty-five",
"56", "fifty-six", "57", "fifty-seven", "58", "fifty-eight", "59", "fifty-nine", "60", "sixty",
"61", "sixty-one", "62", "sixty-two", "63", "sixty-three", "64", "sixty-four", "65", "sixty-five",
"66", "sixty-six", "67", "sixty-seven", "68", "sixty-eight", "69", "sixty-nine", "70", "seventy",
"71", "seventy-one", "72", "seventy-two", "73", "seventy-three", "74", "seventy-four", "75", "seventy-five",
"76", "seventy-six", "77", "seventy-seven", "78", "seventy-eight", "79", "seventy-nine", "80", "eighty",
"81", "eighty-one", "82", "eighty-two", "83", "eighty-three", "84", "eighty-four", "85", "eighty-five",
"86", "eighty-six", "87", "eighty-seven", "88", "eighty-eight", "89", "eighty-nine", "90", "ninety",
"91", "ninety-one", "92", "ninety-two", "93", "ninety-three", "94", "ninety-four", "95", "ninety-five",
"96", "ninety-six", "97", "ninety-seven", "98", "ninety-eight", "99", "ninety-nine", "100", "one hundred"
}

# Gets rid of ad reads and other annoyances.

BADWARDO = {
    "sponsor", "subscribe", "advertisement"
}

# --- Utility Functions ---


def load_models(): # Loads models
    nlp = spacy.load("en_core_web_trf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer("all-mpnet-base-v2").to(device)
    kw_model = KeyBERT(model=embed_model)
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return nlp, embed_model, kw_model, device, emotion_classifier

def load_subtitles(file_path: str, nlp) -> Tuple[List[Dict], List[int]]:  # Reads the subtitle and turns it into readable language for the models.
    subs = pysrt.open(file_path, encoding='utf-8')
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

def detect_boundaries( # Detects topics and there boudaries.
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

def extract_topics(segment_text: str, kw_model, score_threshold: float = 0.45) -> Tuple[str, float, List[str]]: # Extracts topics
    keywords = kw_model.extract_keywords(
        segment_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=100
    )
    all_keywords = [kw for kw, _ in keywords]
    filtered = [(kw, score) for kw, score in keywords if score >= score_threshold]
    if filtered:
        topics = ", ".join([kw for kw, _ in filtered])
        return topics, filtered[0][1], all_keywords
    return "N/A", 0.0, all_keywords

def find_qna_words(text: str) -> List[str]: #finds qna words
    words = set(text.lower().split())
    return [word for word in QNA_WORDS if word in words]
def find_time_words(text: str) -> List[str]: #finds time words
    words = set(text.lower().split())
    return [word for word in TIME_WORD if word in words]

def score_edu_story_boost(text: str) -> float: # the equation to find good videos. go to the top to see what it does.
    text = text.lower()
    edu_score = sum(1 for word in EDUCATIONAL_KEYWORDS if word in text)
    story_score = sum(1 for word in STORY_KEYWORDS if word in text)
    keyword_score = sum(1 for word in KEYWORDS if word in text)
    qna_score = sum(1 for word in QNA_WORDS if word in text)
    time_score = sum(1 for word in TIME_WORD if word in text)
    bad_score = sum(1 for word in BADWARDO if word in text)
    return edu_score * 0.5 + story_score * 0.5 + keyword_score * 1 + qna_score * 0.6 + time_score * 0.5 - bad_score * 50

def named_entity_score(nlp, text: str) -> int:
    doc = nlp(text)
    informative_types = {"PERSON", "ORG", "GPE", "DATE", "TIME", "NORP"}
    return sum(1 for ent in doc.ents if ent.label_ in informative_types)

def emotion_score(emotion_classifier, topics: str, segment_preview: str) -> Tuple[str, float]:
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

def break_segment(segment_info, max_duration=180):
    """Breaks a segment into smaller segments if it exceeds max_duration."""
    segments = []
    start_idx = 0
    while start_idx < len(segment_info):
        start_time = segment_info[start_idx]["start"]
        total_duration = 0
        end_idx = start_idx
        while end_idx < len(segment_info):
            end_time = segment_info[end_idx]["end"]
            duration = (time_to_datetime(end_time) - time_to_datetime(start_time)).total_seconds()
            if duration > max_duration:
                break
            total_duration = duration
            end_idx += 1
        if end_idx == start_idx:
            end_idx += 1  # Ensure at least one sentence per segment
        segments.append((start_idx, end_idx, total_duration))
        start_idx = end_idx
    return segments

def generate_segments(
    boundaries: List[int],
    sentence_entries: List[Dict],
    kw_model,
    nlp,
    emotion_classifier,
    min_sentences: int,
    min_duration: int,
    max_duration: int = 180,
    topic_score_threshold: float = 0.45,
    score_threshold: float = 10
) -> List[Dict]:
    """Generate and score segments, breaking up long ones and filtering by score."""
    segments = []
    for idx in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[idx], boundaries[idx + 1]
        segment_info = sentence_entries[seg_start:seg_end]
        sub_segments = break_segment(segment_info, max_duration)
        for sub_start, sub_end, duration in sub_segments:
            if sub_end - sub_start < min_sentences or duration < min_duration:
                continue
            sub_segment_info = segment_info[sub_start:sub_end]
            start_time = sub_segment_info[0]["start"]
            end_time = sub_segment_info[-1]["end"]
            segment_text = " ".join(entry["text"] for entry in sub_segment_info)
            topics, top_score, all_keywords = extract_topics(segment_text, kw_model, topic_score_threshold)
            qna_words = find_qna_words(segment_text)
            if topics == "N/A":
                continue
            score, emotion, emotion_intensity = segment_score(segment_text, top_score, nlp, emotion_classifier, topics, segment_text)
            if score < score_threshold:
                continue  # Skip segments with low score
            segments.append({
                "TopTopicScore": top_score,
                "Score": score,
                "Emotion": emotion,
                "EmotionIntensity": emotion_intensity,
                "Segment": len(segments) + 1,
                "Start": start_time,
                "End": end_time,
                "Duration": duration,
                "Preview": segment_text[:400] + ("..." if len(segment_text) > 400 else ""),
                "AllKeywords": ", ".join(all_keywords),
                "QnAWords": ", ".join(qna_words),
            })
    top_segments = sorted(segments, key=lambda x: x["Score"], reverse=True)
    return top_segments

def save_segments_to_csv(segments: List[Dict], output_path: str):
    """Save segments to CSV."""
    df = pd.DataFrame(segments)
    df.to_csv(output_path, index=False)
    # print(f"\n✅ Top segments saved to {output_path}.\n")

def segment_srt_pipeline(
    file_path: str,
    dynamic_std_factor: float = 1.5,
    time_gap_threshold: float = 4,
    min_sentences: int = 3,
    min_duration: int = 20,
    max_duration: int = 180,
    topic_score_threshold: float = 0.45,
    score_threshold: float = 10
):
    """Main pipeline to process SRT and output scored segments."""
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
        max_duration,
        topic_score_threshold,
        score_threshold
    )
    output_path = "segments.csv"
    save_segments_to_csv(segments, output_path)

if __name__ == "__main__":
    segment_srt_pipeline('n.srt')
