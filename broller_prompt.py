import json
import re
import urllib.request


ASS_FILE = "subtitles.ass"
MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_TOKENS = 100

def clean_ass_text(text):
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace("\\N", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()
def read_ass_lines(path):
    lines = []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            if not line.startswith("Dialogue:"):
                continue

            parts = line.split(",", 9)
            if len(parts) < 10:
                continue

            text = clean_ass_text(parts[9])
            if text and text not in lines:
                lines.append(text)

    return lines


def clean_ollama_response(text):
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().strip('"')


def generate_with_ollama(prompt):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))

    response = data.get("response")
    return clean_ollama_response(response)


def ask_ollama(subtitle_lines):
    prompt = f"""
You are an expert image-prompt writer for short-form video B-roll.

Use the subtitles to infer the best single image to generate. Return exactly one
concise image-generation prompt in one sentence. Do not return JSON, markdown,
labels, notes, explanations, quotes, or multiple options.

The prompt must naturally include:
- Subject: who or what is in the image
- Action/position: what is happening or where the subject is placed
- Setting: where it is
- Composition: camera angle, framing, distance
- Style/medium: photo, oil painting, 3D render, anime, editorial, etc.
- Lighting/color: time of day, mood, palette
- Details that must be present: specific objects, clothing, text, or layout
- Details to avoid: no extra people, personification, distorted hands, unwanted text, logos, etc.
- Quality constraints: sharp focus, realistic proportions, clean background

Write it as one polished prompt, not a form. Keep it vivid, specific, and useful
for image generation. Keep the final answer short. Do not include spoken lines,
speech bubbles, captions, or any visible text from the subtitles.


Subtitles:
{json.dumps(subtitle_lines, indent=2)}
"""
    return generate_with_ollama(prompt)


subtitle_lines = read_ass_lines(ASS_FILE)
image_prompt = ask_ollama(subtitle_lines)
print(image_prompt)
