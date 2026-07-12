import json
import re
import urllib.request


ASS_FILE = "subtitles.ass"
MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_TOKENS = 1000
OUTPUT_COUNT = 3
IMAGE_DURATION_SECONDS = 5.0
OUTPUT_FILE = "image_prompts.json"

def clean_ass_text(text):
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace("\\N", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()
def ass_time_to_seconds(value):
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def read_ass_lines(path):
    lines = []
    by_text = {}

    with open(path, "r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            if not line.startswith("Dialogue:"):
                continue

            parts = line.split(",", 9)
            if len(parts) < 10:
                continue

            text = clean_ass_text(parts[9])
            if not text:
                continue

            start = ass_time_to_seconds(parts[1])
            end = ass_time_to_seconds(parts[2])
            if text in by_text:
                by_text[text]["start"] = min(by_text[text]["start"], start)
                by_text[text]["end"] = max(by_text[text]["end"], end)
            else:
                item = {"start": start, "end": end, "text": text}
                by_text[text] = item
                lines.append(item)

    return lines


def split_into_sections(lines, section_count=OUTPUT_COUNT):
    """Split subtitle lines into contiguous, evenly sized sections."""
    if section_count < 1:
        raise ValueError("section_count must be at least 1")
    if not lines:
        return []

    section_count = min(section_count, len(lines))
    base_size, remainder = divmod(len(lines), section_count)
    sections = []
    start = 0

    for index in range(section_count):
        size = base_size + (1 if index < remainder else 0)
        end = start + size
        sections.append(lines[start:end])
        start = end

    return sections


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
    timed_subtitles = [
        {
            "start_seconds": line["start"],
            "end_seconds": line["end"],
            "text": line["text"],
        }
        for line in subtitle_lines
    ]
    prompt = f"""
You are an expert image-prompt writer for short-form video B-roll.

Use the timed subtitles to infer the best single image to generate and the best
moment to introduce it. The image will remain visible for exactly five seconds.
Choose start_seconds from within the supplied subtitle section, at the moment
whose spoken idea is best represented by the image.

Return only valid JSON with exactly these fields:
{{"start_seconds": 12.42, "image_prompt": "One concise prompt sentence."}}
Do not return markdown, notes, explanations, or multiple options.

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


Subtitle section:
{json.dumps(timed_subtitles, indent=2)}
"""
    response = generate_with_ollama(prompt)
    response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.IGNORECASE)
    result = json.loads(response)

    section_start = subtitle_lines[0]["start"]
    section_end = subtitle_lines[-1]["end"]
    start = float(result["start_seconds"])
    if not section_start <= start <= section_end:
        start = section_start

    return {
        "start_seconds": start,
        "image_prompt": str(result["image_prompt"]).strip(),
    }


def main():
    subtitle_lines = read_ass_lines(ASS_FILE)
    sections = split_into_sections(subtitle_lines)

    if not sections:
        raise RuntimeError(f"No dialogue lines found in {ASS_FILE}")

    outputs = []
    for index, section in enumerate(sections, start=1):
        output = ask_ollama(section)
        output["output_number"] = index
        outputs.append(output)
        print(f"Output {index}: {json.dumps(output, ensure_ascii=False)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(outputs)} outputs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
