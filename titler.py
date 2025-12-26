import pysrt
import re
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_KEY'))

def generate_video_metadata(transcript):
    if not transcript or len(transcript.strip()) < 5:
        return "Untitled Clip", "No description available."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a viral YouTube master. Based ONLY on the provided transcript:\n"
                        "1. Create a shocking title (4-7 words).\n"
                        "2. Create a 2-sentence description summarizing the hook.\n"
                        "CRITICAL: Do not invent facts. If the transcript doesn't mention the sun/stars, you must not mention them.\n"
                        "Return ONLY a JSON object: {\"title\": \"...\", \"description\": \"...\"}"
                    )
                },
                {"role": "user", "content": f"Transcript: {transcript}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        data = json.loads(chat_completion.choices[0].message.content)
        return data.get("title"), data.get("description")
    except Exception as e:
        print(f"Metadata generation failed: {e}")
        return "Original Clip", "Description unavailable"

def _sanitize_filename(filename: str):
    # Remove forbidden Windows characters: < > : " / \ | ? *
    clean_name = re.sub(r'[<>:"/\\|?*]', '', filename)
    return " ".join(clean_name.split()).strip()[:100]

def get_metadata_package(srt_path: str):
    subs = pysrt.open(srt_path, encoding="utf-8")
    full_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
    
    raw_title, description = generate_video_metadata(full_text)
    safe_name = _sanitize_filename(raw_title)
    
    return {
        "filename": safe_name.title(),
        "title": raw_title,
        "description": description
    }
