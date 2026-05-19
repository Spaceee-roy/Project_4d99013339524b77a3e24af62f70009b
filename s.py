
'''
Get video path ✅
use wisper ✅
track face ✅
get words ✅
get b-roll
add b-roll
crop ✅
subtitles ✅
'''

import cv2
import face_recognition
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import subprocess
from pydub import AudioSegment
os.add_dll_directory(r"C:\Users\pcofp\ffmpeg-master-latest-win64-gpl-shared\bin")
import whisperx
import pysubs2

def crop_video_with_padding(video_path: str, csv_path: str, output_path: str, result: str,face_scale: float = 0.10, ):
    """
    Crops a video to a 9:16 aspect ratio based on horizontal face coordinates (x) from a CSV file.
    Face positions are interpolated per-frame to remove jitter.
    face_scale: fraction to downscale frames for face detection (e.g. 0.10 = 10% size).
    """
    print("📸 Phase 1: Detecting faces and generating timeline...")
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video file")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if face_scale <= 0 or face_scale > 1:
        raise ValueError("face_scale must be between 0 (exclusive) and 1 (inclusive)")

    inv_scale = 1.0 / face_scale
    face_x_positions, timestamps = [], []
    last_x = frame_width // 2  # default to center (in original coordinates)

    # Sample frames once per second to estimate face positions (detect on small frames)
    for frame_idx in tqdm(range(0, total_frames, max(1, int(fps)))):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Downscale for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=face_scale, fy=face_scale, interpolation=cv2.INTER_LINEAR)
        # Convert BGR -> RGB for face_recognition
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces on the small frame
        face_locations = face_recognition.face_locations(small_rgb)

        if face_locations:
            # face_locations returns (top, right, bottom, left) on the small frame
            top, right, bottom, left = face_locations[0]
            # Compute center x in small-frame coordinates then map back to original
            face_center_x_small = (right + left) / 2.0
            face_center_x = face_center_x_small * inv_scale
            last_x = face_center_x
        else:
            # fallback to last known position (already in original coords)
            face_center_x = last_x

        face_x_positions.append(face_center_x)
        timestamps.append(current_time)

    video_capture.release()

    # Save timeline CSV (columns: Time, X)
    df = pd.DataFrame({"Time": timestamps, "X": face_x_positions}).bfill()
    df.to_csv(csv_path, index=False)
    print(f"Saved face timeline CSV -> {csv_path}")

    # --- 1. Load Video and CSV Data ---
    print("Loading video and CSV data...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        cap.release()
        return
    except KeyError:
        print("Error: Your CSV file must contain 'Time' and 'X' columns.")
        cap.release()
        return

    # --- Interpolate face positions per frame ---
    frame_times = np.arange(frame_count) / fps
    interp_series = (
        df.set_index("Time")["X"]
        .reindex(frame_times, method=None)
        .interpolate(method="linear")
        .bfill().ffill()
    )
    coords = interp_series.to_dict()

    # --- 2. Calculate 9:16 Crop Dimensions ---
    crop_h = original_height
    crop_w = int(crop_h * 9 / 16)
    if crop_w > original_width:
        print("Error: Video is not wide enough for a 9:16 crop at full height.")
        cap.release()
        return

    # --- 3. Setup Video Writer ---
    temp_output_path = "temp_video_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (crop_w, crop_h))

    # --- 4. Process Each Frame ---
    print("Processing frames...")

    # Initialize positions
    first_face_x = coords.get(0.0, original_width // 2)
    last_applied_x = first_face_x 
    target_x = first_face_x

    threshold = 60      # Minimum movement to trigger a shift
    smoothing = 0.08    # Cinematic drift (0.05 = very slow, 0.2 = fast)

    for frame_num in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # 1. Get the raw face position for this frame
            current_face_x = coords.get(frame_num / fps, original_width // 2)

            # 2. Update the target ONLY if the face moves beyond the dead-zone
            # This prevents the camera from "vibrating" with the face
            if abs(current_face_x - target_x) > threshold:
                target_x = current_face_x

            # 3. Exponential Smoothing: The "Drift"
            # The camera constantly moves toward target_x, but only by a fraction (smoothing)
            last_applied_x = last_applied_x + smoothing * (target_x - last_applied_x)
            face_center_x = last_applied_x

            # 4. Stay within video bounds
            min_x = crop_w // 2
            max_x = original_width - (crop_w // 2)
            face_center_x = max(min_x, min(face_center_x, max_x))

            # 5. Calculate crop boundaries
            x1 = int(face_center_x - crop_w / 2)
            x2 = x1 + crop_w

            # Edge safety
            if x1 < 0:
                x1, x2 = 0, crop_w
            elif x2 > original_width:
                x2, x1 = original_width, original_width - crop_w

            # Crop and write
            cropped_frame = frame[:, x1:x2]
            if cropped_frame.shape[1] != crop_w:
                cropped_frame = cv2.resize(cropped_frame, (crop_w, crop_h))
                
            out.write(cropped_frame)

        except Exception as e:
            # Error fallback
            center_x = original_width // 2
            out.write(frame[:, center_x - (crop_w//2) : center_x + (crop_w//2)])

# --- 5. Release Video Resources ---
    print("Releasing video resources...")
    cap.release()
    out.release()

    # --- 6. Add audio back using ffmpeg (stream copy) ---
    print("Muxing audio back using ffmpeg (copy)...")
    import shutil
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg not found on PATH — cannot mux audio. Leaving temp video without audio.")

    # Build ffmpeg command:
    # - input 0: processed video (temp, contains video only)
    # - input 1: original video (to grab its audio stream)
    # - map video from input 0, audio from input 1
    # - copy codecs to avoid re-encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", temp_output_path,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", 
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", 
        "-shortest",
        output_path
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # PRINT THE END OF STDERR TO SEE THE ACTUAL ERROR
        print("⚠️ ffmpeg muxing failed. Crucial error details:")
        print(result.stderr.strip()[-1000:])  # Changed from [:1000] to [-1000:]
        
        # Fallback 1: Try muxing WITHOUT video's audio (in case video has no audio track)
        print("🔄 Retrying muxing without audio stream...")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", temp_output_path,
            "-map", "0:v:0",
            "-c:v", "copy", 
            output_path
        ]
        fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
        
        if fallback_result.returncode == 0:
            print(f"✅ Success! Video saved to: {output_path} (No audio track found in original).")
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
        else:
            # Fallback 2: Direct file move if second ffmpeg attempt fails
            try:
                shutil.move(temp_output_path, output_path)
                print(f"⚠️ Fallback: moved temp video to {output_path} (direct copy, no audio).")
            except Exception as e:
                print(f"❌ Fallback move failed: {e}")
    else:
        # Success — remove temp and finish
        try:
            os.remove(temp_output_path)
        except Exception:
            pass
        print(f"✅ Success! Final video with audio saved to: {output_path}")




def create_aesthetic_ass(word_data, output_ass_path):
    # Initialize an empty ASS file
    subs = pysubs2.SSAFile()
    
    # Define the "Aesthetic" Style
    # SecondaryColour is the 'highlight' color used for the karaoke effect
    style = pysubs2.SSAStyle(
        fontname="Montserrat", fontsize=15,
        primarycolor=pysubs2.Color(255, 255, 255),    # White (Default)
        secondarycolor=pysubs2.Color(255, 0, 0),  # Cyan (Active Word)
        outlinecolor=pysubs2.Color(0, 0, 0, 128),    # Semi-transparent black outline
        backcolor=pysubs2.Color(0, 0, 0, 128),
        bold=True, alignment=2, marginv=80, borderstyle=1, outline=2
    )
    subs.styles["Default"] = style

    # Group words into sentences or short phrases (e.g., 3-5 words per line)
    words_per_line = 6
    for i in range(0, len(word_data), words_per_line):
        line_data = word_data[i : i + words_per_line]
        
        start_time = int(line_data[0]['start'] * 1000)
        end_time = int(line_data[-1]['end'] * 1000)
        
        # Build the Karaoke string: {\k(duration)}Word
        # duration is in centiseconds (1/100th of a second)
        text_content = ""
        for word_dict in line_data:
            duration_cs = int((word_dict['end'] - word_dict['start']) * 100)
            text_content += f"{{\\k{duration_cs}}}{word_dict['word']} "

        event = pysubs2.SSAEvent(start=start_time, end=end_time, text=text_content.strip())
        subs.append(event)

    subs.save(output_ass_path)
    print(f"✅ Created Karaoke ASS: {output_ass_path}")
    # ?
def main(audio_file, video_file, cropped_output, final_output, device = "cpu"):
    # 1. Transcribe with WhisperX
    model = whisperx.load_model("small", device)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio)

    # 2. Align (this gives precise word-level timestamps)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # This results in a list of words with 'start' and 'end' keys
    word_segments = result["word_segments"]
    crop_video_with_padding(video_file, "face_position.csv", cropped_output, result = word_segments, face_scale=0.10)
    create_aesthetic_ass(word_segments, "subtitles.ass")
    subs = "subtitles.ass"
    command = [
        'ffmpeg', '-y', 
        '-i', cropped_output,
        '-vf', f"subtitles='{subs}'", # Style is now inside the file!
        '-c:a', 'copy',
        final_output
    ]

    os.system(' '.join(command))
