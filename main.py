import os
import time
import csv
import json
import subprocess
from datetime import timedelta
from tqdm import tqdm
import pandas as pd
import pysrt
from pathlib import Path
# try to import the VideoSegmenter from the viral_segmenter module
try:
    from executioner import VideoSegmenter
except Exception:
    VideoSegmenter = None
    

# import existing helpers (face_recognition, cv2 etc.)
try:
    import face_recognition
    import cv2
    import numpy as np
    from scipy.interpolate import CubicSpline
except Exception:
    # If these imports fail it's okay; the script will notify at runtime
    pass

# -----------------------------
# Utility helpers
# -----------------------------

def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '•' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def parse_time_like(t):
    """Parse a variety of time-like inputs to seconds (float).
    Accepts: floats/ints (seconds), 'MM:SS', 'HH:MM:SS', pandas Timedelta strings, or 'H:MM:SS.sss'."""
    if pd.isna(t):
        return 0.0
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, timedelta):
        return t.total_seconds()
    s = str(t).strip()
    # if looks like a number
    try:
        return float(s)
    except Exception:
        pass
    # try pandas to_timedelta
    try:
        td = pd.to_timedelta(s)
        return td.total_seconds()
    except Exception:
        pass
    # fallback: split by :
    parts = s.split(':')
    try:
        parts = [float(p) for p in parts]
    except Exception:
        return 0.0
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0]*60 + parts[1]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    return 0.0

def cut_video(input_path, output_path, start_time, end_time):
    import shutil

    # Ensure absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    # Check input file exists
    if not os.path.isfile(input_path):
        print(f"[cut_video] ❌ Input file missing: {input_path}")
        return False

    # Check ffmpeg exists on PATH
    if not shutil.which("ffmpeg"):
        print("[cut_video] ❌ ffmpeg not found on PATH")
        return False

    trim_command = [
       "ffmpeg", "-y",
    "-ss", str(start_time),
    "-to", str(end_time),
    "-i", input_path,
    "-c", "copy",
    output_path
    ]

    print(f"[cut_video] Running command: {' '.join(trim_command)}")
    result = subprocess.run(trim_command, capture_output=True, text=True, )


    if result.returncode != 0:
        print(f"[cut_video] ffmpeg error:\n{result.stderr.strip()}")
        return False

    print(f"[cut_video] ✅ Trimmed video saved: {output_path}")
    return True
# -----------------------------
# Existing processing functions (adapted from your main.py)
# - cut_video is no longer used for top clips because viral_segmenter will export trimmed clips
# -----------------------------

def process_video_and_audio(video_path, face_csv_path, output_path):
    video_path = os.path.abspath(video_path)
    face_csv_path = os.path.abspath(face_csv_path)
    output_path = os.path.abspath(output_path)

    """
    1. Locate and record face position in csv file
    2. Retrieve and crop face into a 9:16 format.
    """

    

    print("📸 Phase 1: Detecting faces and generating timeline...")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video file")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    face_x_positions, timestamps = [], []
    last_x = frame_width // 2  # default to center

    # Run detection once per second
    for frame_idx in tqdm(range(0, total_frames, int(fps))):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_capture.read()
        if not ret: # if there is no frames stop trying
            break

        current_time = frame_idx / fps
        face_locations = face_recognition.face_locations(frame)

        if face_locations:
            top, right, bottom, left = face_locations[0] # top and bottom needed to soak up value 0 and 2 outputed from face_locations(0)
            face_center_x = (right + left) / 2 # gets the boundary and finds the center


            last_x = face_center_x
        else:
            face_center_x = last_x # fail safe

        face_x_positions.append(face_center_x)
        timestamps.append(current_time)

    video_capture.release()

    df = pd.DataFrame({'Timestamp': timestamps, 'Face_X_Position': face_x_positions}).bfill()
    df.to_csv(face_csv_path, index=False)

    print("✂️ Phase 2: Cropping and saving frames...")

    # Get video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", video_path
    ]
    proc = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    video_info = json.loads(proc.stdout)
    video_width = int(video_info["streams"][0]["width"])
    video_height = int(video_info["streams"][0]["height"])
    print(f"Source video: {video_width} x {video_height}")

    # Desired crop: 9:16 portrait aspect ratio
    desired_width = (9 / 16) * video_height
    dw_int = int(round(desired_width))
    oh_int = video_height
    half_w = desired_width / 2.0

    # Load CSV data
    times = []
    face_centers = []
    with open(face_csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t = float(r["Timestamp"])
            cx = float(r["Face_X_Position"])
            times.append(t)
            adjustcx = cx - half_w
            face_centers.append(adjustcx)

    if len(times) < 2:
        raise SystemExit("Need at least two CSV points for interpolation.")

    # Smooth with cubic spline
    cs = CubicSpline(times, face_centers, bc_type='natural')

    # Clamp function
    def clamp_left_from_center(center_x):
        left = center_x - half_w
        if left < 0:
            return 0.0
        elif left + desired_width > video_width:
            return float(video_width - desired_width)
        return left

    # Build ffmpeg X-position expression
    seg_terms = []
    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]

        cx0 = float(cs(t0))
        cx1 = float(cs(t1))

        x0 = clamp_left_from_center(cx0)
        x1 = clamp_left_from_center(cx1)

        slope = 0.0 if abs(t1 - t0) < 1e-6 else (x1 - x0) / (t1 - t0)

        term = (
            f"(between(t\\,{t0:.3f}\\,{t1:.3f})*("
            f"{x0:.3f}+({slope:.6f})*(t-{t0:.3f})"
            f"))"
        )
        seg_terms.append(term)

    last_center = clamp_left_from_center(face_centers[-1])
    tail = f"({last_center:.3f})"

    expr = "(" + "+".join(seg_terms) + "+" + tail + ")"

    # Final crop filter
    crop_filter = f"crop={dw_int}:{oh_int}:{expr}:0"
    print("Crop filter length:", len(crop_filter))

    # Run ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", crop_filter,
        "-c:a", "copy",
        output_path
    ]
    print("Running ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"Done — output saved to {output_path}")


def seconds_to_srt_time(seconds):
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=secs, milliseconds=milliseconds)


def extract_srt(input_srt_path, output_srt_path, start_seconds, end_seconds):
    subs = pysrt.open(input_srt_path, encoding='utf-8')
    start = seconds_to_srt_time(start_seconds)
    end = seconds_to_srt_time(end_seconds)

    selected_subs = pysrt.SubRipFile(
        [sub for sub in subs if sub.start >= start and sub.end <= end]
    )

    if selected_subs:
        first_start_seconds = (
            selected_subs[0].start.hours * 3600 +
            selected_subs[0].start.minutes * 60 +
            selected_subs[0].start.seconds +
            selected_subs[0].start.milliseconds / 1000.0
        )
        selected_subs.shift(seconds=-first_start_seconds)

    selected_subs.save(output_srt_path, encoding='utf-8')
    print(f"✅ Extracted subtitles saved to {output_srt_path}")


from pathlib import Path
import subprocess
import sys

def add_subtitles(video_path, subtitle_path, output_path):
    try:
        # Ensure absolute paths and use forward slashes for ffmpeg
        video_path = Path(video_path).resolve().as_posix()
        subtitle_path = Path(subtitle_path).resolve().as_posix()
        output_path = Path(output_path).resolve().as_posix()

        # Escape colons in Windows drive letters for ffmpeg filter
        # (ffmpeg requires '\:' inside the subtitles= filter for Windows drive letters)
        if sys.platform.startswith("win"):
            if ':' in subtitle_path[:3]:
                subtitle_path = subtitle_path.replace(':', '\\:')

        # Build ffmpeg command without URL encoding
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf',
            f"subtitles='{subtitle_path}':force_style='FontName=Roboto,Alignment=2,MarginV=75,MarginL=75,MarginR=75,FontSize=14,BorderStyle=3, Outline=2,Shadow=0,BackColour=&H00620AFA&,Bold=1,PrimaryColour=&HFFFFFF&'",
            '-c:a', 'copy',
            output_path
        ]

        # Run the command
        subprocess.run(command, check=True)
        print(f"✅ Subtitles added to {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error adding subtitles: {e.stderr if hasattr(e, 'stderr') else str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")




# -----------------------------
# Orchestration: create top clips with Viral Segmenter then run your face/crop/subtitle pipeline
# -----------------------------

from pathlib import Path

def force_windows_path(p):
    return Path(p).resolve().as_posix()  # Always C:/... format for ffmpeg

def process_all_segments():
    script_dir = Path(__file__).resolve().parent

    filepath = input("Enter SRT file name (e.g., subtitles.srt): ").strip()
    input_video_name = input("Enter video file name (e.g., video.mp4): ").strip()

    srt_path = force_windows_path(script_dir / filepath)
    input_video = force_windows_path(script_dir / input_video_name)

    override = input("Type override code to use existing segments.csv (leave blank to regenerate): ")
    if override != "y":
        segmenter = VideoSegmenter()
        segmenter.process_file(srt_path)
    else:
        print("⚠️: Using existing viral_clips.csv (no new segmentation will be performed).")

    df = pd.read_csv(script_dir / 'viral_clips.csv')

    df['Start'] = df['Start'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['End'] = df['End'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['Start_seconds'] = pd.to_timedelta(df['Start']).dt.total_seconds().astype(int)
    df['End_seconds'] = pd.to_timedelta(df['End']).dt.total_seconds().astype(int)
    goodname = input_video_name.replace(".mp4", "")

    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    for idx, row in df.iterrows():
        start = row['Start_seconds']
        end = row['End_seconds']

        print(f"\n--- Processing Segment {idx} | {start}s to {end}s ---")

        base_path = temp_dir / f"{goodname.strip()}_topclip_{idx:02d}"

        trimmed_output = force_windows_path(base_path.with_suffix('.mp4'))
        reformatted_output = force_windows_path(base_path.with_name(base_path.name + '_reformatted.mp4'))
        combined_output = force_windows_path(base_path.with_name(base_path.name + '_with_audio.mp4'))
        subtitle_path = force_windows_path(base_path.with_suffix('.srt'))
        final_output = force_windows_path(base_path.with_name(base_path.name + '_final.mp4'))
        print(trimmed_output, reformatted_output, combined_output, subtitle_path, final_output)
        if not cut_video(input_video, trimmed_output, start, end):
            continue

        process_video_and_audio(trimmed_output, face_csv_path='face_position.csv', output_path=combined_output)
        extract_srt(srt_path, subtitle_path, start, end)
        add_subtitles(combined_output, subtitle_path, final_output)

        for f in [trimmed_output, reformatted_output, combined_output]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception as e:
                print(e)
                break

    print("\n✅ All segments processed successfully.")


if __name__ == '__main__':
    
    start = time.time()
    process_all_segments()
    print(f"Total time: {time.time() - start:.1f}s")
