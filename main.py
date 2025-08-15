import os
import time
import csv
import json
import subprocess
from datetime import timedelta
import tqdm
import pandas as pd
import pysrt

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
    """Convert seconds (float/int) to pysrt.SubRipTime."""
    td = timedelta(seconds=seconds)
    # Extract hours, minutes, seconds, milliseconds
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=secs, milliseconds=milliseconds)


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


def add_subtitles(video_path, subtitle_path, output_path):
    try:
        workdir = os.path.dirname(os.path.abspath(video_path))
        video_file = os.path.basename(video_path)
        subtitle_file = os.path.abspath(subtitle_path)
        output_file = os.path.abspath(output_path)

        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"subtitles={subtitle_file}:force_style='FontName=Roboto,Alignment=2,MarginV=75,FontSize=14,Bold=1,PrimaryColour=&HFFFF&'",
            '-c:a', 'copy',
            output_file
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error adding subtitles: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


# -----------------------------
# Orchestration: create top clips with Viral Segmenter then run your face/crop/subtitle pipeline
# -----------------------------

def process_all_top_clips(srt_path: str, input_video_path: str, top_k: int = 5, keep_all: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    srt_path = os.path.abspath(srt_path)
    input_video_path = os.path.abspath(input_video_path)

    if VideoSegmenter is None:
        raise RuntimeError("Could not import VideoSegmenter. Make sure viral_segmenter.py is on the PYTHONPATH and importable.")

    seg = VideoSegmenter()
    # export the top clips (trimmed mp4 files) and write viral_clips.csv
    viral_csv = os.path.join(script_dir, 'viral_clips.csv')
    seg.process_file(srt_path, output_path=os.path.join(script_dir,'segments.csv'), clips_path=viral_csv, top_k=top_k, export_clips=True, keep_all=keep_all)

    if not os.path.exists(viral_csv):
        raise RuntimeError("viral_clips.csv was not created. Aborting.")

    df = pd.read_csv(viral_csv)

    base = os.path.splitext(os.path.basename(input_video_path))[0]
    # iterate top clips: filenames follow the pattern <base>_topclip_##.mp4 in ascending order
    num_clips = len(df)
    for idx in range(1, num_clips + 1):
        clip_filename = f"{base}_topclip_{idx:02d}.mp4"
        clip_path = os.path.join(script_dir, clip_filename)
        if not os.path.exists(clip_path):
            print(f"⚠️ Expected clip not found: {clip_path} (skipping)")
            continue

        # extract start/end from df row (df rows are ordered by score in viral_clips.csv)
        row = df.iloc[idx-1]
        start_s = parse_time_like(row.get('Start', 0))
        end_s = parse_time_like(row.get('End', 0))

        print(f"\n--- Processing top clip {idx}/{num_clips}: {clip_filename} | {start_s:.1f}s → {end_s:.1f}s ---")

        temp_dir = os.path.join(script_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        subtitle_path = os.path.join(temp_dir, f"clip_{idx:02d}.srt")
        # extract and shift SRT for this clip
        extract_srt(srt_path, subtitle_path, start_seconds=start_s, end_seconds=end_s)

        # run face-crop pipeline on the trimmed clip and produce a cropped file
        cropped_output = os.path.join(temp_dir, f"{base}_topclip_{idx:02d}_cropped.mp4")
        try:
            # Call the user's heavy-lifting function - here we assume it's implemented in this file
            # If you kept the original function body in this file, it will run. Otherwise, replace
            # the stub above with the real implementation.
            process_video_and_audio(clip_path, face_csv_path=os.path.join(temp_dir, f"face_clip_{idx:02d}.csv"), output_path=cropped_output)
        except NotImplementedError as e:
            print("Please paste your original 'process_video_and_audio' implementation into this file.\n", e)
            return

        # burn subtitles onto the cropped output
        final_output = os.path.join(script_dir, f"{base}_topclip_{idx:02d}_final.mp4")
        add_subtitles(cropped_output, subtitle_path, final_output)

        # clean up temporary files for this clip
        for f in [clip_path, cropped_output, subtitle_path]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    print("\n✅ All top clips processed.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run viral segmenter -> crop -> burn subtitles pipeline (top clips only)')
    parser.add_argument('srt', help='Path to the SRT file')
    parser.add_argument('video', help='Path to the source video')
    parser.add_argument('--top_k', type=int, default=5, help='How many top clips to process')
    parser.add_argument('--keep_all', action='store_true', help='Keep the full segments.csv')
    args = parser.parse_args()
    srt = input("Enter srt path: ")
    video = input("Enter video path: ")
    start = time.time()
    process_all_top_clips(start, video, top_k=5, keep_all= True)
    print(f"Total time: {time.time() - start:.1f}s")
