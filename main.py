import face_recognition
import cv2
import pandas as pd
import numpy as np
import assemblyai as aai
import os
import subprocess
from executioner import *
import time
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import json
import csv

# AssemblyAI API Key
aai.settings.api_key = ''

def print_progress_bar(iteration, total, prefix='', suffix='', length=100):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '•' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()

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


def generate_subtitles(video_path, subtitle_path):
    # print("🎧 Transcribing audio...")
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano))
    transcript = transcriber.transcribe(video_path)
    subtitles = transcript.export_subtitles_srt()

    with open(subtitle_path, 'w') as f:
        f.write(subtitles)
    # print("✅ Subtitle file created successfully!")

def add_subtitles(video_path, subtitle_path, output_path):
    try:
        workdir = os.path.dirname(os.path.abspath(video_path))
        os.chdir(workdir)
        video_file = os.path.basename(video_path)
        subtitle_file = os.path.basename(subtitle_path)
        output_file = os.path.abspath(output_path)

        command = [
            'ffmpeg',
            # "-loglevel", "error",
            '-i', video_file,
            '-vf', f"subtitles={subtitle_file}:force_style='FontName=Roboto,Alignment=2,MarginV=75,FontSize=14,Bold=1,PrimaryColour=&HFFFF&'",
            '-c:a', 'copy',
            output_file
        ]
        # print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        # print(f"✅ Video with subtitles saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding subtitles: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def process_all_segments():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    filepath = input("Enter SRT file name (e.g., subtitles.srt): ").strip()
    input_video_name = input("Enter video file name (e.g., video.mp4): ").strip()

    srt_path = os.path.join(script_dir, filepath)
    input_video = os.path.join(script_dir, input_video_name)

    override = input("Type override code to use existing segments.csv (leave blank to regenerate): ")
    if override == "y":
        print("⚠️: Using existing segments.csv (no new segmentation will be performed).")
    else:
        segmenter = VideoSegmenter()
        segmenter.process_file(srt_path)

    df = pd.read_csv(os.path.join(script_dir, 'segments.csv'))

    df['Start'] = df['Start'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['End'] = df['End'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['Start_seconds'] = pd.to_timedelta(df['Start']).dt.total_seconds().astype(int)
    df['End_seconds'] = pd.to_timedelta(df['End']).dt.total_seconds().astype(int)

    for idx, row in df.iterrows():
        start = row['Start_seconds']
        end = row['End_seconds']

        print(f"\n--- Processing Segment {idx} | {start}s to {end}s ---")
        temp_dir = os.path.join(script_dir,"temp")
        base_path = os.path.join(temp_dir, f"segment_{idx}")
        trimmed_output = base_path + '.mp4'
        reformatted_output = base_path + '_reformatted.mp4'
        combined_output = base_path + '_with_audio.mp4'
        subtitle_path = base_path + '.srt'
        final_output = base_path + '_final.mp4'

        cut_video(input_video, trimmed_output, start, end)
        process_video_and_audio(trimmed_output,face_csv_path='face_position.csv', output_path= combined_output)
        generate_subtitles(combined_output, subtitle_path)
        add_subtitles(combined_output, subtitle_path, final_output)

        for f in [trimmed_output, reformatted_output, combined_output, subtitle_path]:
            if os.path.exists(f):
                os.remove(f)

    print("\n✅ All segments processed successfully.")


if __name__ == "__main__":
    try:
        start_time = time.time()
        process_all_segments()
        total_time = time.time() - start_time
        print(f"🎉 Total time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")

'''

if you are ever bored:

Add-Type -AssemblyName System.speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.SelectVoice('Microsoft Zira Desktop')
$RandomCatFact = (ConvertFrom-Json (Invoke-WebRequest -Uri "https://catfact.ninja/fact" -UseBasicParsing).Content).fact
Write-Host $RandomCatFact
$speak.Speak("did you know that $RandomCatFact")

type that into powershell
'''
