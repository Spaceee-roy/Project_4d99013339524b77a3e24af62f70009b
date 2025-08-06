import face_recognition
import cv2
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import assemblyai as aai
import os
import subprocess
from executioner import *


# AssemblyAI API Key
aai.settings.api_key = '4d99013339524b77a3e24af62f70009b'

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
        "-i", input_path,
        "-t", str(end_time - start_time),
        "-c", "copy",  # REMUX: copy streams, no encoding, encoding it is way more resource intensive and i lost a peaceful day and sleep cause of this; my eyes hurt note to self dont use -c:a libx256 ever again i have written 246 characters
        output_path,
    ]

    print(f"[cut_video] Running command: {' '.join(trim_command)}")
    result = subprocess.run(trim_command, capture_output=True, text=True)
    audio_path = output_path.replace('.mp4', '.aac')
    audio_command = [
            "ffmpeg", "-y",
            "-loglevel", "error",   
            "-i", output_path,      
            "-vn",                 
            "-acodec", "aac",                         
            "-map", "a",            
    audio_path  
        ]
    subprocess.run(audio_command, check=True)
    print(f"[cut_video] 🎧 Audio extracted: {audio_path}")

    if result.returncode != 0:
        print(f"[cut_video] ffmpeg error:\n{result.stderr.strip()}")
        return False

    print(f"[cut_video] ✅ Trimmed video saved: {output_path}")
    return True

def process_video(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video file")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Phase 1: Detecting faces (1 per second)...")

    # STEP 1: Face detection every 1 second
    face_x_positions = []
    timestamps = []
    last_x = None
    scale_factor = 0.15
    interval = int(fps * 1.0)
    frame_number = 0

    while frame_number < total_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        face_locations = face_recognition.face_locations(small_frame, model='hog')

        if face_locations:
            top, right, bottom, left = face_locations[0]
            center_x = ((right + left) // 2) / scale_factor  # Scale back to original
            last_x = center_x
        else:
            center_x = last_x if last_x is not None else frame_width // 2

        current_time = frame_number / fps
        timestamps.append(current_time)
        face_x_positions.append(center_x)

        frame_number += interval

    video_capture.release()

    # Fill gaps and save CSV
    df = pd.DataFrame({'Timestamp': timestamps, 'Face_X_Position': face_x_positions}).bfill()
    df.to_csv('face_positions.csv', index=False)

    print("Phase 2: Cropping and exporting video...")

    # STEP 2: Crop based on interpolated center positions
    video_capture = cv2.VideoCapture(video_path)
    interpolator = interp1d(df['Timestamp'], df['Face_X_Position'], kind='cubic', fill_value='extrapolate')

    new_height = frame_height
    new_width = int((9 / 16) * new_height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Faster than mp4v
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    frame_count = 0
    last_center = frame_width // 2

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_count / fps
        try:
            x_position = interpolator(current_time)
            crop_center = int(x_position if not np.isnan(x_position) else last_center)
            last_center = crop_center

            crop_left = max(0, crop_center - new_width // 2)
            crop_right = min(frame_width, crop_left + new_width)

            if crop_right - crop_left < new_width:
                crop_left = max(0, crop_right - new_width)

            cropped_frame = frame[:, crop_left:crop_right]

            if cropped_frame.shape[1] != new_width:
                cropped_frame = cv2.resize(cropped_frame, (new_width, new_height))

            out.write(cropped_frame)

        except Exception as e:
            print(f"⚠️ Frame {frame_count} error: {e}")

        frame_count += 1

    out.release()
    video_capture.release()
    cv2.destroyAllWindows()
    print("✅ Done — output saved to:", output_path)




def combine_video_audio(video_path, audio_path, output_path):
    try:
        command = [
    'ffmpeg',
    #'-loglevel', 'error',
    '-y',
    '-i', video_path,       # your original video
    '-i', audio_path,       # your new AAC audio
    '-c:v', 'copy',         # skip video encoding
    '-c:a', 'copy',         # skip audio encoding (since it's AAC already)
    '-map', '0:v:0',        # take video from first input
    '-map', '1:a:0',        # take audio from second input
    '-shortest',            # cut to the shortest stream (avoids trailing silence/black)
    output_path
]

        subprocess.run(command, check=True)
    except Exception as e:
        print(f"Error combining video and audio: {e}")

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
            "-loglevel", "error",
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
        audio_path = base_path + '.aac'
        combined_output = base_path + '_with_audio.mp4'
        subtitle_path = base_path + '.srt'
        final_output = base_path + '_final.mp4'

        cut_video(input_video, trimmed_output, start, end)
        process_video(trimmed_output, reformatted_output)
        combine_video_audio(reformatted_output, audio_path, combined_output)
        generate_subtitles(combined_output, subtitle_path)
        add_subtitles(combined_output, subtitle_path, final_output)

        for f in [trimmed_output, reformatted_output, audio_path, combined_output, subtitle_path]:
            if os.path.exists(f):
                os.remove(f)

    print("\n✅ All segments processed successfully.")


if __name__ == "__main__":
    try:
        process_all_segments()
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

        
