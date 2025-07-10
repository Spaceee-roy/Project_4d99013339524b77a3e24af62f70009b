import face_recognition
import cv2
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from moviepy import VideoFileClip, AudioFileClip
import assemblyai as aai
import os
import subprocess
import executioner


# AssemblyAI API Key
aai.settings.api_key = ''

def print_progress_bar(iteration, total, prefix='', suffix='', length=10):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '•' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def cut_video(input_path, output_path, start_time, end_time):
    try:
        # Trim video with accurate timestamps (frame-accurate)
        trim_command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            output_path
        ]
        subprocess.run(trim_command, check=True)
        print(f"[cut_video] ✅ Video trimmed: {output_path}")

        # Extract audio separately to MP3
        audio_path = output_path.replace('.mp4', '.mp3')
        audio_command = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-q:a", "0",
            "-map", "a",
            audio_path
        ]
        subprocess.run(audio_command, check=True)
        print(f"[cut_video] 🎧 Audio extracted: {audio_path}")

    except subprocess.CalledProcessError as e:
        print(f"[cut_video] ❌ ffmpeg error: {e}")
    except Exception as e:
        print(f"[cut_video] ❌ Unexpected error: {e}")


def process_video(video_path, output_path, detection_interval=0.5):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video file")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    detection_interval_frames = int(fps * detection_interval)

    # Target aspect ratio 9:16 (width:height)
    new_height = frame_height
    new_width = int((9 / 16) * new_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    tracker = None
    last_x = None
    timestamps = []
    face_x_positions = []

    print_progress_bar(0, total_frames, prefix='Progress', suffix='Complete', length=100)

    # PHASE 1: Face detection + tracking & logging
    frame_idx = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_idx / fps
        center_x = None

        if frame_idx % detection_interval_frames == 0 or tracker is None:
            rgb_frame = frame[:, :, ::-1]  # BGR to RGB for face_recognition
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                center_x = (left + right) // 2
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (left, top, right - left, bottom - top))
            else:
                tracker = None
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center_x = x + w // 2

        if center_x is None:
            center_x = last_x if last_x is not None else frame_width // 2
        else:
            last_x = center_x

        face_x_positions.append(center_x)
        timestamps.append(current_time)

        frame_idx += 1
        print_progress_bar(frame_idx, total_frames, prefix='Progress', suffix='Complete', length=100)

    video_capture.release()

    # Interpolate face positions for smooth cropping
    interpolator = interp1d(timestamps, face_x_positions, kind='cubic', fill_value='extrapolate')

    # PHASE 2: Reopen video to crop frames centered on face
    video_capture = cv2.VideoCapture(video_path)
    frame_idx = 0
    last_center = None
    out = None

    print_progress_bar(0, total_frames, prefix='Cropping', suffix='Complete', length=100)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_idx / fps
        try:
            x_position = interpolator(current_time)
            crop_center = int(x_position if not np.isnan(x_position) else last_center or frame_width // 2)
            last_center = crop_center

            crop_left = max(0, crop_center - new_width // 2)
            crop_right = min(frame_width, crop_left + new_width)
            if crop_right - crop_left < new_width:
                crop_left = max(0, crop_right - new_width)

            cropped_frame = frame[:, crop_left:crop_right]
            if cropped_frame.shape[1] != new_width:
                cropped_frame = cv2.resize(cropped_frame, (new_width, new_height))

            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

            out.write(cropped_frame)
        except Exception as e:
            print(f"Error cropping frame {frame_idx}: {e}")

        frame_idx += 1
        print_progress_bar(frame_idx, total_frames, prefix='Cropping', suffix='Complete', length=100)

    if out:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()

    # Save the tracked positions (optional)
    df = pd.DataFrame({'Timestamp': timestamps, 'Face_X_Position': face_x_positions}).bfill()
    df.to_csv('face_positions.csv', index=False)

    print("✅ Video processed, cropped, and saved!")


def combine_video_audio(video_path, audio_path, output_path):
    try:
        command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"Error combining video and audio: {e}")

def generate_subtitles(video_path, subtitle_path):
    print("🎧 Transcribing audio...")
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano))
    transcript = transcriber.transcribe(video_path)
    subtitles = transcript.export_subtitles_srt()

    with open(subtitle_path, 'w') as f:
        f.write(subtitles)
    print("✅ Subtitle file created successfully!")

def add_subtitles(video_path, subtitle_path, output_path):
    try:
        workdir = os.path.dirname(os.path.abspath(video_path))
        os.chdir(workdir)
        video_file = os.path.basename(video_path)
        subtitle_file = os.path.basename(subtitle_path)
        output_file = os.path.abspath(output_path)

        command = [
            'ffmpeg',
            '-i', video_file,
            '-vf', f"subtitles={subtitle_file}:force_style='FontName=Roboto,Alignment=2,MarginV=75,FontSize=14,Bold=1,PrimaryColour=&HFFFF&'",
            '-c:a', 'copy',
            output_file
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"✅ Video with subtitles saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding subtitles: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def process_all_segments():
    filepath = input("SRT file name: ")
    executioner.segment_srt_pipeline(filepath)
    df = pd.read_csv('segments.csv')

    df['Start'] = df['Start'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['End'] = df['End'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['Start_seconds'] = pd.to_timedelta(df['Start']).dt.total_seconds().astype(int)
    df['End_seconds'] = pd.to_timedelta(df['End']).dt.total_seconds().astype(int)

    input_video = f'C:\\Users\\pcofp\\Desktop\\Python\\{input("Enter the video file name: ")}'

    for idx, row in df.iterrows():
        start = row['Start_seconds']
        end = row['End_seconds']

        print(f"\n--- Processing Segment {idx} | {start}s to {end}s ---")

        base_path = f'C:\\Users\\pcofp\\Desktop\\Python\\temp\\segment_{idx}'
        trimmed_output = base_path + '.mp4'
        reformatted_output = base_path + '_reformatted.mp4'
        audio_path = base_path + '.mp3'
        combined_output = base_path + '_with_audio.mp4'
        subtitle_path = base_path + '.srt'
        final_output = base_path + '_final.mp4'

        cut_video(input_video, trimmed_output, start, end)
        process_video(trimmed_output, reformatted_output)
        combine_video_audio(reformatted_output, audio_path, combined_output)
        generate_subtitles(combined_output, subtitle_path)
        add_subtitles(combined_output, subtitle_path, final_output)
        os.remove(trimmed_output)
        os.remove(reformatted_output)
        os.remove(audio_path)
        os.remove(combined_output)
        os.remove(subtitle_path)
    print("\n✅ All segments processed successfully.")

if __name__ == "__main__":
    try:
        process_all_segments()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
