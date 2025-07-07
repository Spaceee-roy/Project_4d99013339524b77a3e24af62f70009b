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
    start_time = float(start_time)
    end_time = float(end_time)
    try:
        with VideoFileClip(input_path) as video:
            cut_clip = video.subclipped(start_time, end_time)
            cut_clip.write_videofile(output_path, codec="libx264", audio_codec='aac')

            # Explicitly export audio from trimmed clip
            audio_path = output_path.replace('.mp4', '.mp3')
            cut_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error in cut_video: {e}")


def process_video(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Could not open video file")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    face_x_positions, timestamps = [], []
    last_x = None
    frame_count = 0
    print("Phase 1: Detecting faces...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_count / fps
        if len(timestamps) == 0 or (current_time - timestamps[-1] >= 0.6):
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                center_x = (right + left) // 2
                last_x = center_x
            else:
                center_x = last_x if last_x is not None else None

            face_x_positions.append(center_x)
            timestamps.append(current_time)

        frame_count += 1
        print_progress_bar(frame_count, total_frames, prefix='Progress', suffix='Complete', length=100)

    video_capture.release()
    df = pd.DataFrame({'Timestamp': timestamps, 'Face_X_Position': face_x_positions}).bfill()
    df.to_csv('face_positions.csv', index=False)

    print("Phase 2: Reformatting video...")
    video_capture = cv2.VideoCapture(video_path)
    interpolator = interp1d(df['Timestamp'], df['Face_X_Position'], kind='cubic', fill_value='extrapolate')

    new_height = frame_height
    new_width = int((9/16) * new_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0
    last_center = None
    print_progress_bar(0, total_frames, prefix='Progress', suffix='Complete', length=100)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_count / fps
        try:
            x_position = interpolator(current_time)
            crop_center = int(x_position if not np.isnan(x_position) else last_center or frame_width // 2)
            last_center = crop_center

            crop_left = max(0, crop_center - new_width//2)
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
            print(f"Error on frame {frame_count}: {e}")
        frame_count += 1
        print_progress_bar(frame_count, total_frames, prefix='Progress', suffix='Complete', length=100)

    if out:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()
    print("✅ Video reformatted and saved!")

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
    executioner.segment_srt()
    df = pd.read_csv('segments.csv')

    df['Start'] = df['Start'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['End'] = df['End'].apply(lambda s: s if ':' in s else f"0:{s}")
    df['Start_seconds'] = pd.to_timedelta(df['Start']).dt.total_seconds().astype(int)
    df['End_seconds'] = pd.to_timedelta(df['End']).dt.total_seconds().astype(int)

    input_video = f'C:\\Users\\pcofp\\Desktop\\Python\\{input("Enter the video file name: ")}.mp4'

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
