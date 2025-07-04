import face_recognition
import cv2
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from scipy.interpolate import interp1d
from moviepy import VideoFileClip, AudioFileClip
import tkinter as tk
from tkinter import filedialog
import assemblyai as aai
import os
import subprocess

# AssemblyAI API Key
aai.settings.api_key = '4d99013339524b77a3e24af62f70009b'
import sys
def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
        percent = f"{100 * (iteration / float(total)):.1f}"
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
        if iteration == total:
            print()
# File Picker
def pick_file(filetypes, title="Select a file"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=filetypes, title=title)

# Save As Dialog
def save_file(default_ext, title="Save file as"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.asksaveasfilename(defaultextension=default_ext, title=title)

# Cut video and extract audio
def cut_video(input_path, output_path, start_time, end_time):
    # Convert string inputs to float for time values
    start_time = float(start_time)
    end_time = float(end_time)
    
    try:
        # Load the video
        with VideoFileClip(input_path) as video:
            # Cut the video
            cut_video = video.subclipped(start_time, end_time)
            
            # Extract and save audio
            audio_path = output_path.replace('.mp4', '.mp3')
            cut_video.audio.write_audiofile(audio_path)
            
            # Write the video
            cut_video.write_videofile(
                output_path,
                codec="libx264"
            )
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Reformat video to follow face position
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
        print_progress_bar(frame_count, total_frames, prefix='Progress', suffix='Complete', length=40)

    video_capture.release()
    df = pd.DataFrame({'Timestamp': timestamps, 'Face_X_Position': face_x_positions}).bfill()
    df.to_csv('face_positions.csv', index=False)

    # Cropping and writing output
    print("Phase 2: Reformatting video...")
    video_capture = cv2.VideoCapture(video_path)
    interpolator = interp1d(df['Timestamp'], df['Face_X_Position'], kind='cubic', fill_value='extrapolate')

    new_height = frame_height
    new_width = int((9/16) * new_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0
    last_center = None

    # Loading bar setup
    

    print_progress_bar(0, total_frames, prefix='Progress', suffix='Complete', length=40)

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
        print_progress_bar(frame_count, total_frames, prefix='Progress', suffix='Complete', length=40)

    if out:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()
    print("Video reformatted and saved!")

# Combine new video with custom audio
def combine_video_audio(video_path, audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        video.audio = audio
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        print(f"Error combining video and audio: {e}")

# Transcribe video audio and create subtitle file
def generate_subtitles(video_path, subtitle_path):
    print("Transcribing audio...")
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano))
    transcript = transcriber.transcribe(video_path)
    subtitles = transcript.export_subtitles_srt()

    with open(subtitle_path, 'w') as f:
        f.write(subtitles)

    print("✅ Subtitle file created successfully!")

# Add subtitles to video using ffmpeg
def add_subtitles(video_path, subtitle_path, output_path):
    try:
        video_path_ff = video_path.replace('\\', '/')
        subtitle_path_ff = subtitle_path.replace('\\', '/')
        output_path_ff = output_path.replace('\\', '/')
        # Use file\' prefix and single quotes for Windows
        command = [
    'ffmpeg',
    '-i', video_path_ff,
    '-vf', (
        f"subtitles=file\\'{subtitle_path_ff}':"
        "force_style='FontName=Roboto,Alignment=2,MarginV=75,FontSize=14,Bold=1,PrimaryColour=&HFFFFFF&'"
    ),
    '-c:a', 'copy',
    output_path_ff
]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"✅ Video with subtitles saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
# Main driver
def main():
    print("🎥 Step 1: Select input video")
    input_path = f'C:\\Users\\pcofp\\Desktop\\Python\\{input("Enter the name of the unedited video : ").strip()}'
    
    if not input_path.endswith('.mp4'):
        input_path += '.mp4'
    
    trimmed_output = f'C:\\Users\\pcofp\\Desktop\\Python\\temp\\{input("Enter what the name of the trimmed video should be : ").strip()}'
    if not trimmed_output.endswith('.mp4'):
        trimmed_output += '.mp4'
    audio_path_temp = trimmed_output.replace('.mp4', '.mp3')
    start = input("Start time (in seconds): ")
    end = input("End time (in seconds): ")
    cut_video(input_path, trimmed_output, start, end)
    video_to_process = trimmed_output
   
    print("\n📐 Step 2: Reformating to 9:16 with face-tracking...")
    reformatted_output = trimmed_output.replace('.mp4', '_reformatted.mp4')
    if not reformatted_output.endswith('.mp4'):
        reformatted_output += '.mp4'
    process_video(video_to_process, reformatted_output)

    print("\n🎧 Step 3: Adding Audio")
    
    audio_path = audio_path_temp
    combined_output = audio_path_temp.replace('.mp3', '_with_audio.mp4')
    combine_video_audio(reformatted_output, audio_path, combined_output)
    final_video_path = combined_output

    print("\n📝 Step 4: Generating Subtitles")

    subtitle_output = 'C:\\Users\\pcofp\\Desktop\\Python\\temp\\subtitles.srt'
    generate_subtitles(final_video_path, subtitle_output)

    print("\n🎬 Step 5: Finishing Makeup")
  
    video_with_subs_output = trimmed_output.replace('.mp4', '_done.mp4')
    add_subtitles(final_video_path, subtitle_output, video_with_subs_output)

    print("\n✅ Video Completed Successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
