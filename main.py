import os, time, subprocess, numpy as np, pandas as pd, sys, assemblyai as aai; from tqdm import tqdm; from executioner import VideoSegmenter; from pathlib import Path; import face_recognition, cv2; from scipy.interpolate import CubicSpline, interp1d;aai.settings.api_key = ''; from moviepy import VideoFileClip

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

def crop_video_with_padding(video_path: str, csv_path: str, output_path: str):
    """
    Crops a video to a 9:16 aspect ratio based on horizontal face coordinates (x) 
    from a CSV file. Face positions are interpolated per-frame to remove jitter.
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

    for frame_idx in tqdm(range(0, total_frames, int(fps))):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_capture.read()
        if not ret:
            break

        current_time = frame_idx / fps
        face_locations = face_recognition.face_locations(frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_center_x = (right + left) / 2
            last_x = face_center_x
        else:
            face_center_x = last_x  # fallback

        face_x_positions.append(face_center_x)
        timestamps.append(current_time)

    video_capture.release()

    df = pd.DataFrame({"Timestamp": timestamps, "Face_X_Position": face_x_positions}).bfill()
    df.to_csv(csv_path, index=False)

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
        return
    except KeyError:
        print("Error: Your CSV file must contain 'Timestamp' and 'Face_X_Position' columns.")
        return

    # --- Interpolate face positions per frame ---
    frame_times = np.arange(frame_count) / fps
    interp_series = (
        df.set_index("Timestamp")["Face_X_Position"]
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
    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        face_center_x = coords.get(frame_num / fps, original_width // 2)

        # Calculate crop boundaries
        x1 = int(face_center_x - crop_w / 2)
        x2 = int(face_center_x + crop_w / 2)

        # Clamp edges
        if x1 < 0:
            x1 = 0
            x2 = crop_w
        elif x2 > original_width:
            x2 = original_width
            x1 = original_width - crop_w

        cropped_frame = frame[:, x1:x2]
        out.write(cropped_frame)

        if (frame_num + 1) % 100 == 0:
            print(f"  Processed {frame_num + 1} / {frame_count} frames")

    # --- 5. Release Video Resources ---
    print("Releasing video resources...")
    cap.release()
    out.release()

    # --- 6. Add audio from original video ---
    print("Adding audio back using MoviePy...")
    try:
        original_clip = VideoFileClip(video_path)
        processed_clip = VideoFileClip(temp_output_path)
        final_clip = processed_clip.with_audio(original_clip.audio)
        time.sleep(3)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        os.remove(temp_output_path)
        print(f"✅ Success! Final video saved to: {output_path}")

    except Exception as e:
        print("⚠️ Audio merge failed:", e)
    
def generate_subtitles(video_path, subtitle_path):
    # print("🎧 Transcribing audio...")
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.best))
    transcript = transcriber.transcribe(video_path)
    subtitles = transcript.export_subtitles_srt()

    with open(subtitle_path, 'w') as f:
        f.write(subtitles)
    # print("✅ Subtitle file created successfully!")

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
            f"subtitles='{subtitle_path}':force_style='FontName=Roboto,Alignment=2,MarginV=75,MarginL=10,MarginR=10,FontSize=14,BorderStyle=3, Outline=2,Shadow=0,BackColour=&H00620AFA&,Bold=1,PrimaryColour=&HFFFFFF&'",
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

def force_windows_path(p):
    return Path(p).resolve().as_posix()  # Always C:/... format for ffmpeg

def process_all_segments():
   

    filepath = input("Enter SRT file name (e.g., subtitles.srt): ").strip()
    input_video_name = input("Enter video file name (e.g., video.mp4): ").strip()
    script_dir = Path(__file__).resolve().parent
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
        trueidx = idx + 1
        print(f"\n--- Processing Segment {trueidx} | {start}s to {end}s ---")

        base_path = temp_dir / f"{goodname.strip()}_topclip_{trueidx:02d}"

        trimmed_output = force_windows_path(base_path.with_suffix('.mp4')) 
        combined_output = force_windows_path(base_path.with_name(base_path.name + '_with_audio.mp4'))
        subtitle_path = force_windows_path(base_path.with_suffix('.srt'))
        final_output = force_windows_path(base_path.with_name(base_path.name + '_final.mp4'))
        cut_video(input_video, trimmed_output, start, end)
        crop_video_with_padding(trimmed_output,'face_position.csv',combined_output)
        generate_subtitles(combined_output, subtitle_path)
        add_subtitles(combined_output, subtitle_path, final_output)

        for f in [trimmed_output, combined_output, subtitle_path]:
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
