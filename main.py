import os, time, subprocess, numpy as np, pandas as pd, sys, assemblyai as aai;
from tqdm import tqdm;
from pathlib import Path;
import face_recognition, cv2;
from titler import get_metadata_package, generate_video_metadata, _sanitize_filename # Added metadata imports
import shutil
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('ASSEMBLYAI_API_KEY')
aai.settings.api_key = key


def cut_video(input_path, output_path, start_time, end_time):
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

    # --- Attempt fast copy cut ---
    fast_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_path,
        "-c", "copy",
        output_path
    ]
    print(f"[cut_video] ⚡ Fast trim: {' '.join(fast_cmd)}")
    fast_result = subprocess.run(fast_cmd, capture_output=True, text=True)

    # Check for success and non-empty output
    if (
        fast_result.returncode == 0
        and os.path.isfile(output_path)
        and os.path.getsize(output_path) > 1024  # 1 KB threshold
    ):
        print(f"[cut_video] ✅ Fast copy successful: {output_path}")
        return True

    # --- Fallback: re-encode cut ---
    print(f"[cut_video] ⚠️ Fast copy failed or empty. Retrying with re-encode...")
    reencode_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_path,
        "-c:v", "libx264",  # force re-encode for safety
        "-c:a", "aac",
        "-movflags", "+faststart",  # smoother playback
        output_path
    ]
    print(f"[cut_video] 🔁 Re-encode trim: {' '.join(reencode_cmd)}")
    re_result = subprocess.run(reencode_cmd, capture_output=True, text=True)

    if re_result.returncode != 0:
        print(f"[cut_video] ❌ ffmpeg re-encode error:\n{re_result.stderr.strip()}")
        return False

    # Validate again after re-encoding
    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
        print(f"[cut_video] ❌ Output file still empty after re-encode.")
        return False

    print(f"[cut_video] ✅ Re-encode successful: {output_path}")
    return True

def crop_video_with_padding(video_path: str, csv_path: str, output_path: str, face_scale: float = 0.10):
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
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    for frame_num in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Get interpolated face position for this frame (coords are in original pixels)
            face_center_x = coords.get(frame_num / fps, original_width // 2)

            # Ensure face_center_x stays within valid crop bounds
            min_x = crop_w // 2
            max_x = original_width - (crop_w // 2)
            face_center_x = max(min_x, min(face_center_x, max_x))

            # Calculate crop boundaries
            x1 = max(0, int(face_center_x - crop_w / 2))
            x2 = min(original_width, int(face_center_x + crop_w / 2))

            # Double check crop width
            if x2 - x1 != crop_w:
                x2 = x1 + crop_w
                if x2 > original_width:
                    x2 = original_width
                    x1 = x2 - crop_w

            # Crop frame
            if x1 >= 0 and x2 <= original_width:
                cropped_frame = frame[:, x1:x2]
                if cropped_frame.shape[1] == crop_w and cropped_frame.shape[0] == crop_h:
                    out.write(cropped_frame)
                else:
                    cropped_frame = cv2.resize(cropped_frame, (crop_w, crop_h))
                    out.write(cropped_frame)
            else:
                center_x = original_width // 2
                x1 = center_x - (crop_w // 2)
                x2 = x1 + crop_w
                cropped_frame = frame[:, x1:x2]
                out.write(cropped_frame)

        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            center_x = original_width // 2
            x1 = center_x - (crop_w // 2)
            x2 = x1 + crop_w
            cropped_frame = frame[:, x1:x2]
            out.write(cropped_frame)

# --- 5. Release Video Resources ---
    print("Releasing video resources...")
    cap.release()
    out.release()

    # --- 6. Add audio back using ffmpeg (stream copy) ---
    print("Muxing audio back using ffmpeg (copy)...")
    import shutil
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg not found on PATH — cannot mux audio. Leaving temp video without audio.")
        try:
            # Move temp to final so pipeline continues, but warn user
            shutil.move(temp_output_path, output_path)
            print(f"⚠️ Moved temp video to {output_path} (no audio).")
        except Exception as e:
            print(f"Failed to move temp file: {e}")
        return

    # Build ffmpeg command:
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", temp_output_path,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        output_path
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️ ffmpeg muxing failed.")
        try:
            shutil.move(temp_output_path, output_path)
        except Exception as e:
            print(f"❌ Fallback move failed: {e}")
    else:
        try:
            os.remove(temp_output_path)
        except Exception:
            pass
        print(f"✅ Success! Final video with audio saved to: {output_path}")

def generate_subtitles(video_path, subtitle_path):
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(speech_model=aai.SpeechModel.best))
    transcript = transcriber.transcribe(video_path)
    subtitles = transcript.export_subtitles_srt()

    with open(subtitle_path, 'w', encoding='utf-8') as f:
        f.write(subtitles)
    return transcript.text # Return text for metadata generation

def add_subtitles(video_path, subtitle_path, output_path, metadata=None):
    try:
        # Ensure absolute paths and use forward slashes for ffmpeg
        video_path = Path(video_path).resolve().as_posix()
        subtitle_path = Path(subtitle_path).resolve().as_posix()
        output_path = Path(output_path).resolve().as_posix()

        if sys.platform.startswith("win"):
            if ':' in subtitle_path[:3]:
                subtitle_path = subtitle_path.replace(':', '\\:')
        
        # Build ffmpeg command
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"subtitles='{subtitle_path}':force_style='FontName=Roboto,Alignment=2,MarginV=75,MarginL=10,MarginR=10,FontSize=14,BorderStyle=3, Outline=2,Shadow=0,BackColour=&H00620AFA&,Bold=1,PrimaryColour=&HFFFFFF&'",
        ]

        # Inject Metadata into the command if provided
        if metadata:
            command += [
                '-metadata', f"title={metadata['title']}",
                '-metadata', f"comment={metadata['description']}",
                '-metadata', f"description={metadata['description']}"
            ]

        command += ['-c:a', 'copy', output_path]
        
        subprocess.run(command, check=True, capture_output=True)
        print(f"✅ Subtitles and Metadata added to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding subtitles/metadata: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def Segmenter(video_path: str):
    from executioner import process_file
    process_file(video_path)

def force_windows_path(p):
    return Path(p).resolve().as_posix()



def process_video(input_video_path, df_clips, temp_dir):
    """Processes a single video file based on the clip data."""
    print(f"\n=======================================================")
    print(f"🎬 Starting processing for video: {input_video_path.name}")
    print(f"=======================================================")

    input_video = force_windows_path(input_video_path)
    input_video_name = input_video_path.name
    goodname = input_video_name.replace(input_video_path.suffix, "")

    for idx, row in df_clips.iterrows():
        start = row['Start_seconds']
        end = row['End_seconds']
        trueidx = idx + 1

        print(f"\n--- Processing Segment {trueidx} | {start}s to {end}s ---")

        base_path = temp_dir / f"{goodname.strip()}_topclip_{trueidx:02d}"
        trimmed_output = force_windows_path(base_path.with_suffix('.mp4'))
        combined_output = force_windows_path(base_path.with_name(base_path.name + '_with_audio.mp4'))
        subtitle_path = force_windows_path(base_path.with_suffix('.srt'))
        final_output = force_windows_path(base_path.with_name(base_path.name + '_final.mp4'))

        try:
            # 1. Cut the video segment
            cut_video(input_video, trimmed_output, start, end)
            
            # 2. Crop the video
            crop_video_with_padding(trimmed_output, 'face_position.csv', combined_output)
            
            # 3. Generate subtitles and get transcript text
            transcript_text = generate_subtitles(combined_output, subtitle_path)
            
            # --- New Metadata Step ---
            title, description = generate_video_metadata(transcript_text)
            meta_dict = {
                "title": title,
                "description": description,
                "clean_filename": _sanitize_filename(title)
            }
            # -------------------------

            # 4. Add subtitles + Metadata to the final output
            add_subtitles(combined_output, subtitle_path, final_output, metadata=meta_dict)
            
            # 5. Handle Final Renaming based on Viral Title
            final_viral_path = Path.cwd() / f"{meta_dict['clean_filename']}.mp4"
            if os.path.exists(final_output):
                os.replace(final_output, str(final_viral_path))
                print(f"🔥 Viral Video Created: {final_viral_path.name}")

        except Exception as e:
            print(f"❌ Error processing segment {trueidx} for {input_video_name}: {e}")
            break 

        # 6. Clean up temporary files
        for f in [trimmed_output, combined_output, subtitle_path, final_output]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
        
    print(f"✅ Finished processing all segments for: {input_video_path.name}")

def main():
    script_dir = Path(__file__).resolve().parent
    input_folder_name = "Videos"
    input_folder = script_dir / input_folder_name
    
    if not input_folder.is_dir():
        print(f"Error: Folder '{input_folder_name}' not found at {input_folder}")
        return

    temp_dir = script_dir / "temp"
    temp_dir.mkdir(exist_ok=True)


    video_files = list(input_folder.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found.")
        return

    for video_path in video_files:
        Segmenter(video_path) 
            
        try:
            df = pd.read_csv(script_dir / 'viral_clips.csv')
        except FileNotFoundError:
            print("Error: 'viral_clips.csv' not found.")
            return
        
        df['Start'] = df['Start'].apply(lambda s: str(s) if ':' in str(s) else f"0:{s}")
        df['End'] = df['End'].apply(lambda s: str(s) if ':' in str(s) else f"0:{s}")
        df['Start_seconds'] = pd.to_timedelta(df['Start']).dt.total_seconds().astype(int)
        df['End_seconds'] = pd.to_timedelta(df['End']).dt.total_seconds().astype(int)
        process_video(video_path, df, temp_dir)

    print("\n🎉 **BATCH PROCESSING COMPLETE!** 🎉")


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed = int(time.time() - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {hours} hours, {minutes} minutes, {seconds} seconds")
