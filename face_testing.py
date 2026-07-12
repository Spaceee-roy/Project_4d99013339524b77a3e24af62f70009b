import os
import shutil
import subprocess
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
import stable_whisper
from tqdm import tqdm


FFMPEG_BIN_DIR = r"C:\ffmpeg-master-latest-win64-gpl-shared\bin"
if os.path.isdir(FFMPEG_BIN_DIR):
    os.add_dll_directory(FFMPEG_BIN_DIR)


def _run_ffmpeg(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip()[-1500:])
    return result.returncode == 0


def _ffmpeg_encoder_args():
    nvenc_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-encoders",
    ]
    result = subprocess.run(nvenc_cmd, capture_output=True, text=True)
    if result.returncode == 0 and "h264_nvenc" in result.stdout:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-cq",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]

    return [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ]


def _video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not read video metadata: {video_path}")

    return fps, width, height, frame_count


def _detect_face_timeline(
    video_path,
    fps,
    frame_width,
    frame_count,
    face_scale=0.10,
    sample_interval_seconds=1.0,
    batch_size=32,
    save_csv_path=None,
):
    if face_scale <= 0 or face_scale > 1:
        raise ValueError("face_scale must be between 0 (exclusive) and 1 (inclusive)")

    sample_step = max(1, int(round(fps * sample_interval_seconds)))
    inv_scale = 1.0 / face_scale
    last_x = frame_width / 2.0
    sampled_frames = []
    sampled_times = []
    small_rgb_frames = []

    print("Phase 1: sampling frames for batched face detection...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    for frame_idx in tqdm(range(0, frame_count, sample_step), desc="Sampling"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=face_scale,
            fy=face_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        small_rgb_frames.append(small_rgb)
        sampled_frames.append(frame_idx)
        sampled_times.append(frame_idx / fps)

    cap.release()

    if not small_rgb_frames:
        return np.full(frame_count, last_x, dtype=np.float32)

    print("Phase 2: detecting faces in batches...")
    face_x_positions = []
    for start in tqdm(range(0, len(small_rgb_frames), batch_size), desc="Detecting"):
        batch = small_rgb_frames[start : start + batch_size]
        locations_batch = face_recognition.batch_face_locations(
            batch,
            number_of_times_to_upsample=0,
            batch_size=batch_size,
        )

        for face_locations in locations_batch:
            if face_locations:
                top, right, bottom, left = face_locations[0]
                last_x = ((right + left) / 2.0) * inv_scale
            face_x_positions.append(last_x)

    if save_csv_path:
        pd.DataFrame({"Time": sampled_times, "X": face_x_positions}).to_csv(
            save_csv_path,
            index=False,
        )
        print(f"Saved face timeline CSV -> {save_csv_path}")

    sample_indices = np.asarray(sampled_frames, dtype=np.float32)
    sample_x = np.asarray(face_x_positions, dtype=np.float32)
    all_indices = np.arange(frame_count, dtype=np.float32)

    return np.interp(all_indices, sample_indices, sample_x).astype(np.float32)


def _smooth_crop_centers(
    raw_centers,
    frame_width,
    crop_width,
    threshold=60.0,
    smoothing=0.08,
):
    min_x = crop_width / 2.0
    max_x = frame_width - (crop_width / 2.0)
    centers = np.empty_like(raw_centers, dtype=np.float32)

    target_x = float(raw_centers[0])
    last_applied_x = target_x

    for idx, current_face_x in enumerate(raw_centers):
        current_face_x = float(current_face_x)
        if abs(current_face_x - target_x) > threshold:
            target_x = current_face_x

        last_applied_x += smoothing * (target_x - last_applied_x)
        centers[idx] = np.clip(last_applied_x, min_x, max_x)

    return centers


def _crop_bounds(center_x, crop_w, frame_width):
    if not np.isfinite(center_x):
        center_x = frame_width / 2.0

    x1 = int(round(center_x - crop_w / 2.0))
    x2 = x1 + crop_w

    if x1 < 0:
        x1 = 0
        x2 = crop_w
    elif x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_w

    return x1, x2


def _valid_cropped_frame(frame, crop_w, crop_h):
    return (
        frame is not None
        and frame.size > 0
        and frame.ndim == 3
        and frame.shape[0] == crop_h
        and frame.shape[1] == crop_w
        and frame.shape[2] == 3
    )


def crop_video_with_padding_cuda(
    video_path,
    output_path,
    face_scale=0.08,
    sample_interval_seconds=1,
    batch_size=64,
    save_face_csv=False,
):
    """
    Faster 9:16 face-follow crop.

    The original version moved one scalar through CUDA per frame, which was slower
    than plain CPU math. This version batches CNN face detection, computes the
    crop path with NumPy, and keeps the frame loop focused on decode/crop/write.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")

    fps, frame_width, frame_height, frame_count = _video_info(video_path)
    crop_h = frame_height
    crop_w = int(crop_h * 9 / 16)
    if crop_w > frame_width:
        raise RuntimeError("Video is not wide enough for a 9:16 crop at full height")

    face_csv = "face_position.csv" if save_face_csv else None
    raw_centers = _detect_face_timeline(
        video_path=video_path,
        fps=fps,
        frame_width=frame_width,
        frame_count=frame_count,
        face_scale=face_scale,
        sample_interval_seconds=sample_interval_seconds,
        batch_size=batch_size,
        save_csv_path=face_csv,
    )
    crop_centers = _smooth_crop_centers(raw_centers, frame_width, crop_w)

    temp_output_path = str(Path(output_path).with_name(Path(output_path).stem + "_no_audio.avi"))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temp_output_path}")

    print("Phase 3: cropping frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video file: {video_path}")

    previous_bounds = _crop_bounds(frame_width / 2.0, crop_w, frame_width)

    for frame_num in tqdm(range(frame_count), desc="Cropping"):
        ret, frame = cap.read()
        if not ret:
            break

        center_x = crop_centers[min(frame_num, len(crop_centers) - 1)]
        x1, x2 = _crop_bounds(center_x, crop_w, frame_width)
        cropped_frame = frame[:, x1:x2]

        if not _valid_cropped_frame(cropped_frame, crop_w, crop_h):
            x1, x2 = previous_bounds
            cropped_frame = frame[:, x1:x2]

        if cropped_frame.size > 0 and (
            cropped_frame.shape[1] != crop_w or cropped_frame.shape[0] != crop_h
        ):
            cropped_frame = cv2.resize(cropped_frame, (crop_w, crop_h))

        if not _valid_cropped_frame(cropped_frame, crop_w, crop_h):
            raise RuntimeError(
                f"Could not create a valid crop for frame {frame_num}: "
                f"bounds=({x1}, {x2}), frame_shape={frame.shape}"
            )

        previous_bounds = (x1, x2)
        cropped_frame = np.ascontiguousarray(cropped_frame)
        writer.write(cropped_frame)

    cap.release()
    writer.release()

    print("Phase 4: muxing audio back...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_output_path,
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        *_ffmpeg_encoder_args(),
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]

    if not _run_ffmpeg(ffmpeg_cmd):
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output_path,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            output_path,
        ]
        if not _run_ffmpeg(fallback_cmd):
            shutil.move(temp_output_path, output_path)
            print(f"Fallback: moved temp video to {output_path}")
            return

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    print(f"Success! Cropped video saved to: {output_path}")


def add_word_highlight_subtitles(video_file, subtitle_file, final_output):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_file,
        "-vf",
        f"subtitles='{subtitle_file}'",
        "-c:a",
        "copy",
        *_ffmpeg_encoder_args(),
        final_output,
    ]

    if not _run_ffmpeg(command):
        raise RuntimeError("Failed to burn subtitles into the video")


def main(video_file, cropped_output, final_output):
    model = stable_whisper.load_model("large-v3-turbo")
    result = model.transcribe(video_file)

    result.to_ass(
        "subtitles.ass",
        font="Montserrat",
        font_size=15,
        highlight_color="0000ff",
        SecondaryColour="ffffff",
        OutlineColour="000000",
        BackColour="000000",
        Bold="-1",
        Alignment="2",
        MarginV="80",
        BorderStyle="1",
        Outline="2",
        tag=-1,
    )

    crop_video_with_padding_cuda(video_file, cropped_output)
    print(result)
    add_word_highlight_subtitles(cropped_output, "subtitles.ass", final_output)


if __name__ == "__main__":
    main("face.mp4", "cropped.mp4", "final.mp4")
