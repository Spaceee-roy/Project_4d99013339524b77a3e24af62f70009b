from pathlib import Path

import local_broll_generator
import panner


PROMPTS = [
    "A serene close-up portrait of an astronaut floating gently in zero gravity inside a sleek spacecraft module with soft blue ambient lighting and metallic textures, sharp focus on calm facial expression looking toward a large window revealing Earth's curvature outside."
]

PANNED_OUTPUT_DIR = "broll_videos"
PAN_DURATION_SECONDS = 5
PAN_DIRECTION = "left_to_right"


def generate_broll_pan_videos(PROMPTS: list[str]):
    image_paths = local_broll_generator.generate_broll_images(PROMPTS)
    output_dir = Path(PANNED_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    for image_path in image_paths:
        image_path = Path(image_path)
        output_video = output_dir / f"{image_path.stem}_pan.mp4"
        video_path = panner.create_panning_video(
            input_image=image_path,
            output_video=output_video,
            duration_seconds=PAN_DURATION_SECONDS,
            pan_direction=PAN_DIRECTION,
        )
        video_paths.append(video_path)
        print(f"Saved {video_path}")

    return video_paths


if __name__ == "__main__":
    generate_broll_pan_videos(PROMPTS)
