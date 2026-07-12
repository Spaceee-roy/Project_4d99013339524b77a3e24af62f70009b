from pathlib import Path

import cv2


INPUT_IMAGE = "broll_images/20260705-163629-000-a-photorealistic-close-up-shot-of-earth-s-surface-with-visible-t.png"
OUTPUT_VIDEO = "panned_video.mp4"

DURATION_SECONDS = 3
FPS = 30
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# Options: "left_to_right" or "right_to_left"
PAN_DIRECTION = "left_to_right"


def _load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0
        color = image[:, :, :3]
        white = 255 * (1 - alpha)
        image = (color * alpha[:, :, None] + white[:, :, None]).astype("uint8")

    return image


def _ease_in_out(progress):
    return progress * progress * (3 - 2 * progress)


def create_panning_video(
    input_image=INPUT_IMAGE,
    output_video=OUTPUT_VIDEO,
    duration_seconds=DURATION_SECONDS,
    fps=FPS,
    output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    pan_direction=PAN_DIRECTION,
):
    input_path = Path(input_image)
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = _load_image(input_path)
    image_height, image_width = image.shape[:2]
    output_width, output_height = output_size
    target_ratio = output_width / output_height

    crop_height = image_height
    crop_width = round(crop_height * target_ratio)

    if crop_width > image_width:
        scale = crop_width / image_width
        new_width = round(image_width * scale)
        new_height = round(image_height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        image_height, image_width = image.shape[:2]
        crop_height = image_height
        crop_width = round(crop_height * target_ratio)

    max_x = image_width - crop_width
    frame_count = max(1, round(duration_seconds * fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_path}")

    try:
        for frame_index in range(frame_count):
            progress = 0 if frame_count == 1 else frame_index / (frame_count - 1)
            eased = _ease_in_out(progress)

            if pan_direction == "right_to_left":
                x = round(max_x * (1 - eased))
            elif pan_direction == "left_to_right":
                x = round(max_x * eased)
            else:
                raise ValueError('pan_direction must be "left_to_right" or "right_to_left"')

            crop = image[0:crop_height, x : x + crop_width]
            frame = cv2.resize(crop, output_size, interpolation=cv2.INTER_CUBIC)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


if __name__ == "__main__":
    create_panning_video()
