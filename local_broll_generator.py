"""Fast local B-roll image generation helpers for vertical shorts."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MODEL = "SG161222/RealVisXL_V5.0"
DEFAULT_LORA_REPO = "ByteDance/SDXL-Lightning"
DEFAULT_LORA_WEIGHT = "sdxl_lightning_4step_lora.safetensors"
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, cgi, 3d render, plastic skin, "
    "overprocessed, blurry, low resolution, distorted face, bad hands, extra fingers, "
    "deformed body, watermark, text, logo, letters, words, captions, subtitles, "
    "signs, labels, typography, writing, numbers, poster, title card, "
    "duplicate subject, extra object, extra planet, duplicate planet, multiple planets, "
    "extra moon, duplicate moon, warped moon, deformed moon, moon rings, spiral trails, "
    "motion trails, smeared surface, stretched planet, surreal scale, cluttered scene,"
)


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    model_path: str | None
    output_dir: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    count: int
    seed: int | None
    negative_prompt: str
    lightning: bool
    lora_repo: str
    lora_weight: str
    cpu_offload: bool


def slugify(value: str, max_length: int = 64) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value[:max_length].strip("-") or "broll"


def import_diffusion_stack():
    try:
        import torch
        from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
    except ImportError as exc:
        raise SystemExit(
            "Missing image-generation packages. Install them in this venv first:\n"
            "  python -m pip install --upgrade diffusers transformers accelerate safetensors torch\n\n"
            f"Original import error: {exc}"
        ) from exc

    return torch, AutoPipelineForText2Image, EulerDiscreteScheduler


def require_peft_for_lora() -> None:
    if importlib.util.find_spec("peft") is not None:
        return

    raise SystemExit(
        "SDXL Lightning LoRA needs the PEFT package, but it is not installed.\n\n"
        "Install it in this venv:\n"
        "  python -m pip install peft\n\n"
        "Then rerun the same command. If you want to test without Lightning, run:\n"
        "  generate_broll_images(['your prompt here'], lightning=False)"
    )


def build_pipeline(config: GenerationConfig):
    torch, AutoPipelineForText2Image, EulerDiscreteScheduler = import_diffusion_stack()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU was not detected. This script is tuned for your RTX 5060; "
            "install a CUDA-enabled PyTorch build before running generation."
        )

    common_kwargs = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
    }

    if config.model_path:
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        pipe = AutoPipelineForText2Image.from_single_file(str(model_path), **common_kwargs)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(config.model, **common_kwargs)

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if config.lightning:
        require_peft_for_lora()
        try:
            pipe.load_lora_weights(config.lora_repo, weight_name=config.lora_weight)
        except ValueError as exc:
            if "PEFT backend is required" not in str(exc):
                raise
            raise SystemExit(
                "Diffusers could not load the SDXL Lightning LoRA because PEFT is missing.\n\n"
                "Install it in this venv:\n"
                "  python -m pip install peft\n\n"
                "Then rerun the same command."
            ) from exc
        pipe.fuse_lora()

    pipe.enable_vae_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    if config.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return torch, pipe


def make_generator(torch_module, seed: int | None, index: int):
    if seed is None:
        return None
    return torch_module.Generator(device="cuda").manual_seed(seed + index)


def expand_broll_prompt(prompt: str) -> str:
    prompt = prompt.replace("outerspace", "outer space")
    prompt = re.sub(r"\bspiraling away from\b", "shown far from", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\bdrifting away from\b", "shown far from", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\bmoving away from\b", "shown far from", prompt, flags=re.IGNORECASE)

    return (
        # " documentary b-roll still, single clean composition, "
        # "one clear main visual idea, accurate scale, realistic lighting, sharp focus, "
        
        f"{prompt}"
    )


def generate_images(config: GenerationConfig, prompts: Iterable[str]) -> list[Path]:
    torch, pipe = build_pipeline(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    image_index = 0

    for prompt in prompts:
        full_prompt = expand_broll_prompt(prompt)

        for _ in range(config.count):
            start = time.perf_counter()
            generator = make_generator(torch, config.seed, image_index)
            result = pipe(
                prompt=full_prompt,
                negative_prompt=config.negative_prompt,
                width=config.width,
                height=config.height,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )

            slug = slugify(prompt)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = output_dir / f"{stamp}-{image_index:03d}-{slug}.png"
            meta_path = image_path.with_suffix(".json")

            result.images[0].save(image_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "expanded_prompt": full_prompt,
                        "seconds": round(time.perf_counter() - start, 3),
                        "config": asdict(config),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            saved_paths.append(image_path)
            print(f"Saved {image_path}")
            image_index += 1

    return saved_paths


def make_config(
    *,
    model: str | None = None,
    model_path: str | None = None,
    output_dir: str = "broll_images",
    width: int = 1280,
    height: int = 720,
    steps: int = 4,
    guidance_scale: float = 1.5,
    count: int = 1,
    seed: int | None = None,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    lightning: bool = True,
    lora_repo: str = DEFAULT_LORA_REPO,
    lora_weight: str = DEFAULT_LORA_WEIGHT,
    cpu_offload: bool = False,
) -> GenerationConfig:
    return GenerationConfig(
        model=model or os.getenv("BROLL_MODEL", DEFAULT_MODEL),
        model_path=model_path or os.getenv("BROLL_MODEL_PATH"),
        output_dir=output_dir,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        count=count,
        seed=seed,
        negative_prompt=negative_prompt,
        lightning=lightning,
        lora_repo=lora_repo,
        lora_weight=lora_weight,
        cpu_offload=cpu_offload,
    )


def generate_broll_images(
    prompts: str | Sequence[str],
    *,
    model: str | None = None,
    model_path: str | None = None,
    output_dir: str = "broll_images",
    width: int = 1280,
    height: int = 720,
    steps: int = 4,
    guidance_scale: float = 1.5,
    count: int = 1,
    seed: int | None = None,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    lightning: bool = True,
    lora_repo: str = DEFAULT_LORA_REPO,
    lora_weight: str = DEFAULT_LORA_WEIGHT,
    cpu_offload: bool = False,
) -> list[Path]:
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = [prompt.strip() for prompt in prompts if prompt.strip()]

    if not prompt_list:
        raise ValueError("At least one prompt is required.")

    config = GenerationConfig(
        model=model or os.getenv("BROLL_MODEL", DEFAULT_MODEL),
        model_path=model_path or os.getenv("BROLL_MODEL_PATH"),
        output_dir=output_dir,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        count=count,
        seed=seed,
        negative_prompt=negative_prompt,
        lightning=lightning,
        lora_repo=lora_repo,
        lora_weight=lora_weight,
        cpu_offload=cpu_offload,
    )
    return generate_images(config, prompt_list)
