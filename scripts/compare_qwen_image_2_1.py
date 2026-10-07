"""Generate the same prompts with BF16, INT4 and NVFP4 Qwen-Image 2.1 and compare them.

Quantized transformers run through the real W4A4 kernels of
``nunchaku_lite`` (github.com/rootonchair/nunchaku-lite), using its generic
``manifest`` adapter driven by the checkpoint's runtime manifest. The BF16
pipeline is loaded once; for each precision a fresh dense transformer is patched
with the checkpoint and swapped in.

Needs ``nunchaku_lite`` and the Hugging Face ``kernels`` package importable, and
network access the first time the prebuilt kernels are resolved.

Outputs under ``--output-dir``:
  <precision>/<index>.png   one image per prompt and precision
  grid.png                  prompts as rows, precisions as columns
  metrics.json              PSNR against BF16, seconds per image, peak memory
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen-Image-2.1"
DEFAULT_PROMPTS = (
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
    "Portrait of an elderly fisherman with a weathered face, natural window light, 85mm photo",
    "A cozy Scandinavian living room with a large window, morning sunlight, plants, wooden floor",
    "A red fox standing in fresh snow in a pine forest, soft overcast light, wildlife photography",
    'A vintage travel poster of Kyoto with the title "KYOTO" in bold Art Deco letters',
    "A bowl of ramen with a soft-boiled egg, steam rising, close-up food photography on a dark table",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Quantized checkpoint per column, e.g. int4=outputs/checkpoints/svdq-int4_r32-qwen-image-2-1.safetensors",
    )
    parser.add_argument("--output-dir", default=str(REPO / "outputs" / "compare" / "qwen-image-2-1"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", action="append", default=[], help="Override the default prompts; repeatable")
    parser.add_argument("--grid-cell", type=int, default=384, help="Grid thumbnail size in pixels")
    parser.add_argument("--device", default="cuda")
    return parser


def generate(pipe, prompts: list[str], args, out_dir: Path) -> tuple[list[Image.Image], list[float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images, seconds = [], []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device=args.device).manual_seed(args.seed + index)
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        image.save(out_dir / f"{index:02d}.png")
        images.append(image)
        print(f"  [{out_dir.name}] {index + 1}/{len(prompts)} in {seconds[-1]:.1f}s", flush=True)
    return images, seconds


def psnr(reference: Image.Image, image: Image.Image) -> float:
    a = np.asarray(reference.convert("RGB"), dtype=np.float64)
    b = np.asarray(image.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def save_grid(columns: dict[str, list[Image.Image]], cell: int, path: Path) -> None:
    names = list(columns)
    rows = len(next(iter(columns.values())))
    header = 32
    grid = Image.new("RGB", (cell * len(names), header + cell * rows), "white")
    draw = ImageDraw.Draw(grid)
    for col, name in enumerate(names):
        draw.text((col * cell + 8, 8), name, fill="black")
        for row, image in enumerate(columns[name]):
            grid.paste(image.convert("RGB").resize((cell, cell)), (col * cell, header + row * cell))
    grid.save(path)


def main() -> None:
    args = build_parser().parse_args()
    from diffusers import DiffusionPipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from nunchaku_lite import patch_transformer

    checkpoints: dict[str, Path] = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got {item!r}")
        checkpoints[name] = Path(path).resolve()
        if not checkpoints[name].is_file():
            raise SystemExit(f"Checkpoint for {name} does not exist: {checkpoints[name]}")
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")

    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device)

    columns: dict[str, list[Image.Image]] = {}
    timings: dict[str, list[float]] = {}
    peak_gb: dict[str, float] = {}

    print("Generating bf16", flush=True)
    torch.cuda.reset_peak_memory_stats()
    columns["bf16"], timings["bf16"] = generate(pipe, prompts, args, output_dir / "bf16")
    peak_gb["bf16"] = torch.cuda.max_memory_allocated() / 2**30

    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()
    for name, checkpoint in checkpoints.items():
        print(f"Generating {name} from {checkpoint}", flush=True)
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        patch_transformer(transformer, checkpoint, target="manifest", device=args.device)
        pipe.transformer = transformer
        torch.cuda.reset_peak_memory_stats()
        columns[name], timings[name] = generate(pipe, prompts, args, output_dir / name)
        peak_gb[name] = torch.cuda.max_memory_allocated() / 2**30
        pipe.transformer = None
        del transformer
        gc.collect()
        torch.cuda.empty_cache()

    metrics: dict[str, dict] = {}
    for name in columns:
        # The first image includes warm-up (kernel load, autotuning), so average the rest.
        steady = timings[name][1:] or timings[name]
        entry = {"seconds_per_image": round(float(np.mean(steady)), 2), "peak_cuda_gb": round(peak_gb[name], 2)}
        if name != "bf16":
            values = [round(psnr(ref, img), 2) for ref, img in zip(columns["bf16"], columns[name], strict=True)]
            entry["psnr_vs_bf16"] = values
            entry["mean_psnr_vs_bf16"] = round(float(np.mean(values)), 2)
        metrics[name] = entry
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_grid(columns, args.grid_cell, output_dir / "grid.png")
    print(json.dumps(metrics, indent=2))
    print(f"Grid: {output_dir / 'grid.png'}")


if __name__ == "__main__":
    main()
