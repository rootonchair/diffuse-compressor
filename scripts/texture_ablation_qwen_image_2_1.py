"""Find where INT4's texture noise comes from: which denoising steps and which layer types.

Variants (default 40-step sampler, seed 42, KV cache off so the transformer can be swapped
between steps):
  bf16                  reference
  int4                  INT4 checkpoint throughout
  int4_early_bf16_late  INT4 for steps [0, switch), BF16 afterwards
  bf16_early_int4_late  BF16 for steps [0, switch), INT4 afterwards
  int4_<group>_bf16     INT4 with one layer group restored to BF16 for all steps

Texture error is the mean absolute difference of the high-pass (image minus Gaussian blur,
sigma 1.5) luminance against BF16; whole-image MAE is reported alongside.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

MODEL_ID = "Qwen/Qwen-Image-2.1"
GROUPS = {
    "qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn_out": ("attn.to_out.0",),
    "mlp_in": ("img_mlp.proj", "img_mlp.gate_layer"),
    "mlp_out": ("img_mlp.out",),
}


def highpass(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(1.5))
    return np.asarray(gray, dtype=np.float32) - np.asarray(blurred, dtype=np.float32)


def scores(image: Image.Image, reference: Image.Image) -> dict:
    a, b = np.asarray(image, dtype=np.float32), np.asarray(reference, dtype=np.float32)
    return {
        "texture_err": float(np.abs(highpass(image) - highpass(reference)).mean()),
        "texture_energy": float(np.abs(highpass(image)).mean()),
        "mae": float(np.abs(a - b).mean() / 255.0),
    }


@contextlib.contextmanager
def swapped(quant, dense, names):
    saved = []
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        q_parent, d_parent = quant.get_submodule(parent_name), dense.get_submodule(parent_name)
        saved.append((q_parent, leaf, getattr(q_parent, leaf)))
        setattr(q_parent, leaf, getattr(d_parent, leaf))
    try:
        yield
    finally:
        for parent, leaf, module in saved:
            setattr(parent, leaf, module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--switch", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffusers import DiffusionPipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from nunchaku_lite import patch_transformer

    torch.set_grad_enabled(False)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    dense = pipe.transformer
    quant = QwenImage21Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
    patch_transformer(quant, args.checkpoint, target="manifest", device="cuda")
    n_blocks = len(dense.transformer_blocks)

    def run(case, first, second=None, switch=None):
        pipe.transformer = first
        images = [Image.open(i["path"]).convert("RGB") for i in case["inputs"]]

        def callback(pipeline, index, _t, kwargs):
            if second is not None and index + 1 == switch:
                pipeline.transformer = second
            return kwargs

        image = pipe(prompt=case["prompt"], image=images, num_inference_steps=args.steps, true_cfg_scale=1.0,
                     output_resolution=1024, use_kv_cache=False, callback_on_step_end=callback,
                     generator=torch.Generator("cuda").manual_seed(args.seed)).images[0]
        pipe.transformer = dense
        return image

    cases = {c["key"]: c for c in json.loads(args.cases.read_text())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in args.keys:
        case = cases[key]
        out = {}
        out["bf16"] = run(case, dense)
        out["int4"] = run(case, quant)
        for k in args.switch:
            out[f"int4_early_bf16_late_{k}"] = run(case, quant, dense, k)
            out[f"bf16_early_int4_late_{k}"] = run(case, dense, quant, k)
        for group, suffixes in GROUPS.items():
            names = [f"transformer_blocks.{i}.{s}" for i in range(n_blocks) for s in suffixes]
            with swapped(quant, dense, names):
                out[f"int4_{group}_bf16"] = run(case, quant)
        with swapped(quant, dense, [f"transformer_blocks.{i}" for i in range(n_blocks - 4, n_blocks)]):
            out["int4_last4blocks_bf16"] = run(case, quant)
        with swapped(quant, dense, [f"transformer_blocks.{i}" for i in range(4)]):
            out["int4_first4blocks_bf16"] = run(case, quant)
        results[key] = {}
        for name, image in out.items():
            image.save(args.output_dir / f"{key.split('__')[-1]}__{name}.png")
            results[key][name] = scores(image, out["bf16"])
            print(f"{key.split('__')[-1]:12s} {name:28s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[key][name].items()), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
