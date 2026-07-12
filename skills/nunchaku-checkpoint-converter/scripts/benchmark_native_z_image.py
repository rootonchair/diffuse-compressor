"""Benchmark native Nunchaku Z-Image against a converted output image."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mem_poll import GpuMemPoller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--converted-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    import diffusers
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import ZImagePipeline
    from nunchaku import NunchakuZImageTransformer2DModel
    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageRopeHook

    # Nunchaku 1.x calls the Diffusers transformer forward positionally. Newer
    # Diffusers inserted ControlNet arguments before return_dict, so use the
    # same native RoPE hooks while forwarding stable arguments by keyword.
    def compatible_forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True, **kwargs):
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return super(NunchakuZImageTransformer2DModel, self).forward(
                x=x,
                t=t,
                cap_feats=cap_feats,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=return_dict,
                **kwargs,
            )
        finally:
            self.unregister_rope_hook()

    NunchakuZImageTransformer2DModel.forward = compatible_forward

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    def run():
        return pipe(
            prompt=args.prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

    for _ in range(args.warmup):
        run()
    latencies = []
    result = None
    with GpuMemPoller() as poller:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = run()
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)

    native_path = output_dir / f"{args.tag}_9step.png"
    result.save(native_path)
    converted = Image.open(args.converted_image).convert("RGB")
    native_array = np.asarray(result.convert("RGB")).astype(np.float64)
    converted_array = np.asarray(converted).astype(np.float64)
    delta = converted_array - native_array
    comparison = Image.new("RGB", (2048, 1024))
    comparison.paste(result, (0, 0))
    comparison.paste(converted, (1024, 0))
    comparison_path = output_dir / "output_comparison.png"
    comparison.save(comparison_path)
    payload = {
        "checkpoint": args.checkpoint,
        "text_encoder": "bf16",
        "prompt": args.prompt,
        "seed": 42,
        "steps": 9,
        "height": 1024,
        "width": 1024,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency_sec_all": latencies,
        "latency_sec_mean": statistics.mean(latencies),
        "latency_sec_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "max_device_memory_used_gib": poller.peak_gib,
        "image_delta_converted_vs_native": {
            "mae_pixel": float(np.abs(delta).mean()),
            "rmse_pixel": float(np.sqrt((delta**2).mean())),
            "max_abs_pixel": int(np.abs(delta).max()),
        },
        "native_image": str(native_path),
        "converted_image": args.converted_image,
        "comparison_image": str(comparison_path),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
    }
    (output_dir / f"{args.tag}_9step.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
