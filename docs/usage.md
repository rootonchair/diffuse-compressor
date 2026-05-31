# Usage Guide

This guide preserves the detailed API and calibration notes from the original
README. For target configuration rules, see [configuration.md](configuration.md).

## Basic API

This example is self-contained: it defines a tiny model and `TargetConfig`
directly, then runs calibration-aware SVDQuant and exports one checkpoint.

```python
from pathlib import Path

import torch
from torch import nn

from diffuse_compressor import (
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    NunchakuSvdqLayout,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    quantize_and_export,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128)

    def forward(self, x):
        return torch.nn.functional.silu(self.proj(x))


class TinyDiffusionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.final = nn.Linear(128, 128)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)


torch.manual_seed(0)
model = TinyDiffusionBackbone().to(torch.bfloat16)

# TargetConfig is the model-specific part. This rule says every
# blocks.<index>.proj Linear becomes one exported quantized target with the
# same checkpoint prefix, which is required for manifest-based runtime loading.
target_config = TargetConfig(
    targets=[
        TargetRule(
            name="block_proj",
            modules=["blocks.*.proj"],
            export_name="blocks.{0}.proj",
            quant=SvdqTargetQuant(weight_layout=NunchakuSvdqLayout()),
        ),
    ],
    unquantized_patterns=["final.*"],
)

# Calibration samples use the model's forward input format. Real diffusion
# examples usually provide prompts and a forward_fn that runs the full pipeline.
samples = [{"x": torch.randn(2, 128, dtype=torch.bfloat16)} for _ in range(2)]

output = Path("outputs/quickstart/tiny.safetensors")
output.parent.mkdir(parents=True, exist_ok=True)

result = quantize_and_export(
    model=model,
    # INT4 SVDQuant with a rank-16 low-rank branch. The tiny model dimensions
    # are chosen so Nunchaku packing works in this minimal example.
    spec=DiffusionQuantSpec(
        precision="int4",
        rank=16,
        group_size=64,
        smooth=False,
    ),
    target_config=target_config,
    calibration=CalibrationSpec(samples=samples),
    export=ExportSpec(output=output),
)

print(result.checkpoint_path)
```

`LoggingConfig` is optional for direct API use. When provided, it writes a text
quantization run log and a `.targets.jsonl` file containing per-target elapsed
time and solver error records. These log files are not embedded in checkpoint
metadata.

## Calibration-Aware SVD

If `CalibrationSpec.samples` is provided, the library first records model
forward inputs to `cache_dir/caches/*.pt` as individual sample records, then
replays those cached records through a loader that honors
`CalibrationSpec.batch_size`, `shuffle`, `drop_last`, `num_workers`, and
`seed`. By default cached records are loaded lazily from disk; set
`eager_load_samples=True` to load selected cached records into the dataset up
front. For plain modules, samples can be passed directly into `model(**sample)`.
For diffusion pipelines, pass a `forward_fn` that closes over the full pipeline:

```python
import torch
from diffuse_compressor import CalibrationSpec

def run_sample(sample: dict) -> None:
    pipe(
        prompt=sample["prompt"],
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(sample["seed"]),
    )

calibration = CalibrationSpec(
    samples=[
        {"prompt": "A glass robot in a greenhouse", "seed": 0},
        {"prompt": "A red train crossing a mountain bridge", "seed": 1},
    ],
    cache_dir="outputs/calibration/my-model",
    cache_mode="reuse",
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    forward_fn=run_sample,
    max_rows_per_target=4096,
    ram_usage_limit=0.90,
)
```

`cache_mode="reuse"` reuses existing cache files, `"refresh"` rewrites them,
and `"disabled"` bypasses disk caching and replays live samples for each scope.
Captured activations are used to compute a weighted SVD branch for each target.
When smoothing is enabled, the same captured activations are used to search
projection smooth factors before weighted SVD and residual weight quantization.
When richer scope capture is configured, each yielded scope also exposes a
DeepCompressor-style layer cache with keyed module inputs, outputs, replay
args, replay kwargs, cache aliases, and repartitioning helpers. These caches are
scope-local and are cleared before the next scope is processed.
If no runnable calibration or reusable cache is provided, the quantizer falls
back to identity smoothing and weight-only SVD.

When `ActivationQuantSpec(enabled=True)` is configured, target input/output
activation ranges are calibrated from the same scoped caches and recorded in
checkpoint metadata, but are not emitted as `input_*` or `output_*` checkpoint
tensors. Nunchaku-style SVDQuant computes activation scales dynamically at
runtime. When `WeightRangeCalibrationSpec(enabled=True)` is configured,
calibrated residual weight range tensors are exported as `weight_range_*`.
These tensors are generic runtime metadata; architecture-specific runtime
consumption remains the exporter/runtime's responsibility.

`CalibrationSpec.artifact_cache` persists quantization artifacts separately
from root model-input caches. The cache writes DeepCompressor-style component
files (`smooth.pt`, `branch.pt`, `wgts.pt`, `acts.pt`, `scale.pt`, and
`model.pt`) and one atomic target cache per completed quantization target. In
`cache_mode="reuse"`, incomplete runs resume from completed target caches and
quantize only missing targets before refreshing the combined files. In
`cache_mode="refresh"`, previous target caches are ignored and rewritten.
`model.pt` is still reused only when the quant spec, target config, and target
export names match the saved cache key.

## Full FLUX.2 Klein 4B Example

The FLUX.2 example lives outside the library core and is just a user config plus
a script:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py --precision int4
```

Override defaults with CLI flags:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py \
  --precision int4 \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --num-samples 128 \
  --batch-size 1 \
  --output outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors
```
