"""Locate where Qwen-Image 2.1 quantization error comes from, using the real Nunchaku Lite kernels.

1. Capture real transformer inputs from a BF16 generation (KV cache off) at a few steps.
2. Per linear: quantized module vs BF16 module on the same BF16 input, split by text/image tokens.
3. Per block: quantized block vs BF16 block on the same BF16 block input.
4. End to end: full quantized model, and ablations that restore parts to BF16.
5. KV cache: quantized model step-1 output with and without the pipeline's prefix cache.

Run with nunchaku_lite and kernels importable, e.g. the PYTHONPATH in scripts/chain_qwen_image_2_1.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen-Image-2.1"
REPO = Path(__file__).resolve().parents[1]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def capture_inputs(pipe, prompt: str, steps: int, seed: int, capture_steps: set[int], size: int):
    calls: list[dict] = []
    counter = {"i": 0}

    def hook(_m, args, kwargs):
        if counter["i"] in capture_steps:
            calls.append({k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in kwargs.items()})
            calls[-1]["_step"] = counter["i"]
        counter["i"] += 1

    handle = pipe.transformer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps,
             generator=torch.Generator("cuda").manual_seed(seed), use_kv_cache=False)
    finally:
        handle.remove()
    return calls


def token_mask(call: dict) -> torch.Tensor:
    """True at image (target or condition) positions of the joint sequence."""

    img_mask = call["img_mask"]
    repeats = torch.where(img_mask, 4, 1)[0]
    return torch.repeat_interleave(img_mask[0], repeats)


def forward(model, call: dict) -> torch.Tensor:
    kwargs = {k: v for k, v in call.items() if not k.startswith("_")}
    kwargs["return_dict"] = False
    return model(**kwargs)[0]


@contextlib.contextmanager
def swapped(quant, dense, names: list[str]):
    """Temporarily replace submodules of ``quant`` with the BF16 ones from ``dense``."""

    saved = []
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        q_parent = quant.get_submodule(parent_name) if parent_name else quant
        d_parent = dense.get_submodule(parent_name) if parent_name else dense
        saved.append((q_parent, leaf, getattr(q_parent, leaf)))
        setattr(q_parent, leaf, getattr(d_parent, leaf))
    try:
        yield
    finally:
        for parent, leaf, module in saved:
            setattr(parent, leaf, module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default='A vintage travel poster of Kyoto with the title "KYOTO" in bold Art Deco letters')
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--capture-steps", default="0,10,20,30,39")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from diffusers import DiffusionPipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from nunchaku_lite import patch_transformer

    torch.set_grad_enabled(False)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    calls = capture_inputs(pipe, args.prompt, args.steps, args.seed,
                           {int(s) for s in args.capture_steps.split(",")}, args.size)
    dense = pipe.transformer
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    quant = QwenImage21Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
    patch_transformer(quant, args.checkpoint, target="manifest", device="cuda")
    report: dict = {"checkpoint": args.checkpoint, "prompt": args.prompt, "steps": [c["_step"] for c in calls]}

    # 1+2. Isolated per-linear and per-block error on real BF16 activations.
    linear_names = [n for n, m in quant.named_modules() if n.startswith("transformer_blocks.") and type(m).__name__.startswith("SVDQ")]
    linear_err: dict[str, dict[str, list[float]]] = {n: {"all": [], "text": [], "image": []} for n in linear_names}
    block_err: dict[int, list[float]] = {i: [] for i in range(len(dense.transformer_blocks))}
    modulation_err: list[float] = []

    for call in calls:
        is_image = token_mask(call).to("cuda")
        handles = []

        def linear_hook(name):
            q_mod = quant.get_submodule(name)

            def hook(_m, inputs, output):
                x = inputs[0]
                q_out = q_mod(x)
                rows_img = is_image[: x.shape[1]] if x.shape[1] == is_image.shape[0] else None
                linear_err[name]["all"].append(rel(q_out, output))
                if rows_img is not None and (~rows_img).any():
                    linear_err[name]["text"].append(rel(q_out[:, ~rows_img], output[:, ~rows_img]))
                    linear_err[name]["image"].append(rel(q_out[:, rows_img], output[:, rows_img]))
            return hook

        def block_hook(index):
            q_block = quant.transformer_blocks[index]

            def hook(_m, _args, kwargs, output):
                block_err[index].append(rel(q_block(**kwargs), output))
            return hook

        def modulation_hook(_m, inputs, output):
            modulation_err.append(rel(quant.modulation(inputs[0]), output))

        for name in linear_names:
            handles.append(dense.get_submodule(name).register_forward_hook(linear_hook(name)))
        for index, block in enumerate(dense.transformer_blocks):
            handles.append(block.register_forward_hook(block_hook(index), with_kwargs=True))
        handles.append(dense.modulation.register_forward_hook(modulation_hook))
        try:
            forward(dense, call)
        finally:
            for handle in handles:
                handle.remove()

    def mean(values):
        return sum(values) / len(values) if values else None

    kinds: dict[str, dict[str, list[float]]] = {}
    for name, errs in linear_err.items():
        kind = ".".join(name.split(".")[2:])
        for part, values in errs.items():
            kinds.setdefault(kind, {}).setdefault(part, []).extend(values)
    report["linear_by_kind"] = {k: {p: mean(v) for p, v in parts.items()} for k, parts in kinds.items()}
    report["worst_linears"] = sorted(
        ((n, mean(e["all"]), mean(e["text"]), mean(e["image"])) for n, e in linear_err.items()), key=lambda x: -x[1]
    )[:15]
    report["block_isolated"] = {i: mean(v) for i, v in block_err.items()}
    report["modulation_isolated"] = mean(modulation_err)

    # 4. End-to-end error and ablations restoring parts to BF16.
    n_blocks = len(dense.transformer_blocks)
    ablations = {
        "full_quant": [],
        "modulation_bf16": ["modulation"],
        "last4_blocks_bf16": [f"transformer_blocks.{i}" for i in range(n_blocks - 4, n_blocks)],
        "first4_blocks_bf16": [f"transformer_blocks.{i}" for i in range(4)],
        "to_out_bf16": [f"transformer_blocks.{i}.attn.to_out" for i in range(n_blocks)],
        "mlp_out_bf16": [f"transformer_blocks.{i}.img_mlp.out" for i in range(n_blocks)],
        "attn_qkv_bf16": [f"transformer_blocks.{i}.attn.{p}" for i in range(n_blocks) for p in ("to_q", "to_k", "to_v")],
    }
    report["end_to_end"] = {}
    for label, names in ablations.items():
        errs = {}
        with swapped(quant, dense, names):
            for call in calls:
                errs[call["_step"]] = round(rel(forward(quant, call), forward(dense, call)), 4)
        report["end_to_end"][label] = errs
        print(label, errs, flush=True)

    # 5. KV cache: step-1 output with prefix cache (extract at step 0, cached at step 1) vs full recompute.
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    step0, step1 = calls[0], calls[1]
    cache = QwenImage21KVCache(n_blocks)
    forward(quant, {**step0, "kv_cache": cache, "kv_cache_mode": "extract"})
    cached = forward(quant, {**step1, "kv_cache": cache, "kv_cache_mode": "cached"})
    full = forward(quant, step1)
    target_tokens = cached.shape[1]
    report["kv_cache_quant_rel_diff"] = rel(cached, full[:, -target_tokens:])
    cache_d = QwenImage21KVCache(n_blocks)
    forward(dense, {**step0, "kv_cache": cache_d, "kv_cache_mode": "extract"})
    report["kv_cache_bf16_rel_diff"] = rel(forward(dense, {**step1, "kv_cache": cache_d, "kv_cache_mode": "cached"}),
                                           forward(dense, step1)[:, -target_tokens:])

    Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("linear_by_kind", "modulation_isolated", "kv_cache_quant_rel_diff",
                                             "kv_cache_bf16_rel_diff")}, indent=2, default=str))


if __name__ == "__main__":
    main()
