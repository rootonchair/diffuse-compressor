# Qwen Image Edit 2509 Example Adapter Notes

This is a model-specific example for the generic Nunchaku-to-Diffusers conversion workflow. Do not treat these module names, layer counts, or target lists as universal.

## Known Repos

- Source Nunchaku checkpoint: `nunchaku-ai/nunchaku-qwen-image-edit-2509`
- Base Diffusers model: `Qwen/Qwen-Image-Edit-2509`
- Known-good style reference: `lite-infer/Qwen-Image-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`

## Fragile Layout Rules

### Fused QKV SVDQ

The source checkpoint stores keys such as:

```text
transformer_blocks.0.attn.to_qkv.qweight
transformer_blocks.0.attn.to_qkv.wscales
transformer_blocks.0.attn.to_qkv.wcscales
transformer_blocks.0.attn.to_qkv.bias
transformer_blocks.0.attn.to_qkv.proj_up
transformer_blocks.0.attn.to_qkv.proj_down
transformer_blocks.0.attn.to_qkv.smooth_factor
```

These are Nunchaku-packed tensors. Convert to Diffusers keys:

```text
transformer_blocks.0.attn.to_q.*
transformer_blocks.0.attn.to_k.*
transformer_blocks.0.attn.to_v.*
```

Do not plain-slice packed `qweight`, `wscales`, `wcscales`, `bias`, or `proj_up`. Use `NunchakuWeightPacker`:

```python
packer = NunchakuWeightPacker(bits=4)
logical = packer.unpack_weight(fused_qweight, rows=9216, columns=3072)
q = packer.pack_weight(packer.pad_weight(logical[:3072]))
```

For packed scale-like tensors:

```python
w = packer.unpack_scale(fused_wscales, rows=9216, groups=192, group_size=16)
q_w = packer.pack_scale(packer.pad_scale(w[:3072], group_size=16), group_size=16)
```

For vector values:

```python
b = packer.unpack_scale(fused_bias, rows=9216, groups=1, group_size=-1)
q_b = packer.pack_scale(packer.pad_scale(b[:3072].view(-1, 1), group_size=-1), group_size=-1)
```

For low-rank `proj_up`:

```python
up = packer.unpack_lowrank_weight(fused_proj_up, down=False, rows=9216, columns=32)
q_up = packer.pack_lowrank_weight(up[:3072], down=False)
```

`proj_down` and `smooth_factor` are input-side tensors, so duplicate them for Q/K/V.

### AWQ Modulation

The source modulation linears:

```text
transformer_blocks.*.img_mod.1
transformer_blocks.*.txt_mod.1
```

are AWQ W4A16 tensors in AdaNorm interleaved output layout. Diffusers normal AWQ modules expect normal output order. Convert by permuting packed output rows, scales, zeros, and bias. Do not dequantize AWQ weights to floating-point weights.

Bias must also undo the AdaNorm offset:

```text
delta[1] -= 1
delta[-2] -= 1
```

after restoring normal output order.

## Validation Snippets

Check real source-block QKV parity after conversion:

```bash
PYTHONPATH=src python - <<'PY'
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker

src = hf_hub_download("nunchaku-ai/nunchaku-qwen-image-edit-2509", "svdq-fp4_r32-qwen-image-edit-2509.safetensors")
out = "outputs/converted/qwen-image-edit-2509-nunchaku-lite-nvfp4_r32-bnb4-text-encoder/transformer/diffusion_pytorch_model.safetensors"
packer = NunchakuWeightPacker(bits=4)
with safe_open(src, framework="pt", device="cpu") as fs, safe_open(out, framework="pt", device="cpu") as fo:
    fused = packer.unpack_weight(fs.get_tensor("transformer_blocks.0.attn.to_qkv.qweight"), rows=9216, columns=3072)
    for name, sl in [("q", slice(0, 3072)), ("k", slice(3072, 6144)), ("v", slice(6144, 9216))]:
        split = packer.unpack_weight(fo.get_tensor(f"transformer_blocks.0.attn.to_{name}.qweight"), rows=3072, columns=3072)
        print(name, bool((split == fused[sl]).all()))
PY
```

Expected output:

```text
q True
k True
v True
```

Check config and BNB4 text encoder:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/converted/qwen-image-edit-2509-nunchaku-lite-nvfp4_r32-bnb4-text-encoder")
t = json.loads((root / "transformer/config.json").read_text())["quantization_config"]
e = json.loads((root / "text_encoder/config.json").read_text())["quantization_config"]
print(t["quant_method"], t["svdq_w4a4"]["precision"], t["svdq_w4a4"]["rank"], len(t["svdq_w4a4"]["targets"]), len(t["awq_w4a16"]["targets"]))
print(e["quant_method"], e["load_in_4bit"], e["bnb_4bit_quant_type"], e["bnb_4bit_compute_dtype"])
PY
```

Expected:

```text
nunchaku_lite nvfp4 32 720 120
bitsandbytes True nf4 bfloat16
```

## Full Inference Requirement

Run a full 40-step edit when judging conversion quality. One-step tests only prove the call path executes. A bad conversion may load successfully and still produce black, speckled, washed-out, or semantically broken images.
