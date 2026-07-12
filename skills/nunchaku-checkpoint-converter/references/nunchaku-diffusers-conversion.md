# Nunchaku to Diffusers Conversion Reference

## Adapter Checklist

For each model family, define:

- Base Diffusers model repo or local path.
- Source Nunchaku checkpoint repo or local safetensors path.
- Pipeline class and `model_index.json` component classes.
- Transformer class and expected module paths.
- SVDQ W4A4/NVFP4 target list.
- AWQ W4A16 target list.
- Fused module groups that must be split, or split modules that must be merged.
- Any model-specific tensor layouts, such as AdaNorm AWQ interleaves.

## Packed SVDQ Rules

Nunchaku-packed tensors can have ordinary-looking shapes but non-row-major byte order. Shape checks and strict state-dict loads are not enough.

Use `diffuse_compressor.backends.nunchaku.packing.NunchakuWeightPacker` for layout-preserving transforms:

```python
packer = NunchakuWeightPacker(bits=4)
logical_weight = packer.unpack_weight(packed_qweight, rows=out_features, columns=in_features)
packed_weight = packer.pack_weight(packer.pad_weight(logical_weight))
```

For packed grouped scales:

```python
logical_scales = packer.unpack_scale(packed_wscales, rows=out_features, groups=num_groups, group_size=group_size)
packed_scales = packer.pack_scale(packer.pad_scale(logical_scales, group_size=group_size), group_size=group_size)
```

For packed vector-like tensors such as `bias`, `smooth_factor`, or `wcscales`:

```python
logical_vector = packer.unpack_scale(packed_vector, rows=out_features, groups=1, group_size=-1)
packed_vector = packer.pack_scale(packer.pad_scale(logical_vector.view(-1, 1), group_size=-1), group_size=-1)
```

For packed low-rank tensors:

```python
logical_up = packer.unpack_lowrank_weight(packed_proj_up, down=False, rows=out_features, columns=rank)
packed_up = packer.pack_lowrank_weight(logical_up, down=False)

logical_down = packer.unpack_lowrank_weight(packed_proj_down, down=True, rows=in_features, columns=rank)
packed_down = packer.pack_lowrank_weight(logical_down, down=True)
```

## Fused Output Split Pattern

For fused output projections such as QKV:

1. Unpack output-side tensors to logical layout.
2. Split output rows in logical layout.
3. Repack each split target.
4. Duplicate input-side tensors that are shared by the fused projection.

Output-side examples:

- `qweight`
- `wscales`
- `wcscales`
- `bias`
- `proj_up`

Input-side examples:

- `smooth_factor`
- `proj_down`

## Split-to-Merged Pattern

The inverse situation of the fused split: the stock Diffusers graph has **one**
linear where the source checkpoint has **two** (example: FLUX single-block
`proj_out(15360→3072)` vs Nunchaku `out_proj(3072→3072)` + `mlp_fc2(12288→3072)`
computing `proj_out(cat(attn_out, mlp_out))`). Because the Lite quantizer can
only replace modules that exist in the stock graph, the checkpoint must carry
the merged target.

**Precondition — group alignment.** Merging two W4A4 calls into one is
numerically safe only if the concat boundary is a multiple of the activation
quantization group (64 for INT4, 16 for NVFP4). Activation scales are computed
per token *per group*, so with an aligned boundary the merged call quantizes
bit-identically to the two split calls. Check this before merging; if the
boundary is unaligned, the merge changes activation quantization and you must
re-quantize instead.

Merge recipe (all steps in logical layout — unpack, edit, repack):

- `qweight`: concat along the input dimension.
- `wscales`: concat along the group dimension, same order as the inputs.
- `smooth_factor`: concat.
- `proj_down`: block-diagonal — `(rank_l + rank_r, in_l + in_r)` with each
  source block on its own rows/columns.
- `proj_up`: concat along the rank dimension.
- `bias`: sum of the two logical biases. (In practice one side often stores an
  all-zero bias — verify with the actual tensor, never assume.)
- NVFP4 only: `wcscales` concat is fine, but `wtscale` is **per-tensor**; two
  different `wtscale` values cannot be merged without rescaling the fp8
  `wscales`. Prefer re-quantization for NVFP4 merges unless both `wtscale`
  values are equal (commonly both 1.0).

The merged target's rank is `rank_l + rank_r`; see Rank Uniformity below for
what that implies for the other targets.

## Signed-Unfused Bias Compensation (INT4 shifted down-projections)

Native Nunchaku runs down-projections after GELU (e.g. `ff.net.2`, `mlp_fc2`)
with GELU **fused** into the quantize kernel, a constant shift added
(`0.171875`, just above GELU's minimum of −0.1700), and **unsigned** INT4
activation quantization. The checkpoint bias for such layers is stored
pre-compensated:

```text
b_ckpt = b_true − shift · Σ_j (Ŵ_ij / smooth_j)
```

The Diffusers Lite runtime computes GELU as a normal graph op and always
quantizes **signed with no shift**, so the conversion must add the term back:

```python
qcodes = packer.unpack_weight(qweight, rows=out_f, columns=in_f).double()
scales = packer.unpack_scale(wscales, rows=out_f, groups=in_f // 64, group_size=64).double()
smooth = packer.unpack_scale(smooth_factor, rows=in_f, groups=1, group_size=-1).double()
w_eff = (qcodes.view(out_f, -1, 64) * scales.view(out_f, -1, 1)).view(out_f, in_f) / smooth
b_true = b_ckpt.double() + 0.171875 * w_eff.sum(dim=1)
```

Apply this to every shifted down-projection **before** any split→merged step,
so only the GELU half of a merged layer carries the correction. The cost of
dropping unsigned quantization (~1 bit on the mostly-positive GELU branch) is
inherent and affects any layout equally; it is not a bug signal.

## Rank Uniformity Padding

The Lite `quantization_config` has a single global `svdq_w4a4.rank`. If a merge
produced a higher-rank target (e.g. 64) while other targets are rank 32, pad
every lower-rank `proj_down`/`proj_up` with zeros up to the global rank —
**in logical layout**:

```python
logical_down = packer.unpack_lowrank_weight(packed_down, down=True, rows=cur_rank, columns=in_f)
padded = torch.zeros(target_rank, in_f, dtype=logical_down.dtype)
padded[:cur_rank] = logical_down
packed = packer.pack_lowrank_weight(padded, down=True)
```

Zero rank columns contribute exactly nothing at runtime, so a padded module
must match its unpadded original bit-for-bit up to kernel accumulation —
verify this, it is a cheap and decisive check. Zero-padding the **packed**
container instead shifts real values into wrong tile positions and corrupts
the low-rank correction of every padded layer.

## AWQ W4A16 Rules

AWQ W4A16 tensors are already quantized. Convert layout by permuting packed rows/scales/zeros/bias when needed; do not dequantize them to floating-point weights.

For a model-specific output interleave:

1. Identify the logical output order expected by Diffusers.
2. Apply the inverse permutation to packed `qweight` rows.
3. Apply the same output permutation to `wscales`, `wzeros`, and `bias`.
4. Undo any model-specific bias offsets introduced by the source layout.

## Diffusers Config Rules

The transformer config must include a compact Nunchaku Lite quantization config:

```json
{
  "quantization_config": {
    "quant_method": "nunchaku_lite",
    "compute_dtype": "bfloat16",
    "svdq_w4a4": {
      "precision": "nvfp4",
      "group_size": 16,
      "rank": 32,
      "targets": ["..."]
    },
    "awq_w4a16": {
      "precision": "int4",
      "group_size": 64,
      "targets": ["..."]
    }
  }
}
```

Use `group_size=16` for NVFP4 SVDQ and `group_size=64` for INT4 SVDQ/AWQ unless the local Diffusers loader says otherwise.

## Validation

Climb this ladder in order; each rung catches what the previous one cannot.

1. **Layout-transform unit tests.** Push tensors with *distinctive* values
   (e.g. `(arange(n) % prime).to(bfloat16)`) through the transform and assert
   against a pack/unpack round-trip of the expected logical result. Constant
   fills (`torch.ones`, `torch.full`) are permutation-invariant and will pass
   even when the packing is scrambled — never use them for these tests.

2. **Exact key match.** Converted state dict keys vs the target list, both
   directions.

3. **Layer-level A/B with real kernels.** For at least one real block per
   conversion situation, instantiate the Lite runtime module (e.g.
   `diffusers.quantizers.nunchaku.utils.SVDQW4A4Linear`) from the converted
   tensors and from the source layout, feed identical bf16 inputs, and compare
   on GPU:
   - fused split: `fused(x)` vs `cat(q(x), k(x), v(x))`
   - split→merged: `merged(cat(x_a, x_b))` vs `left(x_a) + right(x_b)` — also
     with each input half zeroed, which isolates the halves and should match
     exactly;
   - rank padding: padded module vs original module, expect exact match.
   Acceptable: ≤ ~2% relative error (bf16 accumulation-order noise). Tens of
   percent means a layout bug. This rung is the only cheap one that catches
   packed-tensor mistakes — shape checks and clean loads do not.

4. **Pipeline load** with the intended Diffusers checkout, plain
   `from_pretrained`, no patches.

5. **Full multi-step inference** with fixed seed/params, compared against the
   native Nunchaku baseline image (MAE/RMSE in pixel space). Calibrate
   expectations with a known-good pair first: same-class distance (e.g. ~17–20
   MAE for a 28-step FLUX edit) passes; several times that means a bug. A
   one-step smoke test is not enough.

Bad images after a clean load usually mean one of:

- packed tensors were sliced/concatenated/padded directly in container space;
- a shifted (fused-GELU unsigned) down-projection bias was not compensated;
- source AWQ layout was not converted to runtime layout;
- scale keys such as `wtscale`/`wcscales` were synthesized incorrectly;
- config target lists replaced the wrong modules;
- text encoder packaging differs from the intended pipeline.
