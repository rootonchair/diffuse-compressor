# Original DeepCompressor Diffusion SVDQuant Flow

This note summarizes the original DeepCompressor diffusion SVDQuant quantization
flow. It is intended as a reference for this repository, which currently
implements a model-agnostic weighted-SVD path and does not fully reproduce
DeepCompressor's architecture-aware search optimizer.

## Entry Point

The original diffusion PTQ flow starts in the DeepCompressor repository at:

```text
deepcompressor/app/diffusion/ptq.py
```

That entry point builds a `DiffusionModelStruct`, resolves the active
quantization configuration, then orchestrates rotation, smoothing, weight
quantization, low-rank branch calibration, activation quantization, and cache
save/load behavior.

## High-Level Flow

1. Build the diffusion model structure.
   - Wrap the model with `DiffusionModelStruct`.
   - Use architecture-aware metadata for layers, blocks, attention modules,
     QKV projections, skip modules, and parent eval modules.
   - Determine whether weight quantization, input activation quantization,
     output activation quantization, rotation, smoothing, and low-rank paths are
     enabled.

2. Optionally apply rotation.
   - The diffusion PTQ entry point supports a rotation stage when
     `quant.rotation.transforms` is configured.
   - The shipped diffusion SVDQuant presets do not enable rotation; their main
     pre-quantization transform is smoothing.
   - When enabled by a custom config, rotation is applied before smoothing and
     weight quantization.

3. Optionally apply smoothing.
   - Load an existing `smooth.pt` cache if available.
   - Otherwise calibrate smooth factors and apply them to the model.
   - Save smooth state for later reuse.

4. Run or load weight quantization.
   - If a cached quantized model exists, load the quantized model weights,
     weight quantizer state, and low-rank branch state.
   - Otherwise run diffusion weight quantization from:

```text
deepcompressor/app/diffusion/quant/weight.py
```

5. Calibrate low-rank branches when enabled.
   - Iterate calibration activations block by block.
   - For each block, find quantizable modules.
   - If `low_rank.exclusive == false`, group architecture-specific attention
     projections such as self-attention `q_proj`, `k_proj`, `v_proj` or joint
     attention `add_q_proj`, `add_k_proj`, `add_v_proj`.
   - Select an architecture-aware `eval_module`. Depending on the target, this
     can be the target module itself, the parent attention module, or a parent
     transformer block.
   - Call `QuantLowRankCalibrator` with the target modules, target inputs,
     eval-module inputs, eval kwargs, weight quantizer, and activation
     quantizer.

6. Search for the best low-rank branch.
   - The original low-rank calibration is implemented by:

```text
deepcompressor/calib/lowrank.py
```

   - This is a search-based optimizer, not a direct weighted-SVD solve.
   - For each candidate, DeepCompressor:
     - builds a low-rank branch from the residual weight,
     - computes residual weight as original weight minus low-rank branch
       weight,
     - quantizes the residual weight,
     - temporarily installs the quantized residual weight,
     - attaches the candidate low-rank branch as a hook,
     - optionally includes activation quantization in the replay path,
     - replays the selected `eval_module`,
     - scores output error,
     - keeps the best candidate,
     - applies early stopping when configured.

7. Apply the selected branch and residual weight.
   - Save the best low-rank branch into `branch_state_dict`.
   - Subtract the selected branch's effective weight from the original module
     weight.
   - Register the branch hook on the module.
   - The remaining base weight is the residual weight that will be quantized.

8. Calibrate weight quantizer ranges.
   - If the weight quantizer needs calibration data, DeepCompressor performs
     another layer-activation pass.
   - It calibrates dynamic ranges using sampled module inputs.
   - The resulting weight quantizer state is saved for reuse.

9. Quantize weights.
   - Load calibrated quantizer state.
   - Quantize each target module weight.
   - Replace `module.weight.data` with the dequantized quantized weight.
   - Optionally save scale and zero-point state.
   - Save the quantized model if configured.

10. Quantize activations when enabled.
    - Load an existing `acts.pt` cache if available.
    - Otherwise iterate layer activations again and calibrate input/output
      activation quantizers.
    - Register activation quantizer hooks on the model.

## Calibration Data Behavior

The original implementation stores calibration samples as `.pt` files on disk,
then loads the selected samples into a calibration dataset. During quantization,
it does not keep every module activation for the whole model at once. Instead,
it iterates layer by layer, registers hooks for the current layer's modules,
collects the activations needed for that layer, yields them to the quantization
step, then clears the layer cache before moving to the next layer.

This enables DeepCompressor to perform architecture-aware replay while limiting
activation memory to the current layer or block being processed.

## Cache Artifacts

The original flow may read or write these artifacts:

```text
smooth.pt   # smoothing state
branch.pt   # low-rank branch state
wgts.pt     # weight quantizer state
acts.pt     # activation quantizer state
model.pt    # quantized model weights
scale.pt    # optional quant scale/zero state
```

## Difference From This Repository

The current implementation in this repository is intentionally more
model-agnostic:

- It discovers user-specified module targets instead of using
  `DiffusionModelStruct`.
- It supports generic module grouping/splitting through config instead of
  hardcoding Flux, SDXL, or other architecture rules.
- Its default low-rank solver is weighted SVD; a separate opt-in search solver
  provides closer DeepCompressor-style behavior.
- Its calibration replay is generic and scope-configured, not derived from
  DeepCompressor architecture structs.
- It exports Nunchaku-compatible tensors for the supported SVDQuant path.

For closer DeepCompressor parity, this repository now exposes a separate
explicit low-rank search solver behind `LowRankSolverSpec(mode="search")`. The
solver is model-agnostic and supports residual quantization candidate
evaluation, optional eval-module replay, compensation, multiple iterations,
activation fake quantization in the objective, and early stopping. It is still
not a byte-for-byte port of DeepCompressor's architecture-specific calibrator.
