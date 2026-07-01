# DeepCompressor SVDQuant Mapping

DeepCompressor SVDQuant for diffusion is configured by merging its default
diffusion config, `configs/svdquant/__default__.yaml`, precision config such as
`configs/svdquant/int4.yaml`, and a model config such as
`configs/model/flux.1-dev.yaml`. In `diffuse_compressor`, numeric quantization
settings map to `DiffusionQuantSpec`, calibration storage maps to
`CalibrationSpec`, and model architecture choices move into `TargetConfig`.

| DeepCompressor setting | Meaning | `diffuse_compressor` equivalent |
| --- | --- | --- |
| `quant.wgts.dtype: sint4` | INT4 weight quantization | `DiffusionQuantSpec(precision="int4")` |
| `quant.wgts.dtype: sfp4_e2m1_all` | FP4/NVFP4 residual weight quantization | `DiffusionQuantSpec(precision="fp4")` |
| `quant.wgts.group_shapes: [[1, 64, 1, 1, 1]]` | 64-wide input/channel groups | `DiffusionQuantSpec(group_size=64)` |
| `quant.wgts.group_shapes: [[-1, -1], [1, 16, 1, 1, 1]]` | NVFP4 two-level weight quantization: tensor/channel scale plus 16-wide micro groups | `DiffusionQuantSpec(group_size=16, weight_scale_dtypes=(None, "sfp8_e4m3_nan"))`; exports FP8 micro `wscales` plus outer `wcscales`/`wtscale` |
| `quant.wgts.scale_dtypes: [null]` | Weight scales remain unquantized/model dtype | `DiffusionQuantSpec(weight_scale_dtypes=(None,))` |
| `quant.wgts.scale_dtypes: [null, sfp8_e4m3_nan]` | NVFP4 outer scale remains model dtype and micro scales use SFP8 | `DiffusionQuantSpec(weight_scale_dtypes=(None, "sfp8_e4m3_nan"))` |
| `quant.wgts.low_rank.rank: 32` | SVD low-rank branch rank | `DiffusionQuantSpec(rank=32)` |
| `quant.wgts.enable_low_rank: true` | Enable low-rank branch | `rank > 0` and `TargetRule(..., quant=SvdqTargetQuant(shared_low_rank=True))` |
| `quant.wgts.low_rank.exclusive: false` | Share low-rank branch across grouped projections | Group modules in one `TargetRule` |
| `quant.wgts.low_rank.num_iters`, `early_stop`, `compensate` | DeepCompressor-style low-rank search controls | `LowRankSolverSpec(mode="search", num_iters=..., early_stop=..., compensate=...)` |
| `quant.wgts.low_rank.objective` | Low-rank search objective | `LowRankSolverSpec(objective="outputs_error")`; only output error is supported |
| `quant.wgts.low_rank.degree` | Error norm degree for search | `LowRankSolverSpec(degree=...)` |
| `quant.wgts.low_rank.sample_size` | Number of samples for low-rank scoring | `LowRankSolverSpec(sample_size=...)` |
| `quant.wgts.low_rank.sample_batch_size` | Low-rank scoring sample batch size | `CalibrationSpec(sample_batch_size=...)` for replay partitions; `LowRankSolverSpec(sample_size=...)` for solver subsampling |
| DeepCompressor `torch.svd_lowrank` acceleration | Approximate low-rank branch SVD | `LowRankSolverSpec(svd_backend="svd_lowrank", svd_lowrank_oversample=10, svd_lowrank_niter=4)` |
| `quant.wgts.low_rank.skips` / `quant.wgts.skips` | Skip model parts | Do not include those modules in `TargetConfig.targets` |
| `quant.wgts.calib_range.*` | Weight dynamic-range calibration state | `WeightRangeCalibrationSpec(...)` exports calibrated residual weight range tensors |
| `quant.wgts.gptq.*` / `configs/svdquant/gptq.yaml` | GPTQ residual-weight rounding after low-rank residual construction | `DiffusionQuantSpec(gptq=GptqSpec(enabled=True, ...))`; works with `precision="int4"` and FP4/NVFP4 overlays |
| `quant.ipts.dtype: sint4` | Runtime activation quantization | `ActivationQuantSpec(enabled=True, dtype="int4", ...)` exports activation scale/zero tensors |
| `quant.ipts.dtype: sfp4_e2m1_all` | FP4/NVFP4 runtime activation quantization | Not a separate runtime activation packer yet; activation range metadata still uses `ActivationQuantSpec`, while weight FP4 export is supported |
| `quant.ipts.group_shapes: [[1, 64, 1, 1, 1]]` | INT4 runtime activation groups | `RangeCalibrationSpec(granularity="group")` with `DiffusionQuantSpec(group_size=64)` |
| `quant.ipts.group_shapes: [[1, 16, 1, 1, 1]]` | NVFP4 runtime activation groups | `RangeCalibrationSpec(granularity="group")` with `DiffusionQuantSpec(group_size=16)` for exported activation range metadata |
| `quant.ipts.scale_dtypes: [null]` | INT4 activation scales remain model dtype | `ActivationQuantSpec(scale_dtypes=(None,))` |
| `quant.ipts.scale_dtypes: [sfp8_e4m3_nan]` | NVFP4 activation scale dtype metadata | `ActivationQuantSpec(scale_dtypes=("sfp8_e4m3_nan",))` |
| `quant.ipts.static: true` / `false` | Static activation metadata vs dynamic/runtime activation intent | `ActivationQuantSpec(static=True/False)` |
| `quant.ipts.allow_unsigned: true` | Allow unsigned activation paths | `RangeCalibrationSpec(allow_unsigned=True)` |
| `quant.ipts.calib_range.*` | Input activation range calibration | `ActivationQuantSpec(inputs=RangeCalibrationSpec(...))` |
| `quant.opts.calib_range.*` | Output activation range calibration | `ActivationQuantSpec(outputs=RangeCalibrationSpec(...))` |
| `quant.enable_smooth` / `quant.smooth.proj.*` | SmoothQuant-style projection smoothing | `SmoothSpec(...)` passed through `DiffusionQuantSpec.smooth` |
| `quant.smooth.proj.objective`, `strategy`, `alpha`, `beta`, `num_grids`, `spans` | Projection smoothing search objective and search space | `SmoothSpec(objective="outputs_error", strategy=..., alpha=..., beta=..., num_grids=..., spans=...)` |
| `quant.smooth.proj.granularity`, `allow_low_rank`, `fuse_when_possible`, `skips` | Architecture-aware projection smoothing policy | Partially user-owned through `TargetRule`; full parity not modeled yet |
| `quant.smooth.proj.sample_batch_size`, `sample_size` | Projection smoothing calibration batching/subsampling | `CalibrationSpec(sample_batch_size=..., sample_size=...)` plus `SmoothSpec(sample_size=...)` |
| `quant.enable_extra_wgts: true` | Quantize selected modulation or norm linears with AWQ, used by NVFP4 config | Add extra `TargetRule(..., quant=AwqTargetQuant(...))` entries for those modules |
| `quant.extra_wgts.dtype: sint4` | AWQ targets use INT4 instead of FP4 | `TargetRule(..., quant=AwqTargetQuant())` |
| `quant.extra_wgts.group_shapes: [[1, 64, 1, 1, 1]]` | AWQ targets use 64-wide groups | `AwqTargetQuant()` fixes `group_size=64` |
| Runtime naive SVDQ layout | Force logical SVDQ tensors for torch-dequant/debug workflows | `TargetRule(..., quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout()))` |
| Runtime Nunchaku SVDQ layout | Require packed Nunchaku W4A4 tensors and fail if packing cannot be produced | `TargetRule(..., quant=SvdqTargetQuant(weight_layout=NunchakuSvdqLayout()))` |
| Runtime AWQ W4A16 layout | Nunchaku Lite loads selected AWQ linears as W4A16 modules | `TargetRule(..., quant=AwqTargetQuant(layout=AwqW4A16Layout()))` |
| Runtime AdaNorm AWQ layout | DeepCompressor/Nunchaku AdaNorm modulation linears use interleaved W4A16 export | `TargetRule(..., quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=3 or 6)))` |
| `quant.extra_wgts.scale_dtypes: [null]` | AWQ weight scales remain unquantized/model dtype | Use the default `weight_scale_dtypes=(None,)` semantics for those INT4 targets |
| `quant.extra_wgts.includes: [transformer_norm, transformer_add_norm]` | Architecture semantic include list for AWQ targets | Model-agnostic core does not know these labels; user config should add matching `TargetRule`s for the corresponding modules |
| `quant.develop_dtype` | Internal calibration/search dtype | Not exposed; current internals use float32/float64 where needed |
| `pipeline.shift_activations: true` | Shift activation lower-bound outliers into weights | `SvdqTargetQuant(shift_activations=True)` opts specific targets into scalar lower-bound shift calibration; manual `PatchRule(type="shift_linear", ...)` remains available |
| `quant.calib.data` | Named DeepCompressor calibration dataset | User supplies `CalibrationSpec(samples=...)`, `prompts=...`, or `forward_fn=...` |
| `quant.calib.path` | Calibration cache path | `CalibrationSpec(cache_dir=...)` |
| `quant.calib.num_samples: 128` | Number of calibration samples | `CalibrationSpec(num_samples=128)` |
| `quant.calib.batch_size` | Calibration batch size | `CalibrationSpec(batch_size=...)` |
| DeepCompressor calibration dataset/data loader | Cached sample replay | `CalibrationSpec(batch_size=..., shuffle=..., drop_last=..., num_workers=..., eager_load_samples=...)` |
| DeepCompressor model/block structs | Block-wise calibration replay | `TargetConfig.calibration_scopes` |

## Equivalent INT4 SVDQuant Skeleton

```python
spec = DiffusionQuantSpec(
    method="svdquant",
    precision="int4",
    rank=32,
    group_size=64,
    low_rank_solver=LowRankSolverSpec(
        mode="search",
        num_iters=100,
        early_stop=True,
        compensate=False,
        activation_quant=True,
        eval_replay=True,
        svd_backend="svd_lowrank",
        svd_lowrank_oversample=10,
        svd_lowrank_niter=4,
        degree=2,
    ),
    smooth=SmoothSpec(
        enabled=True,
        objective="outputs_error",
        strategy="grid_search",
        alpha=0.5,
        beta=-2,
        num_grids=20,
        spans=(("absmax", "absmax"),),
    ),
    activation_quant=ActivationQuantSpec(
        enabled=True,
        static=True,
        scale_dtypes=(None,),
        inputs=RangeCalibrationSpec(granularity="channel", allow_unsigned=True),
        outputs=RangeCalibrationSpec(granularity="tensor"),
    ),
    weight_range_calibration=WeightRangeCalibrationSpec(
        enabled=True,
        range=RangeCalibrationSpec(granularity="group"),
    ),
)

calibration = CalibrationSpec(
    samples=samples,
    num_samples=128,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    eager_load_samples=False,
    cache_dir="outputs/calibration/flux",
    cache_mode="reuse",
    max_rows_per_target=4096,
    ram_usage_limit=0.90,
    forward_fn=run_sample,
    artifact_cache=QuantizationCacheSpec(
        cache_dir="outputs/calibration/flux/artifacts",
        cache_mode="reuse",
    ),
)

target_config = TargetConfig(
    patches=[...],
    calibration_scopes=[...],
    targets=[...],
)
```

The main translation difference is ownership: DeepCompressor derives many
module groups and block scopes from architecture-specific structs, while
`diffuse_compressor` keeps that knowledge in user-provided config.
