from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence


ScaleDType = str | None


def _validate_scale_dtypes(name: str, scale_dtypes: Sequence[ScaleDType]) -> None:
    """Validate a DeepCompressor-style scale dtype sequence.

    Args:
        name: Configuration field name used in error messages.
        scale_dtypes: Scale dtype names to validate.
    """

    supported = {None, "sfp8_e4m3_nan", "float8_e4m3fn"}
    if not scale_dtypes:
        raise ValueError(f"{name} must contain at least one scale dtype")
    for dtype in scale_dtypes:
        if dtype not in supported:
            raise ValueError(f"Unsupported {name} value: {dtype!r}")


@dataclass(frozen=True)
class LowRankSolverSpec:
    """Configure the solver used to build the SVDQuant low-rank branch.

    Args:
        mode: Solver implementation to use. ``"weighted_svd"`` computes one
            calibrated weighted SVD branch, while ``"search"`` evaluates
            DeepCompressor-style residual candidates.
        num_iters: Maximum number of low-rank search iterations.
        early_stop: Stop candidate search when the objective stops improving.
        compensate: Apply residual compensation during search.
        activation_quant: Include fake activation quantization in the search
            objective.
        objective: Objective name used by the search solver.
        degree: Error norm degree used by the search objective.
        sample_size: Number of calibration samples to score, or ``-1`` for all.
        eval_replay: Whether search may use stored eval-module replay batches.
        svd_backend: SVD routine for low-rank branch initialization. ``"full"``
            uses exact ``torch.linalg.svd``; ``"svd_lowrank"`` uses
            approximate ``torch.svd_lowrank``.
        svd_lowrank_oversample: Extra rank used for ``torch.svd_lowrank``.
        svd_lowrank_niter: Power iterations used for ``torch.svd_lowrank``.
    """

    mode: Literal["weighted_svd", "search"] = "weighted_svd"
    num_iters: int = 1
    early_stop: bool = False
    compensate: bool = False
    activation_quant: bool = False
    objective: Literal["outputs_error"] = "outputs_error"
    degree: int = 2
    sample_size: int = -1
    eval_replay: bool = True
    svd_backend: Literal["full", "svd_lowrank"] = "full"
    svd_lowrank_oversample: int = 10
    svd_lowrank_niter: int = 4

    def __post_init__(self) -> None:
        """Validate solver options after dataclass construction."""
        if self.mode not in {"weighted_svd", "search"}:
            raise ValueError(f"Unsupported low-rank solver mode: {self.mode!r}")
        if self.svd_backend not in {"full", "svd_lowrank"}:
            raise ValueError(f"Unsupported low-rank SVD backend: {self.svd_backend!r}")
        if self.num_iters <= 0:
            raise ValueError("low-rank solver num_iters must be positive")
        if self.objective != "outputs_error":
            raise ValueError(f"Unsupported low-rank solver objective: {self.objective!r}")
        if self.degree <= 0:
            raise ValueError("low-rank solver degree must be positive")
        if self.sample_size == 0 or self.sample_size < -1:
            raise ValueError("low-rank solver sample_size must be -1 or a positive integer")
        if self.svd_lowrank_oversample < 0:
            raise ValueError("svd_lowrank_oversample must be non-negative")
        if self.svd_lowrank_niter < 0:
            raise ValueError("svd_lowrank_niter must be non-negative")


@dataclass(frozen=True)
class SmoothSpec:
    """Configure activation/weight smoothing before SVDQuant residual quantization.

    Args:
        enabled: Whether smoothing is active for eligible targets.
        strategy: Use fixed ``alpha``/``beta`` values or grid-search candidates.
        objective: Objective name used to score smoothing candidates.
        alpha: Activation-span exponent for manual smoothing.
        beta: Weight-span exponent for manual smoothing.
        num_grids: Number of alpha/beta grid points per searched span pair.
        spans: Pairs of activation and weight span estimators to search.
        sample_size: Number of calibration rows to score, or ``-1`` for all.
        eps: Positive numerical floor applied to smoothing scales.
    """

    enabled: bool = True
    strategy: Literal["manual", "grid_search"] = "grid_search"
    objective: Literal["outputs_error"] = "outputs_error"
    alpha: float = 0.5
    beta: float = -2.0
    num_grids: int = 20
    spans: Sequence[tuple[Literal["absmax", "rms"], Literal["absmax", "rms"]]] = (("absmax", "absmax"),)
    sample_size: int = -1
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate smoothing strategy, grid, and numerical bounds."""
        if self.strategy not in {"manual", "grid_search"}:
            raise ValueError(f"Unsupported smoothing strategy: {self.strategy!r}")
        if self.objective != "outputs_error":
            raise ValueError(f"Unsupported smoothing objective: {self.objective!r}")
        if not -3 <= self.alpha <= 1:
            raise ValueError("smooth alpha must be in [-3, 1]")
        if not -3 <= self.beta <= 1:
            raise ValueError("smooth beta must be in [-3, 1]")
        if self.num_grids <= 1:
            raise ValueError("smooth num_grids must be greater than 1")
        if self.sample_size == 0 or self.sample_size < -1:
            raise ValueError("smooth sample_size must be -1 or a positive integer")
        if self.eps <= 0:
            raise ValueError("smooth eps must be positive")
        if not self.spans:
            raise ValueError("smooth spans must contain at least one span pair")
        for alpha_span, beta_span in self.spans:
            if alpha_span not in {"absmax", "rms"} or beta_span not in {"absmax", "rms"}:
                raise ValueError(f"Unsupported smooth span pair: {(alpha_span, beta_span)!r}")


@dataclass(frozen=True)
class RangeCalibrationSpec:
    """Describe how min/max ranges are converted to quantization parameters.

    Args:
        enabled: Whether this range calibration path should run.
        granularity: Range sharing level, currently tensor/channel/group.
        symmetric: Use signed symmetric quantization bounds when true.
        allow_unsigned: Permit unsigned ranges when the observed data is
            non-negative.
        eps: Positive numerical floor for scale computation.
    """

    enabled: bool = True
    granularity: Literal["tensor", "channel", "group"] = "tensor"
    symmetric: bool = True
    allow_unsigned: bool = False
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate range calibration granularity and numerical floor."""
        if self.granularity not in {"tensor", "channel", "group"}:
            raise ValueError(f"Unsupported range calibration granularity: {self.granularity!r}")
        if self.eps <= 0:
            raise ValueError("range calibration eps must be positive")


@dataclass(frozen=True)
class ActivationQuantSpec:
    """Configure optional static activation quantization metadata.

    Args:
        enabled: Whether activation ranges should be calibrated and exported.
        dtype: Activation quantization dtype. Only INT4 is currently accepted.
        static: Whether exported activation parameters are static calibration
            values.
        scale_dtypes: Scale dtype sequence used by exported activation scales.
        inputs: Range calibration settings for target inputs.
        outputs: Range calibration settings for target outputs.
    """

    enabled: bool = False
    dtype: Literal["int4"] = "int4"
    static: bool = True
    scale_dtypes: Sequence[ScaleDType] = (None,)
    inputs: RangeCalibrationSpec = field(default_factory=lambda: RangeCalibrationSpec(enabled=True))
    outputs: RangeCalibrationSpec = field(default_factory=lambda: RangeCalibrationSpec(enabled=True))

    def __post_init__(self) -> None:
        """Validate activation dtype and nested range specs."""
        if self.dtype != "int4":
            raise ValueError(f"Unsupported activation quant dtype: {self.dtype!r}")
        _validate_scale_dtypes("activation_quant.scale_dtypes", self.scale_dtypes)
        if not isinstance(self.inputs, RangeCalibrationSpec):
            raise TypeError("activation_quant.inputs must be a RangeCalibrationSpec")
        if not isinstance(self.outputs, RangeCalibrationSpec):
            raise TypeError("activation_quant.outputs must be a RangeCalibrationSpec")


@dataclass(frozen=True)
class WeightRangeCalibrationSpec:
    """Configure optional range calibration for quantized residual weights.

    Args:
        enabled: Whether calibrated residual weight ranges should override the
            default per-group scale path.
        range: Range calibration settings for residual weights.
    """

    enabled: bool = False
    range: RangeCalibrationSpec = field(default_factory=lambda: RangeCalibrationSpec(enabled=True))

    def __post_init__(self) -> None:
        """Validate the nested range calibration spec."""
        if not isinstance(self.range, RangeCalibrationSpec):
            raise TypeError("weight_range_calibration.range must be a RangeCalibrationSpec")


@dataclass(frozen=True)
class QuantizationCacheSpec:
    """Configure persisted quantization artifacts.

    Args:
        cache_dir: Directory where quantization artifacts are stored.
        cache_mode: Cache behavior: reuse valid artifacts, refresh them, or
            disable artifact caching.
        save_model: Whether to save the combined model artifact as
            ``model.pt`` for direct cache reload.
    """

    cache_dir: str | Path | None = None
    cache_mode: Literal["reuse", "refresh", "disabled"] = "reuse"
    save_model: bool = True

    def __post_init__(self) -> None:
        """Validate artifact cache settings."""

        if self.cache_mode not in {"reuse", "refresh", "disabled"}:
            raise ValueError(f"Unsupported quantization cache_mode: {self.cache_mode!r}")


@dataclass(frozen=True)
class DiffusionQuantSpec:
    """Top-level quantization settings for diffusion SVDQuant export.

    Args:
        method: Quantization method. Currently only ``"svdquant"`` is
            implemented.
        precision: Residual weight precision requested for export.
        rank: Low-rank branch rank. ``0`` disables the low-rank branch.
        group_size: Residual quantization group size along input channels.
        weight_scale_dtypes: Scale dtype sequence used by residual weight
            scales.
        low_rank_solver: Solver settings for the low-rank branch.
        smooth: Smoothing settings, or a boolean to enable/disable defaults.
        activation_quant: Optional activation quantization calibration settings.
        weight_range_calibration: Optional residual weight range calibration.
        shift_activations: Whether shifted wrapper modules should shift inputs.
        torch_dtype: Optional string dtype hint for exported metadata.
    """

    method: Literal["svdquant"] = "svdquant"
    precision: Literal["int4", "fp4"] = "int4"
    rank: int = 32
    group_size: int = 64
    weight_scale_dtypes: Sequence[ScaleDType] = (None,)
    low_rank_solver: LowRankSolverSpec = field(default_factory=LowRankSolverSpec)
    smooth: bool | SmoothSpec = True
    activation_quant: ActivationQuantSpec = field(default_factory=ActivationQuantSpec)
    weight_range_calibration: WeightRangeCalibrationSpec = field(default_factory=WeightRangeCalibrationSpec)
    shift_activations: bool = False
    torch_dtype: str | None = None

    def __post_init__(self) -> None:
        """Validate the top-level quantization configuration."""
        if self.method != "svdquant":
            raise ValueError(f"Unsupported quantization method: {self.method!r}")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        _validate_scale_dtypes("weight_scale_dtypes", self.weight_scale_dtypes)
        if not isinstance(self.low_rank_solver, LowRankSolverSpec):
            raise TypeError("low_rank_solver must be a LowRankSolverSpec")
        if not isinstance(self.smooth, (bool, SmoothSpec)):
            raise TypeError("smooth must be a bool or SmoothSpec")
        if not isinstance(self.activation_quant, ActivationQuantSpec):
            raise TypeError("activation_quant must be an ActivationQuantSpec")
        if not isinstance(self.weight_range_calibration, WeightRangeCalibrationSpec):
            raise TypeError("weight_range_calibration must be a WeightRangeCalibrationSpec")


@dataclass(frozen=True)
class PatchRule:
    """Describe one model rewrite to apply before target collection.

    Args:
        type: Rewrite type, such as splitting a fused linear projection.
        module: Module path pattern that selects modules to rewrite.
        args: Rewrite-specific keyword arguments.
    """

    type: Literal["split_linear", "split_linear_output", "split_conv", "shift_linear", "shift_conv"]
    module: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRule:
    """Describe one quantization target or grouped target pattern.

    Args:
        name: Logical target name template.
        modules: Module path patterns. Shared wildcard captures form grouped
            targets.
        export_name: Optional export name template; defaults to ``name``.
        kind: Target module kind, currently ``"linear"`` or ``"conv"``.
        roles: Optional semantic roles for grouped modules.
        shared_low_rank: Whether grouped modules share one low-rank branch.
        smooth_key: Optional key used to share smoothing ranges across targets.
        precision: Optional precision override for this target.
        group_size: Optional group-size override for this target.
        rank: Optional low-rank rank override for this target.
        smooth: Optional smoothing override for this target.
        activation_quant: Optional activation quantization override for this
            target.
        shift_activations: Optional activation shift override for this target.
    """

    name: str
    modules: Sequence[str]
    export_name: str | None = None
    kind: Literal["linear", "conv"] = "linear"
    roles: Sequence[str] = field(default_factory=tuple)
    shared_low_rank: bool = True
    smooth_key: str | None = None
    precision: Literal["int4", "fp4"] | None = None
    group_size: int | None = None
    rank: int | None = None
    smooth: bool | SmoothSpec | None = None
    activation_quant: bool | ActivationQuantSpec | None = None
    shift_activations: bool | None = None

    def __post_init__(self) -> None:
        """Validate target-level quantization overrides."""

        if self.precision is not None and self.precision not in {"int4", "fp4"}:
            raise ValueError(f"Unsupported target precision override: {self.precision!r}")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("target group_size override must be positive")
        if self.rank is not None and self.rank < 0:
            raise ValueError("target rank override must be non-negative")
        if self.smooth is not None and not isinstance(self.smooth, (bool, SmoothSpec)):
            raise TypeError("target smooth override must be a bool or SmoothSpec")
        if self.activation_quant is not None and not isinstance(self.activation_quant, (bool, ActivationQuantSpec)):
            raise TypeError("target activation_quant override must be a bool or ActivationQuantSpec")
        if self.shift_activations is not None and not isinstance(self.shift_activations, bool):
            raise TypeError("target shift_activations override must be a bool")


@dataclass(frozen=True)
class CalibrationCaptureRule:
    """Describe extra module I/O to cache while replaying a calibration scope.

    Args:
        name: Cache name used for this capture.
        modules: Module path templates resolved within each scope.
        inputs: Capture module inputs when true.
        outputs: Capture module outputs when true.
        input_keys: Optional input tensor keys or argument indices to keep.
        output_keys: Optional output tensor keys or indices to keep.
        channel_dim: Channel dimension used when flattening captured tensors.
    """

    name: str
    modules: Sequence[str]
    inputs: bool = True
    outputs: bool = False
    input_keys: Sequence[str | int] = field(default_factory=tuple)
    output_keys: Sequence[str | int] = field(default_factory=tuple)
    channel_dim: int = -1

    def __post_init__(self) -> None:
        """Validate that the capture rule selects modules and tensor sides."""
        if not self.modules:
            raise ValueError("CalibrationCaptureRule modules must not be empty")
        if not self.inputs and not self.outputs:
            raise ValueError("CalibrationCaptureRule must capture inputs, outputs, or both")


@dataclass(frozen=True)
class CalibrationScopeRule:
    """Group targets into a replay/capture scope for calibration.

    Args:
        name: Optional scope name template. When omitted, the matched module
            path is used as the concrete scope name.
        modules: Target module path patterns assigned to this scope.
        eval_module: Optional module path used to score low-rank search
            candidates.
        replay_module: Optional module replayed instead of the full model.
        capture_modules: Extra module input/output capture rules.
        cache_aliases: Mapping from target cache names to captured cache names.
        replay_arg_indices: Positional replay arguments to forward.
        replay_kwarg_keys: Keyword replay arguments to forward.
        replay_transform: Optional transform applied to replay inputs.
        prev_output_transform: Optional transform from previous scope output to
            replay inputs.
        prev_replay_transform: Optional transform from the previous scope's
            eval replay record to replay inputs.
        use_prev_scope_outputs: Use outputs from the previous scope as replay
            inputs when true.
        recompute: Recompute from full model inputs instead of replaying a
            narrower module.
    """

    name: str | None = None
    modules: Sequence[str] = field(default_factory=tuple)
    eval_module: str | None = None
    replay_module: str | None = None
    capture_modules: Sequence[CalibrationCaptureRule] = field(default_factory=tuple)
    cache_aliases: Mapping[str, str] = field(default_factory=dict)
    replay_arg_indices: Sequence[int] = field(default_factory=tuple)
    replay_kwarg_keys: Sequence[str] = field(default_factory=tuple)
    replay_transform: Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    prev_output_transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    prev_replay_transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    use_prev_scope_outputs: bool = False
    recompute: bool = False

    def __post_init__(self) -> None:
        """Validate scope naming and matching configuration."""

        if self.name is not None and not isinstance(self.name, str) and not self.modules:
            object.__setattr__(self, "modules", tuple(self.name))
            object.__setattr__(self, "name", None)
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("CalibrationScopeRule name must be a string or None")
        if not self.modules:
            raise ValueError("CalibrationScopeRule modules must not be empty")


@dataclass(frozen=True)
class TargetConfig:
    """Model-agnostic configuration for rewrites, targets, and calibration scopes.

    Args:
        targets: Target rules used to discover quantized modules.
        patches: Optional model rewrite rules applied before target discovery.
        calibration_scopes: Optional scope rules for replayed calibration.
        unquantized_patterns: State-dict patterns kept in the exported artifact.
    """

    targets: Sequence[TargetRule]
    patches: Sequence[PatchRule] = field(default_factory=tuple)
    calibration_scopes: Sequence[CalibrationScopeRule] = field(default_factory=tuple)
    unquantized_patterns: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class CalibrationSpec:
    """Configure calibration sample resolution, disk caching, and batching.

    Args:
        samples: Explicit forward samples as dictionaries.
        prompts: Prompt strings, a prompt file, or prompt sequence converted to
            samples.
        num_samples: Optional limit after sample/prompt resolution.
        batch_size: Batch size used by calibration data loaders.
        cache_dir: Root directory for persisted model-input caches.
        cache_mode: Cache behavior: reuse existing, refresh, or disable.
        seed: Optional deterministic shuffle seed.
        forward_fn: Optional callable for custom model invocation.
        output_dir: Optional directory passed to ``output_save_fn``.
        output_save_fn: Optional callable that stores outputs from
            ``forward_fn``.
        shared_input_keys: Input keys whose tensors are shared metadata and
            should be preserved, not concatenated, during cache replay batching.
        max_rows_per_target: Maximum flattened activation rows retained per
            target cache.
        sample_size: Optional sample partition limit, or ``-1`` for all.
        sample_batch_size: Optional sample partition batch size.
        element_size: Optional element-row partition limit, or ``-1`` for all.
        element_batch_size: Optional element-row partition batch size.
        shuffle: Shuffle calibration samples before batching.
        drop_last: Drop incomplete calibration batches.
        num_workers: Number of PyTorch DataLoader worker processes.
        eager_load_samples: Load disk cache records into RAM up front.
        ram_usage_limit: Fraction of system RAM allowed before aborting.
        artifact_cache: Optional quantization artifact cache settings.
    """

    samples: Sequence[dict[str, Any]] | None = None
    prompts: Sequence[str] | str | Path | None = None
    num_samples: int | None = None
    batch_size: int = 1
    cache_dir: str | Path | None = None
    cache_mode: Literal["reuse", "refresh", "disabled"] = "reuse"
    seed: int | None = None
    forward_fn: Callable[[dict[str, Any]], Any] | None = None
    output_dir: str | Path | None = None
    output_save_fn: Callable[[Any, dict[str, Any], Path], None] | None = None
    shared_input_keys: Sequence[str] = field(default_factory=tuple)
    max_rows_per_target: int = 4096
    sample_size: int = -1
    sample_batch_size: int = -1
    element_size: int = -1
    element_batch_size: int = -1
    shuffle: bool = False
    drop_last: bool = False
    num_workers: int = 0
    eager_load_samples: bool = False
    ram_usage_limit: float = 0.90
    artifact_cache: QuantizationCacheSpec | None = None

    def __post_init__(self) -> None:
        """Validate calibration cache, batching, and RAM guard settings."""
        if self.cache_mode not in {"reuse", "refresh", "disabled"}:
            raise ValueError(f"Unsupported cache_mode: {self.cache_mode!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_rows_per_target <= 0:
            raise ValueError("max_rows_per_target must be positive")
        for name in ("sample_size", "sample_batch_size", "element_size", "element_batch_size"):
            value = getattr(self, name)
            if value == 0 or value < -1:
                raise ValueError(f"{name} must be -1 or a positive integer")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not 0 < self.ram_usage_limit <= 1:
            raise ValueError("ram_usage_limit must be in (0, 1]")
        if self.output_save_fn is not None and not callable(self.output_save_fn):
            raise TypeError("output_save_fn must be callable")
        if isinstance(self.shared_input_keys, str) or any(
            not isinstance(key, str) or not key for key in self.shared_input_keys
        ):
            raise TypeError("shared_input_keys must contain non-empty strings")
        object.__setattr__(self, "shared_input_keys", tuple(self.shared_input_keys))
        if self.artifact_cache is not None and not isinstance(self.artifact_cache, QuantizationCacheSpec):
            raise TypeError("artifact_cache must be a QuantizationCacheSpec")


@dataclass(frozen=True)
class ExportSpec:
    """Configure checkpoint export.

    Args:
        target: Runtime/exporter target. Currently only ``"nunchaku"``.
        output: Output checkpoint path.
        checkpoint_format: Serialized checkpoint format.
    """

    target: Literal["nunchaku"] = "nunchaku"
    output: str | Path = "quantized.safetensors"
    checkpoint_format: Literal["single_safetensors"] = "single_safetensors"

    def __post_init__(self) -> None:
        """Validate exporter target and checkpoint format."""
        if self.target != "nunchaku":
            raise ValueError(f"Unsupported export target: {self.target!r}")
        if self.checkpoint_format != "single_safetensors":
            raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_format!r}")
