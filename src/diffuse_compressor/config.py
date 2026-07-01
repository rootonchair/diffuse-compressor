from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence


ScaleDType = str | None


@dataclass(frozen=True)
class LoggingConfig:
    """Configure optional quantization run logging.

    Args:
        enabled: Enable file logging when true.
        log_dir: Directory where run logs are written.
        name: Optional base filename. When omitted, ``"quantization"`` is used.
        text_output: Write quantization run log messages to a text file.
        target_records: Write per-target timing and error records as JSONL.
    """

    enabled: bool = True
    log_dir: str | Path = "outputs/logs"
    name: str | None = None
    text_output: bool = True
    target_records: bool = True


@dataclass(frozen=True)
class SvdqLayout:
    """Export a target in the default auto-selected SVDQuant W4A4 layout."""

    name: Literal["svdq"] = "svdq"


@dataclass(frozen=True)
class NaiveSvdqLayout:
    """Export a target in the logical/naive SVDQuant W4A4 layout."""

    name: Literal["naive_svdq"] = "naive_svdq"


@dataclass(frozen=True)
class NunchakuSvdqLayout:
    """Export a target in Nunchaku-packed SVDQuant W4A4 layout.

    Args:
        outer_scale_splits: Optional output-row chunks for packed NVFP4 outer
            scales. When omitted, grouped targets use source module output
            widths and single-module targets use a scalar outer scale.
    """

    outer_scale_splits: Sequence[int] = field(default_factory=tuple)
    name: Literal["nunchaku_svdq"] = "nunchaku_svdq"

    def __post_init__(self) -> None:
        if any(split <= 0 for split in self.outer_scale_splits):
            raise ValueError("NunchakuSvdqLayout outer_scale_splits must contain positive integers")


@dataclass(frozen=True)
class AwqW4A16Layout:
    """Export a target in plain AWQ W4A16 layout."""

    name: Literal["awq_w4a16"] = "awq_w4a16"


@dataclass(frozen=True)
class AdaNormAwqW4A16Layout:
    """Export an AdaNorm modulation target in DeepCompressor AWQ W4A16 layout."""

    splits: Literal[3, 6]
    name: Literal["adanorm_awq_w4a16"] = "adanorm_awq_w4a16"

    def __post_init__(self) -> None:
        if self.splits not in {3, 6}:
            raise ValueError(f"Unsupported AdaNorm AWQ W4A16 split count: {self.splits!r}")


SvdqWeightLayoutSpec = SvdqLayout | NaiveSvdqLayout | NunchakuSvdqLayout
AwqWeightLayoutSpec = AwqW4A16Layout | AdaNormAwqW4A16Layout
WeightLayoutSpec = SvdqWeightLayoutSpec | AwqWeightLayoutSpec


def weight_layout_metadata(layout: WeightLayoutSpec) -> dict[str, object]:
    """Return JSON-serializable metadata for a weight layout spec."""

    metadata: dict[str, object] = {"name": layout.name}
    if isinstance(layout, NunchakuSvdqLayout):
        metadata["outer_scale_splits"] = list(layout.outer_scale_splits)
    if isinstance(layout, AdaNormAwqW4A16Layout):
        metadata["splits"] = layout.splits
    return metadata


def _target_quant_metadata_value(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    return value


@dataclass(frozen=True)
class SvdqTargetQuant:
    """Target-level SVDQuant behavior and runtime layout.

    Args:
        precision: Optional residual weight precision override.
        group_size: Optional residual quantization group-size override.
        rank: Optional low-rank rank override.
        shared_low_rank: Whether grouped modules share one low-rank branch.
        smooth_key: Optional key used to share smoothing ranges across targets.
        smooth: Optional smoothing override.
        activation_quant: Optional activation quantization override.
        shift_activations: Whether this target should calibrate activation shifts.
        weight_layout: SVDQ export layout for this target.
        bias: Bias export policy.
    """

    precision: Literal["int4", "fp4"] | None = None
    group_size: int | None = None
    rank: int | None = None
    shared_low_rank: bool = True
    smooth_key: str | None = None
    smooth: bool | SmoothSpec | None = None
    activation_quant: bool | ActivationQuantSpec | None = None
    shift_activations: bool | None = None
    weight_layout: SvdqWeightLayoutSpec = field(default_factory=NunchakuSvdqLayout)
    bias: Literal["auto", "zero", "omit"] = "auto"

    def __post_init__(self) -> None:
        if self.precision is not None and self.precision not in {"int4", "fp4"}:
            raise ValueError(f"Unsupported target precision override: {self.precision!r}")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("target group_size override must be positive")
        if self.rank is not None and self.rank < 0:
            raise ValueError("target rank override must be non-negative")
        if not isinstance(self.shared_low_rank, bool):
            raise TypeError("target shared_low_rank must be a bool")
        if self.smooth is not None and not isinstance(self.smooth, (bool, SmoothSpec)):
            raise TypeError("target smooth override must be a bool or SmoothSpec")
        if self.activation_quant is not None and not isinstance(self.activation_quant, (bool, ActivationQuantSpec)):
            raise TypeError("target activation_quant override must be a bool or ActivationQuantSpec")
        if self.shift_activations is not None and not isinstance(self.shift_activations, bool):
            raise TypeError("target shift_activations override must be a bool")
        if self.bias not in {"auto", "zero", "omit"}:
            raise ValueError(f"Unsupported target bias policy: {self.bias!r}")
        if not isinstance(self.weight_layout, (SvdqLayout, NaiveSvdqLayout, NunchakuSvdqLayout)):
            raise ValueError("SvdqTargetQuant weight_layout must be an SVDQ layout")


@dataclass(frozen=True)
class AwqTargetQuant:
    """Target-level AWQ behavior and runtime layout.

    This policy is for selected linear modules such as norm/AdaLN modulation
    projections. It runs weight-only AWQ and exports the result into a W4A16
    runtime layout. It resolves to INT4, 64-wide groups, rank 0, no smoothing,
    no activation quantization, no activation shifting, and no shared low-rank
    branch.
    """

    layout: AwqWeightLayoutSpec = field(default_factory=AwqW4A16Layout)
    bias: Literal["auto", "zero", "omit"] = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.layout, (AwqW4A16Layout, AdaNormAwqW4A16Layout)):
            raise ValueError("AwqTargetQuant layout must be an AWQ W4A16 layout")
        if self.bias not in {"auto", "zero", "omit"}:
            raise ValueError(f"Unsupported target bias policy: {self.bias!r}")


TargetQuantSpec = SvdqTargetQuant | AwqTargetQuant


def target_quant_metadata(quant: TargetQuantSpec) -> dict[str, object]:
    """Return JSON-serializable target quantization policy metadata."""

    if isinstance(quant, SvdqTargetQuant):
        return {
            "type": "svdq",
            "precision": quant.precision,
            "group_size": quant.group_size,
            "rank": quant.rank,
            "shared_low_rank": quant.shared_low_rank,
            "smooth_key": quant.smooth_key,
            "smooth": _target_quant_metadata_value(quant.smooth),
            "activation_quant": _target_quant_metadata_value(quant.activation_quant),
            "shift_activations": quant.shift_activations,
            "bias": quant.bias,
            "weight_layout": weight_layout_metadata(quant.weight_layout),
        }
    return {"type": "awq", "bias": quant.bias, "layout": weight_layout_metadata(quant.layout)}


def target_weight_layout(quant: TargetQuantSpec) -> WeightLayoutSpec:
    """Return the runtime weight layout carried by a target quant policy."""

    if isinstance(quant, SvdqTargetQuant):
        return quant.weight_layout
    return quant.layout


def target_bias_policy(quant: TargetQuantSpec) -> Literal["auto", "zero", "omit"]:
    """Return the bias export policy carried by a target quant policy."""

    return quant.bias


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
        strategy: Use fixed, grid-search, or random-search candidates.
        strategy_options: Strategy-specific options. ``random_search`` supports
            ``random_samples``.
        objective: Objective name used to score smoothing candidates.
        alpha: Activation-span exponent for manual smoothing.
        beta: Weight-span exponent for manual smoothing.
        num_grids: Number of alpha/beta grid points per searched span pair.
        spans: Pairs of activation and weight span estimators to search.
        sample_size: Number of calibration rows to score, or ``-1`` for all.
        eps: Positive numerical floor applied to smoothing scales.
    """

    enabled: bool = True
    strategy: Literal["manual", "grid_search", "random_search"] = "grid_search"
    strategy_options: Mapping[str, Any] = field(default_factory=dict)
    objective: Literal["outputs_error"] = "outputs_error"
    alpha: float = 0.5
    beta: float = -2.0
    num_grids: int = 20
    spans: Sequence[tuple[Literal["absmax", "rms"], Literal["absmax", "rms"]]] = (("absmax", "absmax"),)
    sample_size: int = -1
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate smoothing strategy, grid, and numerical bounds."""
        if self.strategy not in {"manual", "grid_search", "random_search"}:
            raise ValueError(f"Unsupported smoothing strategy: {self.strategy!r}")
        if not isinstance(self.strategy_options, Mapping):
            raise TypeError("smooth strategy_options must be a mapping")
        allowed_options = {"random_samples"} if self.strategy == "random_search" else set()
        unknown_options = set(self.strategy_options) - allowed_options
        if unknown_options:
            raise ValueError(f"Unsupported smooth strategy_options for {self.strategy!r}: {sorted(unknown_options)}")
        if self.strategy == "random_search":
            random_samples = self.strategy_options.get("random_samples", 8)
            if not isinstance(random_samples, int) or random_samples <= 0:
                raise ValueError("smooth strategy_options['random_samples'] must be a positive integer")
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
        compute_device: Optional device used for per-target quantization math.
        offload_model: Move the model back to CPU while quantizing each
            captured calibration scope.
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
    compute_device: str | None = None
    offload_model: bool = False

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
        if self.compute_device is not None and not isinstance(self.compute_device, str):
            raise TypeError("compute_device must be a string or None")
        if not isinstance(self.offload_model, bool):
            raise TypeError("offload_model must be a bool")
        if self.offload_model and self.compute_device is None:
            raise ValueError("offload_model requires compute_device")


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
        name: Logical target name template. Defaults to the matched module path
            when omitted.
        modules: Module path patterns. Shared wildcard captures form grouped
            targets.
        module_classes: Optional module classes used to filter matched modules,
            or to select modules when ``modules`` is omitted.
        scope_module_classes: Optional parent module classes that constrain
            class-only scans to descendants of matching scope modules.
        parent_module_classes: Optional parent module classes used with
            ``member_selector`` for callable grouped targets.
        member_selector: Callable that returns ``role -> module`` members for
            one matched parent module. Mapping order defines group order.
        export_name: Optional export name template; defaults to ``name`` or
            the matched module path when ``name`` is omitted.
        kind: Target module kind, currently ``"linear"`` or ``"conv"``.
        roles: Optional semantic roles for grouped modules.
        quant: Target-level quantization behavior and runtime layout policy.
    """

    name: str | None = None
    modules: Sequence[str] = field(default_factory=tuple)
    export_name: str | None = None
    kind: Literal["linear", "conv"] = "linear"
    roles: Sequence[str] = field(default_factory=tuple)
    module_classes: type | Sequence[type] | None = None
    scope_module_classes: type | Sequence[type] | None = None
    parent_module_classes: type | Sequence[type] | None = None
    member_selector: Callable[[Any], Mapping[str, Any]] | None = None
    quant: TargetQuantSpec = field(default_factory=SvdqTargetQuant)

    def __post_init__(self) -> None:
        """Validate target-level quantization overrides."""

        _normalize_module_classes(self, "module_classes", "target module_classes")
        _normalize_module_classes(self, "scope_module_classes", "target scope_module_classes")
        _normalize_module_classes(self, "parent_module_classes", "target parent_module_classes")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("TargetRule name must be a string or None")
        if self.member_selector is not None:
            if not callable(self.member_selector):
                raise TypeError("TargetRule member_selector must be callable")
            if self.parent_module_classes is None:
                raise ValueError("TargetRule member_selector requires parent_module_classes")
            if self.modules:
                raise ValueError("TargetRule member_selector cannot be combined with modules")
            if self.module_classes is not None:
                raise ValueError("TargetRule member_selector cannot be combined with module_classes")
            if self.scope_module_classes is not None:
                raise ValueError("TargetRule member_selector cannot be combined with scope_module_classes")
            if self.roles:
                raise ValueError("TargetRule member_selector uses mapping keys for roles")
        elif self.parent_module_classes is not None:
            raise ValueError("TargetRule parent_module_classes requires member_selector")
        if self.scope_module_classes is not None and self.modules:
            raise ValueError("TargetRule scope_module_classes cannot be combined with modules")
        if not self.modules and self.module_classes is None and self.member_selector is None:
            raise ValueError("TargetRule requires modules, module_classes, or member_selector")
        if not isinstance(self.quant, (SvdqTargetQuant, AwqTargetQuant)):
            raise TypeError("TargetRule quant must be a SvdqTargetQuant or AwqTargetQuant")


@dataclass(frozen=True)
class SkipRule:
    """Describe modules that broad target scans must leave unquantized.

    Args:
        modules: Optional module path patterns to skip.
        module_classes: Optional module classes to skip, optionally constrained
            by ``scope_module_classes``.
        scope_module_classes: Optional parent module classes used to constrain
            class skips to descendants of matching scope modules, or to skip
            matching scope modules directly when ``module_classes`` is omitted.
    """

    modules: Sequence[str] = field(default_factory=tuple)
    module_classes: type | Sequence[type] | None = None
    scope_module_classes: type | Sequence[type] | None = None

    def __post_init__(self) -> None:
        """Validate skip selectors."""

        _normalize_module_classes(self, "module_classes", "skip module_classes")
        _normalize_module_classes(self, "scope_module_classes", "skip scope_module_classes")
        if not self.modules and self.module_classes is None and self.scope_module_classes is None:
            raise ValueError("SkipRule requires modules, module_classes, or scope_module_classes")


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
        module_classes: Optional module classes used to filter matched modules,
            or to select modules when ``modules`` is omitted.
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
            inputs when true. Defaults to true for sequential block stacks;
            set false for independent scopes that cannot consume the previous
            scope output.
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
    use_prev_scope_outputs: bool = True
    recompute: bool = False
    module_classes: type | Sequence[type] | None = None

    def __post_init__(self) -> None:
        """Validate scope naming and matching configuration."""

        if self.name is not None and not isinstance(self.name, str) and not self.modules:
            object.__setattr__(self, "modules", tuple(self.name))
            object.__setattr__(self, "name", None)
        _normalize_module_classes(self, "module_classes", "calibration scope module_classes")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("CalibrationScopeRule name must be a string or None")
        if not self.modules and self.module_classes is None:
            raise ValueError("CalibrationScopeRule requires modules or module_classes")


@dataclass(frozen=True)
class TargetConfig:
    """Model-agnostic configuration for rewrites, targets, and calibration scopes.

    Args:
        targets: Target rules used to discover quantized modules.
        patches: Optional model rewrite rules applied before target discovery.
        calibration_scopes: Optional scope rules for replayed calibration.
        unquantized_patterns: State-dict patterns kept in the exported artifact.
        skips: Optional rules that exclude modules from broad target scans.
    """

    targets: Sequence[TargetRule]
    patches: Sequence[PatchRule] = field(default_factory=tuple)
    calibration_scopes: Sequence[CalibrationScopeRule] = field(default_factory=tuple)
    unquantized_patterns: Sequence[str] = field(default_factory=tuple)
    skips: Sequence[SkipRule] = field(default_factory=tuple)


def _normalize_module_classes(owner: Any, attr_name: str, field_name: str) -> None:
    classes = getattr(owner, attr_name, None)
    if classes is None:
        return
    if isinstance(classes, type):
        normalized = (classes,)
    else:
        try:
            normalized = tuple(classes)
        except TypeError as exc:
            raise TypeError(f"{field_name} must be a class, sequence of classes, or None") from exc
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not all(isinstance(item, type) for item in normalized):
        raise TypeError(f"{field_name} must contain classes")
    object.__setattr__(owner, attr_name, normalized)


@dataclass(frozen=True)
class CalibrationSpec:
    """Configure calibration sample resolution, disk caching, and batching.

    Args:
        samples: Explicit forward samples as dictionaries.
        prompts: Prompt strings, a prompt file, or prompt sequence converted to
            samples.
        num_samples: Optional limit after sample/prompt resolution. ``-1``
            keeps all samples.
        cache_num_samples: Optional limit after disk-backed cache record
            selection. ``-1`` keeps all cache records.
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
        max_rows_per_target: Optional maximum flattened activation rows
            retained per target cache. ``None`` keeps all captured rows.
        scope_capture_mode: Capture all targets in each scope together, or
            replay each scope once per target to lower peak RAM.
        sample_size: Optional sample partition limit, or ``-1`` for all.
        sample_batch_size: Optional sample partition batch size.
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
    cache_num_samples: int | None = None
    batch_size: int = 1
    cache_dir: str | Path | None = None
    cache_mode: Literal["reuse", "refresh", "disabled"] = "reuse"
    seed: int | None = 0
    forward_fn: Callable[[dict[str, Any]], Any] | None = None
    output_dir: str | Path | None = None
    output_save_fn: Callable[[Any, dict[str, Any], Path], None] | None = None
    shared_input_keys: Sequence[str] = field(default_factory=tuple)
    max_rows_per_target: int | None = None
    scope_capture_mode: Literal["all_targets", "one_target"] = "all_targets"
    sample_size: int = -1
    sample_batch_size: int = -1
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    eager_load_samples: bool = False
    ram_usage_limit: float = 0.90
    artifact_cache: QuantizationCacheSpec | None = None

    def __post_init__(self) -> None:
        """Validate calibration cache, batching, and RAM guard settings."""
        if self.cache_mode not in {"reuse", "refresh", "disabled"}:
            raise ValueError(f"Unsupported cache_mode: {self.cache_mode!r}")
        if self.num_samples is not None and self.num_samples < -1:
            raise ValueError("num_samples must be -1 or a non-negative integer")
        if self.cache_num_samples is not None and (self.cache_num_samples == 0 or self.cache_num_samples < -1):
            raise ValueError("cache_num_samples must be -1, a positive integer, or None")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_rows_per_target is not None and self.max_rows_per_target <= 0:
            raise ValueError("max_rows_per_target must be positive or None")
        if self.scope_capture_mode not in {"all_targets", "one_target"}:
            raise ValueError(f"Unsupported scope_capture_mode: {self.scope_capture_mode!r}")
        for name in ("sample_size", "sample_batch_size"):
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
