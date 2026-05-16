from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence


@dataclass(frozen=True)
class LowRankSolverSpec:
    mode: Literal["weighted_svd", "search"] = "weighted_svd"
    num_iters: int = 1
    early_stop: bool = False
    compensate: bool = False
    activation_quant: bool = False
    objective: Literal["outputs_error"] = "outputs_error"
    sample_size: int = -1
    eval_replay: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"weighted_svd", "search"}:
            raise ValueError(f"Unsupported low-rank solver mode: {self.mode!r}")
        if self.num_iters <= 0:
            raise ValueError("low-rank solver num_iters must be positive")
        if self.objective != "outputs_error":
            raise ValueError(f"Unsupported low-rank solver objective: {self.objective!r}")
        if self.sample_size == 0 or self.sample_size < -1:
            raise ValueError("low-rank solver sample_size must be -1 or a positive integer")


@dataclass(frozen=True)
class SmoothSpec:
    enabled: bool = True
    strategy: Literal["manual", "grid_search"] = "grid_search"
    alpha: float = 0.5
    beta: float = -2.0
    num_grids: int = 20
    spans: Sequence[tuple[Literal["absmax", "rms"], Literal["absmax", "rms"]]] = (("absmax", "absmax"),)
    sample_size: int = -1
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.strategy not in {"manual", "grid_search"}:
            raise ValueError(f"Unsupported smoothing strategy: {self.strategy!r}")
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
class DiffusionQuantSpec:
    method: Literal["svdquant"] = "svdquant"
    precision: Literal["int4", "fp4"] = "int4"
    rank: int = 32
    group_size: int = 64
    low_rank_solver: LowRankSolverSpec = field(default_factory=LowRankSolverSpec)
    smooth: bool | SmoothSpec = True
    shift_activations: bool = True
    torch_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.method != "svdquant":
            raise ValueError(f"Unsupported quantization method: {self.method!r}")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if not isinstance(self.low_rank_solver, LowRankSolverSpec):
            raise TypeError("low_rank_solver must be a LowRankSolverSpec")
        if not isinstance(self.smooth, (bool, SmoothSpec)):
            raise TypeError("smooth must be a bool or SmoothSpec")


@dataclass(frozen=True)
class PatchRule:
    type: Literal["split_linear", "split_linear_output", "split_conv", "shift_linear", "shift_conv"]
    module: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRule:
    name: str
    modules: Sequence[str]
    export_name: str | None = None
    kind: Literal["linear", "conv"] = "linear"
    roles: Sequence[str] = field(default_factory=tuple)
    shared_low_rank: bool = True
    smooth_key: str | None = None


@dataclass(frozen=True)
class CalibrationCaptureRule:
    name: str
    modules: Sequence[str]
    inputs: bool = True
    outputs: bool = False

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("CalibrationCaptureRule modules must not be empty")
        if not self.inputs and not self.outputs:
            raise ValueError("CalibrationCaptureRule must capture inputs, outputs, or both")


@dataclass(frozen=True)
class CalibrationScopeRule:
    name: str
    modules: Sequence[str]
    eval_module: str | None = None
    replay_module: str | None = None
    capture_modules: Sequence[CalibrationCaptureRule] = field(default_factory=tuple)
    use_prev_scope_outputs: bool = False
    recompute: bool = False


@dataclass(frozen=True)
class TargetConfig:
    targets: Sequence[TargetRule]
    patches: Sequence[PatchRule] = field(default_factory=tuple)
    calibration_scopes: Sequence[CalibrationScopeRule] = field(default_factory=tuple)
    unquantized_patterns: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class CalibrationSpec:
    samples: Sequence[dict[str, Any]] | None = None
    prompts: Sequence[str] | str | Path | None = None
    num_samples: int | None = None
    batch_size: int = 1
    cache_dir: str | Path | None = None
    cache_mode: Literal["reuse", "refresh", "disabled"] = "reuse"
    seed: int | None = None
    forward_fn: Callable[[dict[str, Any]], Any] | None = None
    max_rows_per_target: int = 4096
    sample_size: int = -1
    sample_batch_size: int = -1
    element_size: int = -1
    element_batch_size: int = -1
    ram_usage_limit: float = 0.90

    def __post_init__(self) -> None:
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
        if not 0 < self.ram_usage_limit <= 1:
            raise ValueError("ram_usage_limit must be in (0, 1]")


@dataclass(frozen=True)
class ExportSpec:
    target: Literal["nunchaku"] = "nunchaku"
    output: str | Path = "quantized.safetensors"
    checkpoint_format: Literal["single_safetensors"] = "single_safetensors"

    def __post_init__(self) -> None:
        if self.target != "nunchaku":
            raise ValueError(f"Unsupported export target: {self.target!r}")
        if self.checkpoint_format != "single_safetensors":
            raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_format!r}")
