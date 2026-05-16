from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import CalibrationSpec
from .utils import check_ram, to_cpu, to_device


@dataclass(frozen=True)
class ModuleForwardInput:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ModuleForwardInput":
        return ModuleForwardInput(args=to_device(self.args, device), kwargs=to_device(self.kwargs, device))


class CalibrationCacheDataset(Dataset[ModuleForwardInput]):
    def __init__(self, paths: Sequence[Path], *, eager_load: bool = False) -> None:
        self.paths = list(paths)
        self.items = [_load_cached_forward_input(path) for path in self.paths] if eager_load else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ModuleForwardInput:
        if self.items is not None:
            return self.items[index]
        return _load_cached_forward_input(self.paths[index])


class CalibrationSampleDataset(Dataset[ModuleForwardInput]):
    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ModuleForwardInput:
        return _sample_to_forward_input(self.samples[index])


def has_runnable_calibration(calibration: CalibrationSpec | None) -> bool:
    if calibration is None:
        return False
    if calibration.cache_mode != "disabled" and cache_files(calibration):
        return True
    return bool(calibration.samples is not None or calibration.forward_fn is not None)


@torch.inference_mode()
def prepare_calibration_cache(model: nn.Module, calibration: CalibrationSpec | None) -> list[Path]:
    if calibration is None or calibration.cache_mode == "disabled" or calibration.cache_dir is None:
        return []

    cache_root = Path(calibration.cache_dir) / "caches"
    existing = sorted(cache_root.glob("*.pt"))
    if calibration.cache_mode == "reuse" and existing:
        return existing

    if calibration.cache_mode == "refresh" and cache_root.exists():
        for path in cache_root.glob("*.pt"):
            path.unlink()
    cache_root.mkdir(parents=True, exist_ok=True)

    samples = resolve_samples(calibration)
    if not samples:
        return sorted(cache_root.glob("*.pt"))

    paths: list[Path] = []
    counter = 0

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal counter
        path = cache_root / f"{counter:08d}.pt"
        torch.save({"args": to_cpu(args), "kwargs": to_cpu(kwargs)}, path)
        paths.append(path)
        counter += 1

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for forward_input in iter_calibration_forward_inputs(
            calibration,
            samples=samples,
            batch_size=1,
            drop_last=False,
        ):
            run_forward_input(model, calibration, forward_input)
            check_ram(calibration)
    finally:
        handle.remove()
    return paths


def iter_calibration_forward_inputs(
    calibration: CalibrationSpec,
    *,
    cache_paths: Sequence[Path] | None = None,
    samples: Sequence[dict[str, Any]] | None = None,
    batch_size: int | None = None,
    drop_last: bool | None = None,
) -> Iterator[ModuleForwardInput]:
    if cache_paths is not None:
        dataset: Dataset[ModuleForwardInput] = CalibrationCacheDataset(
            cache_paths,
            eager_load=calibration.eager_load_samples,
        )
    else:
        dataset = CalibrationSampleDataset(samples or ())

    batch_size = calibration.batch_size if batch_size is None else batch_size
    generator = None
    if calibration.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(calibration.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=calibration.shuffle,
        drop_last=calibration.drop_last if drop_last is None else drop_last,
        num_workers=calibration.num_workers,
        collate_fn=_batch_forward_inputs,
        generator=generator,
    )
    yield from loader


def run_forward_input(model: nn.Module, calibration: CalibrationSpec, forward_input: ModuleForwardInput) -> None:
    if calibration.forward_fn is not None:
        sample = dict(forward_input.kwargs)
        if forward_input.args:
            sample["__args__"] = forward_input.args
        calibration.forward_fn(sample)
    else:
        model(*forward_input.args, **forward_input.kwargs)


def cache_files(calibration: CalibrationSpec) -> list[Path]:
    if calibration.cache_dir is None:
        return []
    return sorted((Path(calibration.cache_dir) / "caches").glob("*.pt"))


def resolve_samples(calibration: CalibrationSpec) -> list[dict[str, Any]]:
    if calibration.samples is not None:
        samples = list(calibration.samples)
    elif calibration.forward_fn is not None and calibration.prompts is not None:
        prompts = _resolve_prompts(calibration.prompts)
        samples = [{"prompt": prompt} for prompt in prompts]
    else:
        samples = []

    if calibration.num_samples is not None:
        samples = samples[: calibration.num_samples]
    return samples


def _load_cached_forward_input(path: Path) -> ModuleForwardInput:
    item = torch.load(path, map_location="cpu", weights_only=False)
    return ModuleForwardInput(args=tuple(item.get("args", ())), kwargs=dict(item.get("kwargs", {})))


def _sample_to_forward_input(sample: dict[str, Any]) -> ModuleForwardInput:
    return ModuleForwardInput(kwargs=dict(sample))


def _batch_forward_inputs(inputs: Sequence[ModuleForwardInput]) -> ModuleForwardInput:
    if len(inputs) == 1:
        return inputs[0]
    args = _batch_sequence([item.args for item in inputs])
    kwargs = _batch_mapping([item.kwargs for item in inputs])
    return ModuleForwardInput(args=args, kwargs=kwargs)


def _batch_sequence(values: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    if not values or any(len(value) != len(values[0]) for value in values):
        return tuple(values[0]) if values else ()
    return tuple(_batch_values([value[index] for value in values]) for index in range(len(values[0])))


def _batch_mapping(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        return dict(values[0])
    return {key: _batch_values([value[key] for value in values]) for key in values[0]}


def _batch_values(values: Sequence[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if all(torch.is_tensor(value) and value.shape == first.shape for value in values):
        return torch.cat([value for value in values], dim=0)
    if all(isinstance(value, dict) for value in values):
        return _batch_mapping(values)  # type: ignore[arg-type]
    if all(isinstance(value, tuple) and len(value) == len(first) for value in values):
        return _batch_sequence(values)  # type: ignore[arg-type]
    return list(values)


def _resolve_prompts(prompts: Sequence[str] | str | Path) -> list[str]:
    if isinstance(prompts, Path):
        return _read_prompt_file(prompts)
    if isinstance(prompts, str):
        path = Path(prompts)
        if path.exists():
            return _read_prompt_file(path)
        return [prompts]
    return list(prompts)


def _read_prompt_file(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if line:
            lines.append(line.strip("\"'"))
    return lines
