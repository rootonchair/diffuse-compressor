from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import CalibrationSpec
from .utils import check_ram, to_cpu, to_device


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleForwardInput:
    """Positional and keyword arguments for replaying one module/model forward.

    Args:
        args: Positional arguments for the forward call.
        kwargs: Keyword arguments for the forward call.
    """

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ModuleForwardInput":
        """Move all tensor values in this forward input to a device.

        Args:
            device: Destination torch device.

        Returns:
            New forward input with tensors moved to ``device``.
        """

        return ModuleForwardInput(
            args=to_device(self.args, device), kwargs=to_device(self.kwargs, device)
        )


class CalibrationCacheDataset(Dataset[ModuleForwardInput]):
    """Dataset backed by serialized root forward-input cache files.

    Args:
        paths: Cache files created by :func:`prepare_calibration_cache`.
        eager_load: Load all records during construction instead of on demand.
    """

    def __init__(self, paths: Sequence[Path], *, eager_load: bool = False) -> None:
        """Initialize the cache dataset."""

        self.paths = list(paths)
        self.items = (
            [_load_cached_forward_input(path) for path in self.paths]
            if eager_load
            else None
        )

    def __len__(self) -> int:
        """Return the number of cached forward records."""

        return len(self.paths)

    def __getitem__(self, index: int) -> ModuleForwardInput:
        """Load one cached forward record.

        Args:
            index: Dataset index.

        Returns:
            Forward input loaded from memory or disk.
        """

        if self.items is not None:
            return self.items[index]
        return _load_cached_forward_input(self.paths[index])


class CalibrationSampleDataset(Dataset[ModuleForwardInput]):
    """Dataset backed by in-memory calibration sample dictionaries.

    Args:
        samples: Sequence of sample dictionaries used as model kwargs.
    """

    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        """Initialize the sample dataset."""

        self.samples = list(samples)

    def __len__(self) -> int:
        """Return the number of calibration samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> ModuleForwardInput:
        """Convert one sample dictionary to a forward input.

        Args:
            index: Dataset index.

        Returns:
            Forward input using the sample as keyword arguments.
        """

        return _sample_to_forward_input(self.samples[index])


def has_runnable_calibration(calibration: CalibrationSpec | None) -> bool:
    """Return whether calibration has data or cached inputs to replay.

    Args:
        calibration: Optional calibration configuration.

    Returns:
        ``True`` when cache files, explicit samples, or a custom forward
        function are available.
    """

    if calibration is None:
        return False
    if calibration.cache_mode != "disabled" and select_calibration_cache_files(
        cache_files(calibration), calibration
    ):
        return True
    return bool(calibration.samples is not None or calibration.forward_fn is not None)


@torch.inference_mode()
def prepare_calibration_cache(
    model: nn.Module, calibration: CalibrationSpec | None
) -> list[Path]:
    """Create or reuse disk-backed root forward-input caches.

    Args:
        model: Model whose root forward inputs should be captured.
        calibration: Calibration settings controlling samples and cache mode.

    Returns:
        Sorted cache file paths available for replay.
    """

    if (
        calibration is None
        or calibration.cache_mode == "disabled"
        or calibration.cache_dir is None
    ):
        logger.info("- Calibration input cache disabled")
        return []

    cache_root = Path(calibration.cache_dir) / "caches"
    existing = sorted(cache_root.glob("*.pt"))
    if calibration.cache_mode == "reuse" and existing:
        selected = select_calibration_cache_files(existing, calibration)
        if len(selected) == len(existing):
            logger.info(
                "- Reusing %d cached calibration inputs from %s",
                len(existing),
                cache_root,
            )
        else:
            logger.info(
                "- Reusing %d/%d cached calibration inputs from %s",
                len(selected),
                len(existing),
                cache_root,
            )
        return selected

    if calibration.cache_mode == "refresh" and cache_root.exists():
        logger.info("- Refreshing calibration input cache at %s", cache_root)
        for path in cache_root.glob("*.pt"):
            path.unlink()
    cache_root.mkdir(parents=True, exist_ok=True)

    samples = resolve_samples(calibration)
    if not samples:
        logger.info("- No calibration samples available for cache generation")
        paths = sorted(cache_root.glob("*.pt"))
        return paths

    logger.info(
        "- Generating calibration input cache at %s (%d samples)",
        cache_root,
        len(samples),
    )
    paths: list[Path] = []
    counter = 0

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Persist one model forward pre-hook argument set.

        Args:
            _module: Hooked model module, unused.
            args: Positional forward arguments.
            kwargs: Keyword forward arguments.
        """

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
    logger.info("- Saved %d calibration input cache records", len(paths))
    selected = select_calibration_cache_files(paths, calibration)
    if len(selected) != len(paths):
        logger.info(
            "- Selected %d/%d saved calibration input cache records",
            len(selected),
            len(paths),
        )
    return selected


def iter_calibration_forward_inputs(
    calibration: CalibrationSpec,
    *,
    cache_paths: Sequence[Path] | None = None,
    samples: Sequence[dict[str, Any]] | None = None,
    batch_size: int | None = None,
    drop_last: bool | None = None,
) -> Iterator[ModuleForwardInput]:
    """Iterate calibration inputs from cache files or sample dictionaries.

    Args:
        calibration: Calibration settings for batching, shuffling, and loading.
        cache_paths: Optional cache files to read.
        samples: Optional in-memory samples used when cache paths are absent.
        batch_size: Optional batch-size override.
        drop_last: Optional ``drop_last`` override.

    Yields:
        Batched model/module forward inputs.
    """

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
        collate_fn=partial(
            _batch_forward_inputs,
            shared_input_keys=frozenset(calibration.shared_input_keys),
        ),
        generator=generator,
    )
    yield from loader


def run_forward_input(
    model: nn.Module, calibration: CalibrationSpec, forward_input: ModuleForwardInput
) -> None:
    """Execute one calibration forward input.

    Args:
        model: Model to call when no custom forward function is configured.
        calibration: Calibration settings containing an optional ``forward_fn``.
        forward_input: Arguments to replay.
    """

    if calibration.forward_fn is not None:
        sample = dict(forward_input.kwargs)
        if forward_input.args:
            sample["__args__"] = forward_input.args
        result = calibration.forward_fn(sample)
        if (
            calibration.output_dir is not None
            and calibration.output_save_fn is not None
        ):
            calibration.output_save_fn(result, sample, Path(calibration.output_dir))
    else:
        model(*forward_input.args, **forward_input.kwargs)


def cache_files(calibration: CalibrationSpec) -> list[Path]:
    """List existing root forward-input cache files.

    Args:
        calibration: Calibration settings containing ``cache_dir``.

    Returns:
        Sorted cache file paths, or an empty list when no cache directory is
        configured.
    """

    if calibration.cache_dir is None:
        return []
    return sorted((Path(calibration.cache_dir) / "caches").glob("*.pt"))


def select_calibration_cache_files(
    paths: Sequence[Path], calibration: CalibrationSpec
) -> list[Path]:
    """Select cache records according to ``cache_num_samples``.

    Args:
        paths: Candidate cache files.
        calibration: Calibration settings containing ``cache_num_samples`` and
            ``seed``.

    Returns:
        Sorted selected cache paths.
    """

    paths = sorted(paths)
    cache_num_samples = calibration.cache_num_samples
    if (
        cache_num_samples is None
        or cache_num_samples < 0
        or cache_num_samples >= len(paths)
    ):
        return paths
    selected = list(paths)
    random.Random(calibration.seed).shuffle(selected)
    return sorted(selected[:cache_num_samples])


def resolve_samples(calibration: CalibrationSpec) -> list[dict[str, Any]]:
    """Resolve configured samples or prompts into calibration sample dicts.

    Args:
        calibration: Calibration settings with explicit samples or prompts.

    Returns:
        Sample dictionaries, optionally truncated by ``num_samples``.
    """

    if calibration.samples is not None:
        samples = list(calibration.samples)
    elif calibration.forward_fn is not None and calibration.prompts is not None:
        prompts = _resolve_prompts(calibration.prompts)
        samples = [{"prompt": prompt} for prompt in prompts]
    else:
        samples = []

    if calibration.num_samples is not None and calibration.num_samples >= 0:
        samples = samples[: calibration.num_samples]
    return samples


def _load_cached_forward_input(path: Path) -> ModuleForwardInput:
    """Load a serialized forward input from disk.

    Args:
        path: Cache file path.

    Returns:
        Forward input stored in the cache file.
    """

    item = torch.load(path, map_location="cpu", weights_only=False)
    return ModuleForwardInput(
        args=tuple(item.get("args", ())), kwargs=dict(item.get("kwargs", {}))
    )


def _sample_to_forward_input(sample: dict[str, Any]) -> ModuleForwardInput:
    """Convert one sample dictionary to a forward input.

    Args:
        sample: Calibration sample dictionary.

    Returns:
        Forward input using ``sample`` as keyword arguments.
    """

    return ModuleForwardInput(kwargs=dict(sample))


def _batch_forward_inputs(
    inputs: Sequence[ModuleForwardInput],
    *,
    shared_input_keys: frozenset[str] = frozenset(),
) -> ModuleForwardInput:
    """Collate multiple forward inputs for a DataLoader batch.

    Args:
        inputs: Forward inputs to collate.

    Returns:
        A single batched forward input.
    """

    if len(inputs) == 1:
        return inputs[0]
    args = _batch_sequence(
        [item.args for item in inputs], shared_input_keys=shared_input_keys
    )
    kwargs = _batch_mapping(
        [item.kwargs for item in inputs], shared_input_keys=shared_input_keys
    )
    return ModuleForwardInput(args=args, kwargs=kwargs)


def _batch_sequence(
    values: Sequence[tuple[Any, ...]],
    *,
    path: tuple[str, ...] = (),
    shared_input_keys: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    """Batch positional argument tuples elementwise.

    Args:
        values: Sequence of positional argument tuples.

    Returns:
        Tuple of batched values, or the first tuple when structures differ.
    """

    if not values or any(len(value) != len(values[0]) for value in values):
        return tuple(values[0]) if values else ()
    return tuple(
        _batch_values(
            [value[index] for value in values],
            path=(*path, str(index)),
            shared_input_keys=shared_input_keys,
        )
        for index in range(len(values[0]))
    )


def _batch_mapping(
    values: Sequence[dict[str, Any]],
    *,
    path: tuple[str, ...] = (),
    shared_input_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Batch dictionaries with matching keys.

    Args:
        values: Sequence of dictionaries.

    Returns:
        Dictionary of batched values, or the first dictionary when keys differ.
    """

    if not values:
        return {}
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        return dict(values[0])
    return {
        key: _batch_values(
            [value[key] for value in values],
            path=(*path, str(key)),
            shared_input_keys=shared_input_keys,
        )
        for key in values[0]
    }


def _batch_values(
    values: Sequence[Any],
    *,
    path: tuple[str, ...] = (),
    shared_input_keys: frozenset[str] = frozenset(),
) -> Any:
    """Batch homogeneous tensor or nested values.

    Args:
        values: Values at the same structural position across samples.

    Returns:
        Concatenated tensors, recursively batched structures, or a list of raw
        values for unsupported structures.
    """

    if not values:
        return None
    first = values[0]
    if all(value is None for value in values):
        return None
    if all(torch.is_tensor(value) and value.shape == first.shape for value in values):
        if path and path[-1] in shared_input_keys:
            if not all(torch.equal(value, first) for value in values[1:]):
                dotted = ".".join(path)
                raise ValueError(
                    f"Cannot batch inconsistent shared input tensor {dotted}"
                )
            return first
        return torch.cat([value for value in values], dim=0)
    if all(isinstance(value, dict) for value in values):
        return _batch_mapping(values, path=path, shared_input_keys=shared_input_keys)  # type: ignore[arg-type]
    if all(isinstance(value, tuple) and len(value) == len(first) for value in values):
        return _batch_sequence(values, path=path, shared_input_keys=shared_input_keys)  # type: ignore[arg-type]
    return list(values)


def _resolve_prompts(prompts: Sequence[str] | str | Path) -> list[str]:
    """Resolve prompt configuration into prompt strings.

    Args:
        prompts: Prompt sequence, prompt string, or path to a prompt file.

    Returns:
        Prompt strings.
    """

    if isinstance(prompts, Path):
        return _read_prompt_file(prompts)
    if isinstance(prompts, str):
        path = Path(prompts)
        if path.exists():
            return _read_prompt_file(path)
        return [prompts]
    return list(prompts)


def _read_prompt_file(path: Path) -> list[str]:
    """Read prompts from a plain-text or simple YAML-list file.

    Args:
        path: File containing one prompt per line.

    Returns:
        Non-empty, non-comment prompt lines with simple list markers stripped.
    """

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
