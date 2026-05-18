from __future__ import annotations

import random
from typing import Any, Iterable

from torch.utils.data import Dataset


class PromptDataset(Dataset[dict[str, Any]]):
    """Simple prompt dataset using DeepCompressor-compatible filenames."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return prompt_sample(record)


def prompt_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": str(record["filename"]),
        "prompt": str(record["prompt"]),
        "seed": int(record["seed"]),
    }


def select_names(names: Iterable[str], num_samples: int) -> list[str]:
    selected = list(names)
    if num_samples > 0:
        random.Random(0).shuffle(selected)
        selected = selected[:num_samples]
    return sorted(selected)


def require_benchmark_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import yaml  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Benchmark loading requires datasets, PyYAML, and Pillow. "
            "Install evaluation dependencies with python -m pip install -e '.[eval]'."
        ) from exc
