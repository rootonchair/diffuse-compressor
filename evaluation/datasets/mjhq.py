from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .base import prompt_sample, require_benchmark_dependencies, select_names


MJHQ_IMAGE_URL = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/mjhq30k_imgs.zip"
MJHQ_META_URL = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json"


def _hash_str_to_int(value: str) -> int:
    modulus = 10**9 + 7
    hash_int = 0
    for char in value:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


class MJHQDataset(Dataset[dict[str, Any]]):
    """MJHQ prompt and target-image dataset matching DeepCompressor selection."""

    sample_set_name = "MJHQ"

    def __init__(self, num_samples: int) -> None:
        require_benchmark_dependencies()
        import datasets
        from PIL import Image

        manager = datasets.DownloadManager()
        meta_path = manager.download(MJHQ_META_URL)
        image_root = manager.download_and_extract(MJHQ_IMAGE_URL)
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        records = []
        for filename in select_names(meta.keys(), num_samples):
            item = meta[filename]
            image_path = Path(image_root) / str(item["category"]) / f"{filename}.jpg"
            with Image.open(image_path) as image:
                target_image = image.convert("RGB")
            records.append(
                {
                    "filename": str(filename),
                    "prompt": str(item["prompt"]),
                    "seed": _hash_str_to_int(str(filename)),
                    "target_image": target_image,
                }
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return prompt_sample(self.records[index])
