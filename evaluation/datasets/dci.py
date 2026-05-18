from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from examples.upstream_diffusion_svdquant import _hash_str_to_int

from .base import prompt_sample, require_benchmark_dependencies, select_names


DCI_IMAGE_URL = "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.gz"
DCI_PROMPT_URL = "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.yaml"


class DCIDataset(Dataset[dict[str, Any]]):
    """sDCI prompt and target-image dataset matching DeepCompressor selection."""

    sample_set_name = "sDCI"

    def __init__(self, num_samples: int) -> None:
        require_benchmark_dependencies()
        import datasets
        import yaml
        from PIL import Image

        manager = datasets.DownloadManager()
        meta_path = manager.download(DCI_PROMPT_URL)
        image_root = manager.download_and_extract(DCI_IMAGE_URL)
        meta = yaml.safe_load(Path(meta_path).read_text(encoding="utf-8"))
        records = []
        for filename in select_names(meta.keys(), num_samples):
            image_path = Path(image_root) / f"{filename}.jpg"
            with Image.open(image_path) as image:
                target_image = image.convert("RGB")
            records.append(
                {
                    "filename": str(filename),
                    "prompt": str(meta[filename]),
                    "seed": _hash_str_to_int(str(filename)),
                    "target_image": target_image,
                }
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return prompt_sample(self.records[index])
