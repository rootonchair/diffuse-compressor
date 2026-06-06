from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from .base import require_benchmark_dependencies


def _hash_str_to_int(value: str) -> int:
    modulus = 10**9 + 7
    hash_int = 0
    for char in value:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def _resize_image_edit_image(image: object, image_size: int) -> object:
    if image_size <= 0 or not hasattr(image, "size") or not hasattr(image, "resize"):
        return image
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = (
        image.crop((left, top, left + crop, top + crop))
        if hasattr(image, "crop")
        else image
    )
    image = image.resize((image_size, image_size))
    return image.convert("RGB") if hasattr(image, "convert") else image


class LongCatImageEditDataset(Dataset[dict[str, Any]]):
    """LongCat image-edit source/target dataset for held-out evaluation."""

    sample_set_name = "NHR-Edit-Change_Only"

    def __init__(
        self,
        num_samples: int,
        *,
        dataset: str = "VyoJ/NHR-Edit-Change_Only",
        split: str = "test",
        image_size: int = 512,
    ) -> None:
        require_benchmark_dependencies()
        import datasets

        loaded = datasets.load_dataset(dataset, split=split)
        limit = len(loaded) if num_samples < 0 else min(num_samples, len(loaded))
        records = []
        for index in range(limit):
            row = loaded[index]
            filename = str(row.get("filename") or row.get("sample_id") or index)
            prompt = str(row.get("prompt") or row.get("edit_instruction"))
            source = row.get("source_image") or row.get("source")
            target = row.get("target_image") or row.get("edited")
            if source is None:
                raise ValueError(
                    f"Image-edit sample {filename!r} does not contain source_image or source"
                )
            record: dict[str, Any] = {
                "filename": filename,
                "prompt": prompt,
                "seed": _hash_str_to_int(filename),
                "image": _resize_image_edit_image(source, image_size),
            }
            if target is not None:
                record["target_image"] = _resize_image_edit_image(target, image_size)
            records.append(record)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "filename": str(record["filename"]),
            "prompt": str(record["prompt"]),
            "seed": int(record["seed"]),
            "image": record["image"],
        }
