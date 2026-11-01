import json
from pathlib import Path

import pytest

from examples.text_to_image.quantize_moss_image_sft import validate_calibration_recipe
from examples.text_to_image.utils import standard_prompt_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("prompts", [8, 32])
def test_calibration_count_scales_with_prompts_and_requires_complete_cfg_pairs(prompts):
    class Samples:
        metadata = {"prompts": prompts, "steps": 50, "profile": {"steps": 50, "guidance_scale": 3.0}}
        count = prompts * 100
        def __len__(self):
            return self.count
    samples = Samples()
    recipe = {"calibration_prompts": [{}] * prompts, "profile": samples.metadata["profile"].copy(),
              "calibration_calls": prompts * 100}
    assert validate_calibration_recipe(samples, recipe) == prompts * 100
    samples.count -= 1
    with pytest.raises(ValueError, match="CFG calls"):
        validate_calibration_recipe(samples, recipe)
    samples.count += 1
    samples.metadata = {**samples.metadata, "profile": {"steps": 50, "guidance_scale": 1.0}}
    with pytest.raises(ValueError, match="sampling profile"):
        validate_calibration_recipe(samples, recipe)


def test_32_prompt_set_extends_original_qdiff_set_with_original_seeds():
    old = json.loads((ROOT / "examples/prompts/moss_sft_calibration8.json").read_text())["records"]
    new = json.loads((ROOT / "examples/prompts/moss_sft_calibration32.json").read_text())["records"]
    assert len(new) == len({r["prompt"] for r in new}) == 32
    qdiff = [{key: record[key] for key in ("filename", "prompt", "seed")} for record in standard_prompt_records(64)]
    assert new[:8] == old and new == qdiff[:32]
