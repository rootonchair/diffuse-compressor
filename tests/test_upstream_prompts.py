from types import SimpleNamespace

import pytest

from examples.upstream_diffusion_svdquant import batched_samples, save_diffusers_images, standard_prompt_records, svdquant_spec


def test_standard_prompt_records_match_upstream_qdiff_selection(tmp_path):
    prompt_file = tmp_path / "qdiff.yaml"
    prompt_file.write_text(
        "\n".join(
            f"'{index:04d}': prompt {index}"
            for index in range(8)
        ),
        encoding="utf-8",
    )

    records = standard_prompt_records(3, prompt_file=prompt_file)

    assert [record["filename"] for record in records] == ["0001-0", "0004-0", "0005-0"]
    assert [record["prompt"] for record in records] == ["prompt 1", "prompt 4", "prompt 5"]
    assert [record["seed"] for record in records] == [420006749, 420009632, 420010593]


def test_batched_samples_preserve_upstream_prompt_seeds():
    samples = batched_samples(
        [
            {"filename": "0001-0", "prompt": "prompt 1", "seed": 420006749},
            {"filename": "0004-0", "prompt": "prompt 4", "seed": 420009632},
        ],
        batch_size=2,
    )

    assert samples == [
        {
            "filename": ["0001-0", "0004-0"],
            "prompt": ["prompt 1", "prompt 4"],
            "seed": [420006749, 420009632],
        }
    ]


def test_batched_samples_add_synthetic_filenames_for_plain_prompts():
    samples = batched_samples(["prompt 0", "prompt 1", "prompt 2"], batch_size=2)

    assert samples == [
        {"filename": ["0000-0", "0001-0"], "prompt": ["prompt 0", "prompt 1"], "seed": [0, 1]},
        {"filename": "0002-0", "prompt": "prompt 2", "seed": 2},
    ]


def test_svdquant_spec_accepts_svd_lowrank_backend():
    spec = svdquant_spec("int4", svd_backend="svd_lowrank", svd_lowrank_oversample=12, svd_lowrank_niter=3)

    assert spec.low_rank_solver.svd_backend == "svd_lowrank"
    assert spec.low_rank_solver.svd_lowrank_oversample == 12
    assert spec.low_rank_solver.svd_lowrank_niter == 3


def test_save_diffusers_images_uses_calibration_filenames(tmp_path):
    class FakeImage:
        def __init__(self):
            self.path = None

        def save(self, path):
            self.path = path
            path.write_text("image", encoding="utf-8")

    images = [FakeImage(), FakeImage()]
    result = SimpleNamespace(images=images)

    save_diffusers_images(result, {"filename": ["0001-0", "0004-0"]}, tmp_path / "samples")

    assert (tmp_path / "samples" / "0001-0.png").read_text(encoding="utf-8") == "image"
    assert (tmp_path / "samples" / "0004-0.png").read_text(encoding="utf-8") == "image"
    assert images[0].path == tmp_path / "samples" / "0001-0.png"


def test_save_diffusers_images_validates_image_count(tmp_path):
    result = SimpleNamespace(images=[])

    with pytest.raises(ValueError, match="Expected 1 image filenames"):
        save_diffusers_images(result, {"filename": "0001-0"}, tmp_path)
