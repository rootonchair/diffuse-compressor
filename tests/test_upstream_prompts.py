from examples.upstream_diffusion_svdquant import batched_samples, standard_prompt_records


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

    assert samples == [{"prompt": ["prompt 1", "prompt 4"], "seed": [420006749, 420009632]}]
