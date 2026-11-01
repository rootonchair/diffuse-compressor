"""Verify the branch-skipping boundary against original root replay."""
from dataclasses import replace
from pathlib import Path
import torch
import pytest
from functools import partial

from diffuse_compressor import CalibrationSpec
from diffuse_compressor.calibration import iter_calibration_scopes
from diffuse_compressor.targets import collect_quant_targets
from examples.text_to_image.quantize_hf import scan_linear_targets
from examples.text_to_image.quantize_moss_image_sft import (MODEL_ID, REVISION, compatible_source, load_moss_module,
    moss_target_config, moss_spec, resident_replay_forward)


def moss_compat_pipeline(directory):
    """Patched copy of the pinned upstream pipeline (a small code file, fetched or cached)."""
    try:
        from huggingface_hub import hf_hub_download
        upstream=Path(hf_hub_download(MODEL_ID,'pipeline_moss_image.py',revision=REVISION))
    except Exception as error:  # offline without a cached copy
        pytest.skip(f'MOSS pipeline source unavailable: {error}')
    path=directory/'pipeline_moss_image.py'
    path.write_text(compatible_source(upstream.read_text()))
    return path


@pytest.mark.parametrize('device',('cpu','cuda'))
def test_moss_chaining_matches_root_inputs_and_preserves_both_branches_after_skip(device, tmp_path):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    module=load_moss_module(moss_compat_pipeline(tmp_path),name='moss_replay_unit_fixture')
    torch.set_num_threads(4)
    torch.manual_seed(123)
    model=module.MOSSImageTransformer(dim=128,ffn_dim=256,num_heads=8,num_layers=26,
        text_dim=256,freq_dim=32,latent_channels=16).bfloat16().eval().to(device)
    samples=[]
    for t in (1000.,500.):
        latent=torch.randn(1,16,2,2)
        for skipped in ((),(22,)):
            samples.append({'hidden_states':latent,'timestep':torch.tensor(t),
                'encoder_hidden_states':torch.randn(1,5 if not skipped else 3,256).bfloat16(),
                'encoder_attention_mask':torch.ones(1,5 if not skipped else 3,dtype=torch.bool),
                'skip_layers':skipped})
    generic=scan_linear_targets(model,precision='nvfp4',rank=32).target_config
    chained,_=moss_target_config(model,precision='nvfp4',rank=32)
    # One probe per block keeps the test small while checking every block input.
    rules=[r for r in generic.targets if r.modules[0].endswith('.qkv') or r.modules[0].endswith('.proj.q')]
    generic=replace(generic,targets=rules)
    chained=replace(chained,targets=rules)
    calibration=CalibrationSpec(samples=samples,cache_mode='disabled',batch_size=1,
        max_rows_per_target=512,max_score_records=8,shuffle=True,
        forward_fn=partial(resident_replay_forward,model) if device=='cuda' else None)
    targets=collect_quant_targets(model,generic,spec=moss_spec('nvfp4'))
    expected={}
    for batch in iter_calibration_scopes(model,targets,generic,calibration):
        expected[batch.scope.name]=({k:v.clone() for k,v in batch.inputs.items()},len(batch.eval_replays),[r.output.clone() for r in batch.eval_replays])
    observed={}
    for batch in iter_calibration_scopes(model,targets,chained,calibration):
        observed[batch.scope.name]=({k:v.clone() for k,v in batch.inputs.items()},len(batch.eval_replays),[r.output.clone() for r in batch.eval_replays])
    assert expected.keys()==observed.keys() and len(expected)==28
    assert list(observed)==['text_embedder.blocks.0','text_embedder.blocks.1']+[f'layers.{i}' for i in range(26)]
    for name,(inputs,calls,outputs) in expected.items():
        actual,actual_calls,actual_outputs=observed[name]
        assert calls==actual_calls,name
        for key,value in inputs.items():
            assert torch.equal(value,actual[key]),(name,key,(value-actual[key]).abs().max())
        assert all(torch.equal(a,b) for a,b in zip(outputs,actual_outputs)),name
    assert observed['layers.22'][1]==2
    assert observed['layers.23'][1]==4
    assert observed['layers.24'][1]==4
