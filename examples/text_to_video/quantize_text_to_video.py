"""Sketch of a model-agnostic text-to-video target config."""

from diffuse_compressor import TargetConfig, TargetRule


target_config = TargetConfig(
    targets=[
        TargetRule(
            name="spatial_qkv",
            modules=[
                "transformer_blocks.*.attn1.to_q",
                "transformer_blocks.*.attn1.to_k",
                "transformer_blocks.*.attn1.to_v",
            ],
            export_name="transformer_blocks.{0}.attn1.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule("spatial_out", ["transformer_blocks.*.attn1.to_out.0"], "transformer_blocks.{0}.attn1.out_proj"),
        TargetRule("cross_q", ["transformer_blocks.*.attn2.to_q"], "transformer_blocks.{0}.attn2.q_proj"),
        TargetRule(
            name="cross_kv",
            modules=["transformer_blocks.*.attn2.to_k", "transformer_blocks.*.attn2.to_v"],
            export_name="transformer_blocks.{0}.attn2.kv_proj",
            roles=["k", "v"],
        ),
        TargetRule("mlp_fc1", ["transformer_blocks.*.ff.net.0.proj"], "transformer_blocks.{0}.mlp_fc1"),
        TargetRule("mlp_fc2", ["transformer_blocks.*.ff.net.2"], "transformer_blocks.{0}.mlp_fc2"),
    ]
)
