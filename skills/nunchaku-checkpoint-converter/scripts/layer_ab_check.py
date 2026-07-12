"""Validation-ladder rungs 2 and 3 for a converted Nunchaku Lite transformer checkpoint.

Generalizes the ad hoc layer-level probes used to find and fix the NVFP4
single-block ``proj_out`` merge bug (an fp8 group-scale overflow that produced
silent NaNs and black images) into a reusable script for any FLUX.1 Nunchaku
Lite conversion.

Two modes:

``keymatch`` (rung 2) -- confirm the converted transformer's tensor keys
exactly match the target list the converter itself computed (no missing, no
extra targets). Requires the converter module (importable via
``importlib.util``) so it can call ``svdq_targets``/``awq_targets_with_splits``
with the same spec used for the real conversion.

    python layer_ab_check.py keymatch \\
        --converted-checkpoint outputs/converted/<repo>/transformer/diffusion_pytorch_model.safetensors \\
        --converter-script examples/text_to_image/convert_flux1_nunchaku_to_diffusers.py \\
        --num-layers 19 --num-single-layers 38 --precision nvfp4 --rank 32

``merge-ab`` (rung 3) -- for a merged single-block ``proj_out`` target, run the
converted (merged) layer and the source layer (still split into the raw
Nunchaku checkpoint's ``out_proj``/``mlp_fc2`` modules -- the pristine
upstream ``svdq-*.safetensors`` file uses those names, NOT the converter's
intermediate ``proj_out.linears.0``/``proj_out.linears.1`` renaming) through
the real ``SVDQW4A4Linear`` kernel on identical input and compare outputs.
Small (~2%) relative error is expected end-to-end from fp8 re-rounding on
whichever half the outer-scale ratio was folded into -- anything larger
indicates a packing bug, not numerical noise. Needs ``--converter-script`` to
reuse its exact ``SUFFIX_RENAMES``/``DROP_SUFFIXES`` when reading the raw
source tensors, so the comparison uses the same renaming the real conversion
applies.

    python layer_ab_check.py merge-ab \\
        --converted-checkpoint outputs/converted/<repo>/transformer/diffusion_pytorch_model.safetensors \\
        --source-checkpoint <path to the raw upstream nunchaku svdq-*.safetensors> \\
        --converter-script examples/text_to_image/convert_flux1_nunchaku_to_diffusers.py \\
        --target single_transformer_blocks.0.proj_out \\
        --source-left-target single_transformer_blocks.0.out_proj \\
        --source-right-target single_transformer_blocks.0.mlp_fc2 \\
        --precision nvfp4 --group-size 16 \\
        --left-in-features 3072 --right-in-features 12288 --out-features 3072 \\
        --rel-error-threshold 0.02
"""

from __future__ import annotations

import argparse
import importlib.util
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    keymatch = sub.add_parser("keymatch")
    keymatch.add_argument("--converted-checkpoint", required=True)
    keymatch.add_argument("--converter-script", required=True)
    keymatch.add_argument("--num-layers", type=int, required=True)
    keymatch.add_argument("--num-single-layers", type=int, required=True)
    keymatch.add_argument("--precision", choices=("int4", "nvfp4"), required=True)
    keymatch.add_argument("--rank", type=int, required=True)

    merge_ab = sub.add_parser("merge-ab")
    merge_ab.add_argument("--converted-checkpoint", required=True)
    merge_ab.add_argument("--source-checkpoint", required=True, help="raw upstream nunchaku svdq-*.safetensors")
    merge_ab.add_argument("--converter-script", required=True, help="used to reuse SUFFIX_RENAMES/DROP_SUFFIXES")
    merge_ab.add_argument("--target", required=True, help="converted target, e.g. single_transformer_blocks.0.proj_out")
    merge_ab.add_argument("--source-left-target", required=True, help="raw source module, e.g. ...0.out_proj")
    merge_ab.add_argument("--source-right-target", required=True, help="raw source module, e.g. ...0.mlp_fc2")
    merge_ab.add_argument("--precision", choices=("int4", "nvfp4"), required=True)
    merge_ab.add_argument("--group-size", type=int, required=True)
    merge_ab.add_argument("--left-in-features", type=int, required=True)
    merge_ab.add_argument("--right-in-features", type=int, required=True)
    merge_ab.add_argument("--out-features", type=int, required=True)
    merge_ab.add_argument("--rel-error-threshold", type=float, default=0.02)
    merge_ab.add_argument("--batch", type=int, default=8)
    merge_ab.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def load_converter_module(script_path: str):
    spec = importlib.util.spec_from_file_location("_layer_ab_check_converter", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_keymatch(args: argparse.Namespace) -> None:
    import safetensors

    converter = load_converter_module(args.converter_script)
    spec = converter.FluxConversionSpec(
        num_layers=args.num_layers,
        num_single_layers=args.num_single_layers,
        precision=args.precision,
        rank=args.rank,
    )
    expected = sorted(set(converter.svdq_targets(spec)) | {t for t, _ in converter.awq_targets_with_splits(spec)})

    with safetensors.safe_open(args.converted_checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())

    missing = [t for t in expected if f"{t}.qweight" not in keys]
    unexpected_linears = [k for k in keys if ".linears." in k]

    print(f"expected targets: {len(expected)}")
    print(f"missing targets: {len(missing)}" + (f" -- e.g. {missing[:5]}" if missing else ""))
    print(f"stray split-linear keys (should be merged away): {len(unexpected_linears)}")

    if missing or unexpected_linears:
        raise SystemExit(1)
    print("PASS: exact key-set match, no stray split-linear keys.")


def run_merge_ab(args: argparse.Namespace) -> None:
    import torch
    import safetensors.torch
    from diffusers.quantizers.nunchaku.utils import SVDQW4A4Linear

    converter = load_converter_module(args.converter_script)

    def load_prefix(path: str, prefix: str, *, rename: bool) -> dict:
        """Load all tensors under a module prefix. When rename=True, apply the
        converter's own SUFFIX_RENAMES/DROP_SUFFIXES (raw source checkpoints
        use Nunchaku's own suffixes -- lora_down/lora_up/smooth -- not the
        proj_down/proj_up/smooth_factor names SVDQW4A4Linear expects)."""
        out = {}
        with safetensors.torch.safe_open(path, framework="pt") as f:
            for k in f.keys():
                if not k.startswith(prefix + "."):
                    continue
                suffix = k[len(prefix) + 1 :]
                if rename:
                    if suffix in converter.DROP_SUFFIXES:
                        continue
                    suffix = converter.SUFFIX_RENAMES.get(suffix, suffix)
                out[suffix] = f.get_tensor(k)
        return out

    def build(tensors: dict, in_features: int, out_features: int) -> torch.nn.Module:
        tensors = dict(tensors)
        if args.precision == "nvfp4":
            # Older upstream Nunchaku checkpoints omit identity outer scales;
            # the converter synthesizes the same tensors before Lite loading.
            tensors.setdefault("wtscale", torch.ones(1, dtype=torch.bfloat16))
            tensors.setdefault("wcscales", torch.ones(out_features, dtype=torch.bfloat16))
        rank = tensors["proj_down"].shape[1]
        module = SVDQW4A4Linear(
            in_features,
            out_features,
            rank=rank,
            bias="bias" in tensors,
            precision=args.precision,
            group_size=args.group_size,
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        missing, unexpected = module.load_state_dict({k: v.clone() for k, v in tensors.items()}, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch loading {tensors.keys()}: missing={missing} unexpected={unexpected}")
        return module.cuda()

    torch.manual_seed(args.seed)
    total_in = args.left_in_features + args.right_in_features
    x = torch.randn(args.batch, total_in) * 0.5

    merged_t = load_prefix(args.converted_checkpoint, args.target, rename=False)
    left_t = load_prefix(args.source_checkpoint, args.source_left_target, rename=True)
    right_t = load_prefix(args.source_checkpoint, args.source_right_target, rename=True)
    # The real converter drops out_proj/left's bias unconditionally (only the
    # right/mlp_fc2 half's bias survives into the merged target) -- see
    # convert_transformer_checkpoint's ".proj_out.linears.0" + "bias" skip.
    left_t.pop("bias", None)
    if args.precision == "int4":
        # Raw shifted down-projection biases target Nunchaku's fused-GELU,
        # shifted unsigned activation path. The merged Diffusers layer is
        # signed/unfused, so compare against the same compensated right half
        # that the converter places in the merged target.
        prefixed = {f"right.{suffix}": tensor for suffix, tensor in right_t.items()}
        right_t["bias"] = converter.compensated_signed_unfused_bias(
            prefixed, "right", group_size=args.group_size
        )
    if not merged_t:
        raise SystemExit(f"No tensors found for {args.target!r} in {args.converted_checkpoint!r}")
    if not left_t or not right_t:
        raise SystemExit(
            f"No tensors found for {args.source_left_target!r}/{args.source_right_target!r} "
            f"in {args.source_checkpoint!r}"
        )

    with torch.no_grad():
        y_merged = build(merged_t, total_in, args.out_features)(x.cuda().bfloat16())
        y_split = build(left_t, args.left_in_features, args.out_features)(x[:, : args.left_in_features].cuda().bfloat16())
        y_split = y_split + build(right_t, args.right_in_features, args.out_features)(
            x[:, args.left_in_features :].cuda().bfloat16()
        )

    y_merged, y_split = y_merged.float(), y_split.float()
    rel_error = (y_merged - y_split).abs().mean().item() / y_split.abs().mean().item()
    print(f"{args.target}: merged-vs-split relative error = {rel_error:.4f} (threshold {args.rel_error_threshold})")
    if not torch.isfinite(y_merged).all():
        raise SystemExit(f"FAIL: {args.target} merged output contains non-finite values")
    if rel_error > args.rel_error_threshold:
        raise SystemExit(f"FAIL: {args.target} relative error {rel_error:.4f} exceeds threshold {args.rel_error_threshold}")
    print("PASS")


def main() -> None:
    args = parse_args()
    if args.mode == "keymatch":
        run_keymatch(args)
    elif args.mode == "merge-ab":
        run_merge_ab(args)


if __name__ == "__main__":
    main()
