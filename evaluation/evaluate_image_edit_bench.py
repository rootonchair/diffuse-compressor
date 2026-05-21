"""Run LongCat image-edit evaluation on the local image-edit-bench dataset."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_EVALUATION_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVALUATION_DIR.parent
for _path in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
try:
    sys.path.remove(str(_EVALUATION_DIR))
except ValueError:
    pass

import torch
from PIL import Image

from diffuse_compressor.runtime import RuntimePipelineSpec, load_evaluation_pipeline
from examples.upstream_diffusion_svdquant import (
    MODEL_DEFAULTS,
    _call_image_edit_pipeline,
    _hash_str_to_int,
    _resize_image_edit_image,
    configure_logging,
)
from evaluation.evaluate_image_generation import infer_nunchaku_lite_target


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class BenchRecord:
    filename: str
    task: str
    prompt: str
    image_paths: tuple[Path, ...]
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "quantized"), required=True)
    parser.add_argument("--model-key", choices=tuple(MODEL_DEFAULTS), default="longcat-image-edit")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--pipeline-cls", default=None)
    parser.add_argument("--runtime", choices=("none", "nunchaku-lite", "torch-dequant"), default="none")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--nunchaku-lite-target", default=None)
    parser.add_argument("--precision", choices=("int4", "fp4", "nvfp4"), default="nvfp4")
    parser.add_argument("--pipeline-offload", choices=("none", "model", "sequential"), default="model")
    parser.add_argument("--torch-dequant-activation-mode", choices=("none", "input"), default="input")
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-root", default="datasets/image-edit-bench")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--input-image-size",
        type=int,
        default=512,
        help="Center-crop and resize each input image before LongCat prompt encoding. Use <=0 to keep originals.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--html", default=None, help="Optional HTML report path. Defaults to output-dir/report.html.")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    defaults = MODEL_DEFAULTS[args.model_key]
    if defaults.task != "image-edit":
        raise ValueError(f"{args.model_key!r} is not an image-edit model")

    dataset_root = Path(args.dataset_root)
    records = load_bench_records(dataset_root, tasks=args.tasks, num_samples=args.num_samples)
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples" / f"image-edit-bench-{len(records)}"
    input_dir = output_dir / "inputs" / f"image-edit-bench-{len(records)}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_torch_dtype(args.torch_dtype)
    steps = args.steps if args.steps is not None else defaults.steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else defaults.guidance_scale
    model_id = args.model_id or defaults.model_id
    pipeline_cls = _resolve_pipeline_cls(args.pipeline_cls or defaults.pipeline_name)

    load_start = time.perf_counter()
    _reset_cuda_peak(args.device)
    pipe = load_evaluation_pipeline(
        pipeline_cls=pipeline_cls,
        model_id=model_id,
        spec=RuntimePipelineSpec(
            mode=args.mode,
            runtime=args.runtime,
            checkpoint=args.checkpoint,
            model_key=args.model_key,
            nunchaku_lite_target=(
                args.nunchaku_lite_target
                if args.nunchaku_lite_target is not None
                else infer_nunchaku_lite_target(args.model_key)
                if args.runtime == "nunchaku-lite"
                else None
            ),
            precision="fp4" if args.precision == "nvfp4" else args.precision,
            device=args.device,
            torch_dtype=torch_dtype,
            torch_dequant_activation_mode=args.torch_dequant_activation_mode,
            pipeline_offload=args.pipeline_offload,
        ),
    )
    _patch_longcat_image_prompt_encoding(pipe)
    _sync_cuda(args.device)
    load_seconds = time.perf_counter() - load_start
    load_vram = _cuda_peak_gb(args.device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    per_record: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            output_path = sample_dir / f"{record.filename}.png"
            input_paths = save_record_inputs(record, input_dir, image_size=args.input_image_size)
            entry: dict[str, Any] = {
                "filename": record.filename,
                "task": record.task,
                "prompt": record.prompt,
                "seed": record.seed,
                "input_paths": [str(path) for path in input_paths],
                "output_path": str(output_path),
            }
            if args.skip_existing and output_path.exists():
                entry["skipped"] = True
                per_record.append(entry)
                print(f"[{index}/{len(records)}] skip {record.filename}", flush=True)
                continue

            images = [_load_input_image(path, args.input_image_size) for path in record.image_paths]
            image_arg: Image.Image | list[Image.Image] = images[0] if len(images) == 1 else images
            generator = _make_generator(record.seed, args.device)

            try:
                _sync_cuda(args.device)
                _reset_cuda_peak(args.device)
                start = time.perf_counter()
                result = _call_image_edit_pipeline(
                    pipe,
                    height=args.height,
                    width=args.width,
                    image=image_arg,
                    prompt=record.prompt,
                    negative_prompt="",
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                _sync_cuda(args.device)
                latency_seconds = time.perf_counter() - start
                vram = _cuda_peak_gb(args.device)

                generated = getattr(result, "images", None)
                if not generated:
                    raise ValueError("pipeline output must expose a non-empty images attribute")
                generated[0].save(output_path)
                entry.update(
                    {
                        "skipped": False,
                        "latency_seconds": latency_seconds,
                        "peak_allocated_gb": vram["allocated_gb"],
                        "peak_reserved_gb": vram["reserved_gb"],
                        "input_sizes": [list(image.size) for image in images],
                        "output_size": list(generated[0].size),
                    }
                )
                per_record.append(entry)
                print(
                    f"[{index}/{len(records)}] {record.filename} "
                    f"{latency_seconds:.3f}s peak_alloc={vram['allocated_gb']:.2f}GB",
                    flush=True,
                )
            except Exception as exc:
                if args.stop_on_error:
                    raise
                _sync_cuda(args.device)
                if _is_cuda_device(args.device):
                    torch.cuda.empty_cache()
                vram = _cuda_peak_gb(args.device)
                entry.update(
                    {
                        "skipped": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "peak_allocated_gb": vram["allocated_gb"],
                        "peak_reserved_gb": vram["reserved_gb"],
                        "input_sizes": [list(image.size) for image in images],
                    }
                )
                per_record.append(entry)
                print(f"[{index}/{len(records)}] error {record.filename}: {entry['error']}", flush=True)

    summary = summarize_performance(per_record)
    result = {
        "mode": args.mode,
        "model_key": args.model_key,
        "model_id": model_id,
        "runtime": args.runtime,
        "checkpoint": args.checkpoint,
        "torch_dtype": args.torch_dtype,
        "pipeline_offload": args.pipeline_offload,
        "device": args.device,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "height": args.height,
        "width": args.width,
        "input_image_size": args.input_image_size,
        "dataset_root": str(dataset_root),
        "num_records": len(records),
        "sample_dir": str(sample_dir),
        "input_dir": str(input_dir),
        "load_seconds": load_seconds,
        "load_peak_allocated_gb": load_vram["allocated_gb"],
        "load_peak_reserved_gb": load_vram["reserved_gb"],
        "summary": summary,
        "records": per_record,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    html_path = Path(args.html) if args.html is not None else output_dir / "report.html"
    write_report(result, html_path)
    print(json.dumps({"results": str(results_path), "html": str(html_path), "summary": summary}, indent=2))


def load_bench_records(dataset_root: Path, *, tasks: Sequence[str] | None, num_samples: int) -> list[BenchRecord]:
    tasks_path = dataset_root / "tasks.json"
    image_root = dataset_root / "test_images"
    spec = json.loads(tasks_path.read_text(encoding="utf-8"))
    selected_tasks = list(tasks) if tasks is not None else list(spec)
    records: list[BenchRecord] = []
    for task in selected_tasks:
        if task not in spec:
            raise ValueError(f"Task {task!r} is not present in {tasks_path}")
        task_dir = image_root / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Missing image directory for task {task!r}: {task_dir}")
        prompt = str(spec[task]["prompt"])
        num_images = int(spec[task].get("num_images", 1))
        paths = natural_sorted([path for path in task_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])
        for group in group_image_paths(paths, num_images):
            filename = safe_filename(task, group)
            records.append(
                BenchRecord(
                    filename=filename,
                    task=task,
                    prompt=prompt,
                    image_paths=tuple(group),
                    seed=_hash_str_to_int(filename),
                )
            )
    if num_samples >= 0:
        records = records[:num_samples]
    return records


def group_image_paths(paths: Sequence[Path], num_images: int) -> list[tuple[Path, ...]]:
    if num_images <= 1:
        return [(path,) for path in paths]
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        match = re.match(r"^(?P<base>.+)_(?P<index>\d+)$", path.stem)
        base = match.group("base") if match else path.stem
        grouped.setdefault(base, []).append(path)
    groups = []
    for base in natural_sorted(grouped):
        group = tuple(natural_sorted(grouped[base]))
        if len(group) != num_images:
            raise ValueError(f"Expected {num_images} images for group {base!r}, got {len(group)}")
        groups.append(group)
    return groups


def save_record_inputs(record: BenchRecord, input_dir: Path, image_size: int = 512) -> list[Path]:
    saved = []
    for index, source in enumerate(record.image_paths):
        target = input_dir / f"{record.filename}__input{index}.png"
        _load_input_image(source, image_size).save(target)
        saved.append(target)
    return saved


def summarize_performance(records: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    measured = [record for record in records if not record.get("skipped") and "error" not in record]
    errors = [record for record in records if "error" in record]
    latencies = [float(record["latency_seconds"]) for record in measured]
    allocated = [float(record["peak_allocated_gb"]) for record in measured if "peak_allocated_gb" in record]
    reserved = [float(record["peak_reserved_gb"]) for record in measured if "peak_reserved_gb" in record]
    return {
        "measured_records": len(measured),
        "error_records": len(errors),
        "skipped_records": len([record for record in records if record.get("skipped")]),
        "latency_mean_seconds": _mean(latencies),
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_min_seconds": min(latencies) if latencies else None,
        "latency_max_seconds": max(latencies) if latencies else None,
        "peak_allocated_max_gb": max(allocated) if allocated else None,
        "peak_allocated_mean_gb": _mean(allocated),
        "peak_reserved_max_gb": max(reserved) if reserved else None,
        "peak_reserved_mean_gb": _mean(reserved),
    }


def write_report(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.parent.resolve()
    input_dir = Path(result["input_dir"]).resolve()
    sample_dir = Path(result["sample_dir"]).resolve()
    records = []
    for record in result["records"]:
        item = dict(record)
        item["input_paths"] = [relative_url(Path(path).resolve(), root) for path in record["input_paths"]]
        item["output_path"] = relative_url(Path(record["output_path"]).resolve(), root)
        records.append(item)
    payload = {
        "title": title_for_result(result),
        "summary": result["summary"],
        "load_seconds": result["load_seconds"],
        "load_peak_allocated_gb": result["load_peak_allocated_gb"],
        "load_peak_reserved_gb": result["load_peak_reserved_gb"],
        "records": records,
        "input_dir": relative_url(input_dir, root),
        "sample_dir": relative_url(sample_dir, root),
    }
    output.write_text(report_html(payload), encoding="utf-8")


def report_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(payload["title"])}</title>
  <style>
    :root {{ color-scheme: light dark; --line: #d8dee8; --muted: #667085; --bg: #f6f7f9; --panel: #ffffff; --text: #111827; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --line: #2f3847; --muted: #a4adbc; --bg: #11151c; --panel: #171c25; --text: #edf1f7; }} }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: color-mix(in srgb, var(--panel) 92%, transparent); border-bottom: 1px solid var(--line); backdrop-filter: blur(10px); }}
    h1 {{ margin: 0 0 10px; font-size: 18px; font-weight: 650; }}
    .bar {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; }}
    input {{ width: 100%; min-height: 36px; padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ padding: 5px 8px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); font-size: 12px; white-space: nowrap; }}
    main {{ padding: 14px 18px 32px; }}
    .record {{ margin-bottom: 14px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }}
    .meta {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
    .name {{ font-weight: 650; font-size: 14px; }}
    .prompt {{ margin-top: 5px; color: var(--muted); font-size: 13px; line-height: 1.38; }}
    .perf {{ color: var(--muted); font-size: 12px; text-align: right; white-space: nowrap; }}
    .images {{ display: grid; grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr); gap: 1px; background: var(--line); }}
    figure {{ margin: 0; background: var(--panel); }}
    figcaption {{ padding: 7px 9px; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--line); }}
    .input-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px; background: var(--line); }}
    img {{ display: block; width: 100%; height: auto; background: #000; }}
    @media (max-width: 860px) {{ .bar, .meta, .images {{ grid-template-columns: 1fr; }} .perf {{ text-align: left; }} }}
  </style>
</head>
<body>
  <header>
    <h1 id="title"></h1>
    <div class="bar">
      <input id="filter" type="search" placeholder="Filter task, filename, or prompt" autocomplete="off">
      <div class="summary" id="summary"></div>
    </div>
  </header>
  <main id="records"></main>
  <script>
    const payload = {data};
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
    const fmt = (value, digits = 3) => Number.isFinite(value) ? Number(value).toFixed(digits) : "n/a";
    document.getElementById("title").textContent = payload.title;
    document.getElementById("summary").innerHTML = [
      `records ${{payload.summary.measured_records}}`,
      `errors ${{payload.summary.error_records}}`,
      `lat mean ${{fmt(payload.summary.latency_mean_seconds)}}s`,
      `lat median ${{fmt(payload.summary.latency_median_seconds)}}s`,
      `peak alloc ${{fmt(payload.summary.peak_allocated_max_gb, 2)}}GB`,
      `peak reserved ${{fmt(payload.summary.peak_reserved_max_gb, 2)}}GB`,
      `load ${{fmt(payload.load_seconds)}}s`
    ].map((text) => `<span class="pill">${{esc(text)}}</span>`).join("");
    const root = document.getElementById("records");
    function render() {{
      const q = document.getElementById("filter").value.trim().toLowerCase();
      root.innerHTML = payload.records
        .filter((r) => [r.filename, r.task, r.prompt].join(" ").toLowerCase().includes(q))
        .map((r) => {{
          const inputs = r.input_paths.map((path, i) => `<figure><figcaption>Input ${{i + 1}}</figcaption><img loading="lazy" src="${{esc(path)}}"></figure>`).join("");
          const output = r.error
            ? `<figure><figcaption>Error</figcaption><div style="padding:12px;color:#b42318;font-size:13px;line-height:1.35">${{esc(r.error)}}</div></figure>`
            : `<figure><figcaption>Output</figcaption><img loading="lazy" src="${{esc(r.output_path)}}"></figure>`;
          return `<section class="record">
            <div class="meta">
              <div>
                <div class="name">${{esc(r.task)}} / ${{esc(r.filename)}}</div>
                <div class="prompt">${{esc(r.prompt)}}</div>
              </div>
              <div class="perf">
                <div>${{fmt(r.latency_seconds)}}s</div>
                <div>alloc ${{fmt(r.peak_allocated_gb, 2)}}GB</div>
                <div>reserved ${{fmt(r.peak_reserved_gb, 2)}}GB</div>
              </div>
            </div>
            <div class="images">
              <div class="input-grid">${{inputs}}</div>
              ${{output}}
            </div>
          </section>`;
        }}).join("");
    }}
    document.getElementById("filter").addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""


def title_for_result(result: dict[str, Any]) -> str:
    runtime = result["runtime"] if result["runtime"] != "none" else result["mode"]
    return f"LongCat image-edit-bench {runtime} ({result['torch_dtype']})"


def relative_url(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return rel.as_posix()


def natural_sorted(values):
    def key(value):
        text = str(value.name if isinstance(value, Path) else value)
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]

    return sorted(values, key=key)


def safe_filename(task: str, paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        base = paths[0].stem
    else:
        common = re.sub(r"_\d+$", "", paths[0].stem)
        base = common
    text = f"{task}__{base}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or f"{task}__sample"


def _open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_input_image(path: Path, image_size: int) -> Image.Image:
    return _resize_image_edit_image(_open_rgb(path), image_size)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _reset_cuda_peak(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _sync_cuda(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize(torch.device(device))


def _cuda_peak_gb(device: str) -> dict[str, float | None]:
    if not _is_cuda_device(device):
        return {"allocated_gb": None, "reserved_gb": None}
    cuda_device = torch.device(device)
    return {
        "allocated_gb": torch.cuda.max_memory_allocated(cuda_device) / 1024**3,
        "reserved_gb": torch.cuda.max_memory_reserved(cuda_device) / 1024**3,
    }


def _is_cuda_device(device: str) -> bool:
    try:
        return torch.device(device).type == "cuda" and torch.cuda.is_available()
    except RuntimeError:
        return False


def _make_generator(seed: int, device: str) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except RuntimeError:
        return torch.Generator().manual_seed(int(seed))


def _resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _resolve_pipeline_cls(name: str) -> type:
    import importlib

    module_name, class_name = name.rsplit(".", 1) if "." in name else ("diffusers", name)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _patch_longcat_image_prompt_encoding(pipe: Any) -> None:
    """Fix LongCat's image token count for local records with multiple input images."""

    required = (
        "image_processor_vl",
        "image_token",
        "prompt_template_encode_prefix",
        "prompt_template_encode_suffix",
        "tokenizer",
        "text_encoder",
    )
    if not all(hasattr(pipe, name) for name in required):
        return
    module = sys.modules.get(pipe.__class__.__module__)
    split_quotation = getattr(module, "split_quotation", None) if module is not None else None
    if split_quotation is None:
        return

    def _encode_prompt_fixed(self, prompt, image):
        raw_vl_input = self.image_processor_vl(images=image, return_tensors="pt")
        pixel_values = raw_vl_input["pixel_values"]
        image_grid_thw = raw_vl_input["image_grid_thw"]
        all_tokens = []
        for clean_prompt_sub, matched in split_quotation(prompt[0]):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)
            else:
                tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                all_tokens.extend(tokens)

        tokenizer_max_length = getattr(self, "tokenizer_max_length", 512)
        if len(all_tokens) > tokenizer_max_length:
            all_tokens = all_tokens[:tokenizer_max_length]

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": [all_tokens]},
            max_length=tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        text = self.prompt_template_encode_prefix
        merge_length = self.image_processor_vl.merge_size**2
        if image_grid_thw.ndim == 2:
            num_image_tokens = int((image_grid_thw.prod(dim=1).sum() // merge_length).item())
        else:
            num_image_tokens = int((image_grid_thw.prod() // merge_length).item())
        while self.image_token in text:
            text = text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
        text = text.replace("<|placeholder|>", self.image_token)

        prefix_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]

        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        prefix_len = prefix_tokens.index(vision_start_token_id)
        suffix_len = len(suffix_tokens)

        mask_dtype = text_tokens_and_mask.attention_mask[0].dtype
        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=mask_dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=mask_dtype)

        token_dtype = text_tokens_and_mask.input_ids.dtype
        prefix_tokens_tensor = torch.tensor(prefix_tokens, dtype=token_dtype)
        suffix_tokens_tensor = torch.tensor(suffix_tokens, dtype=token_dtype)

        input_ids = torch.cat((prefix_tokens_tensor, text_tokens_and_mask.input_ids[0], suffix_tokens_tensor), dim=-1)
        attention_mask = torch.cat(
            (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1
        )

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]
        return prompt_embeds

    pipe._encode_prompt = types.MethodType(_encode_prompt_fixed, pipe)


if __name__ == "__main__":
    main()
