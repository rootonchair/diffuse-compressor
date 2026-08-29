"""Build a labeled comparison grid from run_edit_cases outputs.

Rows are edit cases; columns are the source image followed by one column per
variant. Usage:

    python -m evaluation.make_edit_comparison \
        --variants original lite-infer datafree \
        --labels "Original BF16" "Calibrated (lite-infer)" "Data-free" \
        --cases restyle-watercolor restyle-ukiyoe remove-bicycle remove-car \
                change-car-color change-armchair add-juice add-sailboat \
        --output outputs/eval/flux2-klein-9b/edits/comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluation.run_edit_cases import CASES

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits-root", default="outputs/eval/flux2-klein-9b/edits")
    parser.add_argument(
        "--source-root",
        default="outputs/eval/flux2-klein-9b/realistic-original/samples/realistic-32",
    )
    parser.add_argument("--variants", nargs="+", default=["original", "lite-infer", "datafree"])
    parser.add_argument("--labels", nargs="+", default=None, help="Column labels, one per variant.")
    parser.add_argument("--cases", nargs="+", default=None, help="Restrict to these case ids.")
    parser.add_argument("--tile", type=int, default=384)
    parser.add_argument("--output", default="outputs/eval/flux2-klein-9b/edits/comparison.png")
    args = parser.parse_args()

    labels = args.labels or args.variants
    if len(labels) != len(args.variants):
        raise SystemExit("--labels must match --variants in length")
    cases = CASES
    if args.cases:
        by_id = {case.case_id: case for case in CASES}
        cases = [by_id[case_id] for case_id in args.cases]

    edits_root = Path(args.edits_root)
    source_root = Path(args.source_root)
    tile = args.tile
    header_h = 48
    caption_h = 56
    margin = 8

    columns = ["Input"] + labels
    grid_w = margin + len(columns) * (tile + margin)
    grid_h = header_h + len(cases) * (tile + caption_h + margin)
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(26)
    caption_font = _load_font(18)

    for col, label in enumerate(columns):
        x = margin + col * (tile + margin)
        w = draw.textlength(label, font=header_font)
        draw.text((x + (tile - w) / 2, (header_h - 30) / 2), label, fill="black", font=header_font)

    for row, case in enumerate(cases):
        y = header_h + row * (tile + caption_h + margin)
        paths = [source_root / case.source] + [
            edits_root / variant / f"{case.case_id}.png" for variant in args.variants
        ]
        for col, path in enumerate(paths):
            x = margin + col * (tile + margin)
            if path.exists():
                image = Image.open(path).convert("RGB").resize((tile, tile), Image.LANCZOS)
                canvas.paste(image, (x, y))
            else:
                draw.rectangle((x, y, x + tile, y + tile), outline="red")
                draw.text((x + 12, y + tile // 2), "missing", fill="red", font=caption_font)
        caption = f"[{case.category}] {case.case_id}: “{case.instruction}”"
        draw.text((margin, y + tile + 8), caption, fill="black", font=caption_font)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
