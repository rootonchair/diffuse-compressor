from __future__ import annotations

import sys
from pathlib import Path


TEXT_TO_IMAGE_EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "text_to_image"
)

if str(TEXT_TO_IMAGE_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_IMAGE_EXAMPLE_DIR))
