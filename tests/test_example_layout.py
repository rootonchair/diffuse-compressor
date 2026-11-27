"""Example scripts run from their folders, and docs only point at scripts that exist."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    path.relative_to(ROOT)
    for path in (ROOT / "examples").glob("*/*/quantize_*.py")
    if path.parent.name in ("calibrated", "data_free")
)
# Scripts with a command line (some, like quantize_text_to_video.py, only hold a config).
CLI_SCRIPTS = [script for script in SCRIPTS if 'if __name__ == "__main__":' in (ROOT / script).read_text()]
HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None


def test_every_task_folder_is_split():
    assert SCRIPTS, "no example scripts found"
    for task in ("text_to_image", "text_to_video", "image_to_image"):
        assert not list((ROOT / "examples" / task).glob("quantize_*.py")), f"script left at the top of {task}/"


@pytest.mark.parametrize("script", CLI_SCRIPTS, ids=str)
def test_script_runs_by_path_from_another_directory(tmp_path, script):
    """``python examples/<task>/<kind>/<script>.py --help`` works from any cwd."""

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    if result.returncode != 0 and not HAS_DIFFUSERS and "No module named 'diffusers'" in result.stderr:
        pytest.skip("this script's help text needs diffusers")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout


_DOC_PATH = re.compile(
    r"(?<![\w/])(?:examples/)?((?:text_to_image|text_to_video|image_to_image)/[\w/]+\.py)|(?<![\w/])(examples/[\w/]+\.py)"
)
_PLACEHOLDERS = {"examples/text_to_image/calibrated/quantize_my_model.py"}


@pytest.mark.parametrize(
    "doc", [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))], ids=lambda p: p.name
)
def test_docs_only_reference_existing_example_scripts(doc):
    missing = []
    for match in _DOC_PATH.finditer(doc.read_text()):
        path = match.group(2) or f"examples/{match.group(1)}"
        if path not in _PLACEHOLDERS and not (ROOT / path).exists():
            missing.append(path)
    assert not missing, f"{doc.name} references missing scripts: {sorted(set(missing))}"
