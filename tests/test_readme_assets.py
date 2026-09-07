"""Protect published claims against incomplete inputs and stale generated files."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "readme_assets", ROOT / "scripts/generate_readme_assets.py"
)
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


@pytest.fixture
def kernel_results(tmp_path, monkeypatch):
    paths = sorted(assets.KERNEL_RESULTS.glob("fused-add-rms-norm-packed-cache*.txt"))
    for path in paths:
        shutil.copyfile(path, tmp_path / path.name)
    monkeypatch.setattr(assets, "KERNEL_RESULTS", tmp_path)
    return tmp_path


def test_rejects_missing_kernel_run(kernel_results):
    next(kernel_results.glob("*.txt")).unlink()
    with pytest.raises(SystemExit, match="Expected five kernel runs"):
        assets.read_speedups("fused-add-rms-norm-packed-cache*.txt", 3)


def test_rejects_duplicate_shape(kernel_results):
    path = next(kernel_results.glob("*.txt"))
    content = path.read_text()
    row = next(line for line in content.splitlines() if line.startswith("| 8 |"))
    path.write_text(content + "\n" + row + "\n")
    with pytest.raises(SystemExit, match="Duplicate token count"):
        assets.read_speedups("fused-add-rms-norm-packed-cache*.txt", 3)


@pytest.mark.parametrize("change", ["missing_run", "incomplete_requests"])
def test_rejects_incomplete_serving_evidence(tmp_path, monkeypatch, change):
    for path in assets.SERVING_RESULTS.glob("*.json"):
        shutil.copyfile(path, tmp_path / path.name)
    path = next(tmp_path.glob("*.json"))
    if change == "missing_run":
        path.unlink()
        expected = "Expected three serving runs"
    else:
        result = json.loads(path.read_text())
        result["completed"] = 31
        path.write_text(json.dumps(result))
        expected = "Expected 32 successful baseline requests"
    monkeypatch.setattr(assets, "SERVING_RESULTS", tmp_path)
    with pytest.raises(SystemExit, match=expected):
        assets.read_serving_results()


def test_check_detects_missing_and_stale_artifacts_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "ASSET_DIR", tmp_path / "assets")
    monkeypatch.setattr(assets, "SUMMARY_PATH", tmp_path / "summary.md")
    monkeypatch.setattr("sys.argv", ["generate_readme_assets.py", "--check"])
    with pytest.raises(SystemExit, match="Missing or stale artifacts"):
        assets.main()
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr("sys.argv", ["generate_readme_assets.py"])
    assets.main()
    monkeypatch.setattr("sys.argv", ["generate_readme_assets.py", "--check"])
    assets.main()
    assets.SUMMARY_PATH.write_text("stale\n")
    with pytest.raises(SystemExit, match="summary.md"):
        assets.main()
    assert assets.SUMMARY_PATH.read_text() == "stale\n"
