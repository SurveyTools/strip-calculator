import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication, QFileDialog


def _make_test_image(path: Path) -> None:
    Image.new("RGB", (400, 300), color=(100, 100, 100)).save(path)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


CASES = [
    # (window_module, window_class, expected_method, cli_command, cli_only_args)
    ("strip_geom", "MainWindow", "calibrate", "calibrate", [
        "--nominal-tilt-deg", "45.0",
    ]),
    ("strip_solver", "SolverWindow", "solve", "solve", [
        "--tilt-nom-deg", "45.0", "--tilt-lo-deg", "42.0", "--tilt-hi-deg", "48.0",
    ]),
]


@pytest.mark.parametrize(
    "window_module,window_class,expected_method,cli_command,cli_only_args", CASES
)
def test_export_marker_set_replays_through_the_cli(
    qapp, tmp_path, monkeypatch, window_module, window_class, expected_method, cli_command, cli_only_args,
):
    # Review finding #3: an earlier draft always replayed through `calibrate`
    # regardless of which window produced the marker set, so the solver
    # case was never actually comparing like with like. `cli_command` and
    # `cli_only_args` are parametrized per window so SolverWindow replays
    # through `solve`, matching what SolverWindow._calculate itself called.
    import importlib

    image_path = tmp_path / "synthetic.jpg"
    _make_test_image(image_path)

    module = importlib.import_module(window_module)
    window = getattr(module, window_class)()

    result = window.view.load_image(str(image_path))
    assert result is not None
    window._img_size = result
    window._img_name = image_path.name
    window._img_path = str(image_path)

    window._pts = [QPointF(200.0, 280.0 - i * 40.0) for i in range(5)]
    window._calculate()
    assert getattr(window, "_last_estimate", None) is not None, "GUI calculation failed"
    assert window._last_estimate.method == expected_method

    export_path = tmp_path / "exported_markers.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(export_path), ""))
    )
    window._export_marker_set()
    assert export_path.exists()

    data = json.loads(export_path.read_text())
    assert data["schema_id"] == "osw:strip-width-marker-set:0.1"
    assert len(data["markers"]) == 5
    assert data["context"]["image_sha256"] != "unknown"

    result = subprocess.run(
        [sys.executable, "-m", "osw_strip_width.cli", cli_command,
         "--input", str(export_path),
         "--focal-mm", "50.0", "--sensor-height-mm", "23.6",
         "--agl-ft", "300.0", "--output", "-",
         *cli_only_args],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    via_cli = json.loads(result.stdout)
    assert via_cli["method"] == expected_method
    assert via_cli["camera_frame_swath_m"] == window._last_estimate.camera_frame_swath_m
    # Review finding #1: the in-process estimate and the exported-then-
    # replayed CLI estimate must record the *same* marker_set_sha256 — if
    # gui_adapter hashes different bytes than what write_artifact actually
    # persists, this is exactly the assertion that catches it (the swath
    # comparison above would not have).
    assert via_cli["input_marker_set_sha256"] == window._last_estimate.input_marker_set_sha256
