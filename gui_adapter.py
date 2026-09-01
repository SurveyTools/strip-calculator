# gui_adapter.py
from __future__ import annotations

import hashlib
from pathlib import Path

from osw_strip_width.io import canonical_text_and_hash, write_artifact
from osw_strip_width.types import CameraContext, ImageContext, Marker, MarkerSet


def build_marker_set(
    *,
    image_path: str,
    image_width_px: int,
    image_height_px: int,
    pts_sorted: list[tuple[float, float]],
    spacing_m: float,
    focal_length_mm: float,
    sensor_height_mm: float,
) -> MarkerSet:
    """The one place both windows (and this module's tests) build a
    MarkerSet from GUI state. exif_orientation_applied=True is honest here
    because ImageView.load_image (strip_geom.py) always loads through
    QImageReader with setAutoTransform(True) — see Step 3."""
    image_sha256 = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    context = ImageContext(
        image_id=Path(image_path).name,
        image_sha256=image_sha256,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        exif_orientation_applied=True,
        pixel_convention="origin-top-left, +x right, +y down, EXIF-oriented pixels",
        camera=CameraContext(
            make_model="unknown",
            focal_length_mm=focal_length_mm,
            sensor_height_mm=sensor_height_mm,
            principal_row_assumption="image_height_px / 2 (uncalibrated)",
            lens_distortion="not modeled",
        ),
        rig_id=None, aircraft_id=None, side=None, observer_id=None, site_id=None,
        package_version="0.1.0", algorithm_version="osw_strip_width.geometry@0.1.0",
    )
    markers = tuple(
        Marker(pixel=(x, y), ground_distance_m=i * spacing_m, station_index=i)
        for i, (x, y) in enumerate(pts_sorted)
    )
    return MarkerSet(context=context, markers=markers)


def marker_set_canonical_text_and_hash(marker_set: MarkerSet) -> tuple[str, str]:
    """The in-process marker_set_sha256 passed to run_calibration/run_solve
    must equal the hash a later `export -> CLI reload` computes from the
    same MarkerSet. A prior revision duplicated the "hash write_artifact's
    output plus a newline" logic here directly; a later review found the
    same duplication had drifted for GeometryEstimate hashing elsewhere
    (Task 12's _solve_all_passes) and asked for one shared byte-level
    convention instead of two copies that can independently go stale.
    Delegate to io.py's canonical_text_and_hash, which every artifact type
    now shares."""
    return canonical_text_and_hash(marker_set)


def export_marker_set(marker_set: MarkerSet, path: str) -> None:
    write_artifact(marker_set, path)
