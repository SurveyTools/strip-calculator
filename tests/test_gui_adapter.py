from pathlib import Path

from PIL import Image

from gui_adapter import build_marker_set, marker_set_canonical_text_and_hash


def _make_test_image(path: Path) -> None:
    Image.new("RGB", (400, 300), color=(100, 100, 100)).save(path)


def test_build_marker_set_computes_a_real_image_hash(tmp_path):
    image_path = tmp_path / "synthetic.jpg"
    _make_test_image(image_path)

    marker_set = build_marker_set(
        image_path=str(image_path), image_width_px=400, image_height_px=300,
        pts_sorted=[(200.0, 280.0), (200.0, 200.0), (200.0, 120.0)],
        spacing_m=20.0, focal_length_mm=50.0, sensor_height_mm=23.6,
    )

    import hashlib

    expected = hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert marker_set.context.image_sha256 == expected
    assert marker_set.context.image_sha256 != "unknown"
    assert len(marker_set.markers) == 3
    assert marker_set.markers[1].ground_distance_m == 20.0


def test_marker_set_canonical_hash_matches_what_export_would_write(tmp_path):
    image_path = tmp_path / "synthetic.jpg"
    _make_test_image(image_path)
    marker_set = build_marker_set(
        image_path=str(image_path), image_width_px=400, image_height_px=300,
        pts_sorted=[(200.0, 280.0), (200.0, 200.0)],
        spacing_m=20.0, focal_length_mm=50.0, sensor_height_mm=23.6,
    )
    text, sha256 = marker_set_canonical_text_and_hash(marker_set)

    import hashlib

    assert sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()

    from osw_strip_width.io import write_artifact

    export_path = tmp_path / "exported.json"
    write_artifact(marker_set, str(export_path))
    # marker_set_canonical_text_and_hash's `text` already includes the
    # trailing newline write_artifact(obj, path) persists — asserting
    # `text + "\n"` here would require two trailing newlines, which nothing
    # writes. Compare directly.
    assert export_path.read_text() == text
