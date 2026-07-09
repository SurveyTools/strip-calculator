# Strip Width Calibration Tool

Calibrates the visible ground strip width for oblique aerial wildlife surveys by clicking ground markers of known spacing in calibration-flight images. Fits a 1-D projective model to derive sensor height AGL, camera tilt angle, and strip width directly from the image — no GPS required.

As long as there are >2 markers clicked (ideally more), then the geometry is constrained and the height AGL and the strip width can be extracted.

Note: this seems to work pretty well with 50mm lenses, but still testing.

## Install and run

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python strip_geom.py
```

## Workflow

1. **Open Image** — load a calibration-flight image (JPEG, TIFF, PNG).
2. **Set parameters** — enter marker spacing (m), expected height AGL (ft for the geometric cross-check), focal length, sensor height, and nominal tilt angle.
3. **Place Markers** — press **P**, then click each visible ground marker from nearest to farthest. Click order does not matter; points are sorted automatically. Press **Z** to undo the last click, **Esc** or **P** to stop placing.
4. **Calculate** — press **Enter**. The result panel shows:
   - Fitted AGL (m / ft) and tilt angle (° from nadir) recovered from the projective fit
   - Empirical strip width (m) — ground distance from bottom to top image edge. This is the MEASURED strip.
   - Geometric strip width — cross-check using your entered camera parameters. This is what it SHOULD have been with entered target height and roll.
   - GSD at the near edge (cm/px)
   - Fit RMSE (px) — meaningful only with ≥ 4 markers
5. **Log Result** — append the pass to the table and update the AGL vs strip width regression plot.
6. Repeat for each pass, then **Export CSV** (Ctrl+S).

## Geometry

Markers laid perpendicular to the flight line project to unequal pixel spacings in the image due to perspective compression. The tool fits the exact pinhole-camera projective model:

```
t = (a·y + b) / (c·y + 1)
```

where `y` is ground distance and `t` is the pixel coordinate along the marker line. From the fit coefficients and the known focal length, height and tilt are recovered directly:

```
h   = f_px / a
φ   = arctan(c · f_px / a)
```

Strip width is the ground distance between the pixel-projected near (bottom) and far (top) image edges under this model.

**Marker count:** 2 markers is the minimum accepted but the fit is under-determined (results are indicative only). 3 markers is exactly determined; 4 or more gives a meaningful RMSE and reliable extrapolation.

## Camera parameters

| Parameter | Description |
|-----------|-------------|
| Focal length (mm) | Lens focal length |
| Sensor height (mm) | Physical sensor height in the tilt direction |
| Nominal tilt (°) | Camera angle from nadir — used only for the geometric cross-check |

The nominal tilt entry does not affect the empirical fit; it is used solely to compare the measured strip width against the analytical prediction.
