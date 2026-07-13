# Strip Width Tools

Two companion tools for measuring the visible ground strip width in oblique aerial wildlife surveys, from clicked ground markers of known spacing:

- **`strip_geom.py` — Calibration.** Many markers visible: fits a 1-D projective model to solve height AGL, camera tilt, *and* strip width directly from the image — no assumptions needed.
- **`strip_solver.py` — Solver.** Survey passes with only 2+ markers visible: assumes the tilt angle (with an uncertainty range, e.g. 42–48°) and a trusted focal length, solves AGL from the markers in closed form, and reports strip width with an uncertainty band. The assumed AGL serves only as a sanity check (warns beyond ±10%).

Note: this seems to work pretty well with 50mm lenses, but still testing.

## Install and run

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python strip_geom.py     # calibration tool
uv run python strip_solver.py   # survey-pass solver
```

---

# Calibration tool (`strip_geom.py`)

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

---

# Survey-pass solver (`strip_solver.py`)

For images where only two (or a few) markers are visible — not enough to solve the full camera pose. Instead the tilt is assumed from flight data, and the markers solve the height AGL.

## Workflow

1. **Open Image** — load a survey-pass image.
2. **Set parameters** — marker spacing (m), assumed AGL (ft), nominal tilt plus its plausible min/max range (°), focal length and sensor height (trusted).
3. **Place Markers** — press **P**, click 2+ visible markers (same controls as the calibration tool).
4. **Calculate** — press **Enter**. The result panel shows:
   - Solved AGL at the nominal tilt, and its deviation from the assumed AGL (warning beyond ±10%)
   - Strip width at the nominal tilt
   - Strip width band across the tilt range (AGL is re-solved at each angle)
   - Geometric cross-check at the fully assumed pose, GSD, and fit RMSE (meaningful with 3+ markers)
5. **Log Result** / **Export CSV** as in the calibration tool. The plot shows strip width vs solved AGL across the tilt uncertainty range.

## Geometry

With tilt φ fixed and focal length f (px) trusted, two markers at pixel coordinates t₁, t₂ (measured from the principal point) separated by ground distance d determine the height in closed form — the markers' unknown offset from nadir cancels exactly:

```
h(φ) = d·(f − t₁u)·(f − t₂u) / (f·(t₂ − t₁)·(1 + u²)),   u = tan φ
```

Strip width follows by inverting the pinhole map `Y(t) = h·(t + f·u)/(f − t·u)` at the bottom and top image edges. Because AGL is re-solved at every tilt in the sweep, the reported band reflects only the uncertainty the markers cannot resolve — considerably narrower than varying tilt and AGL independently.

With 3+ markers the same model is refined by least squares over height and nadir offset, giving a meaningful pixel RMSE; 2 markers are exactly determined (RMSE 0).
