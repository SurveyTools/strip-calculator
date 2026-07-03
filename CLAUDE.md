# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file PyQt5 desktop application for oblique aerial wildlife survey calibration (East Africa). Users click ground markers of known equal spacing in oblique aerial photos; the tool fits a 1-D projective model to derive sensor height AGL, tilt angle, and strip width.

## Commands

```bash
uv run python strip_geom.py   # run the app
uv sync                        # install / sync dependencies
python3 -m py_compile strip_geom.py  # syntax check (no test suite)
```

## Architecture

Everything lives in `strip_geom.py`. No tests, no modules.

**Core geometry** (top of file, pure functions):
- `fit_projective_1d(y_ground, t_pixel)` — fits `t = (ay+b)/(cy+1)` via DLT seed + scipy `least_squares`
- `invert_projective(t, a, b, c)` — recovers ground distance from pixel coordinate
- `fit_camera_params(a, b, c, f_px)` — derives AGL and tilt from fit coefficients: `h = f_px/a`, `φ = arctan(c·f_px/a)`
- `geometric_strip_width(...)` — analytical pinhole cross-check

**Qt classes:**
- `ImageView` — `QGraphicsView` subclass; emits `point_added` signal on click; handles zoom/pan; draws marker overlays and fit result overlay
- `RegressCanvas` — matplotlib canvas; plots fitted AGL vs strip width across logged passes
- `MainWindow` — wires everything together; `_calculate()` is the main computation pipeline

**Calculation pipeline** (`MainWindow._calculate`):
1. Sort clicked points by image y (bottom = nearest ground)
2. SVD line fit through pixel positions → `line_dir`, `centroid`
3. Project each point onto the line → scalar `t_i`
4. Assign ground distances `y_i = i × spacing`
5. Fit projective model → `(a, b, c)`
6. Extrapolate line to image top/bottom edges → `t_near`, `t_far`
7. Invert model → ground distances → `strip_m`
8. Call `fit_camera_params` → `h_fit_m`, `tilt_fit_deg`
9. Compute GSD (finite-difference) and fit RMSE

**Marker count constraints:** minimum 2 (allowed but unreliable — 3 parameters, 2 equations is under-determined); 3 is exactly determined; 4+ gives meaningful RMSE.

**Units:** ground distances in metres, AGL stored both in metres (`h_fit_m`) and feet (`h_fit_ft`), pixel coordinates in image pixels, GSD in cm/px.
