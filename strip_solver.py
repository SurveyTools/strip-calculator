#!/usr/bin/env python3
"""
Strip Width Solver
──────────────────
Oblique aerial wildlife surveys — East Africa

Companion to strip_geom.py (the calibration tool).  Inverse workflow for
survey passes: only 2+ markers of known ground interval are visible, and
AGL / tilt are *assumed* from flight data — but imperfectly (AGL perhaps
±10 %, tilt 42–48° when 45° was attempted).  Focal length is trusted.

Geometry
--------
Pinhole camera at height h, tilt φ from nadir, focal length f (pixels).
With u = tan φ, ground distance from nadir Y maps to pixel coordinate t
(measured from the principal point along the marker line, + toward far):

    t = f·(Y − h·u)/(h + Y·u)        ⇄        Y(t) = h·(t + f·u)/(f − t·u)

Two markers at t₁, t₂ separated by ground distance d constrain h and φ
jointly — the markers' unknown offset from nadir cancels exactly:

    h(φ) = d·(f − t₁u)·(f − t₂u) / (f·(t₂ − t₁)·(1 + u²))

So:  fix tilt at its nominal value → the markers *solve* AGL, and the
assumed AGL becomes a sanity check (warn beyond ±10 %).  Sweeping φ over
the stated uncertainty range re-solves h(φ) at each angle and gives a
strip-width band  W_min / W_nominal / W_max.

With 3+ markers the same model is fit by least squares over (h, y0),
giving a meaningful pixel RMSE; n = 2 is exactly determined (RMSE 0).

Workflow
--------
1.  Open a survey-pass image.
2.  Set marker spacing (m), assumed AGL, nominal tilt + tilt range, camera.
3.  Press P and click the visible markers (2+).
4.  Press Enter -> solved AGL, strip width + uncertainty band, overlay.
5.  Log Result, then open the next pass.
"""

import sys
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QFormLayout, QLabel, QPushButton,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog, QMessageBox, QAction, QStatusBar,
)
from PyQt5.QtCore import Qt, QPointF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from strip_geom import (
    BASE_FONT_SIZE, _font, ImageView, apply_app_style,
)

AGL_TOL_PCT = 10.0   # warn when solved AGL differs from assumed by more than this

# exif_gps_altitude_m used to be defined here directly; it is now
# osw_strip_width.exif.exif_gps_altitude_m (ported unchanged from this exact
# function) and imported locally in _open_image below, so this GUI and the
# CLI/mapping layer share one copy instead of two that can independently
# drift.


# ── pinhole geometry (assumed-pose model) ──────────────────────────────────────

def ground_distance_from_pixel(t: float, h: float, phi_rad: float,
                               f_px: float) -> float:
    """
    Invert the pinhole map: pixel coordinate t (from the principal point,
    + toward the far/top edge) -> ground distance Y from nadir (metres).

        t = f·(Y − h·u)/(h + Y·u)   with  u = tan(phi)
        ⇒  Y = h·(t + f·u)/(f − t·u)

    Must raise ValueError when the pixel is at/above the horizon
    (f − t·u <= 0) — Y would be infinite or behind the camera.
    """
    u = math.tan(phi_rad)
    denom = f_px - t * u
    if denom <= 0:
        raise ValueError("Pixel is at or above the horizon at this tilt — "
                         "ground distance is unbounded there.")
    return h * (t + f_px * u) / denom


def gsd_at_pixel(t: float, h: float, phi_rad: float, f_px: float) -> float:
    """Ground sampling distance dY/dt at pixel t, in cm per pixel."""
    u = math.tan(phi_rad)
    denom = f_px - t * u
    if denom <= 0:
        raise ValueError("Pixel is at or above the horizon.")
    return h * f_px * (1.0 + u * u) / (denom * denom) * 100.0


def solve_agl_given_tilt(t_vals: np.ndarray, y_ground: np.ndarray,
                         phi_rad: float, f_px: float):
    """
    Solve height AGL from marker pixel positions at an assumed tilt.

    t_vals / y_ground must be ordered near -> far (t and y both increasing).
    Uses the closed form on the first/last marker pair (the offset from
    nadir cancels), then for n >= 3 refines (h, y0) by least squares.

    Returns (h_m, y0_m, rmse_px).  Raises ValueError if unphysical.
    """
    u  = math.tan(phi_rad)
    t1 = float(t_vals[0])
    t2 = float(t_vals[-1])
    d  = float(y_ground[-1] - y_ground[0])

    denom = f_px * (t2 - t1) * (1.0 + u * u)
    if denom <= 0:
        raise ValueError("Markers must run near (bottom) to far (top) — "
                         "pixel positions do not increase toward the top.")
    h0 = d * (f_px - t1 * u) * (f_px - t2 * u) / denom
    if not math.isfinite(h0) or h0 <= 0:
        raise ValueError("Solved AGL is unphysical — check marker spacing, "
                         "order and tilt.")
    y0 = ground_distance_from_pixel(t1, h0, phi_rad, f_px) - float(y_ground[0])

    if len(t_vals) == 2:
        return h0, y0, 0.0

    def residuals(p):
        h, y0_ = p
        Y = y0_ + y_ground
        return f_px * (Y - h * u) / (h + Y * u) - t_vals

    result = least_squares(residuals, [h0, y0], method='lm', max_nfev=2000)
    h_fit, y0_fit = result.x
    if not math.isfinite(h_fit) or h_fit <= 0:
        raise ValueError("Least-squares AGL solution is unphysical.")
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return float(h_fit), float(y0_fit), rmse


def strip_width_given_pose(h: float, phi_rad: float, f_px: float,
                           t_near: float, t_far: float) -> float:
    """Ground distance spanned between two pixel coordinates (metres)."""
    return (ground_distance_from_pixel(t_far,  h, phi_rad, f_px)
            - ground_distance_from_pixel(t_near, h, phi_rad, f_px))


def tilt_sweep(t_vals: np.ndarray, y_ground: np.ndarray, f_px: float,
               t_near: float, t_far: float,
               phi_lo_deg: float, phi_hi_deg: float, n: int = 61):
    """
    Sweep tilt across [phi_lo, phi_hi]; at each angle re-solve AGL from the
    markers and evaluate strip width.  Returns (phi_deg, h_m, width_m)
    arrays with NaN where the solve/inversion fails.
    """
    phis = np.linspace(phi_lo_deg, phi_hi_deg, n)
    hs   = np.full(n, np.nan)
    ws   = np.full(n, np.nan)
    for i, pd in enumerate(phis):
        try:
            pr        = math.radians(pd)
            h, _, _   = solve_agl_given_tilt(t_vals, y_ground, pr, f_px)
            hs[i]     = h
            ws[i]     = strip_width_given_pose(h, pr, f_px, t_near, t_far)
        except (ValueError, NotImplementedError):
            continue
    return phis, hs, ws


def profile_cost(alt_res: np.ndarray, marker_costs: np.ndarray) -> float:
    """
    Combine the two evidence terms of the joint solve into one scalar cost.

    alt_res       — standardised altitude residuals, one per pass with an
                    altitude reading: (h_i − alt_i − b) / sigma_i.  Already
                    dimensionless; empty array when <2 readings.
    marker_costs  — n_i · rmse_i² / sigma_px² for each pass with >= 3 markers
                    (already a chi-square-like term); empty array otherwise.

    The CI reported by joint_tilt_profile assumes the returned value behaves
    like a chi-square (sum of squared standardised residuals); re-weighting
    one term trades altimeter trust against click trust and stretches the CI
    accordingly.

    Altitude residuals go through a soft-L1 loss: quadratic for small
    residuals (chi-square behaviour preserved) but linear for outliers, so
    one bad GPS/baro reading is tolerated rather than dragging the tilt.
    Marker terms stay quadratic — clicks rarely have gross outliers.
    """
    robust = 2.0 * (np.sqrt(1.0 + alt_res ** 2) - 1.0)
    return float(np.sum(robust) + np.sum(marker_costs))


def joint_tilt_profile(passes: list, phi_lo_deg: float, phi_hi_deg: float,
                       sigma_dalt_frac: float = 0.02, sigma_px: float = 2.0,
                       n: int = 121):
    """
    Joint mount-tilt profile over all logged passes of one flight.

    passes: dicts with t_vals, y_ground, f_px, t_near, t_far, alt_m
    (altitude reading in metres, or None).  Two information sources
    constrain the shared tilt:

      * altitude term — for passes with a reading, the per-pass solved AGLs
        h_i(phi) must match the readings up to ONE free common offset b
        (absorbs ground elevation / datum), so only the differences between
        passes matter.  Active when >= 2 passes have readings.
      * marker term — passes with >= 3 markers acquire pixel RMSE away from
        the true tilt.

    Returns a dict: phis_deg, costs, phi_best_deg, ci_lo_deg, ci_hi_deg,
    b_m (nan if no altitude term), and per-pass h_m / w_m / rmse_px lists
    evaluated at the best angle.  Raises ValueError when neither term is
    active — the tilt is then unconstrained by the data.

    Sign convention: b = solved AGL − altitude reading, so −b is the ground
    elevation above the altitude datum (plus any altimeter bias).
    """
    t_arrs   = [np.asarray(p['t_vals'],   dtype=float) for p in passes]
    y_arrs   = [np.asarray(p['y_ground'], dtype=float) for p in passes]
    alts     = [p.get('alt_m') for p in passes]
    alt_idx  = [i for i, a in enumerate(alts) if a is not None]
    mark_idx = [i for i, t in enumerate(t_arrs) if len(t) >= 3]
    use_alt  = len(alt_idx) >= 2

    if not use_alt and not mark_idx:
        raise ValueError(
            "Tilt is unconstrained: every pass has only 2 markers and fewer "
            "than 2 passes have an altitude reading.  Add altitude readings "
            "(EXIF or manual) or include a pass with 3+ markers.")

    def solve_all(phi_rad):
        """Per-pass (h_i, rmse_i) at one angle; None if any pass fails."""
        hs, rmses = [], []
        for p, t, y in zip(passes, t_arrs, y_arrs):
            h, _, rmse = solve_agl_given_tilt(t, y, phi_rad, p['f_px'])
            hs.append(h)
            rmses.append(rmse)
        return np.array(hs), np.array(rmses)

    def terms_at(phi_rad):
        hs, rmses = solve_all(phi_rad)
        if use_alt:
            h_a   = hs[alt_idx]
            d     = h_a - np.array([alts[i] for i in alt_idx])
            sig   = sigma_dalt_frac * h_a
            wts   = 1.0 / sig ** 2
            b     = float(np.sum(wts * d) / np.sum(wts))
            alt_res = (d - b) / sig
        else:
            b, alt_res = float('nan'), np.array([])
        marker_costs = np.array(
            [len(t_arrs[i]) * rmses[i] ** 2 / sigma_px ** 2 for i in mark_idx])
        return hs, b, alt_res, marker_costs

    phis  = np.linspace(phi_lo_deg, phi_hi_deg, n)
    costs = np.full(n, np.inf)
    for k, pd in enumerate(phis):
        try:
            _, _, alt_res, marker_costs = terms_at(math.radians(pd))
            costs[k] = profile_cost(alt_res, marker_costs)
        except ValueError:
            continue

    if not np.isfinite(costs).any():
        raise ValueError("Joint solve failed at every tilt in the range — "
                         "check marker order and spacing on each pass.")

    k_best   = int(np.nanargmin(np.where(np.isfinite(costs), costs, np.nan)))
    phi_best = float(phis[k_best])
    in_ci    = np.isfinite(costs) & (costs <= costs[k_best] + 1.0)
    ci_lo    = float(phis[in_ci].min())
    ci_hi    = float(phis[in_ci].max())

    phi_rad = math.radians(phi_best)
    hs, b, _, _ = terms_at(phi_rad)
    _, rmses    = solve_all(phi_rad)
    ws = [strip_width_given_pose(h, phi_rad, p['f_px'], p['t_near'], p['t_far'])
          for h, p in zip(hs, passes)]

    return dict(phis_deg=phis, costs=costs, phi_best_deg=phi_best,
                ci_lo_deg=ci_lo, ci_hi_deg=ci_hi, b_m=b,
                h_m=[float(h) for h in hs], w_m=[float(w) for w in ws],
                rmse_px=[float(r) for r in rmses],
                used_alt=use_alt, n_marker_passes=len(mark_idx))


# ── sensitivity canvas ─────────────────────────────────────────────────────────

class SensitivityCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(4, 3.4), tight_layout=True)
        super().__init__(fig)
        self.fig = fig
        self.ax  = fig.add_subplot(111)
        self._style()

    def _style(self, xlabel="Solved AGL (m)", ylabel="Strip Width (m)",
               title="Strip Width vs Solved AGL"):
        self.fig.patch.set_facecolor("#16161a")
        self.ax.set_facecolor("#1e1f26")
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#3c3e48")
        self.ax.tick_params(colors="#82859a", labelsize=11)
        self.ax.xaxis.label.set_color("#82859a")
        self.ax.yaxis.label.set_color("#82859a")
        self.ax.title.set_color("#dcdee6")
        self.ax.grid(True, color="#2a2b33", linewidth=0.6, zorder=0)
        self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)
        self.ax.set_title(title, fontsize=12, fontweight="bold")

    def _legend(self):
        self.ax.legend(fontsize=10, facecolor="#1e1f26",
                       labelcolor="#dcdee6", framealpha=0.9, loc="upper left")

    def refresh_plot(self, agl_m, widths, agl_nominal, w_nominal, logged=None):
        """Tilt-sweep curve for the current pass + all logged passes as points."""
        self.ax.clear()
        self._style()
        has_label = False
        if agl_m is not None:
            self.ax.plot(agl_m, widths, color="#00d2a0", lw=2.0, zorder=4)
            self.ax.axvline(agl_nominal, color="#ff7828", lw=1.4,
                            linestyle="--", zorder=3)
            if math.isfinite(w_nominal):
                self.ax.scatter([agl_nominal], [w_nominal], color="#ff7828",
                                s=55, zorder=5,
                                label=f"current {agl_nominal:.1f} m → {w_nominal:.1f} m")
                has_label = True
        if logged:
            la = [p[0] for p in logged]
            lw = [p[1] for p in logged]
            self.ax.scatter(la, lw, color="#50a0ff", s=45, zorder=6,
                            marker="D", label=f"logged ({len(logged)})")
            has_label = True
        if has_label:
            self._legend()
        self.draw()

    def refresh_profile(self, phis_deg, costs, phi_best, ci_lo, ci_hi):
        """Joint-solve view: cost vs mount tilt with the Δcost≤1 CI band."""
        self.ax.clear()
        self._style(xlabel="Mount tilt (°)", ylabel="Cost (χ²)",
                    title="Joint Tilt Profile — All Passes")
        ok = np.isfinite(costs)
        self.ax.plot(phis_deg[ok], costs[ok], color="#00d2a0", lw=2.0, zorder=4)
        self.ax.axvspan(ci_lo, ci_hi, color="#50a0ff", alpha=0.18, zorder=2,
                        label=f"CI {ci_lo:.1f}–{ci_hi:.1f}°")
        self.ax.axvline(phi_best, color="#ff7828", lw=1.4, linestyle="--",
                        zorder=3, label=f"tilt {phi_best:.2f}°")
        self._legend()
        self.draw()


# ── main window ────────────────────────────────────────────────────────────────

class SolverWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Strip Width Solver")
        self.resize(1600, 960)

        self._pts:        list         = []
        self._results:    list         = []
        self._last_calc:  dict | None  = None
        self._img_size:   tuple | None = None
        self._img_name    = ""
        self._alt_setting = False          # guards programmatic spin_alt sets
        self._last_sweep: tuple | None = None

        self._build_ui()
        self._build_menus()
        apply_app_style(self)

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(0)

        split = QSplitter(Qt.Horizontal)
        rl.addWidget(split)

        # ── Left: image ───────────────────────────────────────────────
        left = QWidget()
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.setSpacing(6)

        self.view = ImageView()
        self.view.point_added.connect(self._on_point_added)
        ll.addWidget(self.view, stretch=1)

        br = QHBoxLayout()
        br.setSpacing(8)
        self.btn_open  = self._btn("Open Image…", "Ctrl+O", self._open_image, primary=True)
        self.btn_place = self._btn("Place Markers  [P]", None, None)
        self.btn_place.setCheckable(True)
        self.btn_place.toggled.connect(self._toggle_place)
        self.btn_undo  = self._btn("Undo Last  [Z]", None, self._undo_last_marker)
        self.btn_clear = self._btn("Clear Markers", None, self._clear_markers)
        for b in (self.btn_open, self.btn_place, self.btn_undo, self.btn_clear):
            br.addWidget(b)
        ll.addLayout(br)
        split.addWidget(left)

        # ── Right: controls ───────────────────────────────────────────
        right = QWidget()
        rl2   = QVBoxLayout(right)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(10)

        params_row = QHBoxLayout()
        params_row.setSpacing(8)

        mg = QGroupBox("Measurement  (assumed pose)")
        mf = QFormLayout(mg)
        mf.setLabelAlignment(Qt.AlignRight)
        mf.setSpacing(8)

        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(0.1, 99999)
        self.spin_spacing.setValue(50.0)
        self.spin_spacing.setDecimals(2)
        self.spin_spacing.setSuffix(" m")
        mf.addRow("Marker spacing:", self.spin_spacing)

        self.spin_agl = QDoubleSpinBox()
        self.spin_agl.setRange(1, 99999)
        self.spin_agl.setValue(300.0)
        self.spin_agl.setDecimals(1)
        self.spin_agl.setSuffix(" ft")
        mf.addRow("Assumed AGL:", self.spin_agl)

        self.spin_alt = QDoubleSpinBox()
        self.spin_alt.setRange(0, 99999)
        self.spin_alt.setValue(0.0)
        self.spin_alt.setDecimals(1)
        self.spin_alt.setSuffix(" m")
        self.spin_alt.setSpecialValueText("none")
        self.spin_alt.setToolTip(
            "GPS/barometric altitude for this pass (MSL is fine — the joint "
            "solve uses only differences between passes).  Auto-filled from "
            "EXIF GPSAltitude when present; 'none' excludes the pass from "
            "the altitude term.")
        self.spin_alt.valueChanged.connect(self._alt_edited)
        self.lbl_alt_src = QLabel("—")
        self.lbl_alt_src.setFont(_font(BASE_FONT_SIZE - 2))
        alt_row = QHBoxLayout()
        alt_row.setSpacing(6)
        alt_row.addWidget(self.spin_alt, stretch=1)
        alt_row.addWidget(self.lbl_alt_src)
        mf.addRow("Altitude reading:", alt_row)

        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(0, 89)
        self.spin_tilt.setValue(45.0)
        self.spin_tilt.setDecimals(1)
        self.spin_tilt.setSuffix(" °")
        mf.addRow("Nominal tilt:", self.spin_tilt)

        self.spin_tilt_lo = QDoubleSpinBox()
        self.spin_tilt_lo.setRange(0, 89)
        self.spin_tilt_lo.setValue(42.0)
        self.spin_tilt_lo.setDecimals(1)
        self.spin_tilt_lo.setSuffix(" °")
        mf.addRow("Tilt min:", self.spin_tilt_lo)

        self.spin_tilt_hi = QDoubleSpinBox()
        self.spin_tilt_hi.setRange(0, 89)
        self.spin_tilt_hi.setValue(48.0)
        self.spin_tilt_hi.setDecimals(1)
        self.spin_tilt_hi.setSuffix(" °")
        mf.addRow("Tilt max:", self.spin_tilt_hi)

        self.spin_dalt_pct = QDoubleSpinBox()
        self.spin_dalt_pct.setRange(0.1, 20)
        self.spin_dalt_pct.setValue(2.0)
        self.spin_dalt_pct.setDecimals(1)
        self.spin_dalt_pct.setSuffix(" %")
        self.spin_dalt_pct.setToolTip(
            "1σ trust in the altitude *differences* between passes, "
            "as a fraction of AGL.  Used by Solve All Passes.")
        mf.addRow("Alt diff trust:", self.spin_dalt_pct)
        params_row.addWidget(mg)

        cg = QGroupBox("Camera  (trusted)")
        cf = QFormLayout(cg)
        cf.setLabelAlignment(Qt.AlignRight)
        cf.setSpacing(8)

        self.spin_focal = QDoubleSpinBox()
        self.spin_focal.setRange(1, 999)
        self.spin_focal.setValue(50.0)
        self.spin_focal.setDecimals(1)
        self.spin_focal.setSuffix(" mm")
        cf.addRow("Focal length:", self.spin_focal)

        self.spin_sensor_h = QDoubleSpinBox()
        self.spin_sensor_h.setRange(1, 200)
        self.spin_sensor_h.setValue(23.6)
        self.spin_sensor_h.setDecimals(2)
        self.spin_sensor_h.setSuffix(" mm")
        cf.addRow("Sensor height:", self.spin_sensor_h)
        params_row.addWidget(cg)

        rl2.addLayout(params_row)

        # Marker status
        sg  = QGroupBox("Placed Markers")
        svb = QVBoxLayout(sg)
        self.lbl_markers = QLabel("No markers placed.")
        self.lbl_markers.setFont(_font(BASE_FONT_SIZE - 1, mono=True))
        self.lbl_markers.setWordWrap(True)
        svb.addWidget(self.lbl_markers)
        rl2.addWidget(sg)

        # Result
        rg  = QGroupBox("Result")
        rvb = QVBoxLayout(rg)
        self.lbl_result = QLabel("–")
        self.lbl_result.setFont(_font(BASE_FONT_SIZE, bold=True, mono=True))
        self.lbl_result.setWordWrap(True)
        rvb.addWidget(self.lbl_result)

        cr = QHBoxLayout()
        cr.setSpacing(8)
        self.btn_calc = self._btn("Calculate  [Enter]", None, self._calculate, primary=True)
        self.btn_log  = self._btn("Log Result", None, self._log_result)
        self.btn_log.setEnabled(False)
        self.btn_export_markers = self._btn("Export Marker Set", None, self._export_marker_set)
        cr.addWidget(self.btn_calc)
        cr.addWidget(self.btn_log)
        cr.addWidget(self.btn_export_markers)
        rvb.addLayout(cr)
        rl2.addWidget(rg)

        # Logged passes table
        tg  = QGroupBox("Logged Passes")
        tvb = QVBoxLayout(tg)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "AGL solved (m)", "Δ AGL %", "Tilt (°)", "W (m)",
            "W min (m)", "W max (m)", "n", "Image",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setFont(_font(BASE_FONT_SIZE - 1, mono=True))
        tvb.addWidget(self.table)

        tb2 = QHBoxLayout()
        self.btn_del       = self._btn("Delete Row", None, self._delete_row)
        self.btn_solve_all = self._btn("Solve All Passes", None,
                                       self._solve_all_passes, primary=True)
        self.btn_solve_all.setEnabled(False)
        self.btn_solve_all.setToolTip(
            "Joint solve over all logged passes: shared mount tilt from "
            "altitude differences and/or 3+ marker passes.")
        self.btn_export = self._btn("Export CSV…", "Ctrl+S", self._export_csv)
        tb2.addWidget(self.btn_del)
        tb2.addWidget(self.btn_solve_all)
        tb2.addStretch()
        tb2.addWidget(self.btn_export)
        tvb.addLayout(tb2)
        rl2.addWidget(tg)
        self._make_collapsible(tg)

        # Sensitivity plot
        pg  = QGroupBox("Plot")
        pvb = QVBoxLayout(pg)
        self.canvas = SensitivityCanvas()
        self.canvas.setMinimumHeight(240)
        pvb.addWidget(self.canvas)
        rl2.addWidget(pg, stretch=1)
        self._make_collapsible(pg, layout=rl2, stretch=1)

        split.addWidget(right)
        split.setSizes([960, 640])

        self.status = QStatusBar()
        self.status.setFont(_font(BASE_FONT_SIZE - 1))
        self.setStatusBar(self.status)
        self.status.showMessage("Open an image to begin  (Ctrl+O)")

    def _btn(self, label, shortcut, slot, primary=False):
        b = QPushButton(label)
        b.setFont(_font(BASE_FONT_SIZE))
        if shortcut:
            b.setShortcut(shortcut)
        if slot:
            b.clicked.connect(slot)
        if primary:
            b.setProperty("primary", True)
        return b

    def _make_collapsible(self, group: QGroupBox, layout=None, stretch=0):
        """Checkable group box that collapses to its title bar when unchecked."""
        group.setCheckable(True)
        group.setChecked(True)

        def _toggle(on):
            for child in group.findChildren(QWidget):
                child.setVisible(on)
            if layout is not None and stretch:
                layout.setStretchFactor(group, stretch if on else 0)

        group.toggled.connect(_toggle)

    def _build_menus(self):
        m  = self.menuBar()
        m.setFont(_font(BASE_FONT_SIZE))
        fm = m.addMenu("&File")
        for label, sc, slot in [
            ("Open Image…",  "Ctrl+O", self._open_image),
            ("Export CSV…",  "Ctrl+S", self._export_csv),
            (None, None, None),
            ("Quit",         "Ctrl+Q", self.close),
        ]:
            if label is None:
                fm.addSeparator()
            else:
                a = QAction(label, self)
                a.setShortcut(sc)
                a.triggered.connect(slot)
                fm.addAction(a)
        hm   = m.addMenu("&Help")
        about = QAction("About…", self)
        about.triggered.connect(self._about)
        hm.addAction(about)

    # ── image ──────────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Survey Image", "",
            "Images (*.jpg *.jpeg *.tif *.tiff *.png *.bmp)")
        if not path:
            return
        result = self.view.load_image(path)
        if not result:
            QMessageBox.warning(self, "Load Error", f"Could not load:\n{path}")
            return
        self._img_size = result
        self._img_name = Path(path).name
        self._img_path = path
        self._pts.clear()
        self._last_calc = None
        self.btn_log.setEnabled(False)
        self.lbl_result.setText("–")
        self._refresh_marker_label()
        from osw_strip_width.exif import exif_gps_altitude_m

        alt = exif_gps_altitude_m(path)
        self._set_alt_reading(alt, "EXIF" if alt is not None else "—")
        alt_note = f"   alt {alt:.0f} m (EXIF)" if alt is not None else ""
        self.status.showMessage(
            f"{self._img_name}   {result[0]}×{result[1]} px{alt_note}   "
            "— press P and click each marker")

    def _set_alt_reading(self, alt_m, source: str):
        self._alt_setting = True
        self.spin_alt.setValue(alt_m if alt_m is not None else 0.0)
        self.lbl_alt_src.setText(source)
        self._alt_setting = False

    def _alt_edited(self, _value):
        if not self._alt_setting:
            self.lbl_alt_src.setText("manual")

    # ── markers ────────────────────────────────────────────────────────

    def _toggle_place(self, checked: bool):
        self.view.set_placing(checked)
        self.status.showMessage(
            "Click markers — bottom (nearest) to top (farthest) recommended.  "
            "Press P or Esc when done."
            if checked else "Marker placement stopped.")

    def _on_point_added(self, pt: QPointF):
        if not self._img_size:
            return
        self._pts.append(pt)
        self.view.add_marker(self._pts, pt, len(self._pts) - 1)
        self._refresh_marker_label()

    def _clear_markers(self):
        self._pts.clear()
        self.view.clear_overlays()
        self._last_calc = None
        self.lbl_result.setText("–")
        self.btn_log.setEnabled(False)
        self._refresh_marker_label()

    def _undo_last_marker(self):
        if not self._pts:
            return
        self._pts.pop()
        self.view.clear_overlays()
        for i, pt in enumerate(self._pts):
            self.view.add_marker(self._pts[:i + 1], pt, i)
        self._last_calc = None
        self.lbl_result.setText("–")
        self.btn_log.setEnabled(False)
        self._refresh_marker_label()

    def _refresh_marker_label(self):
        n = len(self._pts)
        if n == 0:
            self.lbl_markers.setText("No markers placed.")
            return
        coords = "  ".join(
            f"P{i+1}({p.x():.0f},{p.y():.0f})" for i, p in enumerate(self._pts))
        self.lbl_markers.setText(f"{n} marker(s):  {coords}")

    # ── calculation ────────────────────────────────────────────────────

    def _calculate(self):
        if not self._img_size:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        n = len(self._pts)
        if n < 2:
            QMessageBox.warning(self, "Too Few Markers", "Place at least 2 markers.")
            return

        from osw_strip_width.errors import StripWidthError
        from osw_strip_width.geometry import marker_line_t_values
        from osw_strip_width.solve import run_solve
        from osw_strip_width.solve_geometry import tilt_sweep
        from osw_strip_width.types import AltitudeReading

        from gui_adapter import build_marker_set, marker_set_canonical_text_and_hash

        img_w, img_h = self._img_size
        spacing_m = float(self.spin_spacing.value())
        pts_sorted = sorted(self._pts, key=lambda p: -p.y())
        pixel_tuples = [(p.x(), p.y()) for p in pts_sorted]

        marker_set = build_marker_set(
            image_path=self._img_path, image_width_px=img_w, image_height_px=img_h,
            pts_sorted=pixel_tuples, spacing_m=spacing_m,
            focal_length_mm=float(self.spin_focal.value()),
            sensor_height_mm=float(self.spin_sensor_h.value()),
        )
        _canonical_text, marker_set_sha256 = marker_set_canonical_text_and_hash(marker_set)
        self._last_marker_set = marker_set

        altitude = None
        alt_value = float(self.spin_alt.value())
        if alt_value:
            altitude = AltitudeReading(
                value_m=alt_value, reference="unknown", vertical_datum="unknown",
                source=self.lbl_alt_src.text(), measured_at="image-capture-time",
                uncertainty_m=None, ground_elevation_treatment="not applied",
            )

        tilt_lo = float(self.spin_tilt_lo.value())
        tilt_hi = float(self.spin_tilt_hi.value())
        if tilt_lo > tilt_hi:
            tilt_lo, tilt_hi = tilt_hi, tilt_lo

        try:
            estimate = run_solve(
                marker_set,
                focal_length_mm=float(self.spin_focal.value()),
                sensor_height_mm=float(self.spin_sensor_h.value()),
                tilt_nom_deg=float(self.spin_tilt.value()),
                tilt_lo_deg=tilt_lo, tilt_hi_deg=tilt_hi,
                agl_ft=float(self.spin_agl.value()),
                altitude=altitude,
                marker_set_sha256=marker_set_sha256,
                package_version="0.1.0",
                algorithm_version="osw_strip_width.solve_geometry@0.1.0",
            )
        except StripWidthError as exc:
            QMessageBox.critical(self, "Solve Failed", str(exc))
            return

        self._last_estimate = estimate

        # Recompute the full tilt-sweep curve directly from the engine layer
        # for the plot. This is presentation-only derived data — the public
        # GeometryEstimate deliberately carries only the summary envelope
        # (swath_sensitivity_envelope_m), not the full array (spec §10.1).
        px_arr = np.array(pixel_tuples, dtype=float)
        t_vals, t_near, t_far, line_dir, centroid, t_pp = marker_line_t_values(
            px_arr, float(img_h)
        )
        y_ground = np.array([m.ground_distance_m for m in marker_set.markers], dtype=float)
        f_px = float(self.spin_focal.value()) / float(self.spin_sensor_h.value()) * img_h
        _phis, hs, ws = tilt_sweep(t_vals, y_ground, f_px, t_near, t_far, tilt_lo, tilt_hi)
        self._last_sweep = (hs, ws, estimate.height_agl_m, estimate.camera_frame_swath_m)
        self._update_pass_plot()

        self.view.clear_overlays()
        for i, p in enumerate(pts_sorted):
            self.view.add_marker([pts_sorted[j] for j in range(i + 1)], p, i)
        self.view.draw_fit_overlay(
            pts_sorted, line_dir, centroid,
            t_near + t_pp, t_far + t_pp, estimate.camera_frame_swath_m, img_w, img_h,
        )

        lo, hi = estimate.swath_sensitivity_envelope_m or (None, None)
        result_lines = [
            f"Solved AGL (tilt {estimate.tilt_deg:.1f}°) :  {estimate.height_agl_m} m",
            f"Δ vs assumed AGL           :  {estimate.agl_delta_pct}%",
            f"Camera-frame swath         :  {estimate.camera_frame_swath_m:.2f} m",
            f"Sensitivity envelope       :  {lo} – {hi} m",
            f"Fit RMSE                   :  {estimate.fit_rmse_px} px",
        ]
        for w in estimate.warnings:
            result_lines.append(f"NOTE [{w.code}]: {w.message}")
        self.lbl_result.setText("\n".join(result_lines))
        self.btn_log.setEnabled(True)

    def _logged_points(self):
        return [(e.height_agl_m, e.camera_frame_swath_m) for e in self._results
                if e.height_agl_m is not None]

    def _update_pass_plot(self):
        sweep = self._last_sweep or (None, None, None, float("nan"))
        self.canvas.refresh_plot(*sweep, logged=self._logged_points())

    def _log_result(self):
        if not getattr(self, "_last_estimate", None):
            return
        estimate = self._last_estimate
        self._results.append(estimate)

        def _fmt(v, fmt):
            return fmt % v if v is not None else "n/a"

        lo, hi = estimate.swath_sensitivity_envelope_m or (None, None)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([
            _fmt(estimate.height_agl_m, "%.1f"),
            _fmt(estimate.agl_delta_pct, "%+.1f%%"),
            _fmt(estimate.tilt_deg, "%.1f"),
            f"{estimate.camera_frame_swath_m:.2f}",
            _fmt(lo, "%.2f"),
            _fmt(hi, "%.2f"),
            str(estimate.n_markers),
            estimate.context.image_id,
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        self.btn_log.setEnabled(False)
        self.btn_solve_all.setEnabled(len(self._results) >= 2)
        self._update_pass_plot()
        self.status.showMessage(f"Logged pass #{len(self._results)}.  Load next image for next pass.")

    # ── multi-pass joint solve ─────────────────────────────────────────

    def _solve_all_passes(self):
        from osw_strip_width.errors import StripWidthError
        from osw_strip_width.joint import run_joint_solve
        from osw_strip_width.solve_geometry import joint_tilt_profile

        passes = [e for e in self._results if e.solver_state is not None]
        if len(passes) < 2:
            QMessageBox.information(self, "Joint Solve", "Log at least 2 passes first.")
            return

        tilt_lo = float(self.spin_tilt_lo.value())
        tilt_hi = float(self.spin_tilt_hi.value())
        if tilt_lo > tilt_hi:
            tilt_lo, tilt_hi = tilt_hi, tilt_lo
        sigma_dalt_frac = float(self.spin_dalt_pct.value()) / 100.0

        # In-memory passes have no on-disk geometry-estimate file yet unless the
        # operator exported one. A prior draft used a placeholder string like
        # "in-memory-pass-0" under the estimate_sha256 field — that stopped it
        # from reusing the marker-set hash, but it wrote a non-hash into a field
        # named _sha256, which is its own contract violation. GeometryEstimate
        # serializes deterministically, so hash the canonical bytes even though
        # nothing was written to disk. Use io.py's canonical_text_and_hash, not
        # a local write_artifact(e, None) + manual hash — an earlier revision of
        # this exact line hashed text without the trailing newline
        # write_artifact(e, path) actually persists, so the in-memory hash
        # disagreed with the hash a reload of an exported file would compute.
        # canonical_text_and_hash is the one place that byte convention lives.
        from osw_strip_width.io import canonical_text_and_hash

        estimate_hashes = [canonical_text_and_hash(e)[1] for e in passes]

        try:
            result = run_joint_solve(
                passes, estimate_hashes=estimate_hashes,
                tilt_lo_deg=tilt_lo, tilt_hi_deg=tilt_hi, sigma_dalt_frac=sigma_dalt_frac,
            )
        except StripWidthError as exc:
            QMessageBox.critical(self, "Joint Solve Failed", str(exc))
            return

        lo, hi = result.tilt_profile_cost_interval_deg
        lines = [
            f"JOINT SOLVE — {len(passes)} passes  ({', '.join(result.evidence_sources)})",
            f"Mount tilt              :  {result.mount_tilt_deg:.2f}°  (cost interval {lo:.2f} – {hi:.2f}°)",
        ]
        if result.ground_offset_m is not None:
            lines.append(f"Ground elev (alt − AGL) :  {-result.ground_offset_m:+.1f} m")
        for p, image_id in zip(result.per_pass, [e.context.image_id for e in passes]):
            lines.append(f"  {image_id:<20}  h {p['height_agl_m']:6.1f} m   W {p['camera_frame_swath_m']:7.2f} m")
        for note in result.known_limitations:
            lines.append(f"NOTE: {note}")
        self.lbl_result.setText("\n".join(lines))

        # Recompute the full cost-vs-tilt profile curve directly from the
        # engine layer for the plot — same "public artifact keeps the summary,
        # GUI recomputes the array for display" pattern as _calculate above.
        raw_passes = [
            {
                "t_vals": e.solver_state["t_vals"], "y_ground": e.solver_state["y_ground"],
                "f_px": e.solver_state["f_px"], "t_near": e.solver_state["t_near"],
                "t_far": e.solver_state["t_far"],
                "alt_m": e.altitude.value_m if e.altitude is not None else None,
            }
            for e in passes
        ]
        profile = joint_tilt_profile(raw_passes, tilt_lo, tilt_hi, sigma_dalt_frac=sigma_dalt_frac)
        self.canvas.refresh_profile(
            profile["phis_deg"], profile["costs"],
            profile["phi_best_deg"], profile["ci_lo_deg"], profile["ci_hi_deg"],
        )
        self.status.showMessage(
            f"Joint solve: mount tilt {result.mount_tilt_deg:.2f}° "
            f"[{lo:.2f} – {hi:.2f}°] over {len(passes)} passes"
        )

    # ── table ──────────────────────────────────────────────────────────

    def _delete_row(self):
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            if row < len(self._results):
                self._results.pop(row)
        self.btn_solve_all.setEnabled(len(self._results) >= 2)
        self._update_pass_plot()

    # ── export ─────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            QMessageBox.information(self, "Export", "No results logged yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "strip_width_solved.csv", "CSV (*.csv)")
        if not path:
            return
        # self._results now holds GeometryEstimate objects, not legacy
        # dicts — csv.DictWriter.writerows(self._results) would crash the
        # first time this ran (GeometryEstimate has no .keys()), the same
        # class of "consumer of per-pass state never updated" bug this
        # task's expansion exists to catch elsewhere. Column set also
        # changed for the same reason as strip_geom.py's CSV export:
        # spacing_m/img_w/img_h are dropped because a marker set is no
        # longer assumed uniformly spaced.
        fields = ["agl_solved_m", "agl_solved_ft", "agl_assumed_ft",
                  "delta_agl_pct", "tilt_nom_deg", "tilt_lo_deg", "tilt_hi_deg",
                  "strip_m", "strip_min_m", "strip_max_m", "geom_strip",
                  "gsd_near", "rmse_px", "n_markers", "alt_m", "alt_source", "image"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for e in self._results:
                lo, hi = e.swath_sensitivity_envelope_m or (None, None)
                tilt_lo, tilt_hi = e.tilt_assumed_range_deg or (None, None)
                w.writerow({
                    "agl_solved_m": e.height_agl_m, "agl_solved_ft": e.height_agl_ft,
                    "agl_assumed_ft": e.resolved_parameters.get("agl_ft"),
                    "delta_agl_pct": e.agl_delta_pct, "tilt_nom_deg": e.tilt_deg,
                    "tilt_lo_deg": tilt_lo, "tilt_hi_deg": tilt_hi,
                    "strip_m": e.camera_frame_swath_m, "strip_min_m": lo, "strip_max_m": hi,
                    "geom_strip": e.geometric_cross_check_swath_m,
                    "gsd_near": e.gsd_near_cm_px, "rmse_px": e.fit_rmse_px,
                    "n_markers": e.n_markers,
                    "alt_m": e.altitude.value_m if e.altitude is not None else None,
                    "alt_source": e.altitude.source if e.altitude is not None else None,
                    "image": e.context.image_id,
                })
        self.status.showMessage(f"Exported {len(self._results)} rows -> {path}")

    def _export_marker_set(self):
        if not getattr(self, "_last_marker_set", None):
            QMessageBox.information(self, "Export", "Calculate a result first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Marker Set", "marker_set.json", "JSON (*.json)")
        if not path:
            return
        from gui_adapter import export_marker_set

        export_marker_set(self._last_marker_set, path)
        self.status.showMessage(f"Exported marker set -> {path}")

    # ── keyboard ───────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_P:
            self.btn_place.setChecked(not self.btn_place.isChecked())
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self._calculate()
        elif k == Qt.Key_Z:
            self._undo_last_marker()
        elif k == Qt.Key_Escape and self.btn_place.isChecked():
            self.btn_place.setChecked(False)
        else:
            super().keyPressEvent(event)

    # ── about ──────────────────────────────────────────────────────────

    def _about(self):
        QMessageBox.about(self, "Strip Width Solver",
            "<b>Strip Width Solver</b><br>"
            "Oblique aerial wildlife surveys · East Africa<br><br>"
            "<b>Geometry</b><br>"
            "Assumes tilt (with an uncertainty range) and trusts the focal "
            "length; two or more markers of known interval then <i>solve</i> "
            "the height AGL in closed form, and strip width follows from the "
            "pinhole model.  The assumed AGL is used only as a sanity check "
            f"(warn beyond ±{AGL_TOL_PCT:.0f}%).  Sweeping tilt across its "
            "range gives the strip-width uncertainty band.<br><br>"
            "<b>Workflow</b><br>"
            "1. Open image → set spacing, assumed AGL, tilt + range, camera<br>"
            "2. Press <b>P</b> → click 2+ visible markers<br>"
            "3. Press <b>Enter</b> → solved AGL, strip width + band, overlay<br>"
            "4. Log Result → load next image<br><br>"
            "<b>Notes</b><br>"
            "• Markers are sorted by y-position (bottom = nearest); click "
            "order doesn't matter.<br>"
            "• 2 markers are exactly determined at the assumed tilt; 3+ give "
            "a least-squares fit with a meaningful RMSE.<br>"
            "• Companion tool: strip_geom.py solves AGL <i>and</i> tilt from "
            "many markers (calibration flights).<br><br>"
            "<b>Solve All Passes (joint solve)</b><br>"
            "With 2+ logged passes, scans the mount tilt over the entered "
            "range; at each angle every pass is re-solved and scored.  Tilt "
            "is constrained by (a) altitude differences between passes "
            "(EXIF GPSAltitude, auto-read on open, or manual — a free common "
            "offset absorbs datum and ground elevation), and/or (b) passes "
            "with 3+ markers.  With only 2-marker passes and no altitudes, "
            "tilt is mathematically unconstrained and the solve refuses.  "
            "Reports mount tilt with a Δχ²=1 confidence interval, the common "
            "altitude offset, and refined per-pass AGL / strip widths.<br>"
        )


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
    app = QApplication(sys.argv)
    app.setFont(_font(BASE_FONT_SIZE))
    win = SolverWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
