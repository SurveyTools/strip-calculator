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
    geometric_strip_width, marker_line_t_values,
)

AGL_TOL_PCT = 10.0   # warn when solved AGL differs from assumed by more than this


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


# ── sensitivity canvas ─────────────────────────────────────────────────────────

class SensitivityCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(4, 3.4), tight_layout=True)
        super().__init__(fig)
        self.fig = fig
        self.ax  = fig.add_subplot(111)
        self._style()

    def _style(self):
        self.fig.patch.set_facecolor("#16161a")
        self.ax.set_facecolor("#1e1f26")
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#3c3e48")
        self.ax.tick_params(colors="#82859a", labelsize=11)
        self.ax.xaxis.label.set_color("#82859a")
        self.ax.yaxis.label.set_color("#82859a")
        self.ax.title.set_color("#dcdee6")
        self.ax.grid(True, color="#2a2b33", linewidth=0.6, zorder=0)
        self.ax.set_xlabel("Solved AGL (m)", fontsize=11)
        self.ax.set_ylabel("Strip Width (m)", fontsize=11)
        self.ax.set_title("Strip Width vs Solved AGL", fontsize=12,
                          fontweight="bold")

    def refresh_plot(self, agl_m, widths, agl_nominal, w_nominal):
        self.ax.clear()
        self._style()
        if agl_m is None:
            self.draw()
            return
        self.ax.plot(agl_m, widths, color="#00d2a0", lw=2.0, zorder=4)
        self.ax.axvline(agl_nominal, color="#ff7828", lw=1.4,
                        linestyle="--", zorder=3)
        if math.isfinite(w_nominal):
            self.ax.scatter([agl_nominal], [w_nominal], color="#ff7828",
                            s=55, zorder=5,
                            label=f"nominal {agl_nominal:.1f} m → {w_nominal:.1f} m")
            self.ax.legend(fontsize=10, facecolor="#1e1f26",
                           labelcolor="#dcdee6", framealpha=0.9,
                           loc="upper left")
        self.draw()


# ── main window ────────────────────────────────────────────────────────────────

class SolverWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Strip Width Solver")
        self.resize(1600, 960)

        self._pts:       list         = []
        self._results:   list         = []
        self._last_calc: dict | None  = None
        self._img_size:  tuple | None = None
        self._img_name   = ""

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
        cr.addWidget(self.btn_calc)
        cr.addWidget(self.btn_log)
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
        self.btn_del    = self._btn("Delete Row", None, self._delete_row)
        self.btn_export = self._btn("Export CSV…", "Ctrl+S", self._export_csv)
        tb2.addWidget(self.btn_del)
        tb2.addStretch()
        tb2.addWidget(self.btn_export)
        tvb.addLayout(tb2)
        rl2.addWidget(tg)

        # Sensitivity plot
        self.canvas = SensitivityCanvas()
        self.canvas.setMinimumHeight(240)
        rl2.addWidget(self.canvas, stretch=1)

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
        self._pts.clear()
        self._last_calc = None
        self.btn_log.setEnabled(False)
        self.lbl_result.setText("–")
        self._refresh_marker_label()
        self.status.showMessage(
            f"{self._img_name}   {result[0]}×{result[1]} px   "
            "— press P and click each marker")

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

        img_w, img_h = self._img_size
        spacing_m    = float(self.spin_spacing.value())
        agl_ft       = float(self.spin_agl.value())
        agl_m        = agl_ft / 3.28084
        tilt_nom     = float(self.spin_tilt.value())
        tilt_lo      = float(self.spin_tilt_lo.value())
        tilt_hi      = float(self.spin_tilt_hi.value())
        if tilt_lo > tilt_hi:
            tilt_lo, tilt_hi = tilt_hi, tilt_lo
        f_px = (float(self.spin_focal.value())
                / float(self.spin_sensor_h.value()) * img_h)

        # Sort by y descending: index 0 = bottom (nearest, highest y value)
        pts_sorted = sorted(self._pts, key=lambda p: -p.y())
        px_arr     = np.array([[p.x(), p.y()] for p in pts_sorted], dtype=float)

        try:
            t_vals, t_near, t_far, line_dir, centroid, t_pp = \
                marker_line_t_values(px_arr, img_h)
        except ValueError as exc:
            QMessageBox.critical(self, "Degenerate Line", str(exc))
            return

        y_ground = np.arange(n, dtype=float) * spacing_m
        phi_nom  = math.radians(tilt_nom)

        try:
            h_sol, _y0, rmse_px = solve_agl_given_tilt(
                t_vals, y_ground, phi_nom, f_px)
            w_nom = strip_width_given_pose(h_sol, phi_nom, f_px, t_near, t_far)
            gsd_near = gsd_at_pixel(t_near, h_sol, phi_nom, f_px)
        except NotImplementedError:
            QMessageBox.critical(
                self, "Not Implemented",
                "ground_distance_from_pixel() is not implemented yet — "
                "see the TODO at the top of strip_solver.py.")
            return
        except ValueError as exc:
            QMessageBox.critical(self, "Solve Failed", str(exc))
            return

        # Tilt uncertainty band
        phis, hs, ws = tilt_sweep(t_vals, y_ground, f_px, t_near, t_far,
                                  tilt_lo, tilt_hi)
        w_min = float(np.nanmin(ws)) if np.isfinite(ws).any() else float('nan')
        w_max = float(np.nanmax(ws)) if np.isfinite(ws).any() else float('nan')

        h_sol_ft  = h_sol * 3.28084
        delta_agl = (h_sol - agl_m) / agl_m * 100.0

        # Geometric cross-check at the *assumed* pose
        geom_strip = geometric_strip_width(
            f_mm        = float(self.spin_focal.value()),
            sensor_h_mm = float(self.spin_sensor_h.value()),
            tilt_deg    = tilt_nom,
            agl_ft      = agl_ft,
        )

        # Overlay
        self.view.clear_overlays()
        for i, p in enumerate(pts_sorted):
            self.view.add_marker([pts_sorted[j] for j in range(i+1)], p, i)
        self.view.draw_fit_overlay(
            pts_sorted, line_dir, centroid,
            t_near + t_pp, t_far + t_pp, w_nom, img_w, img_h)

        self.canvas.refresh_plot(hs, ws, h_sol, w_nom)

        self._last_calc = dict(
            agl_assumed_ft = agl_ft,
            agl_solved_m   = round(h_sol, 1),
            agl_solved_ft  = round(h_sol_ft, 0),
            delta_agl_pct  = round(delta_agl, 2),
            tilt_nom_deg   = tilt_nom,
            tilt_lo_deg    = tilt_lo,
            tilt_hi_deg    = tilt_hi,
            strip_m        = round(w_nom, 3),
            strip_min_m    = round(w_min, 3) if math.isfinite(w_min) else float('nan'),
            strip_max_m    = round(w_max, 3) if math.isfinite(w_max) else float('nan'),
            geom_strip     = round(geom_strip, 3),
            gsd_near       = round(gsd_near, 4),
            rmse_px        = round(rmse_px, 2),
            n_markers      = n,
            spacing_m      = spacing_m,
            img_w          = img_w,
            img_h          = img_h,
            image          = self._img_name,
        )

        result_lines = [
            f"{f'Solved AGL (tilt {tilt_nom:.1f}°)':<24}:  {h_sol:.1f} m  "
            f"({h_sol_ft:.0f} ft)",
            f"{'Δ vs assumed AGL':<24}:  {delta_agl:+.1f}%",
            f"{'Strip width (nominal)':<24}:  {w_nom:.2f} m",
            f"{f'Strip width ({tilt_lo:.0f}–{tilt_hi:.0f}°)':<24}:  "
            f"{w_min:.2f} – {w_max:.2f} m",
            f"{'Geometric (assumed pose)':<24}:  {geom_strip:.2f} m",
            f"{'GSD near edge':<24}:  {gsd_near:.2f} cm/px",
            f"{'Fit RMSE':<24}:  {rmse_px:.1f} px  "
            f"({n} markers, {n-1} intervals × {spacing_m:.1f} m)",
        ]
        if abs(delta_agl) > AGL_TOL_PCT:
            result_lines.append(
                f"WARNING: solved AGL differs from assumed by "
                f"{delta_agl:+.1f}% (> {AGL_TOL_PCT:.0f}%) — check AGL, "
                "tilt or marker spacing.")
        if n == 2:
            result_lines.append(
                "NOTE: 2 markers — exactly determined at the assumed tilt; "
                "RMSE is not meaningful.")
        self.lbl_result.setText("\n".join(result_lines))
        self.btn_log.setEnabled(True)
        self.status.showMessage(
            f"AGL solved = {h_sol:.1f} m ({delta_agl:+.1f}%)  |  "
            f"strip = {w_nom:.2f} m  "
            f"[{w_min:.2f} – {w_max:.2f} m over {tilt_lo:.0f}–{tilt_hi:.0f}°]")

    # ── logging ────────────────────────────────────────────────────────

    def _log_result(self):
        if not self._last_calc:
            return
        d = self._last_calc
        self._results.append(d.copy())

        def _fmtv(v, fmt):
            return fmt % v if math.isfinite(v) else "n/a"

        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([
            f"{d['agl_solved_m']:.1f}",
            f"{d['delta_agl_pct']:+.1f}%",
            f"{d['tilt_nom_deg']:.1f}",
            f"{d['strip_m']:.2f}",
            _fmtv(d['strip_min_m'], "%.2f"),
            _fmtv(d['strip_max_m'], "%.2f"),
            str(d['n_markers']),
            d['image'],
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        self.btn_log.setEnabled(False)
        self.status.showMessage(
            f"Logged pass #{len(self._results)}.  Load next image for next pass.")

    # ── table ──────────────────────────────────────────────────────────

    def _delete_row(self):
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            if row < len(self._results):
                self._results.pop(row)

    # ── export ─────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            QMessageBox.information(self, "Export", "No results logged yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "strip_width_solved.csv", "CSV (*.csv)")
        if not path:
            return
        fields = ["agl_solved_m", "agl_solved_ft", "agl_assumed_ft",
                  "delta_agl_pct", "tilt_nom_deg", "tilt_lo_deg", "tilt_hi_deg",
                  "strip_m", "strip_min_m", "strip_max_m", "geom_strip",
                  "gsd_near", "rmse_px", "n_markers", "spacing_m",
                  "img_w", "img_h", "image"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(self._results)
        self.status.showMessage(f"Exported {len(self._results)} rows -> {path}")

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
            "many markers (calibration flights).<br>"
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
