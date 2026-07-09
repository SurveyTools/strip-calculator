#!/usr/bin/env python3
"""
Strip Width Calibration Tool  v3
─────────────────────────────────
Oblique aerial wildlife surveys — East Africa

Geometry
--------
Markers are laid perpendicular to the flight line.  The oblique camera
sees them running bottom (near) to top (far) in the image.  Because the
camera is tilted, equal ground spacings project to *unequal* pixel
intervals — compressed toward the far (top) edge.

The correct model is a 1-D projective (perspective) mapping from ground
distance y  →  pixel coordinate along the marker line:

    t = (a·y + b) / (c·y + 1)        [3 parameters]

The model has 3 unknowns.  n=2 markers gives an underdetermined fit
(strip-width extrapolation unreliable, RMSE always 0); n=3 is exactly
determined; n≥4 is overdetermined and gives a meaningful RMSE.  We:

  1. Fit a line through all clicked pixel positions (SVD/PCA) to handle
     images where the marker line runs at an angle.
  2. Project each clicked point onto that line  ->  scalar t_i.
  3. Fit the projective model to  (y_i, t_i)  pairs  (y_i = i*d).
  4. Find t at the four image corners projected onto the same line
     direction  ->  t_near (bottom), t_far (top).
  5. Invert the model to get ground distances at t_near and t_far.
  6. Strip width = y(t_far) - y(t_near).

Camera parameters (f, sensor height, tilt) are shown as a geometric
estimate alongside the empirical result for cross-checking.

Workflow
--------
1.  Open a calibration-flight image.
2.  Set marker spacing (m), AGL, and camera params.
3.  Press P and click every visible marker.
4.  Press Enter -> projective fit, results, overlay.
5.  Log Result, then open the next pass.
6.  Regression plot (AGL vs strip width) updates after each logged pass.
"""

import sys
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QSplitter, QGroupBox, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QFileDialog,
    QMessageBox, QAction, QStatusBar,
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QFont, QPalette,
    QWheelEvent,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── palette ────────────────────────────────────────────────────────────────────
BG       = QColor(22, 22, 25)
SURFACE  = QColor(32, 33, 38)
SURFACE2 = QColor(42, 43, 50)
BORDER   = QColor(60, 62, 72)
TEXT     = QColor(220, 222, 230)
DIM      = QColor(130, 133, 148)
ORANGE   = QColor(255, 120, 40)
TEAL     = QColor(0, 210, 160)
BLUE     = QColor(80, 160, 255)

BASE_FONT_SIZE = 12   # pt — bump higher if still small on your display


def _font(size=None, bold=False, mono=False):
    sz = size if size is not None else BASE_FONT_SIZE
    f  = QFont("Courier New" if mono else "Helvetica Neue", sz)
    if bold:
        f.setWeight(QFont.DemiBold)
    return f


# ── projective geometry ────────────────────────────────────────────────────────

def fit_projective_1d(y_ground: np.ndarray, t_pixel: np.ndarray):
    """
    Fit  t = (a*y + b) / (c*y + 1)  to (y_ground, t_pixel) pairs.
    Returns (a, b, c) via nonlinear least squares seeded from a linear DLT.
    """
    n  = len(y_ground)
    A  = np.column_stack([y_ground, np.ones(n), -y_ground * t_pixel])
    p0 = np.linalg.lstsq(A, t_pixel, rcond=None)[0]

    def residuals(p):
        a, b, c = p
        denom = c * y_ground + 1.0
        return (a * y_ground + b) / denom - t_pixel

    result = least_squares(residuals, p0, method='lm', max_nfev=2000)
    return result.x[0], result.x[1], result.x[2]


def invert_projective(t_val: float, a: float, b: float, c: float) -> float:
    """Invert  t = (a*y + b)/(c*y + 1)  ->  y = (t - b)/(a - c*t)"""
    denom = a - c * t_val
    if abs(denom) < 1e-12:
        raise ValueError("Projective inversion degenerate at this t value.")
    return (t_val - b) / denom


def geometric_strip_width(f_mm, sensor_h_mm, tilt_deg, agl_ft):
    """Analytical strip width (flat-ground pinhole model), metres."""
    h   = agl_ft / 3.28084
    phi = math.radians(tilt_deg)
    A_v = 2 * math.atan(sensor_h_mm / (2 * f_mm))
    near = phi - A_v / 2
    far  = phi + A_v / 2
    if far >= math.pi / 2:
        far = math.pi / 2 - 1e-4
    d_near = h * math.tan(near) if near > 0 else 0.0
    d_far  = h * math.tan(far)
    return d_far - d_near


def fit_camera_params(a: float, b: float, c: float, f_px: float):
    """
    Recover height AGL (m) and tilt angle (° from nadir) from projective fit.

    REQUIRES t measured from the principal point (image center row projected
    onto the marker line).  The y (ground) origin may be arbitrary — the
    formulas below are invariant to a shift y -> y - y0:

    Pinhole camera at height h, tilt phi from nadir, ground distance Y from
    nadir:  t = f_px*(Y - h*tan(phi)) / (h + Y*tan(phi)).  With Y = y + y0
    (y0 = unknown offset of the first marker from nadir) this is still a
    projective map t = (a*y + b)/(c*y + 1), and:

        h       = f_px * (a - b*c) / (a^2 + c^2 * f_px^2)   [y0-invariant]
        tan phi = c * f_px / a                              [y0-invariant]

    Returns (h_m, tilt_deg) or (nan, nan) if parameters are unphysical.
    """
    if a < 1e-9:
        return float('nan'), float('nan')
    h_m = f_px * (a - b * c) / (a * a + c * c * f_px * f_px)
    if not math.isfinite(h_m) or h_m <= 0:
        return float('nan'), float('nan')
    tilt_deg = math.degrees(math.atan(c * f_px / a))
    return h_m, tilt_deg


# ── image view ─────────────────────────────────────────────────────────────────

class ImageView(QGraphicsView):
    point_added = pyqtSignal(QPointF)

    def __init__(self):
        super().__init__()
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(14, 14, 16)))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QGraphicsView.NoFrame)

        self._placing  = False
        self._img_w    = 0
        self._img_h    = 0
        self._overlays = []

    def load_image(self, path: str):
        pm = QPixmap(path)
        if pm.isNull():
            return None
        self.scene().clear()
        self._overlays.clear()
        self.scene().addPixmap(pm)
        self.scene().setSceneRect(QRectF(pm.rect()))
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._img_w, self._img_h = pm.width(), pm.height()
        return self._img_w, self._img_h

    def set_placing(self, enabled: bool):
        self._placing = enabled
        self.setDragMode(
            QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def _lw(self):
        return max(1.5, self._img_w / 1800)

    def add_marker(self, pts: list, pt: QPointF, idx: int):
        r  = max(8, self._img_w / 320)
        lw = self._lw()
        s  = self.scene()

        el = s.addEllipse(pt.x()-r, pt.y()-r, 2*r, 2*r,
                          QPen(QColor(255,255,255), lw), QBrush(ORANGE))
        el.setZValue(10)
        self._overlays.append(el)

        fsize = max(10, int(self._img_w / 380))
        lbl   = s.addText(str(idx + 1), _font(fsize, bold=True))
        lbl.setDefaultTextColor(QColor(255, 255, 255))
        lbl.setPos(pt.x() + r + 3, pt.y() - r - 3)
        lbl.setZValue(11)
        self._overlays.append(lbl)

        if len(pts) >= 2:
            p0, p1 = pts[-2], pts[-1]
            ln = s.addLine(p0.x(), p0.y(), p1.x(), p1.y(),
                           QPen(QColor(255,200,60,160), lw, Qt.DashLine))
            ln.setZValue(9)
            self._overlays.append(ln)

    def draw_fit_overlay(self, pts_sorted, line_dir, centroid,
                          t_near, t_far, strip_m, img_w, img_h):
        s  = self.scene()
        lw = self._lw()

        # Fitted curve (polyline along the marker-line direction)
        t_vals = np.linspace(t_near, t_far, 80)
        prev   = None
        cpen   = QPen(TEAL, lw * 1.6)
        for tv in t_vals:
            px = float(centroid[0] + tv * line_dir[0])
            py = float(centroid[1] + tv * line_dir[1])
            px = max(0.0, min(float(img_w), px))
            py = max(0.0, min(float(img_h), py))
            if prev is not None:
                it = s.addLine(prev[0], prev[1], px, py, cpen)
                it.setZValue(12)
                self._overlays.append(it)
            prev = (px, py)

        # Near / far horizontal extent lines
        for tv in (t_near, t_far):
            ey = float(centroid[1] + tv * line_dir[1])
            ey = max(0.0, min(float(img_h), ey))
            it = s.addLine(0, ey, img_w, ey,
                           QPen(TEAL, lw, Qt.DashLine))
            it.setZValue(12)
            self._overlays.append(it)
            tw = max(14, img_w / 55)
            for x in (0, img_w):
                tk = s.addLine(x - tw, ey, x + tw, ey,
                               QPen(TEAL, lw * 1.4))
                tk.setZValue(13)
                self._overlays.append(tk)

        # Vertical span bar on right margin — near = bottom edge, far = top edge
        near_y = float(img_h)
        far_y  = 0.0
        near_y = max(0.0, min(float(img_h), near_y))
        far_y  = max(0.0, min(float(img_h), far_y))
        bx     = img_w * 0.93
        tw2    = max(12, img_w / 70)
        bpen   = QPen(TEAL, lw * 1.6)
        s.addLine(bx, near_y, bx, far_y, bpen).setZValue(13)
        for y in (near_y, far_y):
            s.addLine(bx - tw2, y, bx + tw2, y, bpen).setZValue(13)

        fsize = max(11, int(img_w / 340))
        txt   = s.addText(f"{strip_m:.1f} m", _font(fsize, bold=True))
        txt.setDefaultTextColor(TEAL)
        txt.setTransformOriginPoint(0, 0)
        txt.setRotation(-90)
        tw3  = txt.boundingRect().width()
        th3  = txt.boundingRect().height()
        txt.setPos(bx + th3 + 5, (near_y + far_y) / 2 + tw3 / 2)
        txt.setZValue(14)
        self._overlays.append(txt)

    def clear_overlays(self):
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.08 if event.angleDelta().y() > 0 else 1 / 1.08
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self._placing:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if self._placing and event.button() == Qt.LeftButton and self._img_w:
            sp = self.mapToScene(event.pos())
            sp.setX(max(0.0, min(sp.x(), float(self._img_w))))
            sp.setY(max(0.0, min(sp.y(), float(self._img_h))))
            self.point_added.emit(sp)
        else:
            super().mousePressEvent(event)


# ── regression canvas ──────────────────────────────────────────────────────────

class RegressCanvas(FigureCanvas):
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
        self.ax.set_xlabel("Fitted AGL (m)", fontsize=11)
        self.ax.set_ylabel("Strip Width (m)", fontsize=11)
        self.ax.set_title("AGL vs Strip Width", fontsize=12, fontweight="bold")

    def refresh_plot(self, agl, widths):
        self.ax.clear()
        self._style()
        if not agl:
            self.draw()
            return
        a = np.array(agl,    dtype=float)
        w = np.array(widths, dtype=float)
        self.ax.scatter(a, w, color="#ff7828", s=55, zorder=5, label="Measured")

        if len(a) >= 2:
            coeffs = np.polyfit(a, w, 1)
            xi     = np.linspace(a.min() * 0.88, a.max() * 1.12, 200)
            yi     = np.polyval(coeffs, xi)
            yp     = np.polyval(coeffs, a)
            ss_r   = ((w - yp)**2).sum()
            ss_t   = ((w - w.mean())**2).sum()
            r2     = 1 - ss_r/ss_t if ss_t else 0.0
            sign   = '+' if coeffs[1] >= 0 else '-'
            eqn    = (f"W = {coeffs[0]:.4f}·H {sign} "
                      f"{abs(coeffs[1]):.2f}   R²={r2:.4f}")
            self.ax.plot(xi, yi, color="#00d2a0", lw=2.0, label=eqn, zorder=4)
            self.ax.legend(fontsize=10, facecolor="#1e1f26",
                           labelcolor="#dcdee6", framealpha=0.9,
                           loc="upper left")
        self.draw()


# ── main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Strip Width Calibration Tool")
        self.resize(1600, 960)

        self._pts:       list        = []
        self._results:   list        = []
        self._last_calc: dict | None = None
        self._img_size:  tuple | None = None
        self._img_name   = ""

        self._build_ui()
        self._build_menus()
        self._apply_palette()

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

        # Measurement params + Camera params side by side
        params_row = QHBoxLayout()
        params_row.setSpacing(8)

        mg = QGroupBox("Measurement")
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
        mf.addRow("Height AGL:", self.spin_agl)
        params_row.addWidget(mg)

        cg = QGroupBox("Camera  (geometric check)")
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

        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(0, 89)
        self.spin_tilt.setValue(45.0)
        self.spin_tilt.setDecimals(1)
        self.spin_tilt.setSuffix(" °")
        cf.addRow("Nominal tilt:", self.spin_tilt)
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
            "AGL fit (m)", "Tilt fit (°)", "Strip (m)", "Strip geom. (m)",
            "Delta %", "GSD (cm/px)", "n", "Image",
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

        # Regression plot
        self.canvas = RegressCanvas()
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

    def _apply_palette(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        p = QPalette()
        p.setColor(QPalette.Window,          BG)
        p.setColor(QPalette.WindowText,      TEXT)
        p.setColor(QPalette.Base,            SURFACE)
        p.setColor(QPalette.AlternateBase,   SURFACE2)
        p.setColor(QPalette.ToolTipBase,     SURFACE2)
        p.setColor(QPalette.ToolTipText,     TEXT)
        p.setColor(QPalette.Text,            TEXT)
        p.setColor(QPalette.Button,          SURFACE2)
        p.setColor(QPalette.ButtonText,      TEXT)
        p.setColor(QPalette.Highlight,       BLUE)
        p.setColor(QPalette.HighlightedText, QColor(0,0,0))
        p.setColor(QPalette.Mid,             BORDER)
        p.setColor(QPalette.Dark,            QColor(15,15,18))
        p.setColor(QPalette.Light,           SURFACE2)
        app.setPalette(p)

        fs = BASE_FONT_SIZE
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #3c3e48;
                border-radius: 4px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: 600;
                color: #82859a;
                font-size: {fs}pt;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px; padding: 0 4px;
            }}
            QPushButton {{
                background: #2a2b33;
                border: 1px solid #3c3e48;
                border-radius: 3px;
                padding: 7px 16px;
                color: #dcdee6;
                font-size: {fs}pt;
            }}
            QPushButton:hover   {{ background: #34353f; border-color: #5a5c6e; }}
            QPushButton:pressed {{ background: #1e1f26; }}
            QPushButton:checked {{ background: #ff7828; color: #000; border-color: #ff7828; }}
            QPushButton[primary="true"] {{
                background: #2a4d6e; border-color: #3a6a9a; color: #b0d0f0;
            }}
            QPushButton[primary="true"]:hover {{ background: #326082; }}
            QPushButton:disabled {{ color: #4a4c58; border-color: #2e3038; }}
            QSpinBox, QDoubleSpinBox {{
                background: #1e1f26;
                border: 1px solid #3c3e48;
                border-radius: 3px;
                padding: 5px 8px;
                color: #dcdee6;
                font-size: {fs}pt;
            }}
            QLabel {{ font-size: {fs}pt; }}
            QTableWidget {{
                gridline-color: #2a2b33;
                border: 1px solid #3c3e48;
                border-radius: 3px;
                font-size: {fs-1}pt;
            }}
            QHeaderView::section {{
                background: #1e1f26; color: #82859a;
                border: none; border-bottom: 1px solid #3c3e48;
                padding: 5px; font-size: {fs-1}pt;
            }}
            QScrollBar:vertical, QScrollBar:horizontal {{
                background: #1e1f26; width: 9px; height: 9px; border: none;
            }}
            QScrollBar::handle {{ background: #3c3e48; border-radius: 4px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ background: none; }}
            QStatusBar  {{ color: #82859a; font-size: {fs-1}pt; }}
            QMenuBar    {{ background: #16161a; color: #dcdee6; font-size: {fs}pt; }}
            QMenuBar::item:selected {{ background: #2a2b33; }}
            QMenu {{ background: #1e1f26; color: #dcdee6;
                     border: 1px solid #3c3e48; font-size: {fs}pt; }}
            QMenu::item:selected {{ background: #2a4d6e; }}
            QSplitter::handle {{ background: #2a2b33; }}
        """)

    # ── image ──────────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Calibration Image", "",
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
        intervals    = n - 1                        # derived from clicks, no spinbox
        spacing_m    = float(self.spin_spacing.value())
        agl_ft       = float(self.spin_agl.value())
        d            = spacing_m

        # Sort by y descending: index 0 = bottom (nearest, highest y value)
        pts_sorted = sorted(self._pts, key=lambda p: -p.y())
        px_arr     = np.array([[p.x(), p.y()] for p in pts_sorted], dtype=float)

        # 1. Fit line through pixel positions (SVD)
        centroid  = px_arr.mean(axis=0)
        _, _, Vt  = np.linalg.svd(px_arr - centroid)
        line_dir  = Vt[0].copy()
        # Ensure direction points from bottom (high y) toward top (low y)
        if line_dir[1] > 0:
            line_dir = -line_dir

        # 2. Project each point onto the line -> scalar t.
        #    IMPORTANT: shift the t origin to the principal row (image center,
        #    y = img_h/2) so that fit_camera_params' assumptions hold.  With
        #    the origin at the click centroid (as before), the fitted (a,b,c)
        #    — and hence AGL/tilt — depended on WHICH markers were clicked.
        if abs(line_dir[1]) < 1e-6:
            QMessageBox.critical(self, "Degenerate Line",
                                 "Marker line is nearly horizontal — cannot "
                                 "relate it to image rows.")
            return
        t_pp     = (img_h / 2.0 - centroid[1]) / line_dir[1]   # principal row
        t_vals   = (px_arr - centroid) @ line_dir - t_pp

        # 3. Ground positions: y_i = i * d  (0 = nearest)
        y_ground = np.arange(n, dtype=float) * d

        # 4. Fit projective model
        try:
            a, b, c = fit_projective_1d(y_ground, t_vals)
        except Exception as exc:
            QMessageBox.critical(self, "Fit Failed", str(exc))
            return

        # 5. t at image edges: where does the fitted line cross y=img_h (bottom)
        #    and y=0 (top)?  Same principal-point origin as t_vals.
        t_near = (img_h - centroid[1]) / line_dir[1] - t_pp   # bottom edge
        t_far  = (0     - centroid[1]) / line_dir[1] - t_pp   # top edge

        # 6. Invert projective at image edges
        try:
            y_near = invert_projective(t_near, a, b, c)
            y_far  = invert_projective(t_far,  a, b, c)
        except ValueError as exc:
            QMessageBox.critical(self, "Inversion Failed", str(exc))
            return

        strip_m = abs(y_far - y_near)

        # GSD at nearest point: dy/dt at t_near
        dt = max(0.5, abs(t_vals.max() - t_vals.min()) * 0.005)
        try:
            y_near2  = invert_projective(t_near - dt, a, b, c)
            gsd_near = abs(y_near2 - y_near) / dt * 100   # cm/px
        except Exception:
            gsd_near = float('nan')

        # Fit residuals
        t_pred  = (a * y_ground + b) / (c * y_ground + 1)
        rmse_px = float(np.sqrt(np.mean((t_vals - t_pred)**2)))

        # Recover camera geometry from projective parameters
        # f_px: focal length in pixels along the marker-line direction (y-axis)
        f_px = float(self.spin_focal.value()) / float(self.spin_sensor_h.value()) * img_h
        h_fit_m, tilt_fit_deg = fit_camera_params(a, b, c, f_px)
        h_fit_ft = h_fit_m * 3.28084 if math.isfinite(h_fit_m) else float('nan')

        # Extrapolation quality warning
        t_range  = abs(t_vals.max() - t_vals.min())
        extrap   = max(abs(t_near - t_vals.min()), abs(t_far - t_vals.max()))
        extrap_warn = extrap > 2.0 * t_range if t_range > 0 else False

        # Geometric cross-check (uses nominal camera params entered by user)
        geom_strip = geometric_strip_width(
            f_mm        = float(self.spin_focal.value()),
            sensor_h_mm = float(self.spin_sensor_h.value()),
            tilt_deg    = float(self.spin_tilt.value()),
            agl_ft      = agl_ft,
        )
        delta_pct = (strip_m - geom_strip) / geom_strip * 100 if geom_strip else float('nan')

        # Overlay
        self.view.clear_overlays()
        for i, p in enumerate(pts_sorted):
            self.view.add_marker([pts_sorted[j] for j in range(i+1)], p, i)
        self.view.draw_fit_overlay(
            pts_sorted, line_dir, centroid,
            t_near + t_pp, t_far + t_pp, strip_m, img_w, img_h)

        self._last_calc = dict(
            agl_ft       = agl_ft,
            h_fit_m      = round(h_fit_m, 1) if math.isfinite(h_fit_m) else float('nan'),
            h_fit_ft     = round(h_fit_ft, 0) if math.isfinite(h_fit_ft) else float('nan'),
            tilt_fit_deg = round(tilt_fit_deg, 1) if math.isfinite(tilt_fit_deg) else float('nan'),
            strip_m      = round(strip_m, 3),
            geom_strip   = round(geom_strip, 3),
            delta_pct    = round(delta_pct, 2),
            gsd_near     = round(gsd_near, 4),
            rmse_px      = round(rmse_px, 2),
            n_markers    = n,
            spacing_m    = spacing_m,
            intervals    = intervals,
            img_w        = img_w,
            img_h        = img_h,
            image        = self._img_name,
        )

        def _fmt(v, fmt):
            return fmt % v if math.isfinite(v) else "n/a"

        result_lines = [
            f"Fitted AGL               :  {_fmt(h_fit_m, '%.1f')} m  "
            f"({_fmt(h_fit_ft, '%.0f')} ft)",
            f"Fitted tilt (from nadir) :  {_fmt(tilt_fit_deg, '%.1f')}°",
            f"Strip width (empirical)  :  {strip_m:.2f} m",
            f"Strip width (geometric)  :  {geom_strip:.2f} m   ({delta_pct:+.1f}%)",
            f"GSD near edge            :  {gsd_near:.2f} cm/px",
            f"Fit RMSE                 :  {rmse_px:.1f} px  "
            f"({n} markers, {intervals} intervals × {spacing_m:.1f} m)",
        ]
        if n < 4:
            result_lines.append(
                f"NOTE: {n} markers < 4 — fit is {'under' if n < 3 else 'exactly'}-determined. "
                "RMSE is not meaningful; strip width may be unreliable.")
        if extrap_warn:
            result_lines.append(
                "WARNING: markers cover <33% of image height — "
                "strip width extrapolation may be unreliable.")
        self.lbl_result.setText("\n".join(result_lines))
        self.btn_log.setEnabled(True)
        self.status.showMessage(
            f"AGL fit = {_fmt(h_fit_m, '%.1f')} m  |  "
            f"tilt fit = {_fmt(tilt_fit_deg, '%.1f')}°  |  "
            f"strip = {strip_m:.2f} m  |  RMSE = {rmse_px:.1f} px")

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
            _fmtv(d['h_fit_m'],      "%.1f"),
            _fmtv(d['tilt_fit_deg'], "%.1f"),
            f"{d['strip_m']:.3f}",
            f"{d['geom_strip']:.3f}",
            f"{d['delta_pct']:+.1f}%",
            f"{d['gsd_near']:.3f}",
            str(d['n_markers']),
            d['image'],
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, item)

        self._update_plot()
        self.btn_log.setEnabled(False)
        self.status.showMessage(
            f"Logged pass #{len(self._results)}.  Load next image for next pass.")

    def _update_plot(self):
        agls   = [r['h_fit_m']  for r in self._results if math.isfinite(r['h_fit_m'])]
        widths = [r['strip_m']  for r in self._results if math.isfinite(r['h_fit_m'])]
        self.canvas.refresh_plot(agls, widths)

    # ── table ──────────────────────────────────────────────────────────

    def _delete_row(self):
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            if row < len(self._results):
                self._results.pop(row)
        self._update_plot()

    # ── export ─────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            QMessageBox.information(self, "Export", "No results logged yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "strip_width_calibration.csv", "CSV (*.csv)")
        if not path:
            return
        fields = ["h_fit_m", "h_fit_ft", "tilt_fit_deg",
                  "strip_m", "geom_strip", "delta_pct",
                  "gsd_near", "rmse_px", "n_markers",
                  "spacing_m", "intervals", "agl_ft", "img_w", "img_h", "image"]
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
        QMessageBox.about(self, "Strip Width Calibration Tool  v3",
            "<b>Strip Width Calibration Tool  v3</b><br>"
            "Oblique aerial wildlife surveys · East Africa<br><br>"
            "<b>Geometry</b><br>"
            "Fits a 1-D projective model  <i>t = (a·y + b)/(c·y + 1)</i>  to "
            "the clicked marker positions, correctly handling perspective "
            "compression toward the far edge.  Strip width is the ground "
            "distance between the bottom and top image edges as predicted by "
            "the fitted model.<br><br>"
            "<b>Workflow</b><br>"
            "1. Open image → set marker spacing, AGL, camera params<br>"
            "2. Press <b>P</b> → click every visible marker<br>"
            "3. Press <b>Enter</b> → projective fit and overlay<br>"
            "4. Log Result → load next image<br><br>"
            "<b>Notes</b><br>"
            "• Markers are sorted by y-position (bottom = nearest). "
            "Click order doesn't matter.<br>"
            "• Minimum 2 markers; 3 is exactly determined; 4+ gives a meaningful RMSE.<br>"
            "• With 2 markers the fit is under-determined — results are indicative only.<br>"
            "• Fit RMSE > 5 px (with 4+ markers) suggests a mis-click or wrong spacing.<br>"
            "• Geometric strip width uses the flat-ground pinhole model as a "
            "sanity check; Δ% is empirical vs geometric.<br>"
            "• Angled marker lines are handled automatically via SVD line fit.<br>"
        )


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
    app = QApplication(sys.argv)
    app.setFont(_font(BASE_FONT_SIZE))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()