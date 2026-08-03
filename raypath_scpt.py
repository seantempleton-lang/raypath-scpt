"""RayPath SCPT - forward ray-path modelling and SCPT velocity inversion.

This single-file desktop application imports GOnsite/GORILLA ``.GRU`` seismic
records, supports review and manual picking of opposing shear-wave traces, and
estimates a layered shear-wave velocity profile by regularised least squares.

All public engineering quantities use SI units: metres, milliseconds, and
metres per second.  Angles in the numerical core are measured from vertical.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.signal import find_peaks, savgol_filter

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


APP_NAME = "RayPath SCPT"
PROJECT_SUFFIX = ".rpscpt"
VELOCITY_MIN = 50.0
VELOCITY_MAX = 2000.0
PICK_KINDS = ("first_peak", "first_cross", "max_peak")
PICK_LABELS = {
    "first_peak": "First peak",
    "first_cross": "First cross",
    "max_peak": "Maximum peak",
}
CHANNEL_LABELS = {17: "Left", 18: "Right"}


# ---------------------------------------------------------------------------
# Engineering backend
# ---------------------------------------------------------------------------


class RayPathError(RuntimeError):
    """Raised when a physically valid direct ray cannot be calculated."""


@dataclass
class RaySolution:
    """One direct ray solution from the surface source to a receiver."""

    ray_parameter: float
    travel_time_s: float
    angles_rad: np.ndarray
    horizontal_segments_m: np.ndarray


@dataclass
class InversionResult:
    """Complete result of a regularised layered inversion."""

    depths_m: np.ndarray
    thicknesses_m: np.ndarray
    velocities_mps: np.ndarray
    observed_times_s: np.ndarray
    calculated_times_s: np.ndarray
    residuals_s: np.ndarray
    ray_parameters: np.ndarray
    ray_x_segments: list[np.ndarray]
    rmse_s: float
    success: bool
    message: str
    iterations: int


def _validate_model(thicknesses_m: Sequence[float], velocities_mps: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional floating-point model arrays."""

    h = np.asarray(thicknesses_m, dtype=float)
    vs = np.asarray(velocities_mps, dtype=float)
    if h.ndim != 1 or vs.ndim != 1 or h.size == 0 or h.size != vs.size:
        raise ValueError("Layer thickness and velocity arrays must be non-empty and the same length.")
    if not np.all(np.isfinite(h)) or not np.all(h > 0.0):
        raise ValueError("Every layer thickness must be a finite value greater than zero.")
    if not np.all(np.isfinite(vs)) or not np.all(vs > 0.0):
        raise ValueError("Every shear-wave velocity must be a finite value greater than zero.")
    return h, vs


def solve_direct_ray(
    thicknesses_m: Sequence[float],
    velocities_mps: Sequence[float],
    source_offset_m: float,
) -> RaySolution:
    """Solve the unique direct refracted ray using Snell's law.

    The ray parameter ``p = sin(theta_i) / Vs_i`` is common to every crossed
    layer.  ``scipy.optimize.brentq`` finds the root of

    ``sum(h_i * tan(asin(p * Vs_i))) - X = 0``.

    The upper bracket is kept just below the critical value for the fastest
    crossed layer.  For a positive offset the residual tends to positive
    infinity at that limit, so a unique direct-ray root exists.
    """

    h, vs = _validate_model(thicknesses_m, velocities_mps)
    x = float(source_offset_m)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError("Source offset must be a finite value greater than or equal to zero.")

    if x == 0.0:
        angles = np.zeros_like(vs)
        segments = np.zeros_like(vs)
        return RaySolution(0.0, float(np.sum(h / vs)), angles, segments)

    critical_p = 1.0 / float(np.max(vs))
    upper = critical_p * (1.0 - 1.0e-12)

    def offset_residual(p: float) -> float:
        sin_theta = np.clip(p * vs, 0.0, 1.0 - 1.0e-15)
        cos_theta = np.sqrt(np.maximum(1.0 - sin_theta * sin_theta, np.finfo(float).tiny))
        return float(np.sum(h * sin_theta / cos_theta) - x)

    try:
        p = float(brentq(offset_residual, 0.0, upper, xtol=1.0e-14, rtol=1.0e-12, maxiter=150))
    except (ValueError, RuntimeError) as exc:
        raise RayPathError(f"Unable to bracket a physical direct ray: {exc}") from exc

    sin_theta = np.clip(p * vs, 0.0, 1.0 - 1.0e-15)
    angles = np.arcsin(sin_theta)
    cos_theta = np.sqrt(np.maximum(1.0 - sin_theta * sin_theta, np.finfo(float).tiny))
    segments = h * sin_theta / cos_theta
    travel_time = float(np.sum(h / (vs * cos_theta)))
    if not math.isfinite(travel_time):
        raise RayPathError("Calculated travel time is not finite.")
    return RaySolution(p, travel_time, angles, segments)


def forward_model(
    thicknesses_m: Sequence[float],
    velocities_mps: Sequence[float],
    source_offset_m: float,
) -> tuple[np.ndarray, list[RaySolution]]:
    """Calculate travel time to the base of every successively deeper layer."""

    h, vs = _validate_model(thicknesses_m, velocities_mps)
    rays = [solve_direct_ray(h[: index + 1], vs[: index + 1], source_offset_m) for index in range(h.size)]
    return np.asarray([ray.travel_time_s for ray in rays]), rays


def pseudo_interval_velocities(depths_m: Sequence[float], arrival_times_s: Sequence[float]) -> np.ndarray:
    """Return the conventional uncorrected ``delta depth / delta time`` profile."""

    z = np.asarray(depths_m, dtype=float)
    t = np.asarray(arrival_times_s, dtype=float)
    dz = np.diff(np.r_[0.0, z])
    dt = np.diff(np.r_[0.0, t])
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = dz / dt
    raw[(dt <= 0.0) | ~np.isfinite(raw)] = np.nan
    return raw


def _initial_velocity_estimate(depths_m: np.ndarray, times_s: np.ndarray, source_offset_m: float) -> np.ndarray:
    """Build a stable bounded starting profile from apparent and interval speeds."""

    raw = pseudo_interval_velocities(depths_m, times_s)
    path_speed = np.sqrt(depths_m * depths_m + source_offset_m * source_offset_m) / times_s
    finite_path = path_speed[np.isfinite(path_speed) & (path_speed > 0.0)]
    fallback = float(np.median(finite_path)) if finite_path.size else 250.0
    initial = np.where(np.isfinite(raw) & (raw > 0.0), raw, path_speed)
    initial = np.where(np.isfinite(initial) & (initial > 0.0), initial, fallback)
    # A mild median limiter prevents one bad interval from destabilising the first solve.
    if initial.size >= 3:
        limited = initial.copy()
        for i in range(initial.size):
            lo = max(0, i - 1)
            hi = min(initial.size, i + 2)
            local = float(np.median(initial[lo:hi]))
            limited[i] = np.clip(initial[i], 0.4 * local, 2.5 * local)
        initial = limited
    return np.clip(initial, VELOCITY_MIN, VELOCITY_MAX)


def invert_velocity_profile(
    depths_m: Sequence[float],
    observed_times_s: Sequence[float],
    source_offset_m: float,
    regularization: float = 0.35,
    progress_callback: Any | None = None,
) -> InversionResult:
    """Invert arrival times for one velocity per depth interval.

    Velocities are parameterised logarithmically and solved with L-BFGS-B.
    The data term is mean squared time residual in milliseconds.  The
    regularisation term penalises curvature in log velocity (or the single
    gradient for a two-layer model), suppressing isolated spikes while allowing
    broad velocity trends.  The UI maps its 0..1 control directly to
    ``regularization``.
    """

    z = np.asarray(depths_m, dtype=float)
    t_obs = np.asarray(observed_times_s, dtype=float)
    if z.ndim != 1 or t_obs.ndim != 1 or z.size != t_obs.size or z.size < 2:
        raise ValueError("At least two matching receiver depths and arrival times are required.")
    if not np.all(np.isfinite(z)) or not np.all(z > 0.0) or not np.all(np.diff(z) > 0.0):
        raise ValueError("Receiver depths must be finite, positive, and strictly increasing.")
    if not np.all(np.isfinite(t_obs)) or not np.all(t_obs > 0.0):
        raise ValueError("Measured arrival times must be finite and greater than zero.")
    if not math.isfinite(source_offset_m) or source_offset_m < 0.0:
        raise ValueError("Source offset must be finite and non-negative.")
    lam = float(np.clip(regularization, 0.0, 1.0))
    h = np.diff(np.r_[0.0, z])
    initial = _initial_velocity_estimate(z, t_obs, source_offset_m)
    lower = math.log(VELOCITY_MIN)
    upper = math.log(VELOCITY_MAX)
    eval_count = 0

    def objective(log_vs: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal eval_count
        eval_count += 1
        vs = np.exp(log_vs)
        try:
            t_calc, rays = forward_model(h, vs, source_offset_m)
        except (ValueError, RayPathError, FloatingPointError):
            return 1.0e30, np.zeros_like(log_vs)
        residual_ms = (t_calc - t_obs) * 1000.0
        data_cost = float(np.mean(residual_ms * residual_ms))

        # Fermat's principle removes the derivative of the optimally bent path
        # itself.  For receiver i and crossed layer j, the derivative with
        # respect to log(Vs_j) is simply -path_length_ij / Vs_j.  Supplying
        # this lower-triangular Jacobian avoids O(number_of_layers) numerical
        # forward models for every optimiser step.
        travel_jacobian_ms = np.zeros((z.size, z.size), dtype=float)
        for i, ray in enumerate(rays):
            path_lengths = h[: i + 1] / np.cos(ray.angles_rad)
            travel_jacobian_ms[i, : i + 1] = -1000.0 * path_lengths / vs[: i + 1]
        gradient = (2.0 / z.size) * travel_jacobian_ms.T.dot(residual_ms)

        if log_vs.size >= 3:
            curvature = np.diff(log_vs, n=2)
            smooth_gradient = np.zeros_like(log_vs)
            scale = 2.0 / curvature.size
            for index, value in enumerate(curvature):
                smooth_gradient[index : index + 3] += scale * value * np.asarray([1.0, -2.0, 1.0])
        else:
            curvature = np.diff(log_vs)
            smooth_gradient = np.asarray([-2.0 * curvature[0], 2.0 * curvature[0]]) if curvature.size else np.zeros_like(log_vs)
        smooth_cost = float(np.mean(curvature * curvature)) if curvature.size else 0.0
        if progress_callback is not None and eval_count % 10 == 0:
            progress_callback(eval_count, math.sqrt(max(data_cost, 0.0)))
        # At lambda=1 a factor of 100 makes a unit log-curvature equivalent to
        # a 10 ms RMS data error; lambda=0 is an unregularised fit.
        return data_cost + lam * 100.0 * smooth_cost, gradient + lam * 100.0 * smooth_gradient

    result = minimize(
        objective,
        np.log(initial),
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)] * z.size,
        options={"maxiter": 600, "ftol": 1.0e-12, "gtol": 1.0e-7, "maxls": 40},
    )
    velocities = np.exp(result.x)
    calculated, rays = forward_model(h, velocities, source_offset_m)
    residuals = calculated - t_obs
    rmse = float(np.sqrt(np.mean(residuals * residuals)))

    # L-BFGS-B can report a line-search warning after already reaching an
    # excellent finite solution.  Preserve the optimiser status, but only
    # reject a result whose values are not usable.
    usable = bool(np.all(np.isfinite(velocities)) and np.all(np.isfinite(calculated)))
    if not usable:
        raise RayPathError(f"Inversion did not produce a finite model: {result.message}")
    return InversionResult(
        depths_m=z,
        thicknesses_m=h,
        velocities_mps=velocities,
        observed_times_s=t_obs,
        calculated_times_s=calculated,
        residuals_s=residuals,
        ray_parameters=np.asarray([ray.ray_parameter for ray in rays]),
        ray_x_segments=[ray.horizontal_segments_m.copy() for ray in rays],
        rmse_s=rmse,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
    )


# ---------------------------------------------------------------------------
# GRU parsing and arrival picking
# ---------------------------------------------------------------------------


@dataclass
class WaveformRecord:
    """A pair of opposing shear traces at one receiver depth."""

    test_number: int
    depth_m: float
    time_ms: np.ndarray
    left: np.ndarray
    right: np.ndarray
    picks_ms: dict[str, float | None] = field(default_factory=dict)

    @staticmethod
    def pick_key(kind: str, channel: int) -> str:
        return f"{kind}_{channel}"

    def get_pick(self, kind: str, channel: int) -> float | None:
        value = self.picks_ms.get(self.pick_key(kind, channel))
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def set_pick(self, kind: str, channel: int, time_ms: float | None) -> None:
        self.picks_ms[self.pick_key(kind, channel)] = None if time_ms is None else float(time_ms)

    def arrival_ms(self, kind: str) -> float | None:
        values = [self.get_pick(kind, channel) for channel in (17, 18)]
        finite = [value for value in values if value is not None]
        return float(np.mean(finite)) if finite else None


class GruFormatError(ValueError):
    """Raised for malformed or unsupported GRU content."""


SEISMIC_HEADER_RE = re.compile(
    r"^\[SEISMIC\s+TEST\s*=\s*(?P<test>\d+)\s+DEPTH\s*=\s*(?P<depth>[-+0-9.eE]+)\s*\]$",
    re.IGNORECASE,
)
E_RECORD_RE = re.compile(
    r"^\*E:(?P<time>[-+0-9.eE]+)#17:(?P<left>[-+0-9.eE]+)#18:(?P<right>[-+0-9.eE]+)!$",
    re.IGNORECASE,
)


def parse_gru(path: str | Path) -> list[WaveformRecord]:
    """Stream a GORILLA/GOnsite GRU file and return its paired seismic traces."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"GRU file not found: {source}")

    records: list[WaveformRecord] = []
    current: tuple[int, float] | None = None
    times: list[float] = []
    left: list[float] = []
    right: list[float] = []

    def finish_record() -> None:
        nonlocal current, times, left, right
        if current is None:
            return
        if len(times) < 3:
            raise GruFormatError(f"Seismic test {current[0]} at {current[1]:g} m contains too few samples.")
        t = np.asarray(times, dtype=float)
        if not np.all(np.diff(t) > 0.0):
            raise GruFormatError(f"Seismic test {current[0]} has non-increasing sample times.")
        records.append(
            WaveformRecord(
                test_number=current[0],
                depth_m=current[1],
                time_ms=t,
                left=np.asarray(left, dtype=float),
                right=np.asarray(right, dtype=float),
            )
        )
        current = None
        times, left, right = [], [], []

    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            header_match = SEISMIC_HEADER_RE.match(line)
            if header_match:
                finish_record()
                current = (int(header_match.group("test")), float(header_match.group("depth")))
                continue
            if line.upper() == "[SEISMIC TEST END]":
                finish_record()
                continue
            if current is not None and line.startswith("*E:"):
                sample_match = E_RECORD_RE.match(line)
                if not sample_match:
                    raise GruFormatError(f"Malformed seismic sample on line {line_number}.")
                times.append(float(sample_match.group("time")))
                left.append(float(sample_match.group("left")))
                right.append(float(sample_match.group("right")))
    finish_record()
    if not records:
        raise GruFormatError("No paired #17/#18 seismic waveform blocks were found in this GRU file.")
    records.sort(key=lambda record: record.depth_m)
    return records


def _smoothed_trace(values: np.ndarray) -> np.ndarray:
    """Return a lightly smoothed copy suitable for robust automatic suggestions."""

    if values.size < 7:
        return values.astype(float, copy=True)
    window = min(21, values.size if values.size % 2 else values.size - 1)
    window = max(window, 5)
    return savgol_filter(values, window_length=window, polyorder=min(3, window - 2), mode="interp")


def suggest_trace_picks(time_ms: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Suggest first peak, first zero crossing, and maximum peak for one trace.

    Suggestions are deliberately conservative and are only a starting point;
    the picker presents them distinctly and lets the operator move every mark.
    """

    y = _smoothed_trace(np.asarray(values, dtype=float))
    t = np.asarray(time_ms, dtype=float)
    baseline_count = max(10, min(y.size // 8, int(np.searchsorted(t, min(10.0, t[-1] * 0.08)))))
    baseline_count = min(max(baseline_count, 3), y.size)
    baseline = float(np.median(y[:baseline_count]))
    centred = y - baseline
    noise = centred[:baseline_count]
    sigma = 1.4826 * float(np.median(np.abs(noise - np.median(noise))))
    sigma = max(sigma, float(np.std(noise)) * 0.5, np.finfo(float).eps)
    envelope = np.abs(centred)
    threshold = max(5.0 * sigma, 0.04 * float(np.max(envelope)))
    above = envelope >= threshold
    sustained = np.convolve(above.astype(int), np.ones(3, dtype=int), mode="same") >= 2
    candidates = np.flatnonzero(sustained)
    onset = int(candidates[0]) if candidates.size else int(np.argmax(envelope))

    peaks, _ = find_peaks(envelope, prominence=max(2.5 * sigma, 0.015 * float(np.max(envelope))))
    later_peaks = peaks[peaks >= onset]
    first_peak = int(later_peaks[0]) if later_peaks.size else onset

    search_end = int(np.searchsorted(t, t[onset] + 120.0, side="right"))
    search_end = max(search_end, onset + 1)
    search_end = min(search_end, y.size)
    max_peak = onset + int(np.argmax(envelope[onset:search_end]))

    zero_indices = np.flatnonzero(centred[:-1] * centred[1:] <= 0.0)
    later_zeros = zero_indices[zero_indices >= onset]
    zero = int(later_zeros[0]) if later_zeros.size else first_peak
    if zero < y.size - 1 and centred[zero + 1] != centred[zero]:
        fraction = -centred[zero] / (centred[zero + 1] - centred[zero])
        cross_time = float(t[zero] + np.clip(fraction, 0.0, 1.0) * (t[zero + 1] - t[zero]))
    else:
        cross_time = float(t[zero])
    return {
        "first_peak": float(t[first_peak]),
        "first_cross": cross_time,
        "max_peak": float(t[max_peak]),
    }


def add_suggested_picks(records: Iterable[WaveformRecord], overwrite: bool = False) -> None:
    """Populate missing manual-pick fields with automatic trace suggestions."""

    for record in records:
        for channel, values in ((17, record.left), (18, record.right)):
            suggestions = suggest_trace_picks(record.time_ms, values)
            for kind, time_ms in suggestions.items():
                if overwrite or record.get_pick(kind, channel) is None:
                    record.set_pick(kind, channel, time_ms)


# ---------------------------------------------------------------------------
# GUI helpers and widgets
# ---------------------------------------------------------------------------


class PasteTableWidget(QTableWidget):
    """Editable table supporting rectangular Excel/CSV clipboard pastes."""

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_clipboard()
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for item in self.selectedItems():
                item.setText("")
            return
        super().keyPressEvent(event)

    def paste_clipboard(self) -> None:
        text = QApplication.clipboard().text().strip("\r\n")
        if not text:
            return
        lines = text.splitlines()
        delimiter = "\t" if any("\t" in line for line in lines) else ","
        rows = [[cell.strip() for cell in line.split(delimiter)] for line in lines]
        start_row = self.currentRow() if self.currentRow() >= 0 else 0
        start_column = self.currentColumn() if self.currentColumn() >= 0 else 0
        required_rows = start_row + len(rows)
        if required_rows > self.rowCount():
            self.setRowCount(required_rows)
        for row_offset, cells in enumerate(rows):
            for column_offset, value in enumerate(cells):
                column = start_column + column_offset
                if column >= self.columnCount():
                    break
                self.setItem(start_row + row_offset, column, QTableWidgetItem(value))

    def copy_selection(self) -> None:
        ranges = self.selectedRanges()
        if not ranges:
            return
        area = ranges[0]
        rows: list[str] = []
        for row in range(area.topRow(), area.bottomRow() + 1):
            values = []
            for column in range(area.leftColumn(), area.rightColumn() + 1):
                item = self.item(row, column)
                values.append(item.text() if item else "")
            rows.append("\t".join(values))
        QApplication.clipboard().setText("\n".join(rows))


class MplCanvas(FigureCanvas):
    """Matplotlib canvas with one dark-themed axes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.apply_dark_style()

    def apply_dark_style(self) -> None:
        self.figure.set_facecolor("#161b22")
        self.axes.set_facecolor("#0d1117")
        self.axes.tick_params(colors="#c9d1d9")
        for spine in self.axes.spines.values():
            spine.set_color("#48515c")
        self.axes.xaxis.label.set_color("#c9d1d9")
        self.axes.yaxis.label.set_color("#c9d1d9")
        self.axes.title.set_color("#f0f6fc")
        self.axes.grid(True, color="#30363d", alpha=0.55, linewidth=0.7)

    def clear(self) -> None:
        self.axes.clear()
        self.apply_dark_style()


class WaveformPickerDialog(QDialog):
    """Modal reviewer for six manual markers on every paired GRU record."""

    picks_changed = Signal()

    PICK_COLORS = {
        ("first_peak", 17): "#ff9b54",
        ("first_peak", 18): "#ffc857",
        ("first_cross", 17): "#53d8fb",
        ("first_cross", 18): "#3b82f6",
        ("max_peak", 17): "#d97cff",
        ("max_peak", 18): "#9d4edd",
    }

    def __init__(self, records: list[WaveformRecord], source_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.records = records
        self.source_name = source_name
        self.active_kind = "first_cross"
        self.active_channel = 17
        self._click_connection: int | None = None
        self.setWindowTitle(f"Waveform Picker — {source_name}")
        self.resize(1200, 760)
        self.setModal(True)
        self._build_ui()
        self._populate_record_list()
        self.record_list.setCurrentRow(0)
        self._click_connection = self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        guidance = QLabel(
            "Select a marker, then click its time on the trace. Automatic suggestions are review aids only; "
            "the active arrival estimator is chosen in the main window."
        )
        guidance.setWordWrap(True)
        guidance.setObjectName("subtleLabel")
        root.addWidget(guidance)

        body = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(body, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.addWidget(QLabel("Receiver records"))
        self.record_list = QListWidget()
        self.record_list.currentRowChanged.connect(self._record_changed)
        left_layout.addWidget(self.record_list, 1)
        suggest_button = QPushButton("Regenerate suggestions for record")
        suggest_button.clicked.connect(self._resuggest_current)
        left_layout.addWidget(suggest_button)
        clear_button = QPushButton("Clear active marker")
        clear_button.clicked.connect(self._clear_active_pick)
        left_layout.addWidget(clear_button)
        body.addWidget(left_panel)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(8, 0, 0, 0)
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)

        selector = QGroupBox("Marker placed by the next plot click")
        selector_layout = QHBoxLayout(selector)
        self.marker_group = QButtonGroup(self)
        self.active_label = QLabel()
        self.active_label.setObjectName("accentLabel")
        for kind in PICK_KINDS:
            kind_box = QGroupBox(PICK_LABELS[kind])
            kind_layout = QVBoxLayout(kind_box)
            for channel in (17, 18):
                button = QRadioButton(CHANNEL_LABELS[channel])
                button.setProperty("pick_kind", kind)
                button.setProperty("channel", channel)
                button.toggled.connect(self._marker_selected)
                self.marker_group.addButton(button)
                kind_layout.addWidget(button)
                if kind == self.active_kind and channel == self.active_channel:
                    button.setChecked(True)
            selector_layout.addWidget(kind_box)
        plot_layout.addWidget(selector)
        plot_layout.addWidget(self.active_label)
        self.pick_values_label = QLabel()
        self.pick_values_label.setObjectName("subtleLabel")
        self.pick_values_label.setWordWrap(True)
        plot_layout.addWidget(self.pick_values_label)
        body.addWidget(plot_panel)
        body.setSizes([260, 900])

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use reviewed picks")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._update_active_label()

    def _populate_record_list(self) -> None:
        self.record_list.clear()
        for record in self.records:
            self.record_list.addItem(self._record_item_text(record))

    @staticmethod
    def _record_item_text(record: WaveformRecord) -> str:
        complete = sum(record.get_pick(kind, channel) is not None for kind in PICK_KINDS for channel in (17, 18))
        return f"Test {record.test_number:>3}   {record.depth_m:>6.2f} m   [{complete}/6]"

    def _record_changed(self, row: int) -> None:
        if row >= 0:
            self._draw_record()

    def _marker_selected(self, checked: bool) -> None:
        if not checked:
            return
        button = self.sender()
        self.active_kind = str(button.property("pick_kind"))
        self.active_channel = int(button.property("channel"))
        self._update_active_label()

    def _update_active_label(self) -> None:
        self.active_label.setText(
            f"Active: {CHANNEL_LABELS[self.active_channel]} — {PICK_LABELS[self.active_kind]}"
        )

    def _on_plot_click(self, event: Any) -> None:
        row = self.record_list.currentRow()
        if row < 0 or self.toolbar.mode or event.inaxes is not self.canvas.axes or event.xdata is None:
            return
        record = self.records[row]
        time_ms = float(np.clip(event.xdata, record.time_ms[0], record.time_ms[-1]))
        record.set_pick(self.active_kind, self.active_channel, time_ms)
        self._refresh_list_item(row)
        self._draw_record()
        self.picks_changed.emit()

    def _clear_active_pick(self) -> None:
        row = self.record_list.currentRow()
        if row < 0:
            return
        self.records[row].set_pick(self.active_kind, self.active_channel, None)
        self._refresh_list_item(row)
        self._draw_record()

    def _resuggest_current(self) -> None:
        row = self.record_list.currentRow()
        if row < 0:
            return
        add_suggested_picks([self.records[row]], overwrite=True)
        self._refresh_list_item(row)
        self._draw_record()

    def _refresh_list_item(self, row: int) -> None:
        item = self.record_list.item(row)
        if item:
            item.setText(self._record_item_text(self.records[row]))

    def _draw_record(self) -> None:
        row = self.record_list.currentRow()
        if row < 0:
            return
        record = self.records[row]
        ax = self.canvas.axes
        self.canvas.clear()
        ax.axhline(0.0, color="#8b949e", linewidth=0.8, alpha=0.8)
        ax.plot(record.time_ms, record.left, color="#ff7b72", linewidth=1.0, label="Left (#17)")
        ax.plot(record.time_ms, record.right, color="#58a6ff", linewidth=1.0, label="Right (#18)")
        for kind in PICK_KINDS:
            for channel in (17, 18):
                pick = record.get_pick(kind, channel)
                if pick is None:
                    continue
                color = self.PICK_COLORS[(kind, channel)]
                linestyle = {"first_peak": "--", "first_cross": "-", "max_peak": ":"}[kind]
                ax.axvline(pick, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9)
        ax.set_title(f"Test {record.test_number} — receiver depth {record.depth_m:.2f} m")
        ax.set_xlabel("Time after trigger (ms)")
        ax.set_ylabel("Recorded amplitude")
        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        value_parts = []
        for kind in PICK_KINDS:
            left = record.get_pick(kind, 17)
            right = record.get_pick(kind, 18)
            left_text = "—" if left is None else f"{left:.2f} ms"
            right_text = "—" if right is None else f"{right:.2f} ms"
            value_parts.append(f"{PICK_LABELS[kind]}: L {left_text} · R {right_text}")
        self.pick_values_label.setText("   |   ".join(value_parts))
        self.canvas.draw_idle()


class InversionWorker(QObject):
    """Run the numerical inversion away from the Qt GUI thread."""

    finished = Signal(object)
    failed = Signal(str, str)
    progress = Signal(int, float)

    def __init__(self, depths_m: np.ndarray, times_s: np.ndarray, offset_m: float, regularization: float) -> None:
        super().__init__()
        self.depths_m = depths_m
        self.times_s = times_s
        self.offset_m = offset_m
        self.regularization = regularization

    @Slot()
    def run(self) -> None:
        try:
            result = invert_velocity_profile(
                self.depths_m,
                self.times_s,
                self.offset_m,
                self.regularization,
                lambda evaluations, rmse: self.progress.emit(evaluations, rmse),
            )
            self.finished.emit(result)
        except Exception as exc:  # GUI boundary: present all solver/parser failures cleanly.
            self.failed.emit(str(exc), traceback.format_exc())


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class RayPathMainWindow(QMainWindow):
    """Strict left-to-right engineering workflow for RayPath SCPT."""

    SAMPLE_DEPTHS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    SAMPLE_TIMES_MS = [13.56, 20.95, 28.04, 34.15, 39.76, 45.06, 49.62, 54.00]

    def __init__(self) -> None:
        super().__init__()
        self.project_path: Path | None = None
        self.gru_path: Path | None = None
        self.waveform_records: list[WaveformRecord] = []
        self.result: InversionResult | None = None
        self._thread: QThread | None = None
        self._worker: InversionWorker | None = None
        self._dirty = False
        self.setWindowTitle(APP_NAME)
        self.resize(1650, 900)
        self.setMinimumSize(1200, 720)
        self._build_actions_and_menu()
        self._build_ui()
        self._set_sample_data()
        self._set_dirty(False)

    # ---- UI construction -------------------------------------------------

    def _build_actions_and_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        self.new_action = QAction("&New", self, shortcut=QKeySequence.StandardKey.New, triggered=self.new_project)
        self.open_action = QAction("&Open…", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_file)
        self.save_action = QAction("&Save", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.save_project)
        self.save_as_action = QAction("Save &As…", self, shortcut=QKeySequence.StandardKey.SaveAs, triggered=self.save_project_as)
        self.export_action = QAction("Export &CSV…", self, triggered=self.export_csv)
        self.exit_action = QAction("E&xit", self, shortcut=QKeySequence.StandardKey.Quit, triggered=self.close)
        for action in (self.new_action, self.open_action, self.save_action, self.save_as_action):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self.export_action)
        menu.addSeparator()
        menu.addAction(self.exit_action)

        seismic_menu = self.menuBar().addMenu("&Seismic")
        self.review_action = QAction("Review &Waveform Picks…", self, triggered=self.review_waveforms)
        self.review_action.setEnabled(False)
        seismic_menu.addAction(self.review_action)

        units_menu = self.menuBar().addMenu("&Units")
        si_action = QAction("SI — m, ms, m/s", self)
        si_action.setCheckable(True)
        si_action.setChecked(True)
        si_action.setEnabled(False)
        units_menu.addAction(si_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(QAction("&About RayPath SCPT", self, triggered=self.show_about))

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)

        heading_row = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        subtitle = QLabel("Regularised forward ray-path inversion")
        subtitle.setObjectName("subtleLabel")
        heading_row.addWidget(title)
        heading_row.addSpacing(12)
        heading_row.addWidget(subtitle)
        heading_row.addStretch()
        units = QLabel("SI  ·  m  ·  ms  ·  m/s")
        units.setObjectName("unitBadge")
        heading_row.addWidget(units)
        root.addLayout(heading_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_plot_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([380, 820, 450])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ready — sample data loaded")
        status.addWidget(self.status_label, 1)
        self.rmse_label = QLabel("RMSE: — ms")
        status.addPermanentWidget(self.rmse_label)

    def _build_input_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("1  INPUTS")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        form = QFormLayout()
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(0.0, 1000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setValue(2.4)
        self.offset_spin.setSuffix(" m")
        self.offset_spin.valueChanged.connect(self._input_changed)
        form.addRow("Source offset (X)", self.offset_spin)

        self.estimator_combo = QComboBox()
        for kind in PICK_KINDS:
            self.estimator_combo.addItem(PICK_LABELS[kind], kind)
        self.estimator_combo.setCurrentIndex(1)
        self.estimator_combo.setToolTip("Observed arrival time is the mean of the reviewed left and right markers.")
        self.estimator_combo.currentIndexChanged.connect(self._estimator_changed)
        form.addRow("Arrival estimator", self.estimator_combo)
        layout.addLayout(form)

        import_button = QPushButton("Import GRU Waveforms…")
        import_button.clicked.connect(self.import_gru)
        layout.addWidget(import_button)
        self.gru_label = QLabel("No GRU source loaded — editable sample data shown")
        self.gru_label.setWordWrap(True)
        self.gru_label.setObjectName("subtleLabel")
        layout.addWidget(self.gru_label)

        table_label = QLabel("Receiver observations")
        table_label.setObjectName("minorTitle")
        layout.addWidget(table_label)
        self.input_table = PasteTableWidget(0, 2)
        self.input_table.setHorizontalHeaderLabels(["Depth z (m)", "Arrival t_obs (ms)"])
        self.input_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.input_table.verticalHeader().setDefaultSectionSize(27)
        self.input_table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.input_table.itemChanged.connect(self._input_changed)
        layout.addWidget(self.input_table, 1)

        table_buttons = QHBoxLayout()
        add_button = QPushButton("+ Row")
        add_button.clicked.connect(self._add_row)
        remove_button = QPushButton("− Row")
        remove_button.clicked.connect(self._remove_rows)
        paste_button = QPushButton("Paste")
        paste_button.clicked.connect(self.input_table.paste_clipboard)
        table_buttons.addWidget(add_button)
        table_buttons.addWidget(remove_button)
        table_buttons.addWidget(paste_button)
        layout.addLayout(table_buttons)

        smooth_label_row = QHBoxLayout()
        smooth_label_row.addWidget(QLabel("Smoothing / regularisation"))
        self.reg_value_label = QLabel("0.35")
        self.reg_value_label.setObjectName("accentLabel")
        smooth_label_row.addStretch()
        smooth_label_row.addWidget(self.reg_value_label)
        layout.addLayout(smooth_label_row)
        self.reg_slider = QSlider(Qt.Orientation.Horizontal)
        self.reg_slider.setRange(0, 100)
        self.reg_slider.setValue(35)
        self.reg_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.reg_slider.setTickInterval(10)
        self.reg_slider.valueChanged.connect(self._regularization_changed)
        layout.addWidget(self.reg_slider)
        endpoints = QHBoxLayout()
        endpoints.addWidget(QLabel("0.00  fit detail"))
        endpoints.addStretch()
        endpoints.addWidget(QLabel("1.00  smoother"))
        layout.addLayout(endpoints)

        self.run_button = QPushButton("Run RayPath Inversion")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumHeight(48)
        self.run_button.clicked.connect(self.run_inversion)
        layout.addWidget(self.run_button)
        return panel

    def _build_plot_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("2  VISUALISATION")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.plot_tabs = QTabWidget()
        self.velocity_canvas = MplCanvas()
        self.ray_canvas = MplCanvas()
        self.fit_canvas = MplCanvas()
        self.plot_tabs.addTab(self.velocity_canvas, "Velocity Profile")
        self.plot_tabs.addTab(self.ray_canvas, "Ray Paths")
        self.plot_tabs.addTab(self.fit_canvas, "Arrival-Time Fit")
        layout.addWidget(self.plot_tabs, 1)
        self._draw_empty_plots()
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("3  RESULTS")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.result_summary = QLabel("Run the inversion to calculate a layered Vs profile.")
        self.result_summary.setWordWrap(True)
        self.result_summary.setObjectName("subtleLabel")
        layout.addWidget(self.result_summary)
        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(
            ["Layer", "Top depth", "Bottom depth", "Optimized Vs", "Fitting error"]
        )
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.verticalHeader().setVisible(False)
        layout.addWidget(self.result_table, 1)
        export_button = QPushButton("Export Results to CSV…")
        export_button.clicked.connect(self.export_csv)
        layout.addWidget(export_button)
        return panel

    # ---- project state ---------------------------------------------------

    def _set_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        name = self.project_path.name if self.project_path else "Untitled"
        self.setWindowTitle(f"{APP_NAME} — {name}{' *' if dirty else ''}")

    def _input_changed(self, *_args: Any) -> None:
        self.result = None
        self._set_dirty(True)

    def _regularization_changed(self, value: int) -> None:
        self.reg_value_label.setText(f"{value / 100.0:.2f}")
        self._input_changed()

    def _estimator_changed(self, _index: int) -> None:
        if self.waveform_records:
            self._populate_table_from_picks()
        self._set_dirty(True)

    def _set_sample_data(self) -> None:
        self.waveform_records = []
        self.gru_path = None
        self.review_action.setEnabled(False)
        self.gru_label.setText("No GRU source loaded — editable sample data shown")
        self._set_input_rows(zip(self.SAMPLE_DEPTHS, self.SAMPLE_TIMES_MS))
        self.offset_spin.setValue(2.4)
        self.reg_slider.setValue(35)
        self.estimator_combo.setCurrentIndex(1)
        self._clear_results()
        self.status_label.setText("Ready — sample data loaded")

    def _set_input_rows(self, rows: Iterable[tuple[float, float | None]]) -> None:
        self.input_table.blockSignals(True)
        values = list(rows)
        self.input_table.setRowCount(len(values))
        for row, (depth, time_ms) in enumerate(values):
            self.input_table.setItem(row, 0, QTableWidgetItem(f"{depth:.3f}"))
            self.input_table.setItem(row, 1, QTableWidgetItem("" if time_ms is None else f"{time_ms:.3f}"))
        self.input_table.blockSignals(False)

    def _clear_results(self) -> None:
        self.result = None
        self.result_table.setRowCount(0)
        self.result_summary.setText("Run the inversion to calculate a layered Vs profile.")
        self.rmse_label.setText("RMSE: — ms")
        self._draw_empty_plots()

    def _draw_empty_plots(self) -> None:
        configurations = (
            (self.velocity_canvas, "Velocity profile", "Vs (m/s)", "Depth (m)"),
            (self.ray_canvas, "Refracted ray paths", "Horizontal distance (m)", "Depth (m)"),
            (self.fit_canvas, "Observed and calculated arrivals", "Arrival time (ms)", "Depth (m)"),
        )
        for canvas, title, xlabel, ylabel in configurations:
            canvas.clear()
            canvas.axes.set_title(title)
            canvas.axes.set_xlabel(xlabel)
            canvas.axes.set_ylabel(ylabel)
            canvas.axes.text(
                0.5,
                0.5,
                "Run inversion to populate",
                transform=canvas.axes.transAxes,
                ha="center",
                va="center",
                color="#8b949e",
            )
            canvas.draw_idle()

    def _add_row(self) -> None:
        self.input_table.insertRow(self.input_table.rowCount())
        self._input_changed()

    def _remove_rows(self) -> None:
        rows = sorted({index.row() for index in self.input_table.selectedIndexes()}, reverse=True)
        if not rows and self.input_table.currentRow() >= 0:
            rows = [self.input_table.currentRow()]
        for row in rows:
            self.input_table.removeRow(row)
        if rows:
            self._input_changed()

    def _read_input_rows(self) -> tuple[np.ndarray, np.ndarray]:
        rows: list[tuple[float, float]] = []
        errors: list[str] = []
        for row in range(self.input_table.rowCount()):
            depth_text = self.input_table.item(row, 0).text().strip() if self.input_table.item(row, 0) else ""
            time_text = self.input_table.item(row, 1).text().strip() if self.input_table.item(row, 1) else ""
            if not depth_text and not time_text:
                continue
            try:
                depth = float(depth_text)
                time_ms = float(time_text)
                if not math.isfinite(depth) or not math.isfinite(time_ms) or depth <= 0.0 or time_ms <= 0.0:
                    raise ValueError
                rows.append((depth, time_ms))
            except ValueError:
                errors.append(str(row + 1))
        if errors:
            raise ValueError(f"Rows {', '.join(errors)} contain incomplete or invalid positive numeric values.")
        if len(rows) < 2:
            raise ValueError("Enter at least two complete depth and arrival-time rows.")
        rows.sort(key=lambda pair: pair[0])
        depths = np.asarray([pair[0] for pair in rows], dtype=float)
        times_s = np.asarray([pair[1] / 1000.0 for pair in rows], dtype=float)
        if np.any(np.diff(depths) <= 0.0):
            raise ValueError("Receiver depths must be unique and strictly increasing.")
        return depths, times_s

    # ---- file operations -------------------------------------------------

    @Slot()
    def new_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        self.project_path = None
        self._set_sample_data()
        self._set_dirty(False)

    @Slot()
    def open_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RayPath Data",
            "",
            "Supported files (*.GRU *.gru *.rpscpt *.csv *.CSV);;GRU seismic files (*.GRU *.gru);;RayPath projects (*.rpscpt);;CSV observations (*.csv *.CSV)",
        )
        if not path:
            return
        suffix = Path(path).suffix.lower()
        try:
            if suffix == ".gru":
                self._load_gru(Path(path))
            elif suffix == PROJECT_SUFFIX:
                self._load_project(Path(path))
            elif suffix == ".csv":
                self._load_observation_csv(Path(path))
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as exc:
            self._show_error("Unable to open file", exc)

    @Slot()
    def import_gru(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import GRU Seismic File", "", "GRU seismic files (*.GRU *.gru)")
        if not path:
            return
        try:
            self._load_gru(Path(path))
        except Exception as exc:
            self._show_error("Unable to import GRU file", exc)

    def _load_gru(self, path: Path) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            records = parse_gru(path)
            add_suggested_picks(records)
        finally:
            QApplication.restoreOverrideCursor()
        dialog = WaveformPickerDialog(records, path.name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.waveform_records = records
        self.gru_path = path
        self.project_path = None
        self.review_action.setEnabled(True)
        self.gru_label.setText(f"{path.name} — {len(records)} paired seismic records")
        self._populate_table_from_picks()
        self._clear_results()
        self.status_label.setText(f"Imported {len(records)} GRU seismic records — review complete")
        self._set_dirty(True)

    def _populate_table_from_picks(self) -> None:
        kind = str(self.estimator_combo.currentData())
        rows = [(record.depth_m, record.arrival_ms(kind)) for record in self.waveform_records]
        self._set_input_rows(rows)
        missing = sum(time is None for _, time in rows)
        if missing:
            self.status_label.setText(f"{missing} records have no {PICK_LABELS[kind].lower()} pick and need review")

    @Slot()
    def review_waveforms(self) -> None:
        if not self.waveform_records:
            return
        backup = [record.picks_ms.copy() for record in self.waveform_records]
        dialog = WaveformPickerDialog(
            self.waveform_records,
            self.gru_path.name if self.gru_path else "project waveforms",
            self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._populate_table_from_picks()
            self._clear_results()
            self._set_dirty(True)
        else:
            for record, picks in zip(self.waveform_records, backup):
                record.picks_ms = picks

    def _load_observation_csv(self, path: Path) -> None:
        rows: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for number, values in enumerate(reader, 1):
                if not values or all(not value.strip() for value in values):
                    continue
                if len(values) < 2:
                    continue
                try:
                    rows.append((float(values[0]), float(values[1])))
                except ValueError:
                    if number == 1:
                        continue
                    raise ValueError(f"CSV row {number} does not begin with numeric depth and arrival time.")
        if len(rows) < 2:
            raise ValueError("CSV must contain at least two depth/arrival-time rows.")
        rows.sort()
        self.waveform_records = []
        self.gru_path = None
        self.project_path = None
        self.review_action.setEnabled(False)
        self.gru_label.setText(f"{path.name} — CSV observations")
        self._set_input_rows(rows)
        self._clear_results()
        self.status_label.setText(f"Imported {len(rows)} CSV observations")
        self._set_dirty(True)

    def _project_payload(self) -> dict[str, Any]:
        inputs = []
        for row in range(self.input_table.rowCount()):
            depth = self.input_table.item(row, 0).text() if self.input_table.item(row, 0) else ""
            time_ms = self.input_table.item(row, 1).text() if self.input_table.item(row, 1) else ""
            inputs.append({"depth_m": depth, "arrival_time_ms": time_ms})
        payload: dict[str, Any] = {
            "format": "RayPath SCPT Project",
            "version": 1,
            "units": "SI",
            "source_offset_m": self.offset_spin.value(),
            "regularization": self.reg_slider.value() / 100.0,
            "arrival_estimator": self.estimator_combo.currentData(),
            "gru_source": str(self.gru_path) if self.gru_path else None,
            "inputs": inputs,
            "picks": [
                {
                    "test_number": record.test_number,
                    "depth_m": record.depth_m,
                    "picks_ms": record.picks_ms,
                }
                for record in self.waveform_records
            ],
        }
        if self.result is not None:
            payload["last_result"] = {
                "velocities_mps": self.result.velocities_mps.tolist(),
                "calculated_times_ms": (self.result.calculated_times_s * 1000.0).tolist(),
                "rmse_ms": self.result.rmse_s * 1000.0,
            }
        return payload

    @Slot()
    def save_project(self) -> bool:
        if self.project_path is None:
            return self.save_project_as()
        try:
            self.project_path.write_text(json.dumps(self._project_payload(), indent=2), encoding="utf-8")
            self._set_dirty(False)
            self.status_label.setText(f"Saved {self.project_path.name}")
            return True
        except Exception as exc:
            self._show_error("Unable to save project", exc)
            return False

    @Slot()
    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save RayPath Project", "", "RayPath projects (*.rpscpt)")
        if not path:
            return False
        target = Path(path)
        if target.suffix.lower() != PROJECT_SUFFIX:
            target = target.with_suffix(PROJECT_SUFFIX)
        self.project_path = target
        return self.save_project()

    def _load_project(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "RayPath SCPT Project" or payload.get("version") != 1:
            raise ValueError("This is not a supported RayPath SCPT project file.")
        self.project_path = path
        self.offset_spin.setValue(float(payload.get("source_offset_m", 2.4)))
        self.reg_slider.setValue(round(float(payload.get("regularization", 0.35)) * 100))
        estimator = str(payload.get("arrival_estimator", "first_cross"))
        estimator_index = self.estimator_combo.findData(estimator)
        self.estimator_combo.setCurrentIndex(max(0, estimator_index))

        inputs = payload.get("inputs", [])
        rows = [(float(item["depth_m"]), float(item["arrival_time_ms"])) for item in inputs if item.get("depth_m") and item.get("arrival_time_ms")]
        self._set_input_rows(rows)
        self.waveform_records = []
        self.gru_path = Path(payload["gru_source"]) if payload.get("gru_source") else None
        if self.gru_path and self.gru_path.is_file():
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.waveform_records = parse_gru(self.gru_path)
            finally:
                QApplication.restoreOverrideCursor()
            saved = {(int(item["test_number"]), float(item["depth_m"])): item.get("picks_ms", {}) for item in payload.get("picks", [])}
            for record in self.waveform_records:
                record.picks_ms = dict(saved.get((record.test_number, record.depth_m), {}))
            self.gru_label.setText(f"{self.gru_path.name} — {len(self.waveform_records)} paired seismic records")
        elif self.gru_path:
            self.gru_label.setText(f"GRU source unavailable: {self.gru_path}")
        else:
            self.gru_label.setText("Project contains manually entered observations")
        self.review_action.setEnabled(bool(self.waveform_records))
        self._clear_results()
        self.status_label.setText(f"Opened {path.name}")
        self._set_dirty(False)

    @Slot()
    def export_csv(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "No results", "Run the inversion before exporting results.")
            return
        suggested = (self.project_path.stem if self.project_path else "raypath_scpt_results") + ".csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export RayPath Results", suggested, "CSV files (*.csv)")
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        try:
            with target.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Layer",
                        "Top Depth (m)",
                        "Bottom Depth (m)",
                        "Optimized Vs (m/s)",
                        "Observed Arrival (ms)",
                        "Calculated Arrival (ms)",
                        "Fitting Error (ms)",
                        "Ray Parameter (s/m)",
                    ]
                )
                tops = np.r_[0.0, self.result.depths_m[:-1]]
                for i in range(self.result.depths_m.size):
                    writer.writerow(
                        [
                            i + 1,
                            f"{tops[i]:.4f}",
                            f"{self.result.depths_m[i]:.4f}",
                            f"{self.result.velocities_mps[i]:.3f}",
                            f"{self.result.observed_times_s[i] * 1000.0:.4f}",
                            f"{self.result.calculated_times_s[i] * 1000.0:.4f}",
                            f"{self.result.residuals_s[i] * 1000.0:.4f}",
                            f"{self.result.ray_parameters[i]:.10g}",
                        ]
                    )
            self.status_label.setText(f"Exported {target.name}")
        except Exception as exc:
            self._show_error("Unable to export CSV", exc)

    # ---- inversion and output -------------------------------------------

    @Slot()
    def run_inversion(self) -> None:
        if self._thread is not None:
            return
        try:
            depths, times_s = self._read_input_rows()
        except Exception as exc:
            self._show_error("Invalid input data", exc)
            return
        self.run_button.setEnabled(False)
        self.run_button.setText("Inversion running…")
        self.status_label.setText("Solving refracted ray paths and velocity model…")
        self.rmse_label.setText("RMSE: calculating…")
        self._thread = QThread(self)
        self._worker = InversionWorker(
            depths,
            times_s,
            self.offset_spin.value(),
            self.reg_slider.value() / 100.0,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._inversion_finished)
        self._worker.failed.connect(self._inversion_failed)
        self._worker.progress.connect(self._inversion_progress)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    @Slot(int, float)
    def _inversion_progress(self, evaluations: int, rmse_ms: float) -> None:
        self.status_label.setText(f"Optimising model — {evaluations} evaluations, current data RMSE {rmse_ms:.3f} ms")

    @Slot(object)
    def _inversion_finished(self, result: InversionResult) -> None:
        self.result = result
        self._populate_results(result)
        self._draw_results(result)
        state = "Converged" if result.success else "Usable solution (optimizer warning)"
        self.status_label.setText(f"{state} — {result.iterations} iterations — {result.message}")
        self.rmse_label.setText(f"RMSE: {result.rmse_s * 1000.0:.3f} ms")
        self.result_summary.setText(
            f"{state}. {result.depths_m.size} layers, source offset {self.offset_spin.value():.3f} m, "
            f"regularisation {self.reg_slider.value() / 100.0:.2f}. Final RMSE: {result.rmse_s * 1000.0:.3f} ms."
        )
        if not result.success:
            QMessageBox.warning(
                self,
                "Optimizer convergence warning",
                "The solver returned a finite model but did not report formal convergence.\n\n"
                f"{result.message}\n\nReview the arrival picks and smoothing factor before using these results.",
            )
        self._set_dirty(True)

    @Slot(str, str)
    def _inversion_failed(self, message: str, detail: str) -> None:
        self.status_label.setText("Inversion failed — check inputs and solver message")
        self.rmse_label.setText("RMSE: failed")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("RayPath inversion failed")
        box.setText(message or "The numerical solver failed.")
        box.setDetailedText(detail)
        box.exec()

    @Slot()
    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.run_button.setEnabled(True)
        self.run_button.setText("Run RayPath Inversion")

    def _populate_results(self, result: InversionResult) -> None:
        self.result_table.setRowCount(result.depths_m.size)
        tops = np.r_[0.0, result.depths_m[:-1]]
        for row in range(result.depths_m.size):
            values = (
                str(row + 1),
                f"{tops[row]:.2f} m",
                f"{result.depths_m[row]:.2f} m",
                f"{result.velocities_mps[row]:.1f} m/s",
                f"{result.residuals_s[row] * 1000.0:+.3f} ms",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if column == 4:
                    magnitude = abs(result.residuals_s[row] * 1000.0)
                    item.setForeground(QColor("#3fb950" if magnitude < 1.0 else "#d29922" if magnitude < 3.0 else "#f85149"))
                self.result_table.setItem(row, column, item)

    def _draw_results(self, result: InversionResult) -> None:
        z = result.depths_m
        edges = np.r_[0.0, z]
        raw = pseudo_interval_velocities(z, result.observed_times_s)

        ax = self.velocity_canvas.axes
        self.velocity_canvas.clear()
        ax.plot(raw, (edges[:-1] + edges[1:]) / 2.0, "--o", color="#ff5c5c", markersize=4, label="Raw pseudo-interval Vs")
        ax.stairs(result.velocities_mps, edges, orientation="horizontal", color="#4da3ff", linewidth=2.4, label="Optimized forward model")
        ax.set_title("Shear-wave velocity profile")
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1], 0.0)
        finite = np.r_[raw[np.isfinite(raw)], result.velocities_mps]
        if finite.size:
            ax.set_xlim(max(0.0, float(np.min(finite)) * 0.8), float(np.max(finite)) * 1.15)
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        self.velocity_canvas.draw_idle()

        ax = self.ray_canvas.axes
        self.ray_canvas.clear()
        x_offset = self.offset_spin.value()
        for boundary in edges:
            ax.hlines(boundary, 0.0, x_offset, color="#48515c", linewidth=0.7, alpha=0.7)
        color_map = colormaps["viridis"]
        for i, segments in enumerate(result.ray_x_segments):
            ray_x = np.r_[0.0, np.cumsum(segments)]
            ray_z = edges[: i + 2]
            ax.plot(ray_x, ray_z, color=color_map((i + 1) / len(result.ray_x_segments)), linewidth=1.25, alpha=0.9)
        ax.axvline(x_offset, color="#f0f6fc", linewidth=1.8, label="Borehole")
        ax.scatter([0.0], [0.0], marker="*", s=110, color="#ffcc66", zorder=5, label="Source")
        ax.scatter(np.full_like(z, x_offset), z, marker="<", s=28, color="#58a6ff", zorder=5, label="Receivers")
        ax.set_title("Snell's-law refracted direct rays")
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1] * 1.03, -z[-1] * 0.03)
        margin = max(x_offset * 0.08, 0.1)
        ax.set_xlim(-margin, x_offset + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        self.ray_canvas.draw_idle()

        ax = self.fit_canvas.axes
        self.fit_canvas.clear()
        observed_ms = result.observed_times_s * 1000.0
        calculated_ms = result.calculated_times_s * 1000.0
        ax.plot(observed_ms, z, "o", color="#ff9b54", markersize=5, label="Observed")
        ax.plot(calculated_ms, z, "-s", color="#4da3ff", markersize=3.5, label="Calculated")
        for depth, observed, calculated in zip(z, observed_ms, calculated_ms):
            ax.plot([observed, calculated], [depth, depth], color="#8b949e", linewidth=0.8)
        ax.set_title(f"Arrival-time fit — RMSE {result.rmse_s * 1000.0:.3f} ms")
        ax.set_xlabel("Arrival time after trigger (ms)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1], 0.0)
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        self.fit_canvas.draw_idle()

    # ---- lifecycle and dialogs ------------------------------------------

    def _confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes to the current RayPath project?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if self._thread is not None:
            QMessageBox.information(self, "Inversion running", "Wait for the active inversion to finish before closing.")
            event.ignore()
            return
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def _show_error_title_text(parent: QWidget, title: str, text: str) -> None:
        QMessageBox.critical(parent, title, text)

    def _show_error(self, title: str, exc: Exception) -> None:
        self._show_error_title_text(self, title, str(exc))

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            "<h3>RayPath SCPT</h3>"
            "<p>SI-only SCPT arrival-time picking and regularised shear-wave velocity inversion.</p>"
            "<p>The direct-ray forward model solves Snell's-law refraction with SciPy Brent root finding; "
            "layer velocities are estimated with bounded L-BFGS-B least squares.</p>"
            "<p>Automatic waveform markers are suggestions and must be reviewed by a qualified operator.</p>",
        )


def application_stylesheet() -> str:
    """Return the complete dark engineering UI stylesheet."""

    return """
    QMainWindow, QDialog, QWidget {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: "Segoe UI";
        font-size: 10pt;
    }
    QMenuBar { background: #161b22; border-bottom: 1px solid #30363d; }
    QMenuBar::item:selected, QMenu::item:selected { background: #1f6feb; color: white; }
    QMenu { background: #161b22; border: 1px solid #30363d; }
    QFrame#panel { background: #161b22; border: 1px solid #30363d; border-radius: 7px; }
    QLabel#appTitle { color: #f0f6fc; font-size: 20pt; font-weight: 700; }
    QLabel#sectionTitle { color: #58a6ff; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
    QLabel#minorTitle { color: #f0f6fc; font-weight: 600; margin-top: 5px; }
    QLabel#subtleLabel { color: #8b949e; }
    QLabel#accentLabel { color: #58a6ff; font-weight: 600; }
    QLabel#unitBadge { background: #1f2937; color: #79c0ff; border: 1px solid #30363d; border-radius: 10px; padding: 5px 10px; }
    QPushButton {
        background: #21262d; color: #f0f6fc; border: 1px solid #48515c;
        border-radius: 5px; padding: 6px 10px;
    }
    QPushButton:hover { background: #30363d; border-color: #8b949e; }
    QPushButton:pressed { background: #161b22; }
    QPushButton:disabled { color: #6e7681; border-color: #30363d; }
    QPushButton#primaryButton { background: #238636; border: 1px solid #2ea043; font-size: 11pt; font-weight: 700; }
    QPushButton#primaryButton:hover { background: #2ea043; }
    QLineEdit, QDoubleSpinBox, QComboBox {
        background: #0d1117; color: #f0f6fc; border: 1px solid #30363d;
        border-radius: 4px; padding: 5px;
    }
    QTableWidget, QListWidget {
        background: #0d1117; alternate-background-color: #111820; color: #c9d1d9;
        border: 1px solid #30363d; gridline-color: #30363d; selection-background-color: #1f6feb;
    }
    QHeaderView::section { background: #21262d; color: #c9d1d9; border: 0; border-right: 1px solid #30363d; padding: 6px; }
    QTabWidget::pane { border: 1px solid #30363d; background: #0d1117; }
    QTabBar::tab { background: #161b22; color: #8b949e; padding: 8px 14px; border: 1px solid #30363d; }
    QTabBar::tab:selected { color: #f0f6fc; border-bottom: 2px solid #58a6ff; }
    QGroupBox { border: 1px solid #30363d; border-radius: 5px; margin-top: 9px; padding-top: 7px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #8b949e; }
    QSlider::groove:horizontal { height: 5px; background: #30363d; border-radius: 2px; }
    QSlider::handle:horizontal { background: #58a6ff; width: 15px; margin: -5px 0; border-radius: 7px; }
    QStatusBar { background: #161b22; border-top: 1px solid #30363d; color: #8b949e; }
    QSplitter::handle { background: #0d1117; width: 7px; }
    QScrollBar:vertical { background: #0d1117; width: 11px; }
    QScrollBar::handle:vertical { background: #30363d; min-height: 24px; border-radius: 5px; }
    QToolTip { color: #f0f6fc; background: #21262d; border: 1px solid #48515c; }
    """


def run_self_test() -> int:
    """Run a deterministic numerical round-trip test without starting the GUI."""

    depths = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0])
    thicknesses = np.diff(np.r_[0.0, depths])
    expected_vs = np.asarray([180.0, 210.0, 240.0, 280.0, 320.0])
    times, rays = forward_model(thicknesses, expected_vs, 2.4)
    assert times.shape == depths.shape
    assert all(abs(np.sum(ray.horizontal_segments_m) - 2.4) < 1.0e-8 for ray in rays)
    result = invert_velocity_profile(depths, times, 2.4, regularization=0.0)
    assert result.rmse_s < 1.0e-6, result.rmse_s
    print(f"RayPath SCPT self-test passed; RMSE={result.rmse_s * 1000.0:.6f} ms")
    return 0


def main() -> int:
    """Create the Qt application and enter its event loop."""

    if "--self-test" in sys.argv:
        return run_self_test()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("RayPath SCPT")
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0d1117"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#c9d1d9"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0d1117"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#c9d1d9"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#21262d"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f0f6fc"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1f6feb"))
    app.setPalette(palette)
    app.setStyleSheet(application_stylesheet())
    window = RayPathMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
