"""RayPath SCPT - forward ray-path modelling and SCPT velocity inversion.

This single-file desktop application imports GOnsite/GORILLA ``.GRU`` seismic
records, supports review and manual picking of opposing shear-wave traces, and
estimates a layered shear-wave velocity profile by regularised least squares.

All public engineering quantities use SI units: metres, milliseconds, and
metres per second.  Angles in the numerical core are measured from vertical.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


def _relaunch_outside_incompatible_ipython_qt() -> None:
    """Relaunch from Spyder when its kernel has already loaded PyQt5.

    Qt does not permit PyQt5 and PySide6 in the same process.  Spyder commonly
    loads PyQt5 before ``%runfile`` executes user code.  In that one situation,
    start this same script with the same interpreter in a clean child process;
    a guard environment variable prevents accidental recursive relaunches.
    """

    pyqt5_loaded = any(name == "PyQt5" or name.startswith("PyQt5.") for name in sys.modules)
    if __name__ != "__main__" or not pyqt5_loaded:
        return
    if os.environ.get("RAYPATH_SCPT_CLEAN_PROCESS") == "1":
        raise RuntimeError(
            "A clean RayPath process still loaded PyQt5 before PySide6. "
            "Run this file from an Anaconda Prompt instead of inside Spyder."
        )
    environment = os.environ.copy()
    environment["RAYPATH_SCPT_CLEAN_PROCESS"] = "1"
    candidates: list[str] = []

    def add_candidate(value: str | None) -> None:
        if not value:
            return
        resolved = str(Path(value).expanduser())
        if resolved not in candidates and Path(resolved).is_file():
            candidates.append(resolved)

    add_candidate(os.environ.get("RAYPATH_SCPT_PYTHON"))
    add_candidate(sys.executable)
    # The Windows Python launcher provides reliable absolute paths even when
    # Spyder has replaced PATH with its Anaconda environment.
    if os.name == "nt" and shutil.which("py"):
        try:
            discovery = subprocess.run(
                ["py", "-0p"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in discovery.stdout.splitlines():
                match = re.search(r"([A-Za-z]:\\.*?python(?:\.exe)?)\s*$", line, re.IGNORECASE)
                if match:
                    add_candidate(match.group(1))
        except (OSError, subprocess.SubprocessError):
            pass
    add_candidate(shutil.which("python"))

    compatible_python: str | None = None
    probe = "import numpy, scipy, matplotlib, PySide6.QtCore"
    for candidate in candidates:
        try:
            checked = subprocess.run(
                [candidate, "-c", probe],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if checked.returncode == 0:
                compatible_python = candidate
                break
        except (OSError, subprocess.SubprocessError):
            continue
    if compatible_python is None:
        raise RuntimeError(
            "No clean Python interpreter with NumPy, SciPy, Matplotlib, and PySide6 could be found. "
            "Set RAYPATH_SCPT_PYTHON to a compatible python.exe or run `python -m pip install -r requirements.txt`."
        )

    script = Path(__file__).resolve()
    try:
        subprocess.Popen(
            [compatible_python, str(script), *sys.argv[1:]],
            cwd=str(script.parent),
            env=environment,
            close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to launch RayPath SCPT in a clean Qt process: {exc}") from exc
    print("RayPath SCPT opened in a separate process because Spyder is using PyQt5.")
    raise SystemExit(0)


_relaunch_outside_incompatible_ipython_qt()

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
from matplotlib.lines import Line2D


APP_NAME = "RayPath SCPT"
PROJECT_SUFFIX = ".rpscpt"
GRU_PRE_TRIGGER_MS = 50.0
VELOCITY_MIN = 50.0
VELOCITY_MAX = 2000.0
PICK_KINDS = ("first_peak", "first_cross", "max_peak")
PICK_SEQUENCE = tuple((kind, channel) for kind in PICK_KINDS for channel in (17, 18))
PICK_COLUMNS = {kind: index + 1 for index, kind in enumerate(PICK_KINDS)}
PICK_LABELS = {
    "first_peak": "First peak",
    "first_cross": "First cross",
    "max_peak": "Maximum peak",
}
MODEL_COLORS = {
    "first_peak": "#ff9b54",
    "first_cross": "#4da3ff",
    "max_peak": "#d97cff",
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


@dataclass(frozen=True)
class Vs30Result:
    """Travel-time average through the upper 30 metres of a layered model."""

    value_mps: float
    vertical_travel_time_s: float
    included_thicknesses_m: np.ndarray
    included_velocities_mps: np.ndarray
    measured_depth_m: float
    extrapolated_thickness_m: float
    extrapolated_velocity_mps: float | None
    extrapolation_weight_factor: float


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


def calculate_vs30(
    depths_m: Sequence[float],
    velocities_mps: Sequence[float],
    extrapolation_weight_factor: float = 1.0,
) -> Vs30Result:
    """Calculate the travel-time-weighted Vs of the upper 30 metres.

    ``Vs30 = 30 / sum(h_i / Vs_i)`` where the included layer thicknesses sum
    to exactly 30 m.  If a modeled layer crosses 30 m, only its portion above
    30 m is included.  Profiles reaching at least 25 m may be extrapolated to
    30 m.  The missing interval velocity is a thickness- and depth-weighted
    harmonic mean of the measured intervals.  A factor of 1.0 is neutral;
    values below 1.0 emphasize shallow intervals and values above 1.0
    emphasize deep intervals.  The real 0-to-measured-depth travel time is
    never reweighted.
    """

    depths = np.asarray(depths_m, dtype=float)
    velocities = np.asarray(velocities_mps, dtype=float)
    if depths.ndim != 1 or velocities.ndim != 1 or depths.size == 0 or depths.size != velocities.size:
        raise ValueError("Vs30 requires matching non-empty depth and velocity arrays.")
    if not np.all(np.isfinite(depths)) or not np.all(depths > 0.0) or not np.all(np.diff(depths) > 0.0):
        raise ValueError("Vs30 layer depths must be finite, positive, and strictly increasing.")
    if not np.all(np.isfinite(velocities)) or not np.all(velocities > 0.0):
        raise ValueError("Vs30 layer velocities must be finite and greater than zero.")
    factor = float(extrapolation_weight_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("The Vs30 extrapolation weighting factor must be finite and greater than zero.")
    target_depth_m = 30.0
    minimum_measured_depth_m = 25.0
    measured_depth = float(depths[-1])
    if measured_depth < minimum_measured_depth_m - 1.0e-9:
        raise ValueError(
            f"The modeled profile reaches {measured_depth:.2f} m; Vs30 extrapolation requires at least "
            f"{minimum_measured_depth_m:.2f} m of measured profile."
        )
    tops = np.r_[0.0, depths[:-1]]
    measured_included = np.maximum(0.0, np.minimum(depths, min(target_depth_m, measured_depth)) - tops)
    mask = measured_included > 0.0
    included_thicknesses = measured_included[mask]
    included_velocities = velocities[mask]
    extrapolated_thickness = max(0.0, target_depth_m - measured_depth)
    extrapolated_velocity: float | None = None
    if extrapolated_thickness > 1.0e-9:
        interval_tops = tops[mask]
        interval_bottoms = np.minimum(depths[mask], measured_depth)
        interval_midpoints = 0.5 * (interval_tops + interval_bottoms)
        # Exponents run approximately from -1 at the surface to +1 at the
        # measured base, making reciprocal factors mirror each other.
        depth_positions = 2.0 * interval_midpoints / measured_depth - 1.0
        weights = np.power(factor, depth_positions)
        weighted_thickness = weights * included_thicknesses
        extrapolated_velocity = float(
            np.sum(weighted_thickness) / np.sum(weighted_thickness / included_velocities)
        )
        included_thicknesses = np.r_[included_thicknesses, extrapolated_thickness]
        included_velocities = np.r_[included_velocities, extrapolated_velocity]
    if not math.isclose(float(np.sum(included_thicknesses)), target_depth_m, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("The model does not provide continuous velocity coverage from 0 to 30 m.")
    vertical_time = float(np.sum(included_thicknesses / included_velocities))
    if not math.isfinite(vertical_time) or vertical_time <= 0.0:
        raise ValueError("The calculated vertical travel time to 30 m is invalid.")
    return Vs30Result(
        value_mps=target_depth_m / vertical_time,
        vertical_travel_time_s=vertical_time,
        included_thicknesses_m=included_thicknesses,
        included_velocities_mps=included_velocities,
        measured_depth_m=measured_depth,
        extrapolated_thickness_m=extrapolated_thickness,
        extrapolated_velocity_mps=extrapolated_velocity,
        extrapolation_weight_factor=factor,
    )


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
    pre_trigger_ms: float = GRU_PRE_TRIGGER_MS
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


def parse_gru(path: str | Path, pre_trigger_ms: float = GRU_PRE_TRIGGER_MS) -> list[WaveformRecord]:
    """Stream a GRU file and return paired traces on a trigger-relative clock.

    GOnsite/GORILLA GRU files do not declare their 50 ms pre-trigger recording
    period.  The sample times in the file are therefore shifted by
    ``pre_trigger_ms`` during import: a recorded time of 50.0 ms becomes the
    physical trigger at 0.0 ms, and only post-trigger pick times feed the
    velocity inversion.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"GRU file not found: {source}")
    pre_trigger = float(pre_trigger_ms)
    if not math.isfinite(pre_trigger) or pre_trigger < 0.0:
        raise ValueError("GRU pre-trigger duration must be finite and non-negative.")

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
        recorded_time = np.asarray(times, dtype=float)
        if not np.all(np.diff(recorded_time) > 0.0):
            raise GruFormatError(f"Seismic test {current[0]} has non-increasing sample times.")
        t = recorded_time - pre_trigger
        if pre_trigger > 0.0 and not (t[0] < 0.0 < t[-1]):
            raise GruFormatError(
                f"Seismic test {current[0]} does not span the configured {pre_trigger:g} ms trigger time."
            )
        records.append(
            WaveformRecord(
                test_number=current[0],
                depth_m=current[1],
                time_ms=t,
                left=np.asarray(left, dtype=float),
                right=np.asarray(right, dtype=float),
                pre_trigger_ms=pre_trigger,
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
    pre_trigger_indices = np.flatnonzero(t < 0.0)
    if pre_trigger_indices.size >= 3:
        baseline_indices = pre_trigger_indices
        post_trigger_start = int(np.searchsorted(t, 0.0, side="left"))
    else:
        baseline_count = max(10, min(y.size // 8, int(np.searchsorted(t, min(10.0, t[-1] * 0.08)))))
        baseline_count = min(max(baseline_count, 3), y.size)
        baseline_indices = np.arange(baseline_count)
        post_trigger_start = 0
    baseline = float(np.median(y[baseline_indices]))
    centred = y - baseline
    noise = centred[baseline_indices]
    sigma = 1.4826 * float(np.median(np.abs(noise - np.median(noise))))
    sigma = max(sigma, float(np.std(noise)) * 0.5, np.finfo(float).eps)
    envelope = np.abs(centred)
    threshold = max(5.0 * sigma, 0.04 * float(np.max(envelope)))
    above = envelope >= threshold
    sustained = np.convolve(above.astype(int), np.ones(3, dtype=int), mode="same") >= 2
    candidates = np.flatnonzero(sustained & (np.arange(y.size) >= post_trigger_start))
    onset = int(candidates[0]) if candidates.size else int(np.argmax(envelope))
    onset = max(onset, post_trigger_start)

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
        ("first_peak", 17): "#79c0ff",
        ("first_peak", 18): "#ffa198",
        ("first_cross", 17): "#3b82f6",
        ("first_cross", 18): "#f85149",
        ("max_peak", 17): "#1f6feb",
        ("max_peak", 18): "#da3633",
    }

    def __init__(self, records: list[WaveformRecord], source_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.records = records
        self.source_name = source_name
        self.active_kind = "first_peak"
        self.active_channel = 17
        self.marker_buttons: dict[tuple[str, int], QRadioButton] = {}
        self._click_connection: int | None = None
        self.setWindowTitle(f"Waveform Picker — {source_name}")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.resize(1200, 760)
        self.setModal(True)
        self._build_ui()
        self._populate_record_list()
        self.record_list.setCurrentRow(0)
        self._click_connection = self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        guidance = QLabel(
            "Guided picking order: First peak Left/blue, Right/red; First cross Left/blue, Right/red; "
            "Maximum peak Left/blue, Right/red. Each click advances automatically. Automatic suggestions "
            f"are review aids only. GRU times include the {GRU_PRE_TRIGGER_MS:g} ms pre-trigger correction."
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
        navigation_row = QHBoxLayout()
        navigation_row.addWidget(self.toolbar, 1)
        self.pick_mode_button = QPushButton("Return to Pick Mode")
        self.pick_mode_button.setToolTip(
            "Turn off zoom or pan mode. The next waveform click will place the currently selected marker."
        )
        self.pick_mode_button.clicked.connect(self._return_to_pick_mode)
        navigation_row.addWidget(self.pick_mode_button)
        plot_layout.addLayout(navigation_row)
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
                button.setStyleSheet(f"color: {'#58a6ff' if channel == 17 else '#ff7b72'};")
                button.setProperty("pick_kind", kind)
                button.setProperty("channel", channel)
                button.toggled.connect(self._marker_selected)
                self.marker_group.addButton(button)
                self.marker_buttons[(kind, channel)] = button
                kind_layout.addWidget(button)
                if kind == self.active_kind and channel == self.active_channel:
                    button.setChecked(True)
            selector_layout.addWidget(kind_box)
        plot_layout.addWidget(selector)
        plot_layout.addWidget(self.active_label)
        self.pick_values_label = QLabel()
        self.pick_values_label.setObjectName("pickValuesLabel")
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
            self._select_marker("first_peak", 17)
            self._draw_record()

    def _marker_selected(self, checked: bool) -> None:
        if not checked:
            return
        button = self.sender()
        self.active_kind = str(button.property("pick_kind"))
        self.active_channel = int(button.property("channel"))
        self._update_active_label()

    def _update_active_label(self) -> None:
        step = PICK_SEQUENCE.index((self.active_kind, self.active_channel)) + 1
        self.active_label.setText(
            f"Step {step}/6 — Active: {PICK_LABELS[self.active_kind]} — "
            f"{CHANNEL_LABELS[self.active_channel]} ({'blue' if self.active_channel == 17 else 'red'})"
        )

    def _select_marker(self, kind: str, channel: int) -> None:
        """Select a guided marker without requiring a manual radio-button click."""

        button = self.marker_buttons.get((kind, channel))
        if button is not None:
            button.setChecked(True)
        else:
            self.active_kind = kind
            self.active_channel = channel
            self._update_active_label()

    def _return_to_pick_mode(self) -> None:
        """Deactivate Matplotlib navigation so plot clicks place markers again."""

        mode_name = getattr(self.toolbar.mode, "name", str(self.toolbar.mode)).lower()
        if "zoom" in mode_name:
            self.toolbar.zoom()
        elif "pan" in mode_name:
            self.toolbar.pan()
        self.canvas.setFocus(Qt.FocusReason.MouseFocusReason)
        self._update_active_label()

    def _on_plot_click(self, event: Any) -> None:
        row = self.record_list.currentRow()
        if row < 0 or self.toolbar.mode or event.inaxes is not self.canvas.axes or event.xdata is None:
            return
        record = self.records[row]
        positive_samples = record.time_ms[record.time_ms > 0.0]
        earliest_arrival = float(positive_samples[0]) if positive_samples.size else 0.0
        time_ms = float(np.clip(event.xdata, earliest_arrival, record.time_ms[-1]))
        record.set_pick(self.active_kind, self.active_channel, time_ms)
        self._refresh_list_item(row)
        self._draw_record(preserve_view=True)
        self.picks_changed.emit()
        self._advance_pick_workflow()

    def _advance_pick_workflow(self) -> None:
        """Advance to the next marker or prompt when all six are complete."""

        current = (self.active_kind, self.active_channel)
        index = PICK_SEQUENCE.index(current)
        if index < len(PICK_SEQUENCE) - 1:
            self._select_marker(*PICK_SEQUENCE[index + 1])
            return
        self._prompt_interval_complete()

    def _prompt_interval_complete(self) -> None:
        """Save the completed interval and advance, or clear it for re-picking."""

        row = self.record_list.currentRow()
        if row < 0:
            return
        record = self.records[row]
        last_record = row >= len(self.records) - 1
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Receiver interval complete")
        box.setText(f"All six picks are complete for {record.depth_m:.2f} m depth.")
        box.setInformativeText(
            "Save these picks and finish the profile?" if last_record
            else "Save these picks and move to the next depth interval, or clear them and re-pick this interval?"
        )
        save_button = box.addButton(
            "Save & Finish" if last_record else "Save & Next",
            QMessageBox.ButtonRole.AcceptRole,
        )
        repick_button = box.addButton("Re-pick", QMessageBox.ButtonRole.ResetRole)
        box.setDefaultButton(save_button)
        box.exec()
        if box.clickedButton() is repick_button:
            for kind, channel in PICK_SEQUENCE:
                record.set_pick(kind, channel, None)
            self._refresh_list_item(row)
            self._draw_record(preserve_view=True)
            self._select_marker("first_peak", 17)
            return
        if box.clickedButton() is save_button:
            if last_record:
                self.accept()
            else:
                self.record_list.setCurrentRow(row + 1)

    def _clear_active_pick(self) -> None:
        row = self.record_list.currentRow()
        if row < 0:
            return
        self.records[row].set_pick(self.active_kind, self.active_channel, None)
        self._refresh_list_item(row)
        self._draw_record(preserve_view=True)

    def _resuggest_current(self) -> None:
        row = self.record_list.currentRow()
        if row < 0:
            return
        add_suggested_picks([self.records[row]], overwrite=True)
        self._refresh_list_item(row)
        self._draw_record(preserve_view=True)
        self._select_marker("first_peak", 17)

    def _refresh_list_item(self, row: int) -> None:
        item = self.record_list.item(row)
        if item:
            item.setText(self._record_item_text(self.records[row]))

    def _draw_record(self, preserve_view: bool = False) -> None:
        """Render the active traces, optionally retaining the current zoom."""

        row = self.record_list.currentRow()
        if row < 0:
            return
        record = self.records[row]
        ax = self.canvas.axes
        previous_xlim = ax.get_xlim() if preserve_view else None
        previous_ylim = ax.get_ylim() if preserve_view else None
        self.canvas.clear()
        if record.time_ms[0] < 0.0:
            ax.axvspan(record.time_ms[0], 0.0, color="#8b949e", alpha=0.10, label="Pre-trigger")
        ax.axvline(0.0, color="#3fb950", linewidth=1.4, alpha=0.95, label="Trigger")
        ax.axhline(0.0, color="#8b949e", linewidth=0.8, alpha=0.8)
        ax.plot(record.time_ms, record.left, color="#58a6ff", linewidth=1.0, label="Left (#17)")
        ax.plot(record.time_ms, record.right, color="#ff7b72", linewidth=1.0, label="Right (#18)")
        for kind in PICK_KINDS:
            for channel in (17, 18):
                pick = record.get_pick(kind, channel)
                if pick is None:
                    continue
                color = self.PICK_COLORS[(kind, channel)]
                linestyle = {"first_peak": "--", "first_cross": "-", "max_peak": ":"}[kind]
                ax.axvline(pick, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9)
        ax.set_title(f"Test {record.test_number} — receiver depth {record.depth_m:.2f} m")
        ax.set_xlabel("Time relative to trigger (ms)")
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
        if previous_xlim is not None and previous_ylim is not None:
            ax.set_xlim(previous_xlim)
            ax.set_ylim(previous_ylim)
        else:
            self._apply_max_peak_zoom(record)
        self.canvas.draw_idle()

    def _apply_max_peak_zoom(self, record: WaveformRecord) -> None:
        """Show a 50 ms window centered on the mean left/right maximum peak."""

        centers = [record.get_pick("max_peak", channel) for channel in (17, 18)]
        finite_centers = [value for value in centers if value is not None]
        if not finite_centers:
            finite_centers = [
                suggest_trace_picks(record.time_ms, values)["max_peak"]
                for values in (record.left, record.right)
            ]
        center = float(np.mean(finite_centers))
        full_left = float(record.time_ms[0])
        full_right = float(record.time_ms[-1])
        left = center - 25.0
        right = center + 25.0
        if left < full_left:
            right += full_left - left
            left = full_left
        if right > full_right:
            left -= right - full_right
            right = full_right
        self.canvas.axes.set_xlim(max(full_left, left), min(full_right, right))


class InversionWorker(QObject):
    """Run one or more pick-based inversions away from the Qt GUI thread."""

    finished = Signal(object)
    failed = Signal(str, str)
    progress = Signal(str, int, float)

    def __init__(
        self,
        depths_m: np.ndarray,
        times_by_pick_s: dict[str, np.ndarray],
        offset_m: float,
        regularization: float,
    ) -> None:
        super().__init__()
        self.depths_m = depths_m
        self.times_by_pick_s = times_by_pick_s
        self.offset_m = offset_m
        self.regularization = regularization

    @Slot()
    def run(self) -> None:
        try:
            results: dict[str, InversionResult] = {}
            for kind, times_s in self.times_by_pick_s.items():
                results[kind] = invert_velocity_profile(
                    self.depths_m,
                    times_s,
                    self.offset_m,
                    self.regularization,
                    lambda evaluations, rmse, pick_kind=kind: self.progress.emit(pick_kind, evaluations, rmse),
                )
            self.finished.emit(results)
        except Exception as exc:  # GUI boundary: present all solver/parser failures cleanly.
            self.failed.emit(str(exc), traceback.format_exc())


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class RayPathMainWindow(QMainWindow):
    """Strict left-to-right engineering workflow for RayPath SCPT."""

    def __init__(self) -> None:
        super().__init__()
        self.project_path: Path | None = None
        self.gru_path: Path | None = None
        self.waveform_records: list[WaveformRecord] = []
        self.result: InversionResult | None = None
        self.comparison_results: dict[str, InversionResult] = {}
        self.comparison_vs30: dict[str, Vs30Result | None] = {}
        self.comparison_vs30_reasons: dict[str, str] = {}
        self.current_vs30: Vs30Result | None = None
        self.vs30_unavailable_reason: str | None = None
        self.vs30_history: dict[tuple[str, float, float], float] = {}
        self._thread: QThread | None = None
        self._worker: InversionWorker | None = None
        self._dirty = False
        self.setWindowTitle(APP_NAME)
        self.resize(1650, 900)
        self.setMinimumSize(1200, 720)
        self._build_actions_and_menu()
        self._build_ui()
        self._set_empty_project()
        self._set_dirty(False)

    # ---- UI construction -------------------------------------------------

    def _build_actions_and_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        self.new_action = QAction("&New", self, shortcut=QKeySequence.StandardKey.New, triggered=self.new_project)
        self.open_action = QAction("&Open…", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_file)
        self.save_action = QAction("&Save", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.save_project)
        self.save_as_action = QAction("Save &As…", self, shortcut=QKeySequence.StandardKey.SaveAs, triggered=self.save_project_as)
        self.export_action = QAction("Export &CSV…", self, triggered=self.export_csv)
        self.export_pdf_action = QAction("Export PDF &Report…", self, triggered=self.export_pdf_report)
        self.exit_action = QAction("E&xit", self, shortcut=QKeySequence.StandardKey.Quit, triggered=self.close)
        for action in (self.new_action, self.open_action, self.save_action, self.save_as_action):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self.export_action)
        menu.addAction(self.export_pdf_action)
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
        self.status_label = QLabel("Ready — import a GRU file or enter observations")
        status.addWidget(self.status_label, 1)
        self.rmse_label = QLabel("RMSE: — ms")
        status.addPermanentWidget(self.rmse_label)
        self.vs30_status_label = QLabel("Vs30: — m/s")
        status.addPermanentWidget(self.vs30_status_label)

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
        self.gru_label = QLabel("No GRU source loaded")
        self.gru_label.setWordWrap(True)
        self.gru_label.setObjectName("subtleLabel")
        layout.addWidget(self.gru_label)

        table_label = QLabel("Receiver observations")
        table_label.setObjectName("minorTitle")
        layout.addWidget(table_label)
        self.input_table = PasteTableWidget(0, 4)
        self.input_table.setHorizontalHeaderLabels(
            ["Depth z (m)", "First peak (ms)", "First cross (ms)", "Max peak (ms)"]
        )
        self.input_table.setToolTip(
            f"GRU arrivals are measured from the trigger after subtracting the {GRU_PRE_TRIGGER_MS:g} ms pre-trigger period."
        )
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
        self.waterfall_canvas = MplCanvas()
        self.plot_tabs.addTab(self.velocity_canvas, "Velocity Profile")
        self.plot_tabs.addTab(self.ray_canvas, "Ray Paths")
        self.plot_tabs.addTab(self.fit_canvas, "Arrival-Time Fit")
        self.plot_tabs.addTab(self.waterfall_canvas, "Waveform Waterfall")
        self.plot_tabs.addTab(self._build_vs30_tab(), "Vs30 Analysis")
        layout.addWidget(self.plot_tabs, 1)
        self._draw_empty_plots()
        return panel

    def _build_vs30_tab(self) -> QWidget:
        """Create the Vs30 value and smoothing-sensitivity comparison view."""

        tab = QWidget()
        layout = QVBoxLayout(tab)
        metric_row = QHBoxLayout()
        self.vs30_value_label = QLabel("Vs30: — m/s")
        self.vs30_value_label.setObjectName("vs30Value")
        metric_row.addWidget(self.vs30_value_label)
        metric_row.addStretch()
        self.vs30_smoothing_label = QLabel("Smoothing: —")
        self.vs30_smoothing_label.setObjectName("accentLabel")
        metric_row.addWidget(self.vs30_smoothing_label)
        layout.addLayout(metric_row)
        self.vs30_comparison_label = QLabel("First peak: —   |   First cross: —   |   Maximum peak: —")
        self.vs30_comparison_label.setObjectName("minorTitle")
        self.vs30_comparison_label.setWordWrap(True)
        layout.addWidget(self.vs30_comparison_label)
        self.vs30_detail_label = QLabel(
            "Run a model extending to at least 25 m. Profiles from 25–30 m are extrapolated to 30 m."
        )
        self.vs30_detail_label.setWordWrap(True)
        self.vs30_detail_label.setObjectName("subtleLabel")
        layout.addWidget(self.vs30_detail_label)
        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("Extrapolation interval weighting"))
        weight_row.addStretch()
        self.extrapolation_weight_label = QLabel("1.00 — neutral")
        self.extrapolation_weight_label.setObjectName("accentLabel")
        weight_row.addWidget(self.extrapolation_weight_label)
        layout.addLayout(weight_row)
        self.extrapolation_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.extrapolation_weight_slider.setRange(-100, 100)
        self.extrapolation_weight_slider.setValue(0)
        self.extrapolation_weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.extrapolation_weight_slider.setTickInterval(25)
        self.extrapolation_weight_slider.setToolTip(
            "Used only to estimate the unmeasured interval between a 25–30 m profile base and 30 m."
        )
        self.extrapolation_weight_slider.valueChanged.connect(self._extrapolation_weight_changed)
        layout.addWidget(self.extrapolation_weight_slider)
        weight_endpoints = QHBoxLayout()
        weight_endpoints.addWidget(QLabel("0.25  emphasize shallower intervals"))
        weight_endpoints.addStretch()
        weight_endpoints.addWidget(QLabel("1.00  neutral"))
        weight_endpoints.addStretch()
        weight_endpoints.addWidget(QLabel("4.00  emphasize deeper intervals"))
        layout.addLayout(weight_endpoints)
        weight_note = QLabel(
            "Weighting estimates only the missing interval to 30 m. Measured-layer travel times are not altered."
        )
        weight_note.setObjectName("subtleLabel")
        weight_note.setWordWrap(True)
        layout.addWidget(weight_note)
        self.vs30_canvas = MplCanvas()
        layout.addWidget(self.vs30_canvas, 1)
        return tab

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
        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(
            [
                "Layer",
                "Top depth",
                "Bottom depth",
                "First peak Vs",
                "First cross Vs",
                "Max peak Vs",
                "Selected fitting error",
            ]
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
        pdf_button = QPushButton("Export PDF Report…")
        pdf_button.clicked.connect(self.export_pdf_report)
        layout.addWidget(pdf_button)
        return panel

    # ---- project state ---------------------------------------------------

    def _set_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        name = self.project_path.name if self.project_path else "Untitled"
        self.setWindowTitle(f"{APP_NAME} — {name}{' *' if dirty else ''}")

    def _input_changed(self, *_args: Any) -> None:
        if self.result is not None or self.vs30_history or self.result_table.rowCount() > 0:
            self._clear_results(clear_vs30_history=True)
        self._set_dirty(True)

    def _regularization_changed(self, value: int) -> None:
        self.reg_value_label.setText(f"{value / 100.0:.2f}")
        self._clear_results(clear_vs30_history=False)
        if self.vs30_history:
            self.vs30_detail_label.setText(
                "Smoothing changed. Run the inversion to add this setting to the Vs30 comparison."
            )
        self._set_dirty(True)

    def _extrapolation_weight_factor(self) -> float:
        """Map the symmetric slider position to a logarithmic 0.25–4 factor."""

        return float(4.0 ** (self.extrapolation_weight_slider.value() / 100.0))

    def _extrapolation_weight_changed(self, _value: int) -> None:
        factor = self._extrapolation_weight_factor()
        if factor < 1.0 - 1.0e-9:
            direction = "shallower emphasis"
        elif factor > 1.0 + 1.0e-9:
            direction = "deeper emphasis"
        else:
            direction = "neutral"
        self.extrapolation_weight_label.setText(f"{factor:.2f} — {direction}")
        if self.result is not None:
            self._update_all_vs30_results(record_history=True)
            self._activate_comparison_result(str(self.estimator_combo.currentData()))
        else:
            self._draw_vs30_analysis()
        self._set_dirty(True)

    def _estimator_changed(self, _index: int) -> None:
        selected_kind = str(self.estimator_combo.currentData())
        if selected_kind in self.comparison_results:
            self._activate_comparison_result(selected_kind)
        else:
            self._clear_results(clear_vs30_history=False)
        self._set_dirty(True)

    def _set_empty_project(self) -> None:
        """Reset the workspace to empty editable rows with no demonstration data."""

        self.waveform_records = []
        self.gru_path = None
        self.review_action.setEnabled(False)
        self.gru_label.setText("No GRU source loaded")
        self.input_table.blockSignals(True)
        self.input_table.clearContents()
        self.input_table.setRowCount(8)
        self.input_table.blockSignals(False)
        self.offset_spin.setValue(2.4)
        self.reg_slider.setValue(35)
        self.estimator_combo.setCurrentIndex(1)
        self.extrapolation_weight_slider.setValue(0)
        self._clear_results()
        self.status_label.setText("Ready — import a GRU file or enter observations")

    def _set_input_rows(
        self,
        rows: Iterable[tuple[float, float | None]],
        pick_kind: str | None = None,
    ) -> None:
        """Populate depth plus one legacy/CSV arrival column."""

        self.input_table.blockSignals(True)
        values = list(rows)
        self.input_table.setRowCount(len(values))
        self.input_table.clearContents()
        column = PICK_COLUMNS.get(pick_kind or str(self.estimator_combo.currentData()), 2)
        for row, (depth, time_ms) in enumerate(values):
            self.input_table.setItem(row, 0, QTableWidgetItem(f"{depth:.3f}"))
            self.input_table.setItem(row, column, QTableWidgetItem("" if time_ms is None else f"{time_ms:.3f}"))
        self.input_table.blockSignals(False)

    def _set_all_pick_rows(self, rows: Iterable[tuple[float, dict[str, float | None]]]) -> None:
        """Populate all three post-trigger pick columns."""

        values = list(rows)
        self.input_table.blockSignals(True)
        self.input_table.setRowCount(len(values))
        self.input_table.clearContents()
        for row, (depth, picks) in enumerate(values):
            self.input_table.setItem(row, 0, QTableWidgetItem(f"{depth:.3f}"))
            for kind, column in PICK_COLUMNS.items():
                value = picks.get(kind)
                self.input_table.setItem(
                    row,
                    column,
                    QTableWidgetItem("" if value is None else f"{float(value):.3f}"),
                )
        self.input_table.blockSignals(False)

    def _clear_results(self, clear_vs30_history: bool = True) -> None:
        """Clear the active model, optionally retaining smoothing comparisons."""

        self.result = None
        self.comparison_results.clear()
        self.comparison_vs30.clear()
        self.comparison_vs30_reasons.clear()
        self.current_vs30 = None
        self.vs30_unavailable_reason = None
        if clear_vs30_history:
            self.vs30_history.clear()
        self.result_table.setRowCount(0)
        self.result_summary.setText("Run the inversion to calculate a layered Vs profile.")
        self.rmse_label.setText("RMSE: — ms")
        self.vs30_status_label.setText("Vs30: — m/s")
        self.vs30_comparison_label.setText("First peak: —   |   First cross: —   |   Maximum peak: —")
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
        self._draw_waveform_waterfall()
        self._draw_vs30_analysis()

    def _draw_waveform_waterfall(self) -> None:
        """Draw every imported left/right trace at its receiver depth."""

        self.waterfall_canvas.clear()
        self._plot_waveform_waterfall(self.waterfall_canvas.axes, dark_theme=True)
        self.waterfall_canvas.draw_idle()

    def _plot_waveform_waterfall(self, ax: Any, dark_theme: bool = False) -> None:
        """Plot normalized paired waveforms and all six picks on one axes.

        Each trace pair is centred on its receiver depth.  Normalization is
        performed per channel so that low-amplitude intervals remain visible;
        the relative amplitudes within each individual waveform are retained.
        """

        ax.set_title("Paired SCPT waveform waterfall")
        ax.set_xlabel("Time after trigger (ms)")
        ax.set_ylabel("Receiver depth (m)")
        if not self.waveform_records:
            ax.text(
                0.5,
                0.5,
                "Import a GRU file to populate",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#8b949e" if dark_theme else "#5D6D7E",
            )
            return

        records = sorted(self.waveform_records, key=lambda item: item.depth_m)
        depths = np.asarray([record.depth_m for record in records], dtype=float)
        unique_depths = np.unique(depths)
        positive_spacing = np.diff(unique_depths)
        positive_spacing = positive_spacing[positive_spacing > 0.0]
        spacing = float(np.median(positive_spacing)) if positive_spacing.size else 1.0
        trace_height = max(0.12, 0.34 * spacing)

        picked_times = [
            value
            for record in records
            for kind in PICK_KINDS
            for channel in (17, 18)
            if (value := record.get_pick(kind, channel)) is not None
        ]
        earliest_time = min(float(record.time_ms[0]) for record in records)
        latest_time = max(float(record.time_ms[-1]) for record in records)
        if picked_times:
            x_min = max(earliest_time, -10.0)
            x_max = min(latest_time, max(25.0, max(picked_times) + 25.0))
            if x_max <= x_min:
                x_min, x_max = earliest_time, latest_time
        else:
            x_min, x_max = earliest_time, latest_time

        marker_by_kind = {"first_peak": "o", "first_cross": "x", "max_peak": "^"}
        channel_data = ((17, "#58a6ff", "Left"), (18, "#ff7b72", "Right"))
        for record in records:
            for channel, color, _label in channel_data:
                values = record.left if channel == 17 else record.right
                finite = np.asarray(values, dtype=float)
                pre_trigger = finite[record.time_ms < 0.0]
                baseline = float(np.median(pre_trigger)) if pre_trigger.size else float(np.median(finite))
                centred = finite - baseline
                scale = float(np.nanmax(np.abs(centred)))
                normalized = centred / scale if math.isfinite(scale) and scale > 0.0 else np.zeros_like(centred)
                plotted = record.depth_m + trace_height * normalized
                ax.plot(record.time_ms, plotted, color=color, linewidth=0.65, alpha=0.72)
                for kind in PICK_KINDS:
                    pick_time = record.get_pick(kind, channel)
                    if pick_time is None or pick_time < record.time_ms[0] or pick_time > record.time_ms[-1]:
                        continue
                    pick_value = float(np.interp(pick_time, record.time_ms, normalized))
                    ax.scatter(
                        [pick_time],
                        [record.depth_m + trace_height * pick_value],
                        marker=marker_by_kind[kind],
                        s=26,
                        color=color,
                        linewidths=1.1,
                        zorder=5,
                    )

        ax.axvline(0.0, color="#3fb950" if dark_theme else "#238636", linewidth=1.0, alpha=0.9)
        ax.set_xlim(x_min, x_max)
        depth_margin = max(trace_height * 1.5, spacing * 0.35)
        ax.set_ylim(float(depths[-1]) + depth_margin, max(0.0, float(depths[0]) - depth_margin))
        ax.legend(
            handles=[
                Line2D([0], [0], color="#58a6ff", linewidth=1.5, label="Left / channel 17"),
                Line2D([0], [0], color="#ff7b72", linewidth=1.5, label="Right / channel 18"),
                Line2D([0], [0], color="#8b949e", marker="o", linestyle="None", label="First peak"),
                Line2D([0], [0], color="#8b949e", marker="x", linestyle="None", label="First cross"),
                Line2D([0], [0], color="#8b949e", marker="^", linestyle="None", label="Maximum peak"),
            ],
            loc="best",
            fontsize=8,
            frameon=True,
            ncol=2,
        )

    def _draw_vs30_analysis(self) -> None:
        """Draw the current Vs30 result and smoothing-sensitivity history."""

        ax = self.vs30_canvas.axes
        self.vs30_canvas.clear()
        ax.set_title("Vs30 sensitivity to smoothing")
        ax.set_xlabel("Smoothing / regularisation factor")
        ax.set_ylabel("Vs30 (m/s)")
        selected_weight = self._extrapolation_weight_factor()
        histories_by_kind: dict[str, dict[float, float]] = {kind: {} for kind in PICK_KINDS}
        for (kind, smoothing, weighting), value in self.vs30_history.items():
            if math.isclose(weighting, selected_weight, rel_tol=0.0, abs_tol=1.0e-6):
                histories_by_kind.setdefault(kind, {})[smoothing] = value
        all_factors: list[float] = []
        all_values: list[float] = []
        for kind in PICK_KINDS:
            history = histories_by_kind.get(kind, {})
            if not history:
                continue
            factors = np.asarray(sorted(history), dtype=float)
            values = np.asarray([history[factor] for factor in factors], dtype=float)
            all_factors.extend(factors.tolist())
            all_values.extend(values.tolist())
            ax.plot(
                factors,
                values,
                "-o",
                color=MODEL_COLORS[kind],
                linewidth=1.8,
                markersize=5,
                label=PICK_LABELS[kind],
            )
        if all_factors:
            minimum_factor = min(all_factors)
            maximum_factor = max(all_factors)
            if math.isclose(minimum_factor, maximum_factor):
                ax.set_xlim(max(0.0, minimum_factor - 0.1), min(1.0, maximum_factor + 0.1))
            else:
                margin = max(0.03, (maximum_factor - minimum_factor) * 0.08)
                ax.set_xlim(max(0.0, minimum_factor - margin), min(1.0, maximum_factor + margin))
            values_array = np.asarray(all_values, dtype=float)
            value_span = float(np.ptp(values_array))
            value_margin = max(5.0, value_span * 0.15)
            ax.set_ylim(
                max(0.0, float(np.min(values_array)) - value_margin),
                float(np.max(values_array)) + value_margin,
            )
            ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        else:
            ax.text(
                0.5,
                0.5,
                "Run an inversion extending to at least 25 m\nto calculate or extrapolate Vs30",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#8b949e",
            )
            ax.set_xlim(0.0, 1.0)
        if self.current_vs30 is not None:
            factor = self.reg_slider.value() / 100.0
            self.vs30_value_label.setText(f"Vs30: {self.current_vs30.value_mps:.1f} m/s")
            self.vs30_smoothing_label.setText(f"Smoothing: {factor:.2f}")
            if self.current_vs30.extrapolated_thickness_m > 0.0:
                self.vs30_detail_label.setText(
                    f"Measured profile: {self.current_vs30.measured_depth_m:.2f} m. "
                    f"Extrapolated {self.current_vs30.extrapolated_thickness_m:.2f} m at "
                    f"{self.current_vs30.extrapolated_velocity_mps:.1f} m/s using weighting factor "
                    f"{self.current_vs30.extrapolation_weight_factor:.2f}. Total vertical travel time: "
                    f"{self.current_vs30.vertical_travel_time_s * 1000.0:.2f} ms."
                )
            else:
                self.vs30_detail_label.setText(
                    f"Vertical travel time through 30.00 m: {self.current_vs30.vertical_travel_time_s * 1000.0:.2f} ms. "
                    "The modeled profile reaches 30 m, so extrapolation weighting is not applied."
                )
        elif self.vs30_unavailable_reason:
            self.vs30_value_label.setText("Vs30: unavailable")
            self.vs30_smoothing_label.setText(f"Smoothing: {self.reg_slider.value() / 100.0:.2f}")
            self.vs30_detail_label.setText(self.vs30_unavailable_reason)
        elif not all_factors:
            self.vs30_value_label.setText("Vs30: — m/s")
            self.vs30_smoothing_label.setText("Smoothing: —")
            self.vs30_detail_label.setText(
                "Run a model extending to at least 25 m. Change smoothing and rerun to build a comparison."
            )
        else:
            self.vs30_value_label.setText("Vs30: rerun required")
            self.vs30_smoothing_label.setText(f"Selected smoothing: {self.reg_slider.value() / 100.0:.2f}")
        comparison_parts = []
        for kind in PICK_KINDS:
            value = self.comparison_vs30.get(kind)
            comparison_parts.append(
                f"{PICK_LABELS[kind]}: {'—' if value is None else f'{value.value_mps:.1f} m/s'}"
            )
        self.vs30_comparison_label.setText("   |   ".join(comparison_parts))
        self.vs30_canvas.draw_idle()

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

    def _read_input_rows(self, pick_kind: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Read depth and one selected pick-time column."""

        rows: list[tuple[float, float]] = []
        errors: list[str] = []
        kind = pick_kind or str(self.estimator_combo.currentData())
        time_column = PICK_COLUMNS[kind]
        for row in range(self.input_table.rowCount()):
            depth_text = self.input_table.item(row, 0).text().strip() if self.input_table.item(row, 0) else ""
            time_text = (
                self.input_table.item(row, time_column).text().strip()
                if self.input_table.item(row, time_column)
                else ""
            )
            if not depth_text and not time_text:
                continue
            if depth_text and not time_text:
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

    def _read_all_pick_rows(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read rows having valid positive values in all three pick columns."""

        complete: list[tuple[float, dict[str, float]]] = []
        invalid_rows: list[str] = []
        for row in range(self.input_table.rowCount()):
            depth_text = self.input_table.item(row, 0).text().strip() if self.input_table.item(row, 0) else ""
            pick_texts = {
                kind: (
                    self.input_table.item(row, column).text().strip()
                    if self.input_table.item(row, column)
                    else ""
                )
                for kind, column in PICK_COLUMNS.items()
            }
            if not depth_text and not any(pick_texts.values()):
                continue
            if not depth_text or not all(pick_texts.values()):
                continue
            try:
                depth = float(depth_text)
                picks = {kind: float(text) for kind, text in pick_texts.items()}
                if depth <= 0.0 or not math.isfinite(depth):
                    raise ValueError
                if any(value <= 0.0 or not math.isfinite(value) for value in picks.values()):
                    raise ValueError
                complete.append((depth, picks))
            except ValueError:
                invalid_rows.append(str(row + 1))
        if invalid_rows:
            raise ValueError(f"Rows {', '.join(invalid_rows)} contain invalid pick values.")
        if len(complete) < 2:
            raise ValueError("At least two rows require depth and all three positive pick times for comparison.")
        complete.sort(key=lambda item: item[0])
        depths = np.asarray([depth for depth, _ in complete], dtype=float)
        if np.any(np.diff(depths) <= 0.0):
            raise ValueError("Receiver depths must be unique and strictly increasing.")
        times = {
            kind: np.asarray([picks[kind] / 1000.0 for _, picks in complete], dtype=float)
            for kind in PICK_KINDS
        }
        return depths, times

    # ---- file operations -------------------------------------------------

    @Slot()
    def new_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        self.project_path = None
        self._set_empty_project()
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
        self.gru_label.setText(
            f"{path.name} — {len(records)} paired seismic records — {GRU_PRE_TRIGGER_MS:g} ms pre-trigger corrected"
        )
        self._populate_table_from_picks()
        self._clear_results()
        self.status_label.setText(
            f"Imported {len(records)} GRU seismic records — {GRU_PRE_TRIGGER_MS:g} ms pre-trigger correction applied"
        )
        self._set_dirty(True)

    def _populate_table_from_picks(self) -> None:
        rows = [
            (record.depth_m, {kind: record.arrival_ms(kind) for kind in PICK_KINDS})
            for record in self.waveform_records
        ]
        self._set_all_pick_rows(rows)
        missing = sum(value is None for _, picks in rows for value in picks.values())
        if missing:
            self.status_label.setText(f"{missing} pick values are missing and need waveform review")

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
        rows: list[tuple[float, dict[str, float | None]]] = []
        selected_kind = str(self.estimator_combo.currentData())
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for number, values in enumerate(reader, 1):
                if not values or all(not value.strip() for value in values):
                    continue
                if len(values) < 2:
                    continue
                try:
                    depth = float(values[0])
                    picks: dict[str, float | None] = {kind: None for kind in PICK_KINDS}
                    if len(values) >= 4:
                        for kind, value in zip(PICK_KINDS, values[1:4]):
                            picks[kind] = float(value) if value.strip() else None
                    else:
                        picks[selected_kind] = float(values[1])
                    rows.append((depth, picks))
                except ValueError:
                    if number == 1:
                        continue
                    raise ValueError(f"CSV row {number} does not begin with numeric depth and arrival time.")
        if len(rows) < 2:
            raise ValueError("CSV must contain at least two depth/arrival-time rows.")
        rows.sort(key=lambda item: item[0])
        self.waveform_records = []
        self.gru_path = None
        self.project_path = None
        self.review_action.setEnabled(False)
        self.gru_label.setText(f"{path.name} — CSV observations")
        self._set_all_pick_rows(rows)
        self._clear_results()
        self.status_label.setText(f"Imported {len(rows)} CSV observations")
        self._set_dirty(True)

    def _project_payload(self) -> dict[str, Any]:
        inputs = []
        for row in range(self.input_table.rowCount()):
            depth = self.input_table.item(row, 0).text() if self.input_table.item(row, 0) else ""
            item: dict[str, str] = {"depth_m": depth}
            for kind, column in PICK_COLUMNS.items():
                item[f"{kind}_ms"] = self.input_table.item(row, column).text() if self.input_table.item(row, column) else ""
            inputs.append(item)
        payload: dict[str, Any] = {
            "format": "RayPath SCPT Project",
            "version": 3,
            "units": "SI",
            "gru_pre_trigger_ms": GRU_PRE_TRIGGER_MS,
            "pick_time_reference": "relative_to_trigger",
            "source_offset_m": self.offset_spin.value(),
            "regularization": self.reg_slider.value() / 100.0,
            "vs30_extrapolation_weight_factor": self._extrapolation_weight_factor(),
            "arrival_estimator": self.estimator_combo.currentData(),
            "vs30_history": [
                {
                    "pick_kind": kind,
                    "regularization": smoothing,
                    "weight_factor": weighting,
                    "vs30_mps": value,
                }
                for (kind, smoothing, weighting), value in sorted(self.vs30_history.items())
            ],
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
                "vs30_mps": self.current_vs30.value_mps if self.current_vs30 is not None else None,
                "vs30_extrapolated_velocity_mps": (
                    self.current_vs30.extrapolated_velocity_mps if self.current_vs30 is not None else None
                ),
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
        project_version = int(payload.get("version", 0))
        if payload.get("format") != "RayPath SCPT Project" or project_version not in (1, 2, 3):
            raise ValueError("This is not a supported RayPath SCPT project file.")
        self.project_path = path
        self.offset_spin.setValue(float(payload.get("source_offset_m", 2.4)))
        self.reg_slider.setValue(round(float(payload.get("regularization", 0.35)) * 100))
        saved_weight = float(payload.get("vs30_extrapolation_weight_factor", 1.0))
        saved_weight = float(np.clip(saved_weight, 0.25, 4.0))
        self.extrapolation_weight_slider.setValue(round(100.0 * math.log(saved_weight, 4.0)))
        estimator = str(payload.get("arrival_estimator", "first_cross"))
        estimator_index = self.estimator_combo.findData(estimator)
        self.estimator_combo.setCurrentIndex(max(0, estimator_index))

        inputs = payload.get("inputs", [])
        if project_version >= 3:
            rows_all = [
                (
                    float(item["depth_m"]),
                    {
                        kind: (float(item[f"{kind}_ms"]) if item.get(f"{kind}_ms") else None)
                        for kind in PICK_KINDS
                    },
                )
                for item in inputs
                if item.get("depth_m")
            ]
            self._set_all_pick_rows(rows_all)
        else:
            rows = [
                (float(item["depth_m"]), float(item["arrival_time_ms"]))
                for item in inputs
                if item.get("depth_m") and item.get("arrival_time_ms")
            ]
            self._set_input_rows(rows, str(payload.get("arrival_estimator", "first_cross")))
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
                loaded_picks = dict(saved.get((record.test_number, record.depth_m), {}))
                if project_version == 1:
                    # Version 1 stored raw GRU record times and did not account
                    # for the undocumented pre-trigger period.
                    loaded_picks = {
                        key: (None if value is None else float(value) - GRU_PRE_TRIGGER_MS)
                        for key, value in loaded_picks.items()
                    }
                record.picks_ms = loaded_picks
            self.gru_label.setText(
                f"{self.gru_path.name} — {len(self.waveform_records)} paired seismic records — "
                f"{GRU_PRE_TRIGGER_MS:g} ms pre-trigger corrected"
            )
            if project_version < 3:
                self._populate_table_from_picks()
        elif self.gru_path:
            self.gru_label.setText(f"GRU source unavailable: {self.gru_path}")
        else:
            self.gru_label.setText("Project contains manually entered observations")
        self.review_action.setEnabled(bool(self.waveform_records))
        self._clear_results()
        self.vs30_history = {
            (
                str(item.get("pick_kind", payload.get("arrival_estimator", "first_cross"))),
                float(item["regularization"]),
                round(float(item.get("weight_factor", 1.0)), 6),
            ): float(item["vs30_mps"])
            for item in payload.get("vs30_history", [])
            if "regularization" in item and "vs30_mps" in item
        }
        self._draw_vs30_analysis()
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
                        "First Peak Vs (m/s)",
                        "First Cross Vs (m/s)",
                        "Maximum Peak Vs (m/s)",
                        "Observed Post-Trigger Travel Time (ms)",
                        "Calculated Post-Trigger Travel Time (ms)",
                        "Fitting Error (ms)",
                        "Ray Parameter (s/m)",
                        "Regularization Factor",
                        "First Peak Vs30 (m/s)",
                        "First Cross Vs30 (m/s)",
                        "Maximum Peak Vs30 (m/s)",
                        "Vs30 Extrapolation Weight Factor",
                        "Vs30 Extrapolated Thickness (m)",
                        "Vs30 Extrapolated Velocity (m/s)",
                    ]
                )
                tops = np.r_[0.0, self.result.depths_m[:-1]]
                for i in range(self.result.depths_m.size):
                    writer.writerow(
                        [
                            i + 1,
                            f"{tops[i]:.4f}",
                            f"{self.result.depths_m[i]:.4f}",
                            (
                                "" if "first_peak" not in self.comparison_results
                                else f"{self.comparison_results['first_peak'].velocities_mps[i]:.3f}"
                            ),
                            (
                                "" if "first_cross" not in self.comparison_results
                                else f"{self.comparison_results['first_cross'].velocities_mps[i]:.3f}"
                            ),
                            (
                                "" if "max_peak" not in self.comparison_results
                                else f"{self.comparison_results['max_peak'].velocities_mps[i]:.3f}"
                            ),
                            f"{self.result.observed_times_s[i] * 1000.0:.4f}",
                            f"{self.result.calculated_times_s[i] * 1000.0:.4f}",
                            f"{self.result.residuals_s[i] * 1000.0:.4f}",
                            f"{self.result.ray_parameters[i]:.10g}",
                            f"{self.reg_slider.value() / 100.0:.2f}",
                            (
                                "" if self.comparison_vs30.get("first_peak") is None
                                else f"{self.comparison_vs30['first_peak'].value_mps:.3f}"
                            ),
                            (
                                "" if self.comparison_vs30.get("first_cross") is None
                                else f"{self.comparison_vs30['first_cross'].value_mps:.3f}"
                            ),
                            (
                                "" if self.comparison_vs30.get("max_peak") is None
                                else f"{self.comparison_vs30['max_peak'].value_mps:.3f}"
                            ),
                            f"{self._extrapolation_weight_factor():.3f}",
                            "" if self.current_vs30 is None else f"{self.current_vs30.extrapolated_thickness_m:.3f}",
                            (
                                ""
                                if self.current_vs30 is None or self.current_vs30.extrapolated_velocity_mps is None
                                else f"{self.current_vs30.extrapolated_velocity_mps:.3f}"
                            ),
                        ]
                    )
            self.status_label.setText(f"Exported {target.name}")
        except Exception as exc:
            self._show_error("Unable to export CSV", exc)

    @Slot()
    def export_pdf_report(self) -> None:
        """Prompt for and generate a complete printable engineering report."""

        if self.result is None:
            QMessageBox.information(self, "No results", "Run the inversion before exporting a PDF report.")
            return
        suggested = (self.project_path.stem if self.project_path else "raypath_scpt_report") + ".pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export RayPath PDF Report", suggested, "PDF files (*.pdf)")
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".pdf":
            target = target.with_suffix(".pdf")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._build_pdf_report(target)
            if not target.is_file() or target.stat().st_size < 1000:
                raise RuntimeError("The PDF writer did not produce a valid report file.")
            self.status_label.setText(f"Exported PDF report: {target.name}")
        except Exception as exc:
            self._show_error("Unable to export PDF report", exc)
        finally:
            QApplication.restoreOverrideCursor()

    def _build_pdf_report(self, target: Path) -> None:
        """Build a multi-page RayPath SCPT report with charts and tables."""

        try:
            from reportlab.graphics.shapes import Circle, Drawing, Line, Path as ShapePath, Rect
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_LEFT, TA_RIGHT
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                Image,
                KeepTogether,
                LongTable,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PDF reporting requires ReportLab. Install it with `python -m pip install reportlab`."
            ) from exc

        from xml.sax.saxutils import escape

        assert self.result is not None
        selected_kind = str(self.estimator_combo.currentData())
        selected_result = self.result
        report_time = datetime.now().astimezone()
        primary = colors.HexColor("#147D75")
        dark = colors.HexColor("#163F3B")
        copper = colors.HexColor("#C87941")
        mid = colors.HexColor("#5F746F")
        pale = colors.HexColor("#E8F2EF")
        grid = colors.HexColor("#C9D5D2")
        soft = colors.HexColor("#F5F8F7")
        page_width, _ = A4
        content_width = page_width - 30 * mm
        chart_buffers: list[io.BytesIO] = []

        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=dark,
                alignment=TA_LEFT,
                spaceAfter=2 * mm,
            )
        )
        styles.add(
            ParagraphStyle(
                name="BrandWordmark",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=17,
                leading=19,
                textColor=dark,
                spaceAfter=0.8 * mm,
            )
        )
        styles.add(
            ParagraphStyle(
                name="BrandTagline",
                parent=styles["Normal"],
                fontSize=7.8,
                leading=9.5,
                textColor=mid,
            )
        )
        styles.add(
            ParagraphStyle(
                name="BrandMeta",
                parent=styles["Normal"],
                fontSize=7.5,
                leading=10,
                textColor=mid,
                alignment=TA_RIGHT,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportSubtitle",
                parent=styles["Normal"],
                fontSize=9,
                leading=12,
                textColor=mid,
                spaceAfter=5 * mm,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                textColor=primary,
                spaceBefore=4 * mm,
                spaceAfter=2.5 * mm,
                keepWithNext=0,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SmallText",
                parent=styles["Normal"],
                fontSize=8,
                leading=10.5,
                textColor=dark,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportNote",
                parent=styles["Normal"],
                fontSize=8,
                leading=11,
                textColor=mid,
                borderColor=grid,
                borderWidth=0.5,
                borderPadding=6,
                backColor=soft,
            )
        )

        def p(text: Any, style: str = "SmallText") -> Any:
            return Paragraph(escape(str(text)), styles[style])

        def table_style(header: bool = True, font_size: float = 7.5) -> Any:
            commands: list[tuple[Any, ...]] = [
                ("GRID", (0, 0), (-1, -1), 0.35, grid),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, soft]),
            ]
            if header:
                commands.extend(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), dark),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ]
                )
            return TableStyle(commands)

        def chart_image(draw: Any, height_mm: float = 220.0) -> Any:
            """Render one portrait-oriented chart sized to fill an A4 report page."""

            figure = Figure(figsize=(7.2, 8.8), dpi=150, facecolor="white")
            axes = figure.add_subplot(111)
            draw(axes)
            axes.grid(True, color="#D6E0DD", linewidth=0.6, alpha=0.8)
            axes.tick_params(labelsize=8)
            for spine in axes.spines.values():
                spine.set_color("#819790")
            figure.tight_layout(pad=1.1)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=170, facecolor="white")
            buffer.seek(0)
            chart_buffers.append(buffer)
            return Image(buffer, width=content_width, height=height_mm * mm)

        def draw_velocity(ax: Any) -> None:
            for kind in PICK_KINDS:
                comparison = self.comparison_results.get(kind)
                if comparison is None:
                    continue
                ax.stairs(
                    comparison.velocities_mps,
                    np.r_[0.0, comparison.depths_m],
                    orientation="horizontal",
                    color=MODEL_COLORS[kind],
                    linewidth=2.2 if kind == selected_kind else 1.6,
                    label=PICK_LABELS[kind],
                )
            ax.set_title("Optimized shear-wave velocity comparison", fontsize=11, fontweight="bold")
            ax.set_xlabel("Vs (m/s)")
            ax.set_ylabel("Depth (m)")
            ax.set_ylim(float(selected_result.depths_m[-1]), 0.0)
            ax.legend(fontsize=8, frameon=True)

        def draw_fit(ax: Any) -> None:
            for kind in PICK_KINDS:
                comparison = self.comparison_results.get(kind)
                if comparison is None:
                    continue
                ax.plot(
                    comparison.calculated_times_s * 1000.0,
                    comparison.depths_m,
                    "-s",
                    color=MODEL_COLORS[kind],
                    linewidth=2.0 if kind == selected_kind else 1.2,
                    markersize=3,
                    label=f"{PICK_LABELS[kind]} calculated",
                )
                ax.scatter(
                    comparison.observed_times_s * 1000.0,
                    comparison.depths_m,
                    facecolors="none",
                    edgecolors=MODEL_COLORS[kind],
                    s=20,
                )
            ax.set_title("Observed and calculated arrival times", fontsize=11, fontweight="bold")
            ax.set_xlabel("Post-trigger travel time (ms)")
            ax.set_ylabel("Depth (m)")
            ax.set_ylim(float(selected_result.depths_m[-1]), 0.0)
            ax.legend(fontsize=7.5, frameon=True)

        def draw_rays(ax: Any) -> None:
            edges = np.r_[0.0, selected_result.depths_m]
            offset = self.offset_spin.value()
            for boundary in edges:
                ax.hlines(boundary, 0.0, offset, color="#B8C4CC", linewidth=0.55)
            cmap = colormaps["viridis"]
            for index, segments in enumerate(selected_result.ray_x_segments):
                ax.plot(
                    np.r_[0.0, np.cumsum(segments)],
                    edges[: index + 2],
                    color=cmap((index + 1) / len(selected_result.ray_x_segments)),
                    linewidth=1.0,
                )
            ax.axvline(offset, color="#163F3B", linewidth=1.5, label="Borehole")
            ax.scatter([0.0], [0.0], marker="*", s=90, color="#C87941", zorder=5, label="Source")
            ax.scatter(np.full_like(selected_result.depths_m, offset), selected_result.depths_m, marker="<", s=18, color="#147D75", zorder=5, label="Receivers")
            ax.set_title(f"Ray paths - {PICK_LABELS.get(selected_kind, selected_kind)} model", fontsize=11, fontweight="bold")
            ax.set_xlabel("Horizontal distance (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_ylim(float(selected_result.depths_m[-1]) * 1.03, -float(selected_result.depths_m[-1]) * 0.03)
            margin = max(offset * 0.08, 0.1)
            ax.set_xlim(-margin, offset + margin)
            ax.set_aspect("auto")
            ax.legend(fontsize=8, frameon=True)

        def draw_waterfall(ax: Any) -> None:
            self._plot_waveform_waterfall(ax, dark_theme=False)
            ax.set_title("Paired SCPT waveform waterfall", fontsize=11, fontweight="bold")

        def cone_ray_logo(size_mm: float = 18.0) -> Any:
            """Return the Field Teal Cone & Ray mark as vector ReportLab art."""

            size = size_mm * mm
            drawing = Drawing(size, size)
            drawing.add(
                Rect(
                    0.5 * mm,
                    0.5 * mm,
                    size - 1.0 * mm,
                    size - 1.0 * mm,
                    rx=3.2 * mm,
                    ry=3.2 * mm,
                    fillColor=pale,
                    strokeColor=primary,
                    strokeWidth=0.65,
                )
            )
            for y_mm in (5.0, 9.0, 13.0):
                drawing.add(
                    Line(
                        2.3 * mm,
                        y_mm * mm,
                        15.8 * mm,
                        y_mm * mm,
                        strokeColor=colors.HexColor("#8FBDB7"),
                        strokeWidth=0.45,
                    )
                )
            drawing.add(
                Line(15.0 * mm, 2.4 * mm, 15.0 * mm, 15.7 * mm, strokeColor=dark, strokeWidth=1.15)
            )
            ray_points = (
                ((3.0, 15.0), (6.0, 13.2), (10.1, 12.6), (15.0, 13.0)),
                ((3.0, 15.0), (5.4, 10.7), (9.7, 8.7), (15.0, 9.0)),
                ((3.0, 15.0), (5.0, 7.0), (9.2, 4.8), (15.0, 5.0)),
            )
            for points in ray_points:
                path = ShapePath()
                path.moveTo(points[0][0] * mm, points[0][1] * mm)
                for x_mm, y_mm in points[1:]:
                    path.lineTo(x_mm * mm, y_mm * mm)
                path.strokeColor = primary
                path.strokeWidth = 0.9
                path.fillColor = None
                drawing.add(path)
            drawing.add(Circle(3.0 * mm, 15.0 * mm, 1.05 * mm, fillColor=copper, strokeColor=copper))
            for y_mm in (5.0, 9.0, 13.0):
                receiver = ShapePath()
                receiver.moveTo(14.0 * mm, y_mm * mm)
                receiver.lineTo(15.0 * mm, (y_mm + 0.65) * mm)
                receiver.lineTo(15.0 * mm, (y_mm - 0.65) * mm)
                receiver.closePath()
                receiver.fillColor = primary
                receiver.strokeColor = primary
                drawing.add(receiver)
            return drawing

        doc = SimpleDocTemplate(
            str(target),
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title="RayPath SCPT Engineering Report",
            author="RayPath SCPT",
            subject="SCPT ray-path inversion and Vs30 analysis",
        )

        def page_decoration(canvas: Any, document: Any) -> None:
            canvas.saveState()
            canvas.setStrokeColor(grid)
            canvas.setLineWidth(0.4)
            canvas.line(15 * mm, 12 * mm, page_width - 15 * mm, 12 * mm)
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(mid)
            canvas.setFillColor(copper)
            canvas.circle(16 * mm, 8.2 * mm, 1.0 * mm, fill=1, stroke=0)
            canvas.setFillColor(mid)
            canvas.drawString(19 * mm, 7.5 * mm, "RayPath SCPT - Engineering interpretation report")
            canvas.drawRightString(page_width - 15 * mm, 7.5 * mm, f"Page {document.page}")
            canvas.restoreState()

        story: list[Any] = []
        project_name = self.project_path.stem if self.project_path else "Untitled"
        brand_header = Table(
            [
                [
                    cone_ray_logo(),
                    [
                        Paragraph('RayPath <font color="#147D75">SCPT</font>', styles["BrandWordmark"]),
                        Paragraph("Forward ray-path shear-wave interpretation", styles["BrandTagline"]),
                    ],
                    Paragraph(
                        f"<b>PROJECT</b> {escape(project_name)}<br/>{report_time.strftime('%d %b %Y').upper()}",
                        styles["BrandMeta"],
                    ),
                ]
            ],
            colWidths=[21 * mm, 104 * mm, content_width - 125 * mm],
        )
        brand_header.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (0, 0), 0),
                    ("RIGHTPADDING", (0, 0), (0, 0), 3),
                    ("LEFTPADDING", (1, 0), (1, 0), 2),
                    ("RIGHTPADDING", (1, 0), (1, 0), 3),
                    ("LEFTPADDING", (2, 0), (2, 0), 3),
                    ("RIGHTPADDING", (2, 0), (2, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                    ("LINEBELOW", (0, 0), (-1, -1), 2.0, primary),
                ]
            )
        )
        story.append(brand_header)
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("SCPT Engineering Interpretation Report", styles["ReportTitle"]))
        story.append(
            Paragraph(
                "Forward ray-path shear-wave velocity inversion, arrival-time comparison, and Vs30 summary",
                styles["ReportSubtitle"],
            )
        )

        metadata = [
            [p("Report generated"), p(report_time.strftime("%Y-%m-%d %H:%M %Z"))],
            [p("Project"), p(self.project_path.name if self.project_path else "Untitled")],
            [p("GRU source"), p(self.gru_path.name if self.gru_path else "Manual / CSV observations")],
            [p("Source offset"), p(f"{self.offset_spin.value():.3f} m")],
            [p("Selected detailed model"), p(PICK_LABELS.get(selected_kind, selected_kind))],
            [p("Smoothing / regularisation"), p(f"{self.reg_slider.value() / 100.0:.2f}")],
            [p("Vs30 extrapolation weighting"), p(f"{self._extrapolation_weight_factor():.2f}")],
            [p("Deepest receiver"), p(f"{selected_result.depths_m[-1]:.2f} m")],
            [p("GRU pre-trigger correction"), p(f"{GRU_PRE_TRIGGER_MS:.1f} ms")],
        ]
        metadata_table = Table(metadata, colWidths=[58 * mm, content_width - 58 * mm])
        metadata_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.35, grid),
                    ("BACKGROUND", (0, 0), (0, -1), pale),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(metadata_table)
        story.append(Paragraph("Model comparison", styles["SectionHeading"]))
        comparison_rows: list[list[Any]] = [
            ["Pick definition", "Layers", "RMSE (ms)", "Vs30 (m/s)", "Extrapolation"]
        ]
        for kind in PICK_KINDS:
            comparison = self.comparison_results.get(kind)
            vs30 = self.comparison_vs30.get(kind)
            if comparison is None:
                comparison_rows.append([p(PICK_LABELS[kind]), p("-"), p("-"), p("-"), p("Not calculated")])
                continue
            if vs30 is None:
                vs30_text = "Unavailable"
                extrapolation_text = self.comparison_vs30_reasons.get(kind, "Insufficient depth")
            else:
                vs30_text = f"{vs30.value_mps:.1f}"
                extrapolation_text = (
                    "None - measured to 30 m"
                    if vs30.extrapolated_thickness_m <= 0.0
                    else f"{vs30.extrapolated_thickness_m:.2f} m at {vs30.extrapolated_velocity_mps:.1f} m/s"
                )
            comparison_rows.append(
                [
                    p(PICK_LABELS[kind]),
                    p(comparison.depths_m.size),
                    p(f"{comparison.rmse_s * 1000.0:.3f}"),
                    p(vs30_text),
                    p(extrapolation_text),
                ]
            )
        comparison_table = Table(
            comparison_rows,
            colWidths=[37 * mm, 19 * mm, 27 * mm, 29 * mm, content_width - 112 * mm],
            repeatRows=1,
        )
        comparison_table.setStyle(table_style(font_size=7.5))
        story.append(comparison_table)
        story.append(Spacer(1, 3 * mm))
        story.append(
            Paragraph(
                "Vs30 is calculated as 30 divided by the summed vertical shear-wave travel time. Profiles "
                "between 25 m and 30 m use the configured interval weighting only to estimate the missing depth. "
                "Automatic waveform suggestions should be reviewed by a qualified operator before relying on this report.",
                styles["ReportNote"],
            )
        )

        full_page_plots = (
            ("Velocity profile comparison", draw_velocity),
            ("Arrival-time fit comparison", draw_fit),
            ("Selected-model ray paths", draw_rays),
            ("Waveform waterfall and reviewed picks", draw_waterfall),
        )
        for heading, draw_plot in full_page_plots:
            story.append(PageBreak())
            story.append(
                KeepTogether(
                    [
                        Paragraph(heading, styles["SectionHeading"]),
                        chart_image(draw_plot),
                    ]
                )
            )

        story.append(PageBreak())
        story.append(Paragraph("Receiver pick times", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))
        pick_rows: list[list[Any]] = [
            ["Depth (m)", "First peak (ms)", "First cross (ms)", "Maximum peak (ms)"]
        ]
        for row in range(self.input_table.rowCount()):
            values = []
            for column in range(4):
                item = self.input_table.item(row, column)
                values.append(item.text().strip() if item else "")
            if not any(values):
                continue
            pick_rows.append([p(value or "-") for value in values])
        pick_table = LongTable(pick_rows, colWidths=[35 * mm, 48 * mm, 48 * mm, 49 * mm], repeatRows=1)
        pick_table.setStyle(table_style(font_size=7.2))
        story.append(pick_table)

        story.append(PageBreak())
        story.append(Paragraph("Layer velocity results", styles["SectionHeading"]))
        story.append(Spacer(1, 2 * mm))
        layer_rows: list[list[Any]] = [
            [
                "Layer",
                "Top (m)",
                "Bottom (m)",
                "First peak Vs",
                "First cross Vs",
                "Max peak Vs",
                "Selected residual (ms)",
            ]
        ]
        tops = np.r_[0.0, selected_result.depths_m[:-1]]
        for index in range(selected_result.depths_m.size):
            velocity_cells = []
            for kind in PICK_KINDS:
                comparison = self.comparison_results.get(kind)
                velocity_cells.append("-" if comparison is None else f"{comparison.velocities_mps[index]:.1f}")
            layer_rows.append(
                [
                    p(index + 1),
                    p(f"{tops[index]:.2f}"),
                    p(f"{selected_result.depths_m[index]:.2f}"),
                    *[p(value) for value in velocity_cells],
                    p(f"{selected_result.residuals_s[index] * 1000.0:+.3f}"),
                ]
            )
        layer_table = LongTable(
            layer_rows,
            colWidths=[14 * mm, 22 * mm, 24 * mm, 29 * mm, 30 * mm, 28 * mm, 33 * mm],
            repeatRows=1,
        )
        layer_table.setStyle(table_style(font_size=6.9))
        story.append(layer_table)

        doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)

    # ---- inversion and output -------------------------------------------

    @Slot()
    def run_inversion(self) -> None:
        if self._thread is not None:
            return
        try:
            if self.waveform_records:
                depths, times_by_pick_s = self._read_all_pick_rows()
            else:
                try:
                    depths, times_by_pick_s = self._read_all_pick_rows()
                except ValueError:
                    depths, times_s = self._read_input_rows()
                    times_by_pick_s = {str(self.estimator_combo.currentData()): times_s}
        except Exception as exc:
            self._show_error("Invalid input data", exc)
            return
        self.run_button.setEnabled(False)
        self.run_button.setText("Inversion running…")
        self.input_table.setEnabled(False)
        self.offset_spin.setEnabled(False)
        self.estimator_combo.setEnabled(False)
        self.reg_slider.setEnabled(False)
        self.extrapolation_weight_slider.setEnabled(False)
        self.status_label.setText("Solving refracted ray paths and velocity model…")
        self.rmse_label.setText("RMSE: calculating…")
        self._thread = QThread(self)
        self._worker = InversionWorker(
            depths,
            times_by_pick_s,
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

    @Slot(str, int, float)
    def _inversion_progress(self, kind: str, evaluations: int, rmse_ms: float) -> None:
        self.status_label.setText(
            f"Optimising {PICK_LABELS.get(kind, kind)} model — {evaluations} evaluations, "
            f"current data RMSE {rmse_ms:.3f} ms"
        )

    @Slot(object)
    def _inversion_finished(self, results: dict[str, InversionResult]) -> None:
        self.comparison_results = dict(results)
        self._update_all_vs30_results(record_history=True)
        selected_kind = str(self.estimator_combo.currentData())
        if selected_kind not in self.comparison_results:
            selected_kind = next(iter(self.comparison_results))
        self._activate_comparison_result(selected_kind)
        result = self.result
        assert result is not None
        state = "Converged" if result.success else "Usable solution (optimizer warning)"
        self.status_label.setText(f"{state} — {result.iterations} iterations — {result.message}")
        self.rmse_label.setText(f"RMSE: {result.rmse_s * 1000.0:.3f} ms")
        self._update_result_summary(result)
        if not result.success:
            QMessageBox.warning(
                self,
                "Optimizer convergence warning",
                "The solver returned a finite model but did not report formal convergence.\n\n"
                f"{result.message}\n\nReview the arrival picks and smoothing factor before using these results.",
            )
        self._set_dirty(True)

    def _update_all_vs30_results(self, record_history: bool) -> None:
        """Recalculate Vs30 for every available pick-based velocity model."""

        self.comparison_vs30.clear()
        self.comparison_vs30_reasons.clear()
        weight_factor = self._extrapolation_weight_factor()
        smoothing = self.reg_slider.value() / 100.0
        for kind, result in self.comparison_results.items():
            try:
                value = calculate_vs30(result.depths_m, result.velocities_mps, weight_factor)
                self.comparison_vs30[kind] = value
                if record_history:
                    self.vs30_history[(kind, smoothing, round(weight_factor, 6))] = value.value_mps
            except ValueError as exc:
                self.comparison_vs30[kind] = None
                self.comparison_vs30_reasons[kind] = str(exc)

    def _activate_comparison_result(self, kind: str) -> None:
        """Show one comparison model in detailed views while retaining overlays."""

        if kind not in self.comparison_results:
            return
        self.result = self.comparison_results[kind]
        self.current_vs30 = self.comparison_vs30.get(kind)
        self.vs30_unavailable_reason = self.comparison_vs30_reasons.get(kind)
        if self.current_vs30 is not None:
            self.vs30_status_label.setText(f"Vs30: {self.current_vs30.value_mps:.1f} m/s")
        else:
            self.vs30_status_label.setText("Vs30: unavailable")
        self._populate_results(self.result)
        self._draw_results(self.result)
        self._update_result_summary(self.result)
        self.rmse_label.setText(f"RMSE: {self.result.rmse_s * 1000.0:.3f} ms")
        self.status_label.setText(
            f"Showing {PICK_LABELS.get(kind, kind)} model — {self.result.message}"
        )

    def _update_result_summary(self, result: InversionResult) -> None:
        """Update the textual engineering summary for the active result."""

        state = "Converged" if result.success else "Usable solution (optimizer warning)"
        if self.current_vs30 is not None:
            if self.current_vs30.extrapolated_thickness_m > 0.0:
                vs30_note = (
                    f" Vs30: {self.current_vs30.value_mps:.1f} m/s, including "
                    f"{self.current_vs30.extrapolated_thickness_m:.2f} m extrapolated with weighting "
                    f"{self.current_vs30.extrapolation_weight_factor:.2f}."
                )
            else:
                vs30_note = f" Vs30: {self.current_vs30.value_mps:.1f} m/s (fully measured to 30 m)."
        else:
            vs30_note = f" Vs30 unavailable: {self.vs30_unavailable_reason or 'insufficient profile data'}."
        self.result_summary.setText(
            f"{state}. {result.depths_m.size} layers, source offset {self.offset_spin.value():.3f} m, "
            f"{PICK_LABELS.get(str(self.estimator_combo.currentData()), 'Selected')} model, "
            f"regularisation {self.reg_slider.value() / 100.0:.2f}. Final RMSE: {result.rmse_s * 1000.0:.3f} ms."
            f"{vs30_note}"
        )

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
        self.input_table.setEnabled(True)
        self.offset_spin.setEnabled(True)
        self.estimator_combo.setEnabled(True)
        self.reg_slider.setEnabled(True)
        self.extrapolation_weight_slider.setEnabled(True)

    def _populate_results(self, result: InversionResult) -> None:
        self.result_table.setRowCount(result.depths_m.size)
        tops = np.r_[0.0, result.depths_m[:-1]]
        for row in range(result.depths_m.size):
            velocity_values = []
            for kind in PICK_KINDS:
                comparison = self.comparison_results.get(kind)
                if comparison is not None and row < comparison.velocities_mps.size:
                    velocity_values.append(f"{comparison.velocities_mps[row]:.1f} m/s")
                else:
                    velocity_values.append("—")
            values = (
                str(row + 1),
                f"{tops[row]:.2f} m",
                f"{result.depths_m[row]:.2f} m",
                *velocity_values,
                f"{result.residuals_s[row] * 1000.0:+.3f} ms",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 3, 4, 5, 6):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if column == 6:
                    magnitude = abs(result.residuals_s[row] * 1000.0)
                    item.setForeground(QColor("#3fb950" if magnitude < 1.0 else "#d29922" if magnitude < 3.0 else "#f85149"))
                self.result_table.setItem(row, column, item)

    def _draw_results(self, result: InversionResult) -> None:
        z = result.depths_m
        edges = np.r_[0.0, z]
        raw = pseudo_interval_velocities(z, result.observed_times_s)

        ax = self.velocity_canvas.axes
        self.velocity_canvas.clear()
        ax.plot(raw, (edges[:-1] + edges[1:]) / 2.0, "--", color="#8b949e", linewidth=1.0, label="Selected raw pseudo-interval")
        all_velocities: list[np.ndarray] = []
        selected_kind = str(self.estimator_combo.currentData())
        for kind in PICK_KINDS:
            comparison = self.comparison_results.get(kind)
            if comparison is None:
                continue
            comparison_edges = np.r_[0.0, comparison.depths_m]
            all_velocities.append(comparison.velocities_mps)
            ax.stairs(
                comparison.velocities_mps,
                comparison_edges,
                orientation="horizontal",
                color=MODEL_COLORS[kind],
                linewidth=2.8 if kind == selected_kind else 1.8,
                alpha=1.0 if kind == selected_kind else 0.82,
                label=PICK_LABELS[kind],
            )
        ax.set_title("Pick-based shear-wave velocity comparison")
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1], 0.0)
        finite = np.concatenate([raw[np.isfinite(raw)], *all_velocities]) if all_velocities else raw[np.isfinite(raw)]
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
        ax.set_title(f"Snell's-law rays — {PICK_LABELS.get(selected_kind, selected_kind)} model")
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1] * 1.03, -z[-1] * 0.03)
        margin = max(x_offset * 0.08, 0.1)
        ax.set_xlim(-margin, x_offset + margin)
        ax.set_aspect("auto")
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        self.ray_canvas.draw_idle()

        ax = self.fit_canvas.axes
        self.fit_canvas.clear()
        for kind in PICK_KINDS:
            comparison = self.comparison_results.get(kind)
            if comparison is None:
                continue
            observed_kind_ms = comparison.observed_times_s * 1000.0
            calculated_kind_ms = comparison.calculated_times_s * 1000.0
            linewidth = 2.2 if kind == selected_kind else 1.2
            ax.plot(
                calculated_kind_ms,
                comparison.depths_m,
                "-s",
                color=MODEL_COLORS[kind],
                linewidth=linewidth,
                markersize=3.2,
                label=f"{PICK_LABELS[kind]} calculated",
            )
            ax.scatter(
                observed_kind_ms,
                comparison.depths_m,
                facecolors="none",
                edgecolors=MODEL_COLORS[kind],
                s=24,
                alpha=0.8,
            )
        observed_ms = result.observed_times_s * 1000.0
        calculated_ms = result.calculated_times_s * 1000.0
        for depth, observed, calculated in zip(z, observed_ms, calculated_ms):
            ax.plot([observed, calculated], [depth, depth], color="#8b949e", linewidth=0.8)
        ax.set_title(
            f"Arrival-time comparison — selected {PICK_LABELS.get(selected_kind, selected_kind)} "
            f"RMSE {result.rmse_s * 1000.0:.3f} ms"
        )
        ax.set_xlabel("Arrival time after trigger (ms)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1], 0.0)
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        self.fit_canvas.draw_idle()
        self._draw_vs30_analysis()

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
            f"<p>GRU imports apply the undocumented {GRU_PRE_TRIGGER_MS:g} ms pre-trigger correction before picking "
            "or velocity calculation.</p>"
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
    QLabel#pickValuesLabel {
        color: #f0f6fc; font-size: 11pt; font-weight: 700;
        background: #161b22; border: 1px solid #30363d; border-radius: 4px;
        padding: 7px 9px;
    }
    QLabel#vs30Value { color: #f0f6fc; font-size: 20pt; font-weight: 700; }
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
    vs30 = calculate_vs30([10.0, 20.0, 40.0], [200.0, 400.0, 800.0])
    assert abs(vs30.value_mps - (30.0 / (10.0 / 200.0 + 10.0 / 400.0 + 10.0 / 800.0))) < 1.0e-10
    assert abs(float(np.sum(vs30.included_thicknesses_m)) - 30.0) < 1.0e-10
    shallow_weighted = calculate_vs30([5.0, 10.0, 15.0, 20.0, 25.0], [150.0, 200.0, 250.0, 300.0, 400.0], 0.25)
    neutral_weighted = calculate_vs30([5.0, 10.0, 15.0, 20.0, 25.0], [150.0, 200.0, 250.0, 300.0, 400.0], 1.0)
    deep_weighted = calculate_vs30([5.0, 10.0, 15.0, 20.0, 25.0], [150.0, 200.0, 250.0, 300.0, 400.0], 4.0)
    assert shallow_weighted.extrapolated_velocity_mps < neutral_weighted.extrapolated_velocity_mps
    assert neutral_weighted.extrapolated_velocity_mps < deep_weighted.extrapolated_velocity_mps
    assert shallow_weighted.value_mps < neutral_weighted.value_mps < deep_weighted.value_mps
    print(
        f"RayPath SCPT self-test passed; RMSE={result.rmse_s * 1000.0:.6f} ms; "
        f"Vs30={vs30.value_mps:.3f} m/s"
    )
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
