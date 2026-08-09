"""RayPath SCPT - forward ray-path modelling and SCPT velocity inversion.

This single-file desktop application imports GOnsite/GORILLA ``.GRU`` seismic
records, supports review and manual picking of opposing shear-wave traces, and
estimates an analyst-defined piecewise-constant shear-wave velocity profile by weighted least squares.

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
<<<<<<< Updated upstream
PROJECT_SUFFIX = ".rpscpt"
=======
APP_VERSION = "0.9.0-alpha.1"
PROJECT_SUFFIX = ".rpscpt"
PROJECT_SCHEMA_VERSION = 11
SUPPORTED_PROJECT_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
>>>>>>> Stashed changes
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
    """Complete result of a layered ray-path inversion."""

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
<<<<<<< Updated upstream
=======
    weighted_rmse_s: float = math.nan
    data_cost_ms2: float = math.nan
    regularization_cost: float = math.nan
    roughness_norm: float = math.nan
    objective_value: float = math.nan
    observation_std_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    standardized_residuals: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    resolution_diagonal: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    observation_leverage: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    influence_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    outlier_flags: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    influential_flags: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    bound_active_flags: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    robust_loss: str = "linear"
    warnings: tuple[str, ...] = ()
    model_layer_bottoms_m: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    model_layer_velocities_mps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    model_layer_resolution: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    model_layer_bound_active: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    ray_z_nodes_m: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class RegularizationSelectionResult:
    """Repeatable L-curve selection audit for one arrival-time model."""

    selected_factor: float
    candidate_factors: np.ndarray
    weighted_rmse_ms: np.ndarray
    roughness_norm: np.ndarray
    chord_distances: np.ndarray


@dataclass(frozen=True)
class VelocityUncertaintyResult:
    """Monte Carlo velocity and Vs30 percentiles from pick-time uncertainty."""

    random_seed: int
    requested_models: int
    successful_models: int
    percentile_levels: tuple[float, float, float]
    velocity_lower_mps: np.ndarray
    velocity_median_mps: np.ndarray
    velocity_upper_mps: np.ndarray
    vs30_samples_mps: np.ndarray
    vs30_lower_mps: float | None
    vs30_median_mps: float | None
    vs30_upper_mps: float | None
    warnings: tuple[str, ...] = ()


def uncertainty_ensemble_classification(requested_models: int) -> str:
    """Return the user/report label for an uncertainty ensemble size."""

    requested = int(requested_models)
    if requested <= 0:
        return "Off"
    if requested < REPORT_QUALITY_ENSEMBLE_MINIMUM:
        return "Quick preview - preliminary"
    return "Final/report-quality"
>>>>>>> Stashed changes


@dataclass(frozen=True)
class IntervalVelocitySeries:
    """Observed interval velocities for one receiver-pair selection."""

    recorded_top_depths_m: np.ndarray
    recorded_bottom_depths_m: np.ndarray
    top_depths_m: np.ndarray
    bottom_depths_m: np.ndarray
    centre_depths_m: np.ndarray
    path_length_differences_m: np.ndarray
    arrival_time_differences_s: np.ndarray
    velocities_mps: np.ndarray
    standard_deviations_mps: np.ndarray
    valid_flags: np.ndarray
    status_messages: tuple[str, ...]


@dataclass(frozen=True)
class StaggeredIntervalComparison:
    """Adjacent and staggered 1 m interval-velocity sensitivity series."""

    adjacent: IntervalVelocitySeries
    phase_a: IntervalVelocitySeries
    phase_b: IntervalVelocitySeries
    slowness_mean_depths_m: np.ndarray
    slowness_mean_velocities_mps: np.ndarray
    target_interval_m: float
    tolerance_m: float


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


def _validate_layered_geometry(
    receiver_depths_m: Sequence[float],
    layer_bottoms_m: Sequence[float],
    layer_velocities_mps: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate receiver depths and an interpreted piecewise-constant model."""

    receivers = np.asarray(receiver_depths_m, dtype=float)
    bottoms = np.asarray(layer_bottoms_m, dtype=float)
    if receivers.ndim != 1 or receivers.size < 2:
        raise ValueError("A layered interpretation requires at least two receiver observations.")
    if not np.all(np.isfinite(receivers)) or not np.all(receivers > 0.0) or np.any(np.diff(receivers) <= 0.0):
        raise ValueError("Receiver depths must be finite, positive, and strictly increasing.")
    if bottoms.ndim != 1 or bottoms.size < 1:
        raise ValueError("Enter at least one interpreted layer ending at the deepest receiver.")
    if not np.all(np.isfinite(bottoms)) or not np.all(bottoms > 0.0) or np.any(np.diff(bottoms) <= 0.0):
        raise ValueError("Interpreted layer bottoms must be finite, positive, and strictly increasing.")
    tolerance = max(1.0e-6, float(receivers[-1]) * 1.0e-8)
    if not math.isclose(float(bottoms[-1]), float(receivers[-1]), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("The final interpreted layer must end at the deepest corrected receiver depth.")
    if bottoms.size >= receivers.size:
        raise ValueError(
            "The layered model must contain fewer velocity parameters than receiver observations. "
            "Remove at least one internal boundary."
        )
    velocities: np.ndarray | None = None
    if layer_velocities_mps is not None:
        velocities = np.asarray(layer_velocities_mps, dtype=float)
        if velocities.ndim != 1 or velocities.size != bottoms.size:
            raise ValueError("Provide one velocity for every interpreted layer.")
        if not np.all(np.isfinite(velocities)) or not np.all(velocities > 0.0):
            raise ValueError("Layer velocities must be finite and greater than zero.")
    return receivers, bottoms, velocities


def forward_layered_model(
    receiver_depths_m: Sequence[float],
    layer_bottoms_m: Sequence[float],
    layer_velocities_mps: Sequence[float],
    source_offset_m: float,
    receiver_offsets_m: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[RaySolution], list[np.ndarray]]:
    """Calculate direct refracted rays through interpreted constant-Vs layers.

    Receiver depths may fall inside a model layer. The final crossed layer is
    truncated at that receiver while the same fitted velocity is retained.
    """

    receivers, bottoms, velocities_raw = _validate_layered_geometry(
        receiver_depths_m, layer_bottoms_m, layer_velocities_mps
    )
    assert velocities_raw is not None
    velocities = velocities_raw
    if receiver_offsets_m is None:
        offsets = np.full(receivers.size, float(source_offset_m), dtype=float)
    else:
        offsets = np.asarray(receiver_offsets_m, dtype=float)
        if offsets.ndim != 1 or offsets.size != receivers.size:
            raise ValueError("Receiver offsets must contain one value for every receiver depth.")
    if not np.all(np.isfinite(offsets)) or not np.all(offsets >= 0.0):
        raise ValueError("Receiver offsets must be finite and greater than or equal to zero.")

    rays: list[RaySolution] = []
    z_nodes: list[np.ndarray] = []
    for receiver_depth, offset in zip(receivers, offsets):
        last_layer = int(np.searchsorted(bottoms, receiver_depth, side="left"))
        segment_bottoms = np.r_[bottoms[:last_layer], receiver_depth]
        thicknesses = np.diff(np.r_[0.0, segment_bottoms])
        ray = solve_direct_ray(thicknesses, velocities[: thicknesses.size], float(offset))
        rays.append(ray)
        z_nodes.append(np.r_[0.0, segment_bottoms])
    return np.asarray([ray.travel_time_s for ray in rays]), rays, z_nodes


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


<<<<<<< Updated upstream
=======
def staggered_interval_velocity_comparison(
    recorded_depths_m: Sequence[float],
    corrected_vertical_depths_m: Sequence[float],
    receiver_offsets_m: Sequence[float],
    arrival_times_s: Sequence[float],
    observation_std_s: Sequence[float] | None = None,
    target_interval_m: float = 1.0,
    tolerance_m: float = 0.15,
) -> StaggeredIntervalComparison:
    """Calculate adjacent and alternating wider observed interval velocities.

    The distance coordinate is the geometric source-to-receiver path length,
    ``sqrt(vertical_depth**2 + radial_offset**2)``. Each velocity is the
    difference in path length divided by the picked arrival-time difference.
    Candidate wider windows are paired when their *recorded* receiver-depth
    separation is within ``tolerance_m`` of ``target_interval_m``. Phase A is
    anchored to whole-metre recorded depths and phase B to the grid half a
    target interval deeper.

    The optional combined curve interpolates each phase in slowness space and
    averages slowness, not velocity. It is a display sensitivity only because
    neighbouring 1 m windows overlap and are not independent observations.
    """

    recorded = np.asarray(recorded_depths_m, dtype=float)
    vertical = np.asarray(corrected_vertical_depths_m, dtype=float)
    offsets = np.asarray(receiver_offsets_m, dtype=float)
    arrivals = np.asarray(arrival_times_s, dtype=float)
    size = recorded.size
    if size < 3 or any(array.ndim != 1 or array.size != size for array in (vertical, offsets, arrivals)):
        raise ValueError("Staggered interval comparison requires at least three matching receiver observations.")
    if not all(np.all(np.isfinite(array)) for array in (recorded, vertical, offsets, arrivals)):
        raise ValueError("Staggered interval inputs must be finite.")
    if (
        not np.all(recorded > 0.0)
        or not np.all(vertical > 0.0)
        or not np.all(offsets >= 0.0)
        or not np.all(arrivals > 0.0)
        or not np.all(np.diff(recorded) > 0.0)
        or not np.all(np.diff(vertical) > 0.0)
    ):
        raise ValueError("Receiver depths and arrivals must be positive, with depths strictly increasing.")
    target = float(target_interval_m)
    tolerance = float(tolerance_m)
    if not math.isfinite(target) or target <= 0.0 or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Target interval must be positive and tolerance must be non-negative.")
    if observation_std_s is None:
        standard_deviations = np.full(size, np.nan, dtype=float)
    else:
        standard_deviations = np.asarray(observation_std_s, dtype=float)
        if standard_deviations.ndim != 1 or standard_deviations.size != size:
            raise ValueError("Arrival uncertainty must contain one value per receiver observation.")
        if not np.all(np.isfinite(standard_deviations)) or not np.all(standard_deviations > 0.0):
            raise ValueError("Arrival uncertainties must be finite and greater than zero.")

    path_lengths = np.hypot(vertical, offsets)

    def make_series(pairs: Sequence[tuple[int, int]]) -> IntervalVelocitySeries:
        recorded_tops: list[float] = []
        recorded_bottoms: list[float] = []
        tops: list[float] = []
        bottoms: list[float] = []
        centres: list[float] = []
        path_differences: list[float] = []
        time_differences: list[float] = []
        velocities: list[float] = []
        velocity_std: list[float] = []
        valid_flags: list[bool] = []
        status_messages: list[str] = []
        for first, second in pairs:
            delta_time = float(arrivals[second] - arrivals[first])
            delta_path = float(path_lengths[second] - path_lengths[first])
            recorded_tops.append(float(recorded[first]))
            recorded_bottoms.append(float(recorded[second]))
            tops.append(float(vertical[first]))
            bottoms.append(float(vertical[second]))
            centres.append(0.5 * float(vertical[first] + vertical[second]))
            path_differences.append(delta_path)
            time_differences.append(delta_time)
            if delta_path <= 0.0:
                velocities.append(math.nan)
                velocity_std.append(math.nan)
                valid_flags.append(False)
                status_messages.append("Invalid: non-positive geometric path-length difference")
                continue
            if delta_time <= 0.0:
                velocities.append(math.nan)
                velocity_std.append(math.nan)
                valid_flags.append(False)
                status_messages.append("Invalid: non-positive picked arrival-time difference; review picks")
                continue
            velocity = delta_path / delta_time
            velocities.append(velocity)
            if np.all(np.isfinite(standard_deviations[[first, second]])):
                delta_time_std = math.hypot(
                    float(standard_deviations[first]), float(standard_deviations[second])
                )
                velocity_std.append(velocity * delta_time_std / delta_time)
            else:
                velocity_std.append(math.nan)
            valid_flags.append(True)
            status_messages.append("Valid")
        return IntervalVelocitySeries(
            recorded_top_depths_m=np.asarray(recorded_tops, dtype=float),
            recorded_bottom_depths_m=np.asarray(recorded_bottoms, dtype=float),
            top_depths_m=np.asarray(tops, dtype=float),
            bottom_depths_m=np.asarray(bottoms, dtype=float),
            centre_depths_m=np.asarray(centres, dtype=float),
            path_length_differences_m=np.asarray(path_differences, dtype=float),
            arrival_time_differences_s=np.asarray(time_differences, dtype=float),
            velocities_mps=np.asarray(velocities, dtype=float),
            standard_deviations_mps=np.asarray(velocity_std, dtype=float),
            valid_flags=np.asarray(valid_flags, dtype=bool),
            status_messages=tuple(status_messages),
        )

    adjacent_pairs = [(index, index + 1) for index in range(size - 1)]
    wider_pairs: list[tuple[int, int]] = []
    for first in range(size - 1):
        separations = recorded[first + 1 :] - recorded[first]
        relative_index = int(np.argmin(np.abs(separations - target)))
        second = first + 1 + relative_index
        if abs(float(separations[relative_index] - target)) <= tolerance:
            wider_pairs.append((first, second))
    # Anchor the two phases to absolute recorded receiver depth rather than the
    # retained-row index, so excluding one observation does not swap every
    # deeper window. For a 1 m target, phase A starts on whole-metre depths and
    # phase B starts on half-metre depths.
    phase_a_pairs: list[tuple[int, int]] = []
    phase_b_pairs: list[tuple[int, int]] = []
    for pair in wider_pairs:
        phase_position = float(recorded[pair[0]] % target)
        distance_to_a = min(phase_position, target - phase_position)
        distance_to_b = abs(phase_position - 0.5 * target)
        (phase_a_pairs if distance_to_a <= distance_to_b else phase_b_pairs).append(pair)
    adjacent = make_series(adjacent_pairs)
    phase_a = make_series(phase_a_pairs)
    phase_b = make_series(phase_b_pairs)

    mean_depths = np.empty(0, dtype=float)
    mean_velocities = np.empty(0, dtype=float)
    finite_a = np.isfinite(phase_a.velocities_mps)
    finite_b = np.isfinite(phase_b.velocities_mps)
    phase_a_depths = phase_a.centre_depths_m[finite_a]
    phase_b_depths = phase_b.centre_depths_m[finite_b]
    phase_a_velocities = phase_a.velocities_mps[finite_a]
    phase_b_velocities = phase_b.velocities_mps[finite_b]
    if phase_a_depths.size >= 2 and phase_b_depths.size >= 2:
        overlap_top = max(float(phase_a_depths[0]), float(phase_b_depths[0]))
        overlap_bottom = min(float(phase_a_depths[-1]), float(phase_b_depths[-1]))
        candidate_depths = np.unique(
            np.r_[phase_a_depths, phase_b_depths].round(9)
        )
        mean_depths = candidate_depths[
            (candidate_depths >= overlap_top - 1.0e-9) & (candidate_depths <= overlap_bottom + 1.0e-9)
        ]
        if mean_depths.size:
            slowness_a = np.interp(
                mean_depths, phase_a_depths, 1.0 / phase_a_velocities
            )
            slowness_b = np.interp(
                mean_depths, phase_b_depths, 1.0 / phase_b_velocities
            )
            mean_velocities = 1.0 / (0.5 * (slowness_a + slowness_b))

    return StaggeredIntervalComparison(
        adjacent=adjacent,
        phase_a=phase_a,
        phase_b=phase_b,
        slowness_mean_depths_m=mean_depths,
        slowness_mean_velocities_mps=mean_velocities,
        target_interval_m=target,
        tolerance_m=tolerance,
    )


def _validated_vs_profile(
    depths_m: Sequence[float], velocities_mps: Sequence[float], calculation_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional layer-bottom depths and velocities."""

    depths = np.asarray(depths_m, dtype=float)
    velocities = np.asarray(velocities_mps, dtype=float)
    if depths.ndim != 1 or velocities.ndim != 1 or depths.size == 0 or depths.size != velocities.size:
        raise ValueError(f"{calculation_name} requires matching non-empty depth and velocity arrays.")
    if not np.all(np.isfinite(depths)) or not np.all(depths > 0.0) or not np.all(np.diff(depths) > 0.0):
        raise ValueError(f"{calculation_name} layer depths must be finite, positive, and strictly increasing.")
    if not np.all(np.isfinite(velocities)) or not np.all(velocities > 0.0):
        raise ValueError(f"{calculation_name} layer velocities must be finite and greater than zero.")
    return depths, velocities


def _interval_overlap(
    tops_m: np.ndarray, bottoms_m: np.ndarray, range_top_m: float, range_bottom_m: float
) -> np.ndarray:
    """Return layer thickness lying within one closed-open depth range."""

    return np.maximum(0.0, np.minimum(bottoms_m, range_bottom_m) - np.maximum(tops_m, range_top_m))


def ts1170_5_vs30_band(value_mps: float) -> str:
    """Return the numerical TS 1170.5 Vs30 screening band for one value.

    This is deliberately not a final site-class assignment because Classes I,
    II, III, V, and VI include criteria that cannot be resolved from Vs30
    alone.
    """

    value = float(value_mps)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("A TS 1170.5 Vs30 screening value must be finite and greater than zero.")
    if value <= 150.0:
        return "VII"
    if value <= 200.0:
        return "VI"
    if value <= 250.0:
        return "V"
    if value <= 300.0:
        return "IV"
    if value <= 450.0:
        return "III"
    if value <= 750.0:
        return "II"
    return "I"


def _vs30_bands_over_range(lower_mps: float, upper_mps: float) -> tuple[str, ...]:
    """Return every numerical Vs30 band intersected by an uncertainty range."""

    lower = float(lower_mps)
    upper = float(upper_mps)
    if lower > upper:
        lower, upper = upper, lower
    probes = [lower, upper]
    for threshold in (150.0, 200.0, 250.0, 300.0, 450.0, 750.0):
        if lower < threshold < upper:
            probes.extend((threshold, float(np.nextafter(threshold, math.inf))))
    ordered = ("VII", "VI", "V", "IV", "III", "II", "I")
    present = {ts1170_5_vs30_band(value) for value in probes}
    return tuple(band for band in ordered if band in present)


def calculate_ts1170_5_method1_vs30(
    depths_m: Sequence[float], velocities_mps: Sequence[float]
) -> Ts1170Method1Vs30Result:
    """Calculate TS 1170.5:2025 Method 1 Vs30 for a direct SCPT profile.

    ``depths_m`` are successive layer-bottom depths and ``velocities_mps``
    are the corresponding interval velocities.  Continuous direct Vs coverage
    from the surface to at least 25 m is required.  The calculation follows
    the published Method 1 SCPT/downhole treatment: the 0-to-3 m interval uses
    the thickness-weighted arithmetic mean of measured Vs from 2.5-to-3.5 m,
    and the final measured layer is extended to 30 m where necessary.  The
    5 percent uncertainty bounds are central/1.05 and central*1.05.
    """

    depths, velocities = _validated_vs_profile(depths_m, velocities_mps, "TS 1170.5 Method 1 Vs30")
    target_depth_m = 30.0
    minimum_measured_depth_m = 25.0
    measured_depth = float(depths[-1])
    if measured_depth < minimum_measured_depth_m - 1.0e-9:
        raise ValueError(
            f"The directly measured Vs profile reaches {measured_depth:.2f} m; TS 1170.5:2025 "
            f"Method 1 requires direct field measurement to at least {minimum_measured_depth_m:.2f} m."
        )

    original_tops = np.r_[0.0, depths[:-1]]
    reference_overlap = _interval_overlap(original_tops, depths, 2.5, 3.5)
    reference_thickness = float(np.sum(reference_overlap))
    if not math.isclose(reference_thickness, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("The measured profile does not continuously cover the required 2.5-to-3.5 m interval.")
    shallow_reference_velocity = float(np.sum(reference_overlap * velocities) / reference_thickness)

    boundaries = sorted(
        {
            0.0,
            3.0,
            target_depth_m,
            *[float(depth) for depth in depths if 0.0 < depth < target_depth_m],
        }
    )
    interval_tops = np.asarray(boundaries[:-1], dtype=float)
    interval_bottoms = np.asarray(boundaries[1:], dtype=float)
    included_thicknesses = interval_bottoms - interval_tops
    interval_midpoints = 0.5 * (interval_tops + interval_bottoms)
    source_indices = np.searchsorted(depths, np.minimum(interval_midpoints, measured_depth), side="right")
    source_indices = np.minimum(source_indices, velocities.size - 1)
    raw_velocities = velocities[source_indices].astype(float, copy=True)
    adjusted_velocities = raw_velocities.copy()
    adjusted_velocities[interval_bottoms <= 3.0 + 1.0e-10] = shallow_reference_velocity

    extrapolated_thickness = max(0.0, target_depth_m - measured_depth)
    extrapolated_velocity = float(velocities[-1]) if extrapolated_thickness > 1.0e-9 else None
    raw_time = float(np.sum(included_thicknesses / raw_velocities))
    adjusted_time = float(np.sum(included_thicknesses / adjusted_velocities))
    if not math.isfinite(adjusted_time) or adjusted_time <= 0.0:
        raise ValueError("The standards-adjusted vertical travel time to 30 m is invalid.")
    raw_value = target_depth_m / raw_time
    value = target_depth_m / adjusted_time
    uncertainty = 1.05
    lower = value / uncertainty
    upper = value * uncertainty

    slow_overlap = _interval_overlap(original_tops, depths, 0.0, min(20.0, measured_depth))
    slow_thickness = float(np.sum(slow_overlap[velocities <= 150.0]))
    bands = _vs30_bands_over_range(lower, upper)
    notes = [
        "Method 1 direct SCPT/downhole shallow-layer adjustment applied from 0 to 3 m.",
        "Indicative bands use Vs30 thresholds only; final TS site classification requires all additional criteria.",
    ]
    if extrapolated_thickness > 1.0e-9:
        notes.append(
            f"The last directly measured layer velocity was extended through the final {extrapolated_thickness:.2f} m."
        )
    if slow_thickness > 10.0 + 1.0e-9:
        notes.append(
            "More than 10 m of the upper 20 m has Vs at or below 150 m/s; the TS Site Class VI soil criteria require review."
        )
    if len(bands) > 1:
        notes.append("The 5% Method 1 uncertainty range intersects multiple numerical Vs30 bands.")

    return Ts1170Method1Vs30Result(
        method="TS 1170.5:2025 Method 1 — direct measured Vs",
        value_mps=value,
        lower_bound_mps=lower,
        upper_bound_mps=upper,
        uncertainty_factor=0.05,
        raw_value_mps=raw_value,
        vertical_travel_time_s=adjusted_time,
        raw_vertical_travel_time_s=raw_time,
        included_thicknesses_m=included_thicknesses,
        included_velocities_mps=adjusted_velocities,
        raw_included_velocities_mps=raw_velocities,
        measured_depth_m=measured_depth,
        adjusted_shallow_thickness_m=3.0,
        shallow_reference_velocity_mps=shallow_reference_velocity,
        extrapolated_thickness_m=extrapolated_thickness,
        extrapolated_velocity_mps=extrapolated_velocity,
        slow_velocity_thickness_top20_m=slow_thickness,
        indicative_vs30_bands=bands,
        spans_multiple_vs30_bands=len(bands) > 1,
        notes=tuple(notes),
    )


>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
        weighted_rmse_s=weighted_rmse_ms / 1000.0,
        data_cost_ms2=final_data_cost,
        regularization_cost=regularization_contribution,
        roughness_norm=roughness_norm,
        objective_value=final_data_cost + regularization_contribution,
        observation_std_s=observation_std.copy(),
        standardized_residuals=standardized_residuals,
        resolution_diagonal=resolution,
        observation_leverage=leverage,
        influence_scores=influence,
        outlier_flags=outlier_flags,
        influential_flags=influential_flags,
        bound_active_flags=bound_active,
        robust_loss=loss_name,
        warnings=tuple(warnings),
    )


def invert_layered_velocity_profile(
    depths_m: Sequence[float],
    observed_times_s: Sequence[float],
    source_offset_m: float,
    layer_bottoms_m: Sequence[float],
    progress_callback: Any | None = None,
    receiver_offsets_m: Sequence[float] | None = None,
    observation_std_s: Sequence[float] | None = None,
    robust_loss: str = "linear",
    huber_threshold_sigma: float = 1.5,
) -> InversionResult:
    """Fit one constant Vs per analyst-defined layer without smoothing.

    The number and depths of model layers are fixed by the interpretation.
    Arrival-time uncertainty controls data weighting, while the optional Huber
    loss limits the influence of outliers. No roughness, smoothing, or
    regularisation term is applied.
    """

    z, layer_bottoms, _unused = _validate_layered_geometry(depths_m, layer_bottoms_m)
    t_obs = np.asarray(observed_times_s, dtype=float)
    if t_obs.ndim != 1 or t_obs.size != z.size or not np.all(np.isfinite(t_obs)) or not np.all(t_obs > 0.0):
        raise ValueError("Provide one finite positive arrival time for every receiver depth.")
    if not math.isfinite(source_offset_m) or source_offset_m < 0.0:
        raise ValueError("Source offset must be finite and non-negative.")
    if receiver_offsets_m is None:
        receiver_offsets = np.full(z.size, float(source_offset_m), dtype=float)
    else:
        receiver_offsets = np.asarray(receiver_offsets_m, dtype=float)
        if receiver_offsets.ndim != 1 or receiver_offsets.size != z.size:
            raise ValueError("Receiver offsets must contain one value for every receiver depth.")
        if not np.all(np.isfinite(receiver_offsets)) or not np.all(receiver_offsets >= 0.0):
            raise ValueError("Receiver offsets must be finite and non-negative.")
    if observation_std_s is None:
        observation_std = np.full(z.size, 0.001, dtype=float)
    else:
        observation_std = np.asarray(observation_std_s, dtype=float)
        if observation_std.ndim != 1 or observation_std.size != z.size:
            raise ValueError("Observation uncertainty must contain one standard deviation per arrival.")
        if not np.all(np.isfinite(observation_std)) or not np.all(observation_std > 0.0):
            raise ValueError("Observation standard deviations must be finite and greater than zero.")
    loss_name = str(robust_loss).strip().lower()
    if loss_name not in {"linear", "huber"}:
        raise ValueError("Robust loss must be 'linear' or 'huber'.")
    huber_delta = float(huber_threshold_sigma)
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("The Huber threshold must be finite and greater than zero.")

    layer_count = layer_bottoms.size
    interval_layer_indices = np.searchsorted(layer_bottoms, z, side="left")
    receiver_initial = _initial_velocity_estimate(z, t_obs, receiver_offsets)
    finite_initial = receiver_initial[np.isfinite(receiver_initial)]
    fallback = float(np.median(finite_initial)) if finite_initial.size else 250.0
    initial = np.full(layer_count, fallback, dtype=float)
    for layer_index in range(layer_count):
        candidates = receiver_initial[interval_layer_indices == layer_index]
        candidates = candidates[np.isfinite(candidates) & (candidates > 0.0)]
        if candidates.size:
            initial[layer_index] = float(np.median(candidates))
    initial = np.clip(initial, VELOCITY_MIN, VELOCITY_MAX)
    lower = math.log(VELOCITY_MIN)
    upper = math.log(VELOCITY_MAX)
    sigma_ms = observation_std * 1000.0
    inverse_variance = 1.0 / (sigma_ms * sigma_ms)
    sigma_reference_ms = math.sqrt(z.size / float(np.sum(inverse_variance)))
    eval_count = 0

    def layered_jacobian_ms(velocities: np.ndarray, rays: Sequence[RaySolution], z_nodes: Sequence[np.ndarray]) -> np.ndarray:
        jacobian = np.zeros((z.size, layer_count), dtype=float)
        for receiver_index, (ray, nodes) in enumerate(zip(rays, z_nodes)):
            thicknesses = np.diff(nodes)
            path_lengths = thicknesses / np.cos(ray.angles_rad)
            crossed = thicknesses.size
            jacobian[receiver_index, :crossed] = -1000.0 * path_lengths / velocities[:crossed]
        return jacobian

    def objective(log_vs: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal eval_count
        eval_count += 1
        velocities = np.exp(log_vs)
        try:
            calculated, rays, z_nodes = forward_layered_model(
                z,
                layer_bottoms,
                velocities,
                source_offset_m,
                receiver_offsets,
            )
        except (ValueError, RayPathError, FloatingPointError):
            return 1.0e30, np.zeros_like(log_vs)
        residual_ms = (calculated - t_obs) * 1000.0
        standardized = residual_ms / sigma_ms
        if loss_name == "huber":
            absolute = np.abs(standardized)
            rho = np.where(
                absolute <= huber_delta,
                standardized * standardized,
                2.0 * huber_delta * absolute - huber_delta * huber_delta,
            )
            psi = np.where(
                absolute <= huber_delta,
                2.0 * standardized,
                2.0 * huber_delta * np.sign(standardized),
            )
        else:
            rho = standardized * standardized
            psi = 2.0 * standardized
        data_cost = float(sigma_reference_ms * sigma_reference_ms * np.mean(rho))
        jacobian = layered_jacobian_ms(velocities, rays, z_nodes)
        gradient = sigma_reference_ms * sigma_reference_ms / z.size * jacobian.T.dot(psi / sigma_ms)
        if progress_callback is not None and eval_count % 10 == 0:
            progress_callback(eval_count, math.sqrt(max(data_cost, 0.0)))
        return data_cost, gradient

    optimisation = minimize(
        objective,
        np.log(initial),
        method="L-BFGS-B",
        jac=True,
        bounds=[(lower, upper)] * layer_count,
        options={"maxiter": 600, "ftol": 1.0e-12, "gtol": 1.0e-7, "maxls": 40},
    )
    layer_velocities = np.exp(optimisation.x)
    calculated, rays, ray_z_nodes = forward_layered_model(
        z,
        layer_bottoms,
        layer_velocities,
        source_offset_m,
        receiver_offsets,
    )
    residuals = calculated - t_obs
    residual_ms = residuals * 1000.0
    standardized_residuals = residuals / observation_std
    if loss_name == "huber":
        absolute_standardized = np.abs(standardized_residuals)
        robust_weight = np.where(
            absolute_standardized <= huber_delta,
            1.0,
            huber_delta / np.maximum(absolute_standardized, np.finfo(float).eps),
        )
        rho = np.where(
            absolute_standardized <= huber_delta,
            standardized_residuals * standardized_residuals,
            2.0 * huber_delta * absolute_standardized - huber_delta * huber_delta,
        )
    else:
        robust_weight = np.ones(z.size, dtype=float)
        rho = standardized_residuals * standardized_residuals
    normalised_weight = inverse_variance / float(np.mean(inverse_variance))
    weighted_rmse_ms = float(np.sqrt(np.mean(normalised_weight * residual_ms * residual_ms)))
    final_jacobian = layered_jacobian_ms(layer_velocities, rays, ray_z_nodes)
    weighted_jacobian = np.sqrt(normalised_weight * robust_weight / z.size)[:, np.newaxis] * final_jacobian
    if int(np.linalg.matrix_rank(weighted_jacobian, tol=1.0e-10)) < layer_count:
        raise RayPathError(
            "The interpreted layer model is not independently resolvable from these observations. "
            "Remove or revise layer boundaries."
        )
    data_hessian = weighted_jacobian.T.dot(weighted_jacobian)
    information = data_hessian + np.eye(layer_count) * max(
        1.0e-12, float(np.trace(data_hessian)) * 1.0e-12 / layer_count
    )
    inverse_information = np.linalg.pinv(information, rcond=1.0e-10)
    layer_resolution = np.clip(np.diag(inverse_information.dot(data_hessian)), 0.0, 1.0)
    hat_matrix = weighted_jacobian.dot(inverse_information).dot(weighted_jacobian.T)
    leverage = np.clip(np.diag(hat_matrix), 0.0, 1.0 - 1.0e-9)
    influence = np.abs(standardized_residuals) * np.sqrt(
        leverage / np.maximum(1.0 - leverage, 1.0e-9)
    )
    outlier_flags = np.abs(standardized_residuals) > 3.0
    influential_flags = influence > 2.0
    bound_tolerance = 1.0e-3
    layer_bound_active = (optimisation.x <= lower + bound_tolerance) | (optimisation.x >= upper - bound_tolerance)
    expanded_velocities = layer_velocities[interval_layer_indices]
    expanded_resolution = layer_resolution[interval_layer_indices]
    expanded_bound_active = layer_bound_active[interval_layer_indices]
    final_data_cost = float(sigma_reference_ms * sigma_reference_ms * np.mean(rho))
    warnings: list[str] = []
    if np.any(layer_bound_active):
        warnings.append(f"{int(np.count_nonzero(layer_bound_active))} interpreted layer velocity parameter(s) are at solver bounds.")
    if np.any(layer_resolution < 0.20):
        warnings.append(f"{int(np.count_nonzero(layer_resolution < 0.20))} interpreted layer(s) have low linearised resolution (<0.20).")
    if np.any(outlier_flags):
        warnings.append(f"{int(np.count_nonzero(outlier_flags))} observation(s) exceed 3 standard deviations.")
    if np.any(influential_flags):
        warnings.append(f"{int(np.count_nonzero(influential_flags))} observation(s) have influence score above 2.")
    usable = bool(np.all(np.isfinite(layer_velocities)) and np.all(np.isfinite(calculated)))
    if not usable:
        raise RayPathError(f"Layered inversion did not produce a finite model: {optimisation.message}")
    return InversionResult(
        depths_m=z,
        thicknesses_m=np.diff(np.r_[0.0, z]),
        velocities_mps=expanded_velocities,
        observed_times_s=t_obs,
        calculated_times_s=calculated,
        residuals_s=residuals,
        ray_parameters=np.asarray([ray.ray_parameter for ray in rays]),
        ray_x_segments=[ray.horizontal_segments_m.copy() for ray in rays],
        receiver_offsets_m=receiver_offsets.copy(),
        rmse_s=float(np.sqrt(np.mean(residuals * residuals))),
        success=bool(optimisation.success),
        message=f"Piecewise-constant layered model; {optimisation.message}",
        iterations=int(getattr(optimisation, "nit", 0)),
        weighted_rmse_s=weighted_rmse_ms / 1000.0,
        data_cost_ms2=final_data_cost,
        regularization_cost=0.0,
        roughness_norm=0.0,
        objective_value=final_data_cost,
        observation_std_s=observation_std.copy(),
        standardized_residuals=standardized_residuals,
        resolution_diagonal=expanded_resolution,
        observation_leverage=leverage,
        influence_scores=influence,
        outlier_flags=outlier_flags,
        influential_flags=influential_flags,
        bound_active_flags=expanded_bound_active,
        robust_loss=loss_name,
        warnings=tuple(warnings),
        model_layer_bottoms_m=layer_bottoms.copy(),
        model_layer_velocities_mps=layer_velocities,
        model_layer_resolution=layer_resolution,
        model_layer_bound_active=layer_bound_active,
        ray_z_nodes_m=[nodes.copy() for nodes in ray_z_nodes],
    )


def result_model_profile(result: InversionResult) -> tuple[np.ndarray, np.ndarray]:
    """Return the actual interpreted layer profile represented by a result."""

    if (
        result.model_layer_bottoms_m.size
        and result.model_layer_bottoms_m.size == result.model_layer_velocities_mps.size
    ):
        return result.model_layer_bottoms_m, result.model_layer_velocities_mps
    return result.depths_m, result.velocities_mps


def select_regularization_lcurve(
    depths_m: Sequence[float],
    observed_times_s: Sequence[float],
    source_offset_m: float,
    receiver_offsets_m: Sequence[float] | None = None,
    observation_std_s: Sequence[float] | None = None,
    robust_loss: str = "linear",
    candidate_factors: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75, 1.0),
) -> tuple[RegularizationSelectionResult, InversionResult]:
    """Select regularisation using maximum distance from the log-log L-curve chord.

    All candidate models use identical observations, uncertainty weights, loss,
    bounds, and geometry.  The end points define a chord after both log axes are
    normalised to 0..1; the interior candidate farthest toward the L-curve
    corner is selected.  This deterministic criterion supplements, rather than
    removes, the manual slider.
    """

    factors = np.asarray(candidate_factors, dtype=float)
    if factors.ndim != 1 or factors.size < 3 or not np.all(np.isfinite(factors)):
        raise ValueError("L-curve selection requires at least three finite candidate factors.")
    if np.any(factors < 0.0) or np.any(factors > 1.0) or np.any(np.diff(factors) <= 0.0):
        raise ValueError("L-curve candidate factors must be strictly increasing within 0..1.")
    models = [
        invert_velocity_profile(
            depths_m,
            observed_times_s,
            source_offset_m,
            regularization=float(factor),
            receiver_offsets_m=receiver_offsets_m,
            observation_std_s=observation_std_s,
            robust_loss=robust_loss,
        )
        for factor in factors
    ]
    misfit = np.asarray([max(model.weighted_rmse_s * 1000.0, 1.0e-12) for model in models])
    roughness = np.asarray([max(model.roughness_norm, 1.0e-12) for model in models])
    x = np.log10(roughness)
    y = np.log10(misfit)

    def normalise(values: np.ndarray) -> np.ndarray:
        span = float(np.ptp(values))
        return np.zeros_like(values) if span <= 1.0e-12 else (values - float(np.min(values))) / span

    x_norm = normalise(x)
    y_norm = normalise(y)
    start = np.asarray([x_norm[0], y_norm[0]])
    end = np.asarray([x_norm[-1], y_norm[-1]])
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    distances = np.zeros(factors.size, dtype=float)
    if chord_length > 1.0e-12:
        for index in range(1, factors.size - 1):
            point = np.asarray([x_norm[index], y_norm[index]])
            offset = point - start
            distances[index] = abs(float(chord[0] * offset[1] - chord[1] * offset[0])) / chord_length
    selected_index = int(np.argmax(distances[1:-1])) + 1
    selection = RegularizationSelectionResult(
        selected_factor=float(factors[selected_index]),
        candidate_factors=factors,
        weighted_rmse_ms=misfit,
        roughness_norm=roughness,
        chord_distances=distances,
    )
    return selection, models[selected_index]


def generate_velocity_uncertainty_ensemble(
    central_result: InversionResult,
    source_offset_m: float,
    regularization: float,
    ensemble_size: int = QUICK_ENSEMBLE_SIZE,
    random_seed: int = DEFAULT_UNCERTAINTY_SEED,
    robust_loss: str | None = None,
    progress_callback: Any | None = None,
) -> VelocityUncertaintyResult:
    """Perturb arrivals by their 1-sigma uncertainties and reinvert repeatedly."""

    requested = int(ensemble_size)
    if requested < 1:
        raise ValueError("The uncertainty ensemble must request at least one model.")
    seed = int(random_seed)
    if central_result.observation_std_s.size != central_result.depths_m.size:
        raise ValueError("The central model does not contain one uncertainty for every observation.")
    rng = np.random.default_rng(seed)
    velocity_models: list[np.ndarray] = []
    vs30_values: list[float] = []
    failure_count = 0
    selected_loss = robust_loss or central_result.robust_loss
    for index in range(requested):
        perturbed = central_result.observed_times_s + rng.normal(
            0.0, central_result.observation_std_s, central_result.depths_m.size
        )
        if np.any(perturbed <= 0.0):
            failure_count += 1
            continue
        try:
            model = invert_velocity_profile(
                central_result.depths_m,
                perturbed,
                source_offset_m,
                regularization=regularization,
                receiver_offsets_m=central_result.receiver_offsets_m,
                observation_std_s=central_result.observation_std_s,
                robust_loss=selected_loss,
            )
            velocity_models.append(model.velocities_mps)
            try:
                vs30_values.append(
                    calculate_ts1170_5_method1_vs30(model.depths_m, model.velocities_mps).value_mps
                )
            except ValueError:
                pass
        except (ValueError, RayPathError, FloatingPointError):
            failure_count += 1
        if progress_callback is not None:
            progress_callback(index + 1, requested)
    minimum_success = max(3, int(math.ceil(0.80 * requested)))
    if len(velocity_models) < minimum_success:
        raise RayPathError(
            f"Only {len(velocity_models)} of {requested} uncertainty models succeeded; "
            f"at least {minimum_success} are required."
        )
    levels = (2.5, 50.0, 97.5)
    velocity_array = np.asarray(velocity_models, dtype=float)
    velocity_percentiles = np.percentile(velocity_array, levels, axis=0)
    vs30_array = np.asarray(vs30_values, dtype=float)
    if vs30_array.size:
        vs30_percentiles = np.percentile(vs30_array, levels)
        vs30_lower, vs30_median, vs30_upper = (float(value) for value in vs30_percentiles)
    else:
        vs30_lower = vs30_median = vs30_upper = None
    warnings = []
    if requested < REPORT_QUALITY_ENSEMBLE_MINIMUM:
        warnings.append(
            f"{requested} models provide a preliminary sensitivity preview; use at least "
            f"{REPORT_QUALITY_ENSEMBLE_MINIMUM} models for report-quality percentile intervals."
        )
    if failure_count:
        warnings.append(f"{failure_count} perturbed model(s) failed and were excluded from the percentiles.")
    return VelocityUncertaintyResult(
        random_seed=seed,
        requested_models=requested,
        successful_models=len(velocity_models),
        percentile_levels=levels,
        velocity_lower_mps=velocity_percentiles[0],
        velocity_median_mps=velocity_percentiles[1],
        velocity_upper_mps=velocity_percentiles[2],
        vs30_samples_mps=vs30_array,
        vs30_lower_mps=vs30_lower,
        vs30_median_mps=vs30_median,
        vs30_upper_mps=vs30_upper,
        warnings=tuple(warnings),
>>>>>>> Stashed changes
    )


def generate_layered_velocity_uncertainty_ensemble(
    central_result: InversionResult,
    source_offset_m: float,
    ensemble_size: int = QUICK_ENSEMBLE_SIZE,
    random_seed: int = DEFAULT_UNCERTAINTY_SEED,
    robust_loss: str | None = None,
    progress_callback: Any | None = None,
) -> VelocityUncertaintyResult:
    """Perturb picks and refit the same fixed interpreted layer geometry."""

    if central_result.model_layer_bottoms_m.size == 0:
        raise ValueError("The central result does not contain interpreted layer boundaries.")
    requested = int(ensemble_size)
    if requested < 1:
        raise ValueError("The uncertainty ensemble must request at least one model.")
    if central_result.observation_std_s.size != central_result.depths_m.size:
        raise ValueError("The central model does not contain one uncertainty for every observation.")
    seed = int(random_seed)
    rng = np.random.default_rng(seed)
    velocity_models: list[np.ndarray] = []
    vs30_values: list[float] = []
    failure_count = 0
    selected_loss = robust_loss or central_result.robust_loss
    for index in range(requested):
        perturbed = central_result.observed_times_s + rng.normal(
            0.0, central_result.observation_std_s, central_result.depths_m.size
        )
        if np.any(perturbed <= 0.0):
            failure_count += 1
            continue
        try:
            model = invert_layered_velocity_profile(
                central_result.depths_m,
                perturbed,
                source_offset_m,
                central_result.model_layer_bottoms_m,
                receiver_offsets_m=central_result.receiver_offsets_m,
                observation_std_s=central_result.observation_std_s,
                robust_loss=selected_loss,
            )
            velocity_models.append(model.velocities_mps)
            model_depths, model_velocities = result_model_profile(model)
            try:
                vs30_values.append(
                    calculate_ts1170_5_method1_vs30(model_depths, model_velocities).value_mps
                )
            except ValueError:
                pass
        except (ValueError, RayPathError, FloatingPointError):
            failure_count += 1
        if progress_callback is not None:
            progress_callback(index + 1, requested)
    minimum_success = max(3, int(math.ceil(0.80 * requested)))
    if len(velocity_models) < minimum_success:
        raise RayPathError(
            f"Only {len(velocity_models)} of {requested} layered uncertainty models succeeded; "
            f"at least {minimum_success} are required."
        )
    levels = (2.5, 50.0, 97.5)
    velocity_percentiles = np.percentile(np.asarray(velocity_models, dtype=float), levels, axis=0)
    vs30_array = np.asarray(vs30_values, dtype=float)
    if vs30_array.size:
        vs30_lower, vs30_median, vs30_upper = (
            float(value) for value in np.percentile(vs30_array, levels)
        )
    else:
        vs30_lower = vs30_median = vs30_upper = None
    warnings: list[str] = []
    if requested < REPORT_QUALITY_ENSEMBLE_MINIMUM:
        warnings.append(
            f"{requested} models provide a preliminary sensitivity preview; use at least "
            f"{REPORT_QUALITY_ENSEMBLE_MINIMUM} models for report-quality percentile intervals."
        )
    if failure_count:
        warnings.append(f"{failure_count} perturbed layered model(s) failed and were excluded.")
    return VelocityUncertaintyResult(
        random_seed=seed,
        requested_models=requested,
        successful_models=len(velocity_models),
        percentile_levels=levels,
        velocity_lower_mps=velocity_percentiles[0],
        velocity_median_mps=velocity_percentiles[1],
        velocity_upper_mps=velocity_percentiles[2],
        vs30_samples_mps=vs30_array,
        vs30_lower_mps=vs30_lower,
        vs30_median_mps=vs30_median,
        vs30_upper_mps=vs30_upper,
        warnings=tuple(warnings),
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


<<<<<<< Updated upstream
=======
@dataclass(frozen=True)
class InversionBatchResult:
    """Central pick models and optional uncertainty for fixed interpreted layers."""

    results: dict[str, InversionResult]
    uncertainty_results: dict[str, VelocityUncertaintyResult]
    layer_bottoms_m: np.ndarray


>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        regularization: float,
=======
        receiver_offsets_m: np.ndarray,
        layer_bottoms_m: np.ndarray,
        observation_std_s: np.ndarray,
        robust_loss: str,
        selected_kind: str,
        ensemble_size: int,
        random_seed: int,
>>>>>>> Stashed changes
    ) -> None:
        super().__init__()
        self.depths_m = depths_m
        self.times_by_pick_s = times_by_pick_s
        self.offset_m = offset_m
<<<<<<< Updated upstream
        self.regularization = regularization
=======
        self.receiver_offsets_m = receiver_offsets_m
        self.layer_bottoms_m = layer_bottoms_m
        self.observation_std_s = observation_std_s
        self.robust_loss = robust_loss
        self.selected_kind = selected_kind
        self.ensemble_size = ensemble_size
        self.random_seed = random_seed
>>>>>>> Stashed changes

    @Slot()
    def run(self) -> None:
        try:
            results: dict[str, InversionResult] = {}
            for kind, times_s in self.times_by_pick_s.items():
<<<<<<< Updated upstream
                results[kind] = invert_velocity_profile(
                    self.depths_m,
                    times_s,
                    self.offset_m,
                    self.regularization,
=======
                results[kind] = invert_layered_velocity_profile(
                    self.depths_m,
                    times_s,
                    self.offset_m,
                    self.layer_bottoms_m,
>>>>>>> Stashed changes
                    lambda evaluations, rmse, pick_kind=kind: self.progress.emit(pick_kind, evaluations, rmse),
                )
<<<<<<< Updated upstream
            self.finished.emit(results)
=======
            uncertainty_results: dict[str, VelocityUncertaintyResult] = {}
            if self.ensemble_size > 0:
                uncertainty_results[self.selected_kind] = generate_layered_velocity_uncertainty_ensemble(
                    results[self.selected_kind],
                    self.offset_m,
                    ensemble_size=self.ensemble_size,
                    random_seed=self.random_seed,
                    robust_loss=self.robust_loss,
                    progress_callback=lambda complete, total: self.progress.emit(
                        f"{self.selected_kind}:ensemble", complete, float(total)
                    ),
                )
            self.finished.emit(
                InversionBatchResult(
                    results=results,
                    uncertainty_results=uncertainty_results,
                    layer_bottoms_m=self.layer_bottoms_m.copy(),
                )
            )
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
        self.gru_pre_trigger_ms: float | None = None
        self.survey_geometry = SurveyGeometry()
        self.active_geometry: CorrectedGeometry | None = None
        self.active_layer_recorded_bottoms_m = np.empty(0, dtype=float)
        self.active_layer_notes: list[str] = []
>>>>>>> Stashed changes
        self.waveform_records: list[WaveformRecord] = []
        self.result: InversionResult | None = None
        self.comparison_results: dict[str, InversionResult] = {}
        self.comparison_vs30: dict[str, Vs30Result | None] = {}
        self.comparison_vs30_reasons: dict[str, str] = {}
        self.current_vs30: Vs30Result | None = None
        self.vs30_unavailable_reason: str | None = None
<<<<<<< Updated upstream
        self.vs30_history: dict[tuple[str, float, float], float] = {}
=======
        self.uncertainty_results: dict[str, VelocityUncertaintyResult] = {}
        self.picker_half_width_ms = DEFAULT_PICKER_HALF_WIDTH_MS
>>>>>>> Stashed changes
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
        subtitle = QLabel("Interpretive layered forward ray-path inversion")
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

<<<<<<< Updated upstream
=======
    @Slot(bool)
    def _theme_changed(self, dark_mode: bool) -> None:
        """Apply the selected theme to Qt widgets and all embedded plots."""

        self.dark_mode = bool(dark_mode)
        app = QApplication.instance()
        if app is not None:
            apply_application_theme(app, self.dark_mode)
        canvases = (
            self.velocity_canvas,
            self.interval_canvas,
            self.ray_canvas,
            self.fit_canvas,
            self.waterfall_canvas,
            self.vs30_canvas,
        )
        for canvas in canvases:
            canvas.set_dark_mode(self.dark_mode)
        if self.waveform_records:
            self._apply_waveform_review_to_input_table()
        elif self.observation_review:
            self._apply_saved_review_to_input_table()
        self._update_geometry_status()
        if self.result is None:
            self._draw_empty_plots()
        else:
            self._draw_results(self.result)

>>>>>>> Stashed changes
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

        layer_heading = QLabel("Interpreted geological layers")
        layer_heading.setObjectName("minorTitle")
        layout.addWidget(layer_heading)
        layer_note = QLabel(
            "Enter internal layer bottoms using recorded receiver depth. The deepest receiver closes the final "
            "layer automatically. One constant Vs is fitted per layer; no smoothing is applied."
        )
        layer_note.setObjectName("subtleLabel")
        layer_note.setWordWrap(True)
        layout.addWidget(layer_note)
        self.layer_boundary_table = QTableWidget(0, 2)
        self.layer_boundary_table.setHorizontalHeaderLabels(
            ["Internal bottom depth (m)", "Interpretive basis / unit"]
        )
        self.layer_boundary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.layer_boundary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.layer_boundary_table.verticalHeader().setVisible(False)
        self.layer_boundary_table.setMinimumHeight(105)
        self.layer_boundary_table.setMaximumHeight(155)
        self.layer_boundary_table.itemChanged.connect(self._layer_boundaries_changed)
        layout.addWidget(self.layer_boundary_table)
        layer_buttons = QHBoxLayout()
        self.add_selected_boundary_button = QPushButton("Use Selected Depth")
        self.add_selected_boundary_button.setToolTip(
            "Add the receiver depth selected in the observations table as an interpreted layer bottom."
        )
        self.add_selected_boundary_button.clicked.connect(self._add_selected_layer_boundary)
        self.add_boundary_button = QPushButton("+ Boundary")
        self.add_boundary_button.clicked.connect(self._add_blank_layer_boundary)
        self.remove_boundary_button = QPushButton("- Boundary")
        self.remove_boundary_button.clicked.connect(self._remove_layer_boundaries)
        self.clear_boundary_button = QPushButton("Clear")
        self.clear_boundary_button.clicked.connect(self._clear_layer_boundaries)
        layer_buttons.addWidget(self.add_selected_boundary_button)
        layer_buttons.addWidget(self.add_boundary_button)
        layer_buttons.addWidget(self.remove_boundary_button)
        layer_buttons.addWidget(self.clear_boundary_button)
        layout.addLayout(layer_buttons)

<<<<<<< Updated upstream
        self.run_button = QPushButton("Run RayPath Inversion")
=======
        inversion_options = QFormLayout()
        self.robust_loss_combo = QComboBox()
        self.robust_loss_combo.addItem("Linear weighted least squares", "linear")
        self.robust_loss_combo.addItem("Huber robust loss (1.5σ)", "huber")
        self.robust_loss_combo.currentIndexChanged.connect(self._inversion_option_changed)
        inversion_options.addRow("Data loss", self.robust_loss_combo)
        self.manual_uncertainty_spin = QDoubleSpinBox()
        self.manual_uncertainty_spin.setRange(0.001, 50.0)
        self.manual_uncertainty_spin.setDecimals(3)
        self.manual_uncertainty_spin.setValue(DEFAULT_MANUAL_PICK_UNCERTAINTY_MS)
        self.manual_uncertainty_spin.setSuffix(" ms (1σ)")
        self.manual_uncertainty_spin.setToolTip(
            "Fallback one-standard-deviation uncertainty for manual/CSV observations and any GRU interval "
            "without a recorded per-depth value."
        )
        self.manual_uncertainty_spin.valueChanged.connect(self._inversion_option_changed)
        inversion_options.addRow("Fallback uncertainty", self.manual_uncertainty_spin)
        ensemble_controls = QWidget()
        ensemble_controls_layout = QHBoxLayout(ensemble_controls)
        ensemble_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.ensemble_preset_combo = QComboBox()
        self.ensemble_preset_combo.addItem("Off - fastest", DEFAULT_ENSEMBLE_SIZE)
        self.ensemble_preset_combo.addItem("Quick preview - 20 (preliminary)", QUICK_ENSEMBLE_SIZE)
        self.ensemble_preset_combo.addItem("Final report - 200", FINAL_ENSEMBLE_SIZE)
        self.ensemble_preset_combo.addItem("Custom count", -1)
        self.ensemble_preset_combo.setToolTip(
            "Keep uncertainty sampling off while reviewing picks and layer boundaries. The 20-model preview is "
            "preliminary; use at least 100 models for a report-quality percentile interval."
        )
        ensemble_controls_layout.addWidget(self.ensemble_preset_combo, 1)
        self.ensemble_size_spin = QSpinBox()
        self.ensemble_size_spin.setRange(0, 500)
        self.ensemble_size_spin.setValue(DEFAULT_ENSEMBLE_SIZE)
        self.ensemble_size_spin.setSpecialValueText("Off")
        self.ensemble_size_spin.setFixedWidth(80)
        self.ensemble_size_spin.setToolTip("Custom number of pick-time perturbation models (maximum 500).")
        self.ensemble_size_spin.setVisible(False)
        ensemble_controls_layout.addWidget(self.ensemble_size_spin)
        self.ensemble_preset_combo.currentIndexChanged.connect(self._ensemble_preset_changed)
        self.ensemble_size_spin.valueChanged.connect(self._ensemble_size_changed)
        inversion_options.addRow("Uncertainty ensemble", ensemble_controls)
        self.uncertainty_seed_spin = QSpinBox()
        self.uncertainty_seed_spin.setRange(0, 2_147_483_647)
        self.uncertainty_seed_spin.setValue(DEFAULT_UNCERTAINTY_SEED)
        self.uncertainty_seed_spin.setToolTip("Stored random seed used to make uncertainty bands exactly repeatable.")
        self.uncertainty_seed_spin.valueChanged.connect(self._inversion_option_changed)
        inversion_options.addRow("Random seed", self.uncertainty_seed_spin)
        layout.addLayout(inversion_options)

        self.run_button = QPushButton("Run Layered RayPath Inversion")
>>>>>>> Stashed changes
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
        self.interval_canvas = MplCanvas()
        self.ray_canvas = MplCanvas()
        self.fit_canvas = MplCanvas()
        self.waterfall_canvas = MplCanvas()
        self.plot_tabs.addTab(self.velocity_canvas, "Velocity Profile")
        self.plot_tabs.addTab(self.interval_canvas, "1 m Interval Comparison")
        self.plot_tabs.addTab(self.ray_canvas, "Ray Paths")
        self.plot_tabs.addTab(self.fit_canvas, "Arrival-Time Fit")
        self.plot_tabs.addTab(self.waterfall_canvas, "Waveform Waterfall")
        self.plot_tabs.addTab(self._build_vs30_tab(), "Vs30 Analysis")
        layout.addWidget(self.plot_tabs, 1)
        self._draw_empty_plots()
        return panel

    def _build_vs30_tab(self) -> QWidget:
        """Create the Vs30 value and pick-interpretation comparison view."""

        tab = QWidget()
        layout = QVBoxLayout(tab)
        metric_row = QHBoxLayout()
        self.vs30_value_label = QLabel("Vs30: — m/s")
        self.vs30_value_label.setObjectName("vs30Value")
        metric_row.addWidget(self.vs30_value_label)
        metric_row.addStretch()
        self.vs30_layer_label = QLabel("Interpreted layers: —")
        self.vs30_layer_label.setObjectName("accentLabel")
        metric_row.addWidget(self.vs30_layer_label)
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
<<<<<<< Updated upstream
        if self.result is not None or self.vs30_history or self.result_table.rowCount() > 0:
=======
        self.active_geometry = None
        self.active_layer_recorded_bottoms_m = np.empty(0, dtype=float)
        self.active_layer_notes = []
        if self.result is not None or self.result_table.rowCount() > 0:
>>>>>>> Stashed changes
            self._clear_results(clear_vs30_history=True)
        self._set_dirty(True)

<<<<<<< Updated upstream
    def _regularization_changed(self, value: int) -> None:
        self.reg_value_label.setText(f"{value / 100.0:.2f}")
        self._clear_results(clear_vs30_history=False)
        if self.vs30_history:
            self.vs30_detail_label.setText(
                "Smoothing changed. Run the inversion to add this setting to the Vs30 comparison."
            )
=======
    def _ensemble_preset_changed(self, _index: int) -> None:
        """Apply an Off, preview, final, or custom ensemble preset."""

        preset = int(self.ensemble_preset_combo.currentData())
        if preset >= 0:
            self.ensemble_size_spin.blockSignals(True)
            self.ensemble_size_spin.setValue(preset)
            self.ensemble_size_spin.blockSignals(False)
            self.ensemble_size_spin.setVisible(False)
        else:
            if self.ensemble_size_spin.value() in {
                DEFAULT_ENSEMBLE_SIZE,
                QUICK_ENSEMBLE_SIZE,
                FINAL_ENSEMBLE_SIZE,
            }:
                self.ensemble_size_spin.blockSignals(True)
                self.ensemble_size_spin.setValue(50)
                self.ensemble_size_spin.blockSignals(False)
            self.ensemble_size_spin.setVisible(True)
        self._inversion_option_changed()

    def _ensemble_size_changed(self, value: int) -> None:
        """Keep the preset label synchronized with a loaded or custom count."""

        preset = value if value in {DEFAULT_ENSEMBLE_SIZE, QUICK_ENSEMBLE_SIZE, FINAL_ENSEMBLE_SIZE} else -1
        index = self.ensemble_preset_combo.findData(preset)
        self.ensemble_preset_combo.blockSignals(True)
        self.ensemble_preset_combo.setCurrentIndex(max(0, index))
        self.ensemble_preset_combo.blockSignals(False)
        self.ensemble_size_spin.setVisible(preset == -1)
        self._inversion_option_changed()

    def _inversion_option_changed(self, *_args: Any) -> None:
        """Invalidate results when weighting, loss, selection, or ensemble settings change."""

        if self.result is not None:
            self._clear_results(clear_vs30_history=False)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        self.review_action.setEnabled(False)
=======
        self.gru_pre_trigger_ms = None
        self.survey_geometry = SurveyGeometry()
        self.active_geometry = None
        self.active_layer_recorded_bottoms_m = np.empty(0, dtype=float)
        self.active_layer_notes = []
        self._set_waveform_review_available(False)
>>>>>>> Stashed changes
        self.gru_label.setText("No GRU source loaded")
        self.input_table.blockSignals(True)
        self.input_table.clearContents()
        self.input_table.setRowCount(8)
        self.input_table.blockSignals(False)
        self.offset_spin.setValue(2.4)
<<<<<<< Updated upstream
        self.reg_slider.setValue(35)
=======
        self.layer_boundary_table.blockSignals(True)
        self.layer_boundary_table.setRowCount(0)
        self.layer_boundary_table.blockSignals(False)
        self.robust_loss_combo.setCurrentIndex(0)
        self.manual_uncertainty_spin.setValue(DEFAULT_MANUAL_PICK_UNCERTAINTY_MS)
        self.ensemble_size_spin.setValue(DEFAULT_ENSEMBLE_SIZE)
        self.uncertainty_seed_spin.setValue(DEFAULT_UNCERTAINTY_SEED)
>>>>>>> Stashed changes
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
        """Clear the active model; the argument is retained for legacy callers."""

        self.result = None
        self.comparison_results.clear()
        self.comparison_vs30.clear()
        self.comparison_vs30_reasons.clear()
        self.current_vs30 = None
        self.vs30_unavailable_reason = None
<<<<<<< Updated upstream
        if clear_vs30_history:
            self.vs30_history.clear()
=======
        self.uncertainty_results.clear()
>>>>>>> Stashed changes
        self.result_table.setRowCount(0)
        self.result_summary.setText("Run the inversion to calculate a layered Vs profile.")
        self.rmse_label.setText("RMSE: — ms")
        self.vs30_status_label.setText("Vs30: — m/s")
        self.vs30_comparison_label.setText("First peak: —   |   First cross: —   |   Maximum peak: —")
        self._draw_empty_plots()

    def _draw_empty_plots(self) -> None:
        configurations = (
            (self.velocity_canvas, "Velocity profile", "Vs (m/s)", "Depth (m)"),
            (
                self.interval_canvas,
                "Staggered 1 m observed interval comparison",
                "Vs (m/s)",
                "Corrected vertical depth (m)",
            ),
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
<<<<<<< Updated upstream
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
=======
        """Draw Method 1 Vs30 across picks for the fixed interpreted layers."""

        ax = self.vs30_canvas.axes
        self.vs30_canvas.clear()
        ax.set_title("TS 1170.5:2025 Method 1 Vs30 by pick interpretation")
        ax.set_ylabel("Vs30 (m/s)")
        available = [kind for kind in PICK_KINDS if self.comparison_vs30.get(kind) is not None]
        if available:
            values = np.asarray([self.comparison_vs30[kind].value_mps for kind in available], dtype=float)
            lower = np.asarray([self.comparison_vs30[kind].lower_bound_mps for kind in available], dtype=float)
            upper = np.asarray([self.comparison_vs30[kind].upper_bound_mps for kind in available], dtype=float)
            positions = np.arange(len(available), dtype=float)
            ax.bar(
                positions,
>>>>>>> Stashed changes
                values,
                color=[MODEL_COLORS[kind] for kind in available],
                alpha=0.82,
                width=0.68,
            )
            ax.errorbar(
                positions,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                fmt="none",
                ecolor=self.vs30_canvas.foreground_color,
                elinewidth=1.0,
                capsize=4,
            )
<<<<<<< Updated upstream
            ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
=======
            ax.set_xticks(positions, [PICK_LABELS[kind] for kind in available], rotation=12, ha="right")
            value_span = float(np.ptp(np.r_[lower, upper]))
            margin = max(8.0, value_span * 0.15)
            ax.set_ylim(max(0.0, float(np.min(lower)) - margin), float(np.max(upper)) + margin)
>>>>>>> Stashed changes
        else:
            ax.text(
                0.5,
                0.5,
                "Run a layered model extending to at least 25 m\nto calculate or extrapolate Vs30",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#8b949e",
            )
            ax.set_xticks([])
        layer_count = (
            int(self.result.model_layer_bottoms_m.size)
            if self.result is not None and self.result.model_layer_bottoms_m.size
            else 0
        )
        self.vs30_layer_label.setText(
            f"Interpreted layers: {layer_count}" if layer_count else "Interpreted layers: —"
        )
        if self.current_vs30 is not None:
<<<<<<< Updated upstream
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
=======
            bands = ", ".join(self.current_vs30.indicative_vs30_bands)
            self.vs30_value_label.setText(
                f"TS Method 1 Vs30: {self.current_vs30.value_mps:.1f} m/s "
                f"({self.current_vs30.lower_bound_mps:.1f}–{self.current_vs30.upper_bound_mps:.1f})"
            )
            extrapolation_note = (
                "No extension was required because the profile reaches at least 30 m."
                if self.current_vs30.extrapolated_thickness_m <= 0.0
                else (
                    f"The last measured layer was extended {self.current_vs30.extrapolated_thickness_m:.2f} m "
                    f"at {self.current_vs30.extrapolated_velocity_mps:.1f} m/s."
>>>>>>> Stashed changes
                )
            else:
                self.vs30_detail_label.setText(
                    f"Vertical travel time through 30.00 m: {self.current_vs30.vertical_travel_time_s * 1000.0:.2f} ms. "
                    "The modeled profile reaches 30 m, so extrapolation weighting is not applied."
                )
        elif self.vs30_unavailable_reason:
<<<<<<< Updated upstream
            self.vs30_value_label.setText("Vs30: unavailable")
            self.vs30_smoothing_label.setText(f"Smoothing: {self.reg_slider.value() / 100.0:.2f}")
            self.vs30_detail_label.setText(self.vs30_unavailable_reason)
        elif not all_factors:
            self.vs30_value_label.setText("Vs30: — m/s")
            self.vs30_smoothing_label.setText("Smoothing: —")
=======
            self.vs30_value_label.setText("TS Method 1 Vs30: unavailable")
            self.vs30_detail_label.setText(self.vs30_unavailable_reason)
        elif not available:
            self.vs30_value_label.setText("TS Method 1 Vs30: — m/s")
>>>>>>> Stashed changes
            self.vs30_detail_label.setText(
                "Run a layered model extending to at least 25 m. Revise the interpreted boundaries and rerun "
                "to compare a different geological model."
            )
        else:
<<<<<<< Updated upstream
            self.vs30_value_label.setText("Vs30: rerun required")
            self.vs30_smoothing_label.setText(f"Selected smoothing: {self.reg_slider.value() / 100.0:.2f}")
=======
            self.vs30_value_label.setText("TS Method 1 Vs30: rerun required")
>>>>>>> Stashed changes
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

    def _layer_boundaries_changed(self, *_args: Any) -> None:
        """Invalidate the model when the analyst changes layer geometry."""

        self.active_layer_recorded_bottoms_m = np.empty(0, dtype=float)
        self.active_layer_notes = []
        if self.result is not None:
            self._clear_results(clear_vs30_history=True)
        self._set_dirty(True)

    def _add_selected_layer_boundary(self) -> None:
        """Add the selected receiver depth as an internal interpreted boundary."""

        row = self.input_table.currentRow()
        item = self.input_table.item(row, 0) if row >= 0 else None
        if item is None or not item.text().strip():
            QMessageBox.information(
                self,
                "Select a receiver depth",
                "Select a populated observation row before adding its depth as a layer boundary.",
            )
            return
        try:
            depth = float(item.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid receiver depth", "The selected receiver depth is not numeric.")
            return
        self._insert_layer_boundary(depth, "Boundary selected from receiver observations")

    def _add_blank_layer_boundary(self) -> None:
        """Append an editable blank boundary row."""

        row = self.layer_boundary_table.rowCount()
        self.layer_boundary_table.insertRow(row)
        self.layer_boundary_table.setItem(row, 0, QTableWidgetItem(""))
        self.layer_boundary_table.setItem(row, 1, QTableWidgetItem(""))
        self.layer_boundary_table.setCurrentCell(row, 0)
        self.layer_boundary_table.editItem(self.layer_boundary_table.item(row, 0))
        self._layer_boundaries_changed()

    def _insert_layer_boundary(self, depth_m: float, note: str = "") -> None:
        row = self.layer_boundary_table.rowCount()
        self.layer_boundary_table.blockSignals(True)
        self.layer_boundary_table.insertRow(row)
        self.layer_boundary_table.setItem(row, 0, QTableWidgetItem(f"{depth_m:.3f}"))
        self.layer_boundary_table.setItem(row, 1, QTableWidgetItem(note))
        self.layer_boundary_table.blockSignals(False)
        self._layer_boundaries_changed()

    def _remove_layer_boundaries(self) -> None:
        rows = sorted({index.row() for index in self.layer_boundary_table.selectedIndexes()}, reverse=True)
        if not rows and self.layer_boundary_table.currentRow() >= 0:
            rows = [self.layer_boundary_table.currentRow()]
        for row in rows:
            self.layer_boundary_table.removeRow(row)
        if rows:
            self._layer_boundaries_changed()

    def _clear_layer_boundaries(self) -> None:
        if self.layer_boundary_table.rowCount() == 0:
            return
        self.layer_boundary_table.setRowCount(0)
        self._layer_boundaries_changed()

    def _read_layer_boundaries_recorded(
        self, deepest_recorded_depth_m: float
    ) -> tuple[np.ndarray, list[str]]:
        """Return sorted analyst-entered internal boundaries and their basis."""

        entries: list[tuple[float, str]] = []
        invalid_rows: list[str] = []
        for row in range(self.layer_boundary_table.rowCount()):
            depth_item = self.layer_boundary_table.item(row, 0)
            note_item = self.layer_boundary_table.item(row, 1)
            depth_text = depth_item.text().strip() if depth_item else ""
            note = note_item.text().strip() if note_item else ""
            try:
                depth = float(depth_text)
                if (
                    not math.isfinite(depth)
                    or depth <= 0.0
                    or depth >= float(deepest_recorded_depth_m) - 1.0e-6
                ):
                    raise ValueError
                entries.append((depth, note))
            except ValueError:
                invalid_rows.append(str(row + 1))
        if invalid_rows:
            raise ValueError(
                "Layer-boundary rows "
                + ", ".join(invalid_rows)
                + " must contain a positive depth shallower than the deepest receiver."
            )
        entries.sort(key=lambda item: item[0])
        depths = np.asarray([entry[0] for entry in entries], dtype=float)
        if depths.size and np.any(np.diff(depths) <= 1.0e-6):
            raise ValueError("Interpreted layer-boundary depths must be unique.")
        return depths, [entry[1] for entry in entries]

    def _corrected_interpreted_layers(
        self,
        recorded_depths_m: np.ndarray,
        corrected_geometry: CorrectedGeometry,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Map recorded-depth boundaries to corrected vertical layer bottoms."""

        internal_recorded, notes = self._read_layer_boundaries_recorded(float(recorded_depths_m[-1]))
        all_recorded = np.r_[internal_recorded, recorded_depths_m[-1]]
        corrected_bottoms = np.interp(
            all_recorded,
            np.r_[0.0, corrected_geometry.recorded_depths_m],
            np.r_[0.0, corrected_geometry.vertical_depths_m],
        )
        if corrected_bottoms.size >= recorded_depths_m.size:
            raise ValueError(
                "The interpretation has as many layers as observations. Remove at least one internal boundary."
            )
        receiver_layer = np.searchsorted(all_recorded, recorded_depths_m, side="left")
        counts = np.bincount(receiver_layer, minlength=all_recorded.size)
        if np.any(counts < 2):
            sparse = ", ".join(str(index + 1) for index in np.flatnonzero(counts < 2))
            raise ValueError(
                f"Interpreted layer(s) {sparse} contain fewer than two receiver observations. Revise the "
                "boundaries so every fitted layer is supported by at least two observations."
            )
        return all_recorded, corrected_bottoms, [*notes, "Final layer to deepest receiver"]

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
<<<<<<< Updated upstream
=======
        self.gru_label.setToolTip("\n".join(import_messages))
        if import_messages:
            self.gru_label.setText(
                f"{self.gru_label.text()}; {len(import_messages)} import note(s)"
            )
        if imported_deviation_points:
            self.gru_label.setText(
                f"{self.gru_label.text()}; cone tilt imported for {len(imported_deviation_points)} intervals"
            )
        self.layer_boundary_table.blockSignals(True)
        self.layer_boundary_table.setRowCount(0)
        self.layer_boundary_table.blockSignals(False)
>>>>>>> Stashed changes
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
        self.layer_boundary_table.blockSignals(True)
        self.layer_boundary_table.setRowCount(0)
        self.layer_boundary_table.blockSignals(False)
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
        interpreted_boundaries = []
        for row in range(self.layer_boundary_table.rowCount()):
            depth_item = self.layer_boundary_table.item(row, 0)
            note_item = self.layer_boundary_table.item(row, 1)
            interpreted_boundaries.append(
                {
                    "bottom_recorded_depth_m": depth_item.text().strip() if depth_item else "",
                    "interpretive_basis": note_item.text().strip() if note_item else "",
                }
            )
        payload: dict[str, Any] = {
            "format": "RayPath SCPT Project",
            "version": 3,
            "units": "SI",
            "gru_pre_trigger_ms": GRU_PRE_TRIGGER_MS,
            "pick_time_reference": "relative_to_trigger",
            "source_offset_m": self.offset_spin.value(),
<<<<<<< Updated upstream
            "regularization": self.reg_slider.value() / 100.0,
=======
            "survey_geometry": self.survey_geometry.to_dict(),
            "geometry_audit": geometry_audit,
            "model_parameterization": "analyst_defined_piecewise_constant_layers",
            "smoothing_status": "disabled",
            "interpreted_layer_boundaries": interpreted_boundaries,
            "robust_loss": str(self.robust_loss_combo.currentData()),
            "fallback_pick_uncertainty_ms": self.manual_uncertainty_spin.value(),
            "uncertainty_ensemble_size": self.ensemble_size_spin.value(),
            "uncertainty_ensemble_classification": uncertainty_ensemble_classification(
                self.ensemble_size_spin.value()
            ),
            "uncertainty_random_seed": self.uncertainty_seed_spin.value(),
>>>>>>> Stashed changes
            "vs30_extrapolation_weight_factor": self._extrapolation_weight_factor(),
            "arrival_estimator": self.estimator_combo.currentData(),
<<<<<<< Updated upstream
            "vs30_history": [
                {
                    "pick_kind": kind,
                    "regularization": smoothing,
                    "weight_factor": weighting,
                    "vs30_mps": value,
                }
                for (kind, smoothing, weighting), value in sorted(self.vs30_history.items())
            ],
=======
>>>>>>> Stashed changes
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
                "interpreted_layer_bottoms_corrected_m": self.result.model_layer_bottoms_m.tolist(),
                "interpreted_layer_bottoms_recorded_m": self.active_layer_recorded_bottoms_m.tolist(),
                "interpreted_layer_velocities_mps": self.result.model_layer_velocities_mps.tolist(),
                "interpreted_layer_notes": list(self.active_layer_notes),
                "interpreted_layer_resolution": self.result.model_layer_resolution.tolist(),
                "interpreted_layer_bound_active": self.result.model_layer_bound_active.tolist(),
                "calculated_times_ms": (self.result.calculated_times_s * 1000.0).tolist(),
                "rmse_ms": self.result.rmse_s * 1000.0,
<<<<<<< Updated upstream
=======
                "weighted_rmse_ms": self.result.weighted_rmse_s * 1000.0,
                "data_cost_ms2": self.result.data_cost_ms2,
                "regularization_contribution": self.result.regularization_cost,
                "model_penalty_contribution": 0.0,
                "roughness_norm": self.result.roughness_norm,
                "objective_value": self.result.objective_value,
                "robust_loss": self.result.robust_loss,
                "observation_std_ms": (self.result.observation_std_s * 1000.0).tolist(),
                "standardized_residuals": self.result.standardized_residuals.tolist(),
                "resolution_diagonal": self.result.resolution_diagonal.tolist(),
                "observation_leverage": self.result.observation_leverage.tolist(),
                "influence_scores": self.result.influence_scores.tolist(),
                "outlier_flags": self.result.outlier_flags.tolist(),
                "influential_flags": self.result.influential_flags.tolist(),
                "bound_active_flags": self.result.bound_active_flags.tolist(),
                "diagnostic_warnings": list(self.result.warnings),
>>>>>>> Stashed changes
                "vs30_mps": self.current_vs30.value_mps if self.current_vs30 is not None else None,
                "vs30_extrapolated_velocity_mps": (
                    self.current_vs30.extrapolated_velocity_mps if self.current_vs30 is not None else None
                ),
            }
<<<<<<< Updated upstream
=======
            selected_kind = str(self.estimator_combo.currentData())
            ensemble = self.uncertainty_results.get(selected_kind)
            if ensemble is not None:
                payload["last_result"]["uncertainty_ensemble"] = {
                    "pick_kind": selected_kind,
                    "random_seed": ensemble.random_seed,
                    "requested_models": ensemble.requested_models,
                    "successful_models": ensemble.successful_models,
                    "classification": uncertainty_ensemble_classification(ensemble.requested_models),
                    "percentile_levels": list(ensemble.percentile_levels),
                    "velocity_lower_mps": ensemble.velocity_lower_mps.tolist(),
                    "velocity_median_mps": ensemble.velocity_median_mps.tolist(),
                    "velocity_upper_mps": ensemble.velocity_upper_mps.tolist(),
                    "vs30_lower_mps": ensemble.vs30_lower_mps,
                    "vs30_median_mps": ensemble.vs30_median_mps,
                    "vs30_upper_mps": ensemble.vs30_upper_mps,
                    "warnings": list(ensemble.warnings),
                }
            try:
                interval_comparison = self._interval_comparison_for_result(self.result)

                def interval_series_payload(series: IntervalVelocitySeries) -> dict[str, Any]:
                    def nullable(values: np.ndarray) -> list[float | None]:
                        return [float(value) if math.isfinite(float(value)) else None for value in values]

                    return {
                        "recorded_top_depths_m": series.recorded_top_depths_m.tolist(),
                        "recorded_bottom_depths_m": series.recorded_bottom_depths_m.tolist(),
                        "corrected_top_depths_m": series.top_depths_m.tolist(),
                        "corrected_bottom_depths_m": series.bottom_depths_m.tolist(),
                        "corrected_centre_depths_m": series.centre_depths_m.tolist(),
                        "path_length_differences_m": series.path_length_differences_m.tolist(),
                        "arrival_time_differences_ms": (
                            series.arrival_time_differences_s * 1000.0
                        ).tolist(),
                        "velocities_mps": nullable(series.velocities_mps),
                        "velocity_standard_deviations_mps": nullable(
                            series.standard_deviations_mps
                        ),
                        "valid_flags": series.valid_flags.tolist(),
                        "status_messages": list(series.status_messages),
                    }

                payload["last_result"]["staggered_interval_comparison"] = {
                    "method": (
                        "geometric source-to-receiver path-length difference divided by "
                        "picked arrival-time difference"
                    ),
                    "target_interval_m": interval_comparison.target_interval_m,
                    "tolerance_m": interval_comparison.tolerance_m,
                    "phase_definition": (
                        "A starts on whole-metre recorded depths; B starts approximately half a "
                        "target interval deeper"
                    ),
                    "independence_warning": (
                        "Staggered windows overlap and are correlated; the slowness mean is an "
                        "experimental sensitivity display, not an independent layer model."
                    ),
                    "adjacent": interval_series_payload(interval_comparison.adjacent),
                    "phase_a": interval_series_payload(interval_comparison.phase_a),
                    "phase_b": interval_series_payload(interval_comparison.phase_b),
                    "slowness_mean_corrected_depths_m": (
                        interval_comparison.slowness_mean_depths_m.tolist()
                    ),
                    "slowness_mean_velocities_mps": (
                        interval_comparison.slowness_mean_velocities_mps.tolist()
                    ),
                }
            except ValueError:
                pass
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        self.reg_slider.setValue(round(float(payload.get("regularization", 0.35)) * 100))
=======
        self.survey_geometry = SurveyGeometry.from_dict(
            payload.get("survey_geometry") if project_version >= 7 else None
        )
        self.active_geometry = None
        robust_loss = str(payload.get("robust_loss", "linear"))
        robust_index = self.robust_loss_combo.findData(robust_loss)
        self.robust_loss_combo.setCurrentIndex(max(0, robust_index))
        self.manual_uncertainty_spin.setValue(
            float(payload.get("fallback_pick_uncertainty_ms", DEFAULT_MANUAL_PICK_UNCERTAINTY_MS))
        )
        self.ensemble_size_spin.setValue(int(payload.get("uncertainty_ensemble_size", DEFAULT_ENSEMBLE_SIZE)))
        self.uncertainty_seed_spin.setValue(int(payload.get("uncertainty_random_seed", DEFAULT_UNCERTAINTY_SEED)))
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            self._set_input_rows(rows, str(payload.get("arrival_estimator", "first_cross")))
=======
            legacy_kind = str(payload.get("arrival_estimator", "first_cross"))
            self._set_input_rows(rows, "zero_cross" if legacy_kind == "first_cross" else legacy_kind)
        self.layer_boundary_table.blockSignals(True)
        self.layer_boundary_table.setRowCount(0)
        for boundary in payload.get("interpreted_layer_boundaries", []):
            row = self.layer_boundary_table.rowCount()
            self.layer_boundary_table.insertRow(row)
            self.layer_boundary_table.setItem(
                row,
                0,
                QTableWidgetItem(str(boundary.get("bottom_recorded_depth_m", ""))),
            )
            self.layer_boundary_table.setItem(
                row,
                1,
                QTableWidgetItem(str(boundary.get("interpretive_basis", ""))),
            )
        self.layer_boundary_table.blockSignals(False)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        self.vs30_history = {
            (
                str(item.get("pick_kind", payload.get("arrival_estimator", "first_cross"))),
                float(item["regularization"]),
                round(float(item.get("weight_factor", 1.0)), 6),
            ): float(item["vs30_mps"])
            for item in payload.get("vs30_history", [])
            if "regularization" in item and "vs30_mps" in item
        }
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
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
=======
                        "Observation",
                        "Previous Corrected Receiver Depth (m)",
                        "Corrected Receiver Depth (m)",
                        "Recorded Receiver Depth (m)",
                        "Corrected Source-to-Receiver Offset (m)",
                        "Receiver East Relative to Source (m)",
                        "Receiver North Relative to Source (m)",
                        *[f"{PICK_LABELS[kind]} Vs (m/s)" for kind in PICK_KINDS],
                        "Observed Post-Trigger Travel Time (ms)",
                        "Calculated Post-Trigger Travel Time (ms)",
                        "Fitting Error (ms)",
                        "Observation Standard Deviation (ms)",
                        "Standardized Residual (sigma)",
                        "Observation Leverage",
                        "Influence Score",
                        "Layer Resolution",
                        "Outlier Flag",
                        "Influential Flag",
                        "Velocity Bound Active",
                        "Weighted RMSE (ms)",
                        "Data Cost (ms^2)",
                        "Interpreted Layer Count",
                        "Robust Loss",
                        "Ray Parameter (s/m)",
                        "Model Parameterization",
                        "Smoothing Status",
                        "Ensemble Velocity 2.5% (m/s)",
                        "Ensemble Velocity Median (m/s)",
                        "Ensemble Velocity 97.5% (m/s)",
                        "Uncertainty Random Seed",
                        "Ensemble Requested Models",
                        "Ensemble Classification",
                        *[f"{PICK_LABELS[kind]} TS Method 1 Vs30 (m/s)" for kind in PICK_KINDS],
                        *[f"{PICK_LABELS[kind]} TS Method 1 Lower Bound (m/s)" for kind in PICK_KINDS],
                        *[f"{PICK_LABELS[kind]} TS Method 1 Upper Bound (m/s)" for kind in PICK_KINDS],
                        *[f"{PICK_LABELS[kind]} Raw RayPath Vs30 (m/s)" for kind in PICK_KINDS],
                        "TS Method 1 Indicative Vs30 Bands",
                        "TS Method 1 Shallow Reference Vs (m/s)",
                        "TS Method 1 Extended Thickness (m)",
                        "TS Method 1 Extended Last-Layer Vs (m/s)",
                        "Experimental Extrapolation Weight Factor",
                        "Experimental Weighted Vs30 (m/s)",
                        "Applied GRU Pre-Trigger Correction (ms)",
                        "Arrival-Time Reference",
                        "Application Version",
                        "Project Schema Version",
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                            f"{self.result.ray_parameters[i]:.10g}",
                            f"{self.reg_slider.value() / 100.0:.2f}",
=======
                            f"{self.result.observation_std_s[i] * 1000.0:.4f}",
                            f"{self.result.standardized_residuals[i]:.4f}",
                            f"{self.result.observation_leverage[i]:.6f}",
                            f"{self.result.influence_scores[i]:.6f}",
                            f"{self.result.resolution_diagonal[i]:.6f}",
                            bool(self.result.outlier_flags[i]),
                            bool(self.result.influential_flags[i]),
                            bool(self.result.bound_active_flags[i]),
                            f"{self.result.weighted_rmse_s * 1000.0:.4f}",
                            f"{self.result.data_cost_ms2:.8g}",
                            int(self.result.model_layer_bottoms_m.size),
                            self.result.robust_loss,
                            f"{self.result.ray_parameters[i]:.10g}",
                            "analyst-defined piecewise-constant layers",
                            "disabled",
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            self.status_label.setText(f"Exported {target.name}")
        except Exception as exc:
            self._show_error("Unable to export CSV", exc)

=======
            exported_names = [target.name]
            layer_target = target.with_name(f"{target.stem}_layer_model.csv")
            self._export_layer_model_csv(layer_target)
            exported_names.append(layer_target.name)
            try:
                interval_target = target.with_name(f"{target.stem}_interval_comparison.csv")
                self._export_interval_comparison_csv(interval_target)
                exported_names.append(interval_target.name)
            except ValueError:
                pass
            if self.observation_review:
                qc_target = target.with_name(f"{target.stem}_waveform_qc.csv")
                self._export_waveform_qc_csv(qc_target)
                exported_names.append(qc_target.name)
            self.status_label.setText(f"Exported {', '.join(exported_names)}")
        except Exception as exc:
            self._show_error("Unable to export CSV", exc)

    def _export_layer_model_csv(self, target: Path) -> None:
        """Write the analyst-defined layer geometry and fitted Vs values."""

        if self.result is None or self.result.model_layer_bottoms_m.size == 0:
            raise ValueError("Run a layered inversion before exporting the interpreted model.")
        corrected_bottoms = self.result.model_layer_bottoms_m
        corrected_tops = np.r_[0.0, corrected_bottoms[:-1]]
        recorded_bottoms = (
            self.active_layer_recorded_bottoms_m
            if self.active_layer_recorded_bottoms_m.size == corrected_bottoms.size
            else corrected_bottoms
        )
        recorded_tops = np.r_[0.0, recorded_bottoms[:-1]]
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Interpreted Layer",
                    "Top Recorded Depth (m)",
                    "Bottom Recorded Depth (m)",
                    "Top Corrected Vertical Depth (m)",
                    "Bottom Corrected Vertical Depth (m)",
                    "Interpretive Basis / Unit",
                    *[f"{PICK_LABELS[kind]} Layer Vs (m/s)" for kind in PICK_KINDS],
                    "Selected Layer Resolution",
                    "Selected Velocity Bound Active",
                    "Model Parameterization",
                    "Smoothing Status",
                    "Application Version",
                    "Project Schema Version",
                ]
            )
            for index in range(corrected_bottoms.size):
                writer.writerow(
                    [
                        index + 1,
                        f"{recorded_tops[index]:.4f}",
                        f"{recorded_bottoms[index]:.4f}",
                        f"{corrected_tops[index]:.4f}",
                        f"{corrected_bottoms[index]:.4f}",
                        self.active_layer_notes[index] if index < len(self.active_layer_notes) else "",
                        *[
                            ""
                            if kind not in self.comparison_results
                            or self.comparison_results[kind].model_layer_velocities_mps.size <= index
                            else f"{self.comparison_results[kind].model_layer_velocities_mps[index]:.3f}"
                            for kind in PICK_KINDS
                        ],
                        (
                            ""
                            if self.result.model_layer_resolution.size <= index
                            else f"{self.result.model_layer_resolution[index]:.6f}"
                        ),
                        (
                            ""
                            if self.result.model_layer_bound_active.size <= index
                            else bool(self.result.model_layer_bound_active[index])
                        ),
                        "analyst-defined piecewise-constant layers",
                        "disabled",
                        APP_VERSION,
                        PROJECT_SCHEMA_VERSION,
                    ]
                )

    def _export_interval_comparison_csv(self, target: Path) -> None:
        """Write an auditable companion schedule for observed interval velocities."""

        if self.result is None:
            raise ValueError("Run the inversion before exporting interval comparisons.")
        comparison = self._interval_comparison_for_result(self.result)
        selected_pick = PICK_LABELS.get(str(self.estimator_combo.currentData()), "Selected")
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Interpretation",
                    "Recorded Top Receiver Depth (m)",
                    "Recorded Bottom Receiver Depth (m)",
                    "Corrected Vertical Top Depth (m)",
                    "Corrected Vertical Bottom Depth (m)",
                    "Corrected Vertical Centre Depth (m)",
                    "Geometric Path-Length Difference (m)",
                    "Picked Arrival-Time Difference (ms)",
                    "Interval Vs (m/s)",
                    "Approximate Vs Standard Deviation (m/s)",
                    "Approximate 95% Lower Vs (m/s)",
                    "Approximate 95% Upper Vs (m/s)",
                    "Arrival Pick Definition",
                    "Method / Status",
                    "Application Version",
                    "Project Schema Version",
                ]
            )

            def write_series(name: str, series: IntervalVelocitySeries) -> None:
                for index, velocity in enumerate(series.velocities_mps):
                    standard_deviation = float(series.standard_deviations_mps[index])
                    is_valid = bool(series.valid_flags[index]) and math.isfinite(float(velocity))
                    has_uncertainty = is_valid and math.isfinite(standard_deviation)
                    writer.writerow(
                        [
                            name,
                            f"{series.recorded_top_depths_m[index]:.4f}",
                            f"{series.recorded_bottom_depths_m[index]:.4f}",
                            f"{series.top_depths_m[index]:.4f}",
                            f"{series.bottom_depths_m[index]:.4f}",
                            f"{series.centre_depths_m[index]:.4f}",
                            f"{series.path_length_differences_m[index]:.6f}",
                            f"{series.arrival_time_differences_s[index] * 1000.0:.6f}",
                            "" if not is_valid else f"{velocity:.3f}",
                            "" if not has_uncertainty else f"{standard_deviation:.3f}",
                            (
                                ""
                                if not has_uncertainty
                                else f"{max(0.0, velocity - 1.96 * standard_deviation):.3f}"
                            ),
                            (
                                ""
                                if not has_uncertainty
                                else f"{velocity + 1.96 * standard_deviation:.3f}"
                            ),
                            selected_pick,
                            (
                                f"{series.status_messages[index]}; observed geometric path-length difference / "
                                "picked time difference; uncertainty assumes independent pick errors"
                            ),
                            APP_VERSION,
                            PROJECT_SCHEMA_VERSION,
                        ]
                    )

            write_series("Adjacent receiver intervals", comparison.adjacent)
            write_series("1 m phase A - whole-metre grid", comparison.phase_a)
            write_series("1 m phase B - approximately 0.5 m stagger", comparison.phase_b)
            for depth, velocity in zip(
                comparison.slowness_mean_depths_m,
                comparison.slowness_mean_velocities_mps,
            ):
                writer.writerow(
                    [
                        "Experimental staggered slowness mean",
                        "",
                        "",
                        "",
                        "",
                        f"{depth:.4f}",
                        "",
                        "",
                        f"{velocity:.3f}",
                        "",
                        "",
                        "",
                        selected_pick,
                        (
                            "Interpolated mean slowness of overlapping phase A/B windows; correlated "
                            "sensitivity display only, not an independent layer model"
                        ),
                        APP_VERSION,
                        PROJECT_SCHEMA_VERSION,
                    ]
                )

    def _export_waveform_qc_csv(self, target: Path) -> None:
        """Write a companion receiver-level QC and exclusions schedule."""

        geometry_by_depth: dict[float, tuple[float, float]] = {}
        try:
            geometry_depths = self._geometry_recorded_depths()
            corrected = calculate_corrected_geometry(
                geometry_depths,
                self.offset_spin.value(),
                self.survey_geometry,
            )
            geometry_by_depth = {
                round(float(depth), 6): (float(vertical), float(offset))
                for depth, vertical, offset in zip(
                    corrected.recorded_depths_m,
                    corrected.vertical_depths_m,
                    corrected.receiver_offsets_m,
                )
            }
        except (ValueError, TypeError):
            pass
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Test",
                    "Recorded Receiver Depth (m)",
                    "Corrected Vertical Depth (m)",
                    "Corrected Source-to-Receiver Offset (m)",
                    "Applied Pre-Trigger Correction (ms)",
                    "Pick Time Reference",
                    "Review State",
                    "Included in Inversion",
                    "Pick Uncertainty (ms)",
                    "Pick Uncertainty Source",
                    "Pick Uncertainty Basis",
                    "Left Pre-Trigger Noise RMS (recorded units)",
                    "Right Pre-Trigger Noise RMS (recorded units)",
                    "Left SNR (dB)",
                    "Right SNR (dB)",
                    "Sign-Reversed Correlation",
                    "Correlation Lag (ms)",
                    "First Peak/Trough Disagreement (ms)",
                    "Individual Zero-Cross Disagreement (ms)",
                    "Maximum-Peak Disagreement (ms)",
                    "Reversed Polarity",
                    "Sample Interval (ms)",
                    "Sample Interval Maximum Deviation (%)",
                    "Possible Clipping",
                    "Constant Trace",
                    "QC Warnings",
                    "Analyst Comment",
                    "Application Version",
                    "Project Schema Version",
                ]
            )

            def number(value: float | None, decimals: int = 3) -> str:
                if value is None:
                    return ""
                if math.isinf(value):
                    return "inf" if value > 0.0 else "-inf"
                return f"{value:.{decimals}f}"

            for record in self.waveform_records:
                qc = calculate_waveform_qc(record)
                corrected_depth, corrected_offset = geometry_by_depth.get(
                    round(record.depth_m, 6),
                    (record.depth_m, self.offset_spin.value()),
                )
                writer.writerow(
                    [
                        record.test_number,
                        f"{record.depth_m:.4f}",
                        f"{corrected_depth:.4f}",
                        f"{corrected_offset:.4f}",
                        f"{record.pre_trigger_ms:.3f}",
                        "relative to physical trigger",
                        REVIEW_LABELS.get(record.review_state, record.review_state),
                        "No" if record.is_excluded else "Yes",
                        number(record.pick_uncertainty_ms),
                        record.pick_uncertainty_source,
                        record.pick_uncertainty_basis,
                        number(qc.noise_rms_left, 6),
                        number(qc.noise_rms_right, 6),
                        number(qc.snr_left_db, 2),
                        number(qc.snr_right_db, 2),
                        number(qc.sign_reversed_correlation, 4),
                        number(qc.correlation_lag_ms),
                        number(qc.first_peak_disagreement_ms),
                        number(qc.zero_cross_disagreement_ms),
                        number(qc.max_peak_disagreement_ms),
                        "Yes" if qc.polarity_reversed else "No",
                        number(qc.sample_interval_ms),
                        number(qc.sample_interval_deviation_pct, 4),
                        "Yes" if qc.clipped_left or qc.clipped_right else "No",
                        "Yes" if qc.constant_left or qc.constant_right else "No",
                        "; ".join(qc.warnings),
                        record.review_comment,
                        APP_VERSION,
                        PROJECT_SCHEMA_VERSION,
                    ]
                )
            if not self.waveform_records:
                for depth, metadata in sorted(self.observation_review.items()):
                    state = str(metadata.get("review_state", "not_reviewed"))
                    corrected_depth, corrected_offset = geometry_by_depth.get(
                        round(depth, 6),
                        (depth, self.offset_spin.value()),
                    )
                    writer.writerow(
                        [
                            "",
                            f"{depth:.4f}",
                            f"{corrected_depth:.4f}",
                            f"{corrected_offset:.4f}",
                            "" if self.gru_pre_trigger_ms is None else f"{self.gru_pre_trigger_ms:.3f}",
                            "relative to physical trigger",
                            REVIEW_LABELS.get(state, state),
                            "No" if state == "rejected" else "Yes",
                            number(metadata.get("pick_uncertainty_ms")),
                            str(metadata.get("pick_uncertainty_source", "fallback")),
                            str(metadata.get("pick_uncertainty_basis", "Fallback project setting.")),
                            *([""] * 15),
                            str(metadata.get("review_comment", "")),
                            APP_VERSION,
                            PROJECT_SCHEMA_VERSION,
                        ]
                    )

>>>>>>> Stashed changes
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
        try:
            interval_comparison = self._interval_comparison_for_result(selected_result)
        except ValueError:
            interval_comparison = None
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
                comparison_bottoms, comparison_velocities = result_model_profile(comparison)
                ax.stairs(
                    comparison_velocities,
                    np.r_[0.0, comparison_bottoms],
                    orientation="horizontal",
                    color=MODEL_COLORS[kind],
                    linewidth=2.2 if kind == selected_kind else 1.6,
                    label=PICK_LABELS[kind],
                )
<<<<<<< Updated upstream
            ax.set_title("Optimized shear-wave velocity comparison", fontsize=11, fontweight="bold")
=======
                ensemble = self.uncertainty_results.get(kind)
                if kind == selected_kind and ensemble is not None:
                    envelope_edges = np.r_[0.0, comparison.depths_m]
                    ax.fill_betweenx(
                        envelope_edges,
                        np.r_[ensemble.velocity_lower_mps, ensemble.velocity_lower_mps[-1]],
                        np.r_[ensemble.velocity_upper_mps, ensemble.velocity_upper_mps[-1]],
                        step="post",
                        color=MODEL_COLORS[kind],
                        alpha=0.18,
                        label=(
                            "Preliminary 2.5-97.5% pick-time ensemble"
                            if ensemble.requested_models < REPORT_QUALITY_ENSEMBLE_MINIMUM
                            else "Report-quality 2.5-97.5% pick-time ensemble"
                        ),
                    )
            selected_bottoms, _selected_velocities = result_model_profile(selected_result)
            for boundary in selected_bottoms[:-1]:
                ax.axhline(boundary, color="#819790", linewidth=0.8, linestyle=":", alpha=0.8)
            ax.set_title("Interpreted layered shear-wave velocity comparison", fontsize=11, fontweight="bold")
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
            selected_bottoms, _selected_velocities = result_model_profile(selected_result)
            for boundary in selected_bottoms[:-1]:
                ax.axhline(boundary, color="#819790", linewidth=0.8, linestyle=":", alpha=0.8)
            ax.tick_params(axis="y", labelleft=True, colors="#111111")
            ax.yaxis.label.set_color("#111111")
>>>>>>> Stashed changes
            ax.legend(fontsize=7.5, frameon=True)

        def draw_interval_comparison(ax: Any) -> None:
            if interval_comparison is None:
                ax.text(
                    0.5,
                    0.5,
                    "At least three valid receiver observations are required.",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                adjacent = interval_comparison.adjacent
                if adjacent.velocities_mps.size:
                    ax.plot(
                        adjacent.velocities_mps,
                        adjacent.centre_depths_m,
                        "--o",
                        color="#8B949E",
                        linewidth=1.0,
                        markersize=3,
                        alpha=0.65,
                        label="Adjacent receiver intervals",
                    )
                phase_specs = (
                    (
                        interval_comparison.phase_a,
                        "#2F81F7",
                        "1 m phase A - whole-metre grid",
                    ),
                    (
                        interval_comparison.phase_b,
                        "#F0883E",
                        "1 m phase B - approximately 0.5 m stagger",
                    ),
                )
                for series, color, label in phase_specs:
                    if not series.velocities_mps.size:
                        continue
                    finite = np.isfinite(series.standard_deviations_mps)
                    if np.any(finite):
                        ax.errorbar(
                            series.velocities_mps[finite],
                            series.centre_depths_m[finite],
                            xerr=1.96 * series.standard_deviations_mps[finite],
                            fmt="none",
                            ecolor=color,
                            elinewidth=0.8,
                            capsize=2,
                            alpha=0.35,
                        )
                    ax.plot(
                        series.velocities_mps,
                        series.centre_depths_m,
                        "-o",
                        color=color,
                        linewidth=1.7,
                        markersize=4,
                        label=label,
                    )
                if interval_comparison.slowness_mean_velocities_mps.size:
                    ax.plot(
                        interval_comparison.slowness_mean_velocities_mps,
                        interval_comparison.slowness_mean_depths_m,
                        "-s",
                        color="#147D75",
                        linewidth=2.6,
                        markersize=4,
                        label="Experimental staggered slowness mean",
                    )
                selected_bottoms, selected_velocities = result_model_profile(selected_result)
                ax.stairs(
                    selected_velocities,
                    np.r_[0.0, selected_bottoms],
                    orientation="horizontal",
                    color="#163F3B",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Selected RayPath model",
                )
                for boundary in selected_bottoms[:-1]:
                    ax.axhline(boundary, color="#819790", linewidth=0.8, linestyle=":", alpha=0.8)
                ax.text(
                    0.01,
                    0.01,
                    "The 1 m windows overlap and are correlated.\n"
                    "The slowness mean is an experimental sensitivity display,\n"
                    "not an independent layer model.",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color="#5F746F",
                    bbox={"facecolor": "white", "edgecolor": "#C9D5D2", "alpha": 0.92, "pad": 4},
                )
                ax.legend(fontsize=7.5, frameon=True)
            ax.set_title(
                f"Observed path-length interval sensitivity - {PICK_LABELS.get(selected_kind, selected_kind)} picks",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Vs (m/s)")
            ax.set_ylabel("Corrected vertical depth (m)")
            ax.set_xlim(0.0, VELOCITY_PROFILE_DISPLAY_MAX)
            ax.set_ylim(float(selected_result.depths_m[-1]), 0.0)

        def draw_rays(ax: Any) -> None:
            model_bottoms, _model_velocities = result_model_profile(selected_result)
            edges = np.r_[0.0, model_bottoms]
            offset = self.offset_spin.value()
            for boundary in edges:
                ax.hlines(boundary, 0.0, offset, color="#B8C4CC", linewidth=0.55)
            cmap = colormaps["viridis"]
            for index, segments in enumerate(selected_result.ray_x_segments):
                ray_depths = (
                    selected_result.ray_z_nodes_m[index]
                    if len(selected_result.ray_z_nodes_m) == len(selected_result.ray_x_segments)
                    else np.r_[0.0, selected_result.depths_m[: index + 1]]
                )
                ax.plot(
                    np.r_[0.0, np.cumsum(segments)],
                    ray_depths,
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
<<<<<<< Updated upstream
            [p("Smoothing / regularisation"), p(f"{self.reg_slider.value() / 100.0:.2f}")],
            [p("Vs30 extrapolation weighting"), p(f"{self._extrapolation_weight_factor():.2f}")],
=======
            [
                p("Inversion objective"),
                p(
                    f"Analyst-defined piecewise-constant model with "
                    f"{selected_result.model_layer_bottoms_m.size} fitted layers and no smoothing or "
                    f"regularisation; {selected_result.robust_loss} loss; "
                    f"RMSE {selected_result.rmse_s * 1000.0:.3f} ms; weighted RMSE "
                    f"{selected_result.weighted_rmse_s * 1000.0:.3f} ms; data cost "
                    f"{selected_result.data_cost_ms2:.4g} ms^2"
                ),
            ],
            [
                p("Observed interval comparator"),
                p(
                    "Adjacent and 1.0 m +/- 0.15 m staggered receiver pairs; geometric source-to-receiver "
                    "path-length difference divided by picked arrival-time difference. Phase A is anchored "
                    "to whole-metre recorded depths and phase B is approximately 0.5 m staggered. "
                    "Overlapping windows are correlated; the slowness mean is experimental."
                ),
            ],
            [
                p("Arrival uncertainty weighting"),
                p(
                    f"One sigma per observation; selected-model range "
                    f"{np.min(selected_result.observation_std_s) * 1000.0:.3f}-"
                    f"{np.max(selected_result.observation_std_s) * 1000.0:.3f} ms; fallback "
                    f"{self.manual_uncertainty_spin.value():.3f} ms"
                ),
            ],
            [
                p("Uncertainty ensemble"),
                p(
                    "Off"
                    if selected_kind not in self.uncertainty_results
                    else (
                        f"{uncertainty_ensemble_classification(self.uncertainty_results[selected_kind].requested_models)}; "
                        f"{self.uncertainty_results[selected_kind].successful_models}/"
                        f"{self.uncertainty_results[selected_kind].requested_models} successful; seed "
                        f"{self.uncertainty_results[selected_kind].random_seed}; velocity and Vs30 percentiles "
                        "2.5/50/97.5; Vs30 median/95% interval "
                        f"{self.uncertainty_results[selected_kind].vs30_median_mps:.1f} / "
                        f"{self.uncertainty_results[selected_kind].vs30_lower_mps:.1f}-"
                        f"{self.uncertainty_results[selected_kind].vs30_upper_mps:.1f} m/s"
                    )
                ),
            ],
            [
                p("Primary Vs30 method"),
                p(
                    "TS 1170.5:2025 Method 1 - direct measured Vs"
                    if self.current_vs30 is None
                    else (
                        f"TS 1170.5:2025 Method 1; raw/adjusted "
                        f"{self.current_vs30.raw_value_mps:.1f}/{self.current_vs30.value_mps:.1f} m/s; "
                        f"5% bounds {self.current_vs30.lower_bound_mps:.1f}-"
                        f"{self.current_vs30.upper_bound_mps:.1f} m/s; 0-3 m set to "
                        f"{self.current_vs30.shallow_reference_velocity_mps:.1f} m/s; bands "
                        f"{'/'.join(self.current_vs30.indicative_vs30_bands)}"
                    )
                ),
            ],
            [
                p("Experimental extrapolation weighting"),
                p(f"{self._extrapolation_weight_factor():.2f} - excluded from the primary Method 1 result"),
            ],
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        comparison_rows: list[list[Any]] = [
            ["Pick definition", "Layers", "RMSE (ms)", "Vs30 (m/s)", "Extrapolation"]
        ]
=======
        comparison_rows: list[list[Any]] = [[
            p("Pick definition", "TableHeader"),
            p("Layers", "TableHeader"),
            p("RMSE / weighted (ms)", "TableHeader"),
            p("Parameters / data cost", "TableHeader"),
            p("TS M1 Vs30 (m/s)", "TableHeader"),
            p("5% range (m/s)", "TableHeader"),
            p("Vs30 bands", "TableHeader"),
            p("30 m extension", "TableHeader"),
        ]]
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                    p(comparison.depths_m.size),
                    p(f"{comparison.rmse_s * 1000.0:.3f}"),
=======
                    p(comparison.model_layer_bottoms_m.size or comparison.depths_m.size),
                    p(f"{comparison.rmse_s * 1000.0:.3f} / {comparison.weighted_rmse_s * 1000.0:.3f}"),
                    p(f"{comparison.model_layer_bottoms_m.size} / {comparison.data_cost_ms2:.3g}"),
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                "Vs30 is calculated as 30 divided by the summed vertical shear-wave travel time. Profiles "
                "between 25 m and 30 m use the configured interval weighting only to estimate the missing depth. "
                "Automatic waveform suggestions should be reviewed by a qualified operator before relying on this report.",
=======
                "Velocity parameterisation: layer bottoms are analyst-defined using the recorded-depth "
                "interpretation table and are converted to corrected vertical depths using the survey geometry. "
                "One constant Vs is fitted per layer to all applicable arrival observations. No smoothing, "
                "roughness penalty, or automatic L-curve selection is applied. Boundary selection remains an "
                "engineering interpretation that should be supported by CPT stratigraphy, corrected-time slope "
                "changes, or another recorded ground-model basis.",
                styles["ReportNote"],
            )
        )
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                "Arrival definitions: First peak/trough is the mean of the reviewed left and right extrema; "
                "Pair crossover is one reviewed time where the reversed traces intersect after arrival; "
                "Individual zero crossing is the mean of two per-trace zero-axis crossings and is experimental; "
                "Maximum peak is the mean of two maximum-amplitude times and is experimental.",
                styles["ReportNote"],
            )
        )
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                "The primary Vs30 follows TS 1170.5:2025 Method 1 for direct SCPT measurements: 0-3 m is "
                "assigned the depth-average measured from 2.5-3.5 m, profiles reaching at least 25 m extend "
                "the last measured layer to 30 m, and the reported bounds are Vs30/1.05 and 1.05 x Vs30. "
                "The listed bands use Vs30 thresholds only and are not a final site classification; all additional "
                "TS soil and rock criteria require engineering review. Experimental interval weighting is excluded "
                "from these primary results.",
>>>>>>> Stashed changes
                styles["ReportNote"],
            )
        )
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                "The observed 1 m comparator remains a pseudo-interval interpretation when receiver records "
                "come from separate source impacts. Wider time differences reduce sensitivity to a fixed pick "
                "error but reduce vertical resolution and may average across a real layer boundary. The two "
                "staggered series are compared as sampling-phase sensitivity; their slowness mean is not used "
                "as the forward inversion or primary Vs30 model.",
                styles["ReportNote"],
            )
        )

        full_page_plots = (
            ("Velocity profile comparison", draw_velocity),
            ("Staggered 1 m interval comparison", draw_interval_comparison),
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
<<<<<<< Updated upstream
                "Layer",
                "Top (m)",
                "Bottom (m)",
                "First peak Vs",
                "First cross Vs",
                "Max peak Vs",
                "Selected residual (ms)",
=======
                p("Layer", "TableHeader"),
                p("Recorded top-bottom (m)", "TableHeader"),
                p("Corrected top-bottom (m)", "TableHeader"),
                p("Interpretive basis / unit", "TableHeader"),
                *[p(f"{PICK_LABELS[kind]} Vs", "TableHeader") for kind in PICK_KINDS],
                p("Selected diagnostics", "TableHeader"),
>>>>>>> Stashed changes
            ]
        ]
        corrected_bottoms, _selected_velocities = result_model_profile(selected_result)
        corrected_tops = np.r_[0.0, corrected_bottoms[:-1]]
        recorded_bottoms = (
            self.active_layer_recorded_bottoms_m
            if self.active_layer_recorded_bottoms_m.size == corrected_bottoms.size
            else corrected_bottoms
        )
        recorded_tops = np.r_[0.0, recorded_bottoms[:-1]]
        observation_layers = np.searchsorted(corrected_bottoms, selected_result.depths_m, side="left")
        for index in range(corrected_bottoms.size):
            velocity_cells = []
            for kind in PICK_KINDS:
                comparison = self.comparison_results.get(kind)
<<<<<<< Updated upstream
                velocity_cells.append("-" if comparison is None else f"{comparison.velocities_mps[index]:.1f}")
=======
                if comparison is None:
                    velocity_cells.append("-")
                else:
                    _bottoms, velocities = result_model_profile(comparison)
                    velocity_cells.append("-" if index >= velocities.size else f"{velocities[index]:.1f}")
            observations = observation_layers == index
            residuals_ms = selected_result.residuals_s[observations] * 1000.0
            layer_rmse_ms = float(np.sqrt(np.mean(residuals_ms * residuals_ms)))
            diagnostic_parts = []
            if selected_result.model_layer_resolution.size == corrected_bottoms.size:
                diagnostic_parts.append(f"R={selected_result.model_layer_resolution[index]:.2f}")
            if selected_result.model_layer_bound_active.size == corrected_bottoms.size and selected_result.model_layer_bound_active[index]:
                diagnostic_parts.append("BOUND")
            if np.any(selected_result.outlier_flags[observations]):
                diagnostic_parts.append("OUTLIER")
            if np.any(selected_result.influential_flags[observations]):
                diagnostic_parts.append("INFLUENTIAL")
>>>>>>> Stashed changes
            layer_rows.append(
                [
                    p(index + 1),
                    p(f"{recorded_tops[index]:.2f}-{recorded_bottoms[index]:.2f}"),
                    p(f"{corrected_tops[index]:.2f}-{corrected_bottoms[index]:.2f}"),
                    p(self.active_layer_notes[index] if index < len(self.active_layer_notes) else "Not recorded"),
                    *[p(value) for value in velocity_cells],
<<<<<<< Updated upstream
                    p(f"{selected_result.residuals_s[index] * 1000.0:+.3f}"),
=======
                    p(
                        f"{layer_rmse_ms:.3f} ms RMS / {int(np.count_nonzero(observations))} obs; "
                        + (", ".join(diagnostic_parts) or "diagnostics unavailable")
                    ),
>>>>>>> Stashed changes
                ]
            )
        layer_table = LongTable(
            layer_rows,
<<<<<<< Updated upstream
            colWidths=[14 * mm, 22 * mm, 24 * mm, 29 * mm, 30 * mm, 28 * mm, 33 * mm],
=======
            colWidths=[8 * mm, 21 * mm, 23 * mm, 28 * mm, *([16 * mm] * len(PICK_KINDS)), 36 * mm],
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        except Exception as exc:
            self._show_error("Invalid input data", exc)
            return
=======
            corrected_geometry = calculate_corrected_geometry(
                depths,
                self.offset_spin.value(),
                self.survey_geometry,
            )
            recorded_layer_bottoms, corrected_layer_bottoms, layer_notes = (
                self._corrected_interpreted_layers(depths, corrected_geometry)
            )
            observation_std_s = self._observation_uncertainties_s(depths)
        except Exception as exc:
            self._show_error("Invalid input data", exc)
            return
        if corrected_geometry.warnings:
            warning_lines = "\n• " + "\n• ".join(corrected_geometry.warnings)
            answer = QMessageBox.question(
                self,
                "Geometry applicability warnings",
                "The ray model can run, but the following survey geometry items are incomplete or "
                f"inconsistent:{warning_lines}\n\nContinue using the calculated/nominal geometry?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.active_geometry = corrected_geometry
        self.active_layer_recorded_bottoms_m = recorded_layer_bottoms
        self.active_layer_notes = layer_notes
>>>>>>> Stashed changes
        self.run_button.setEnabled(False)
        self.run_button.setText("Inversion running…")
        self.input_table.setEnabled(False)
        self.offset_spin.setEnabled(False)
        self.estimator_combo.setEnabled(False)
        self.layer_boundary_table.setEnabled(False)
        self.add_selected_boundary_button.setEnabled(False)
        self.add_boundary_button.setEnabled(False)
        self.remove_boundary_button.setEnabled(False)
        self.clear_boundary_button.setEnabled(False)
        self.extrapolation_weight_slider.setEnabled(False)
<<<<<<< Updated upstream
        self.status_label.setText("Solving refracted ray paths and velocity model…")
=======
        self.geometry_button.setEnabled(False)
        self.robust_loss_combo.setEnabled(False)
        self.manual_uncertainty_spin.setEnabled(False)
        self.ensemble_preset_combo.setEnabled(False)
        self.ensemble_size_spin.setEnabled(False)
        self.uncertainty_seed_spin.setEnabled(False)
        self.status_label.setText(
            f"Solving {corrected_layer_bottoms.size}-layer refracted ray-path model…"
        )
>>>>>>> Stashed changes
        self.rmse_label.setText("RMSE: calculating…")
        self._thread = QThread(self)
        self._worker = InversionWorker(
            depths,
            times_by_pick_s,
            self.offset_spin.value(),
<<<<<<< Updated upstream
            self.reg_slider.value() / 100.0,
=======
            corrected_geometry.receiver_offsets_m,
            corrected_layer_bottoms,
            observation_std_s,
            str(self.robust_loss_combo.currentData()),
            str(self.estimator_combo.currentData()),
            self.ensemble_size_spin.value(),
            self.uncertainty_seed_spin.value(),
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    def _inversion_finished(self, results: dict[str, InversionResult]) -> None:
        self.comparison_results = dict(results)
=======
    def _inversion_finished(self, batch: InversionBatchResult | dict[str, InversionResult]) -> None:
        if isinstance(batch, InversionBatchResult):
            self.comparison_results = dict(batch.results)
            self.uncertainty_results = dict(batch.uncertainty_results)
        else:
            self.comparison_results = dict(batch)
            self.uncertainty_results.clear()
>>>>>>> Stashed changes
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
                f"{result.message}\n\nReview the arrival picks and interpreted layer boundaries before using these results.",
            )
        self._set_dirty(True)

    def _update_all_vs30_results(self, record_history: bool) -> None:
        """Recalculate Vs30 for every available pick-based velocity model."""

        self.comparison_vs30.clear()
        self.comparison_vs30_reasons.clear()
        weight_factor = self._extrapolation_weight_factor()
        for kind, result in self.comparison_results.items():
            profile_depths, profile_velocities = result_model_profile(result)
            try:
<<<<<<< Updated upstream
                value = calculate_vs30(result.depths_m, result.velocities_mps, weight_factor)
=======
                value = calculate_ts1170_5_method1_vs30(profile_depths, profile_velocities)
>>>>>>> Stashed changes
                self.comparison_vs30[kind] = value
            except ValueError as exc:
                self.comparison_vs30[kind] = None
                self.comparison_vs30_reasons[kind] = str(exc)
<<<<<<< Updated upstream
=======
            try:
                self.comparison_experimental_vs30[kind] = calculate_vs30(
                    profile_depths, profile_velocities, weight_factor
                )
            except ValueError:
                self.comparison_experimental_vs30[kind] = None
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
            vs30_note = f" Vs30 unavailable: {self.vs30_unavailable_reason or 'insufficient profile data'}."
        self.result_summary.setText(
            f"{state}. {result.depths_m.size} layers, source offset {self.offset_spin.value():.3f} m, "
            f"{PICK_LABELS.get(str(self.estimator_combo.currentData()), 'Selected')} model, "
            f"regularisation {self.reg_slider.value() / 100.0:.2f}. Final RMSE: {result.rmse_s * 1000.0:.3f} ms."
            f"{vs30_note}"
=======
            vs30_note = f" TS Method 1 Vs30 unavailable: {self.vs30_unavailable_reason or 'insufficient profile data'}."
        if self.observation_review:
            _accepted, rejected, unreviewed = self._review_counts()
            review_note = f" Waveform review: {rejected} rejected/excluded; {unreviewed} not reviewed."
        else:
            review_note = ""
        geometry_warning_count = len(self.active_geometry.warnings) if self.active_geometry is not None else 0
        geometry_note = (
            f" Receiver offset range {np.min(result.receiver_offsets_m):.3f}–"
            f"{np.max(result.receiver_offsets_m):.3f} m; {geometry_warning_count} geometry warning(s)."
        )
        layer_count = int(result.model_layer_bottoms_m.size) or int(result.depths_m.size)
        self.result_summary.setText(
            f"{state}. {layer_count} interpreted constant-Vs layers fitted to {result.depths_m.size} observations, "
            f"collar source offset {self.offset_spin.value():.3f} m, "
            f"{PICK_LABELS.get(str(self.estimator_combo.currentData()), 'Selected')} model, "
            f"no smoothing/regularisation, {result.robust_loss} loss. "
            f"RMSE: {result.rmse_s * 1000.0:.3f} ms; weighted RMSE: {result.weighted_rmse_s * 1000.0:.3f} ms; "
            f"data cost: {result.data_cost_ms2:.4g} ms²."
            f"{geometry_note}{vs30_note}{review_note}"
            + (" Diagnostics: " + " ".join(result.warnings) if result.warnings else "")
>>>>>>> Stashed changes
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
        self.run_button.setText("Run Layered RayPath Inversion")
        self.input_table.setEnabled(True)
        self.offset_spin.setEnabled(True)
        self.estimator_combo.setEnabled(True)
<<<<<<< Updated upstream
        self.reg_slider.setEnabled(True)
        self.extrapolation_weight_slider.setEnabled(True)
=======
        self.layer_boundary_table.setEnabled(True)
        self.add_selected_boundary_button.setEnabled(True)
        self.add_boundary_button.setEnabled(True)
        self.remove_boundary_button.setEnabled(True)
        self.clear_boundary_button.setEnabled(True)
        self.extrapolation_weight_slider.setEnabled(True)
        self.geometry_button.setEnabled(True)
        self.robust_loss_combo.setEnabled(True)
        self.manual_uncertainty_spin.setEnabled(True)
        self.ensemble_preset_combo.setEnabled(True)
        self.ensemble_size_spin.setEnabled(True)
        self.uncertainty_seed_spin.setEnabled(True)
>>>>>>> Stashed changes

    def _populate_results(self, result: InversionResult) -> None:
        model_bottoms, _selected_velocities = result_model_profile(result)
        self.result_table.setRowCount(model_bottoms.size)
        tops = np.r_[0.0, model_bottoms[:-1]]
        observation_layers = np.searchsorted(model_bottoms, result.depths_m, side="left")
        for row in range(model_bottoms.size):
            velocity_values = []
            for kind in PICK_KINDS:
                comparison = self.comparison_results.get(kind)
                if comparison is not None:
                    _comparison_bottoms, comparison_velocities = result_model_profile(comparison)
                else:
                    comparison_velocities = np.empty(0, dtype=float)
                if row < comparison_velocities.size:
                    velocity_values.append(f"{comparison_velocities[row]:.1f} m/s")
                else:
                    velocity_values.append("—")
<<<<<<< Updated upstream
=======
            observations = observation_layers == row
            layer_residuals_ms = result.residuals_s[observations] * 1000.0
            layer_rmse_ms = (
                float(np.sqrt(np.mean(layer_residuals_ms * layer_residuals_ms)))
                if layer_residuals_ms.size
                else math.nan
            )
            diagnostic_parts = []
            if result.model_layer_resolution.size == model_bottoms.size:
                diagnostic_parts.append(f"R={result.model_layer_resolution[row]:.2f}")
            if result.model_layer_bound_active.size == model_bottoms.size and result.model_layer_bound_active[row]:
                diagnostic_parts.append("BOUND")
            if result.outlier_flags.size == result.depths_m.size and np.any(result.outlier_flags[observations]):
                diagnostic_parts.append("CONTAINS OUTLIER >3σ")
            if result.influential_flags.size == result.depths_m.size and np.any(result.influential_flags[observations]):
                diagnostic_parts.append("INFLUENTIAL")
>>>>>>> Stashed changes
            values = (
                str(row + 1),
                f"{tops[row]:.2f} m",
                f"{model_bottoms[row]:.2f} m",
                *velocity_values,
<<<<<<< Updated upstream
                f"{result.residuals_s[row] * 1000.0:+.3f} ms",
=======
                f"{layer_rmse_ms:.3f} ms RMS ({int(np.count_nonzero(observations))} obs)",
                "; ".join(diagnostic_parts) or "—",
>>>>>>> Stashed changes
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 3, 4, 5, 6):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
<<<<<<< Updated upstream
                if column == 6:
                    magnitude = abs(result.residuals_s[row] * 1000.0)
                    item.setForeground(QColor("#3fb950" if magnitude < 1.0 else "#d29922" if magnitude < 3.0 else "#f85149"))
=======
                if column == residual_column:
                    magnitude = (
                        float(np.max(np.abs(result.standardized_residuals[observations])))
                        if result.standardized_residuals.size == result.depths_m.size and np.any(observations)
                        else 0.0
                    )
                    item.setForeground(QColor("#3fb950" if magnitude < 2.0 else "#d29922" if magnitude <= 3.0 else "#f85149"))
                if column == residual_column + 1 and diagnostic_parts[1 if diagnostic_parts and diagnostic_parts[0].startswith("R=") else 0:]:
                    item.setForeground(QColor("#f85149"))
>>>>>>> Stashed changes
                self.result_table.setItem(row, column, item)

    def _interval_comparison_for_result(self, result: InversionResult) -> StaggeredIntervalComparison:
        """Return the audited staggered-interval comparison for one inversion."""

        recorded_depths = (
            self.active_geometry.recorded_depths_m
            if self.active_geometry is not None
            and self.active_geometry.recorded_depths_m.size == result.depths_m.size
            else result.depths_m
        )
        return staggered_interval_velocity_comparison(
            recorded_depths,
            result.depths_m,
            result.receiver_offsets_m,
            result.observed_times_s,
            result.observation_std_s,
        )

    def _draw_interval_comparison(self, result: InversionResult) -> None:
        """Plot adjacent and two staggered 1 m observed interval series."""

        canvas = self.interval_canvas
        canvas.clear()
        ax = canvas.axes
        try:
            comparison = self._interval_comparison_for_result(result)
        except ValueError as exc:
            ax.set_title("Staggered 1 m observed interval comparison")
            ax.set_xlabel("Vs (m/s)")
            ax.set_ylabel("Corrected vertical depth (m)")
            ax.text(0.5, 0.5, str(exc), transform=ax.transAxes, ha="center", va="center", color=canvas.muted_color)
            canvas.draw_idle()
            return

        adjacent = comparison.adjacent
        if adjacent.velocities_mps.size:
            ax.plot(
                adjacent.velocities_mps,
                adjacent.centre_depths_m,
                "--o",
                color="#8b949e",
                linewidth=1.0,
                markersize=3.0,
                alpha=0.65,
                label="Adjacent receiver intervals",
            )
        phase_specs = (
            (comparison.phase_a, "#2f81f7", "1 m phase A - whole-metre grid"),
            (comparison.phase_b, "#f0883e", "1 m phase B - approximately 0.5 m stagger"),
        )
        for series, color, label in phase_specs:
            if not series.velocities_mps.size:
                continue
            finite_uncertainty = np.isfinite(series.standard_deviations_mps)
            if np.any(finite_uncertainty):
                ax.errorbar(
                    series.velocities_mps[finite_uncertainty],
                    series.centre_depths_m[finite_uncertainty],
                    xerr=1.96 * series.standard_deviations_mps[finite_uncertainty],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.35,
                )
            ax.plot(
                series.velocities_mps,
                series.centre_depths_m,
                "-o",
                color=color,
                linewidth=1.6,
                markersize=4.0,
                label=label,
            )
        if comparison.slowness_mean_velocities_mps.size:
            ax.plot(
                comparison.slowness_mean_velocities_mps,
                comparison.slowness_mean_depths_m,
                "-s",
                color="#147d75",
                linewidth=2.6,
                markersize=4.0,
                label="Experimental staggered slowness mean",
            )
        model_bottoms, model_velocities = result_model_profile(result)
        ax.stairs(
            model_velocities,
            np.r_[0.0, model_bottoms],
            orientation="horizontal",
            color=canvas.foreground_color,
            linewidth=1.5,
            alpha=0.8,
            label="Selected RayPath model",
        )
        for boundary in model_bottoms[:-1]:
            ax.axhline(boundary, color=canvas.muted_color, linewidth=0.8, linestyle=":", alpha=0.75)
        ax.set_title(
            f"Path-length interval sensitivity - {PICK_LABELS.get(str(self.estimator_combo.currentData()), 'selected')} picks"
        )
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Corrected vertical depth (m)")
        ax.set_xlim(0.0, VELOCITY_PROFILE_DISPLAY_MAX)
        ax.set_ylim(float(result.depths_m[-1]), 0.0)
        ax.text(
            0.01,
            0.01,
            "1 m windows overlap by approximately 0.5 m and are correlated.\n"
            "The slowness mean is an experimental sensitivity display, not an independent layer model.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=canvas.muted_color,
            bbox={"facecolor": canvas.figure_color, "edgecolor": canvas.grid_color, "alpha": 0.9, "pad": 3},
        )
        ax.legend(**canvas.legend_kwargs())
        canvas.draw_idle()

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
            comparison_bottoms, comparison_velocities = result_model_profile(comparison)
            comparison_edges = np.r_[0.0, comparison_bottoms]
            all_velocities.append(comparison_velocities)
            ax.stairs(
                comparison_velocities,
                comparison_edges,
                orientation="horizontal",
                color=MODEL_COLORS[kind],
                linewidth=2.8 if kind == selected_kind else 1.8,
                alpha=1.0 if kind == selected_kind else 0.82,
                label=PICK_LABELS[kind],
            )
<<<<<<< Updated upstream
        ax.set_title("Pick-based shear-wave velocity comparison")
=======
            uncertainty = self.uncertainty_results.get(kind)
            if kind == selected_kind and uncertainty is not None:
                envelope_edges = np.r_[0.0, comparison.depths_m]
                lower = np.r_[uncertainty.velocity_lower_mps, uncertainty.velocity_lower_mps[-1]]
                upper = np.r_[uncertainty.velocity_upper_mps, uncertainty.velocity_upper_mps[-1]]
                ax.fill_betweenx(
                    envelope_edges,
                    lower,
                    upper,
                    step="post",
                    color=MODEL_COLORS[kind],
                    alpha=0.18,
                    label=(
                        ("Preliminary " if uncertainty.requested_models < REPORT_QUALITY_ENSEMBLE_MINIMUM else "Report-quality ")
                        + f"{uncertainty.percentile_levels[0]:g}–{uncertainty.percentile_levels[2]:g}% "
                        "pick-time ensemble"
                    ),
                )
        ax.set_title("Interpreted layered shear-wave velocity comparison")
>>>>>>> Stashed changes
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1], 0.0)
<<<<<<< Updated upstream
        finite = np.concatenate([raw[np.isfinite(raw)], *all_velocities]) if all_velocities else raw[np.isfinite(raw)]
        if finite.size:
            ax.set_xlim(max(0.0, float(np.min(finite)) * 0.8), float(np.max(finite)) * 1.15)
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
=======
        ax.set_xlim(0.0, VELOCITY_PROFILE_DISPLAY_MAX)
        ax.set_xticks(np.arange(0.0, VELOCITY_PROFILE_DISPLAY_MAX + 1.0, 100.0))
        selected_bottoms, _selected_velocities = result_model_profile(result)
        for boundary in selected_bottoms[:-1]:
            ax.axhline(boundary, color=self.velocity_canvas.muted_color, linewidth=0.8, linestyle=":", alpha=0.75)
        raw_above_limit = int(np.count_nonzero(np.isfinite(raw) & (raw > VELOCITY_PROFILE_DISPLAY_MAX)))
        model_above_limit = sum(
            int(np.count_nonzero(values > VELOCITY_PROFILE_DISPLAY_MAX))
            for values in all_velocities
        )
        if raw_above_limit or model_above_limit:
            clipped_parts = []
            if raw_above_limit:
                clipped_parts.append(f"{raw_above_limit} raw")
            if model_above_limit:
                clipped_parts.append(f"{model_above_limit} modeled")
            ax.text(
                0.99,
                0.01,
                f"Values above {VELOCITY_PROFILE_DISPLAY_MAX:.0f} m/s outside view: "
                + ", ".join(clipped_parts),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="#d29922",
                fontsize=8,
                bbox={
                    "facecolor": self.velocity_canvas.figure_color,
                    "edgecolor": "#6e5b25",
                    "alpha": 0.9,
                    "pad": 3,
                },
            )
        ax.legend(**self.velocity_canvas.legend_kwargs())
>>>>>>> Stashed changes
        self.velocity_canvas.draw_idle()

        ax = self.ray_canvas.axes
        self.ray_canvas.clear()
        x_offset = self.offset_spin.value()
<<<<<<< Updated upstream
        for boundary in edges:
            ax.hlines(boundary, 0.0, x_offset, color="#48515c", linewidth=0.7, alpha=0.7)
=======
        plot_min = min(0.0, float(np.min(receiver_offsets)), x_offset)
        plot_max = max(0.0, float(np.max(receiver_offsets)), x_offset)
        model_bottoms, _model_velocities = result_model_profile(result)
        for boundary in np.r_[0.0, model_bottoms]:
            ax.hlines(boundary, plot_min, plot_max, color=self.ray_canvas.grid_color, linewidth=0.7, alpha=0.7)
>>>>>>> Stashed changes
        color_map = colormaps["viridis"]
        for i, segments in enumerate(result.ray_x_segments):
            ray_x = np.r_[0.0, np.cumsum(segments)]
            ray_z = (
                result.ray_z_nodes_m[i]
                if len(result.ray_z_nodes_m) == len(result.ray_x_segments)
                else edges[: i + 2]
            )
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
        selected_bottoms, _selected_velocities = result_model_profile(result)
        for boundary in selected_bottoms[:-1]:
            ax.axhline(boundary, color=self.fit_canvas.muted_color, linewidth=0.8, linestyle=":", alpha=0.75)
        ax.set_title(
            f"Arrival-time comparison — selected {PICK_LABELS.get(selected_kind, selected_kind)} "
            f"RMSE {result.rmse_s * 1000.0:.3f} ms"
        )
        ax.set_xlabel("Arrival time after trigger (ms)")
        ax.set_ylabel("Depth (m)")
        ax.set_ylim(z[-1], 0.0)
        ax.legend(facecolor="#161b22", edgecolor="#48515c", labelcolor="#c9d1d9")
        self.fit_canvas.draw_idle()
        self._draw_interval_comparison(result)
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
<<<<<<< Updated upstream
            "<h3>RayPath SCPT</h3>"
            "<p>SI-only SCPT arrival-time picking and regularised shear-wave velocity inversion.</p>"
=======
            f"<h3>RayPath SCPT {APP_VERSION}</h3>"
            f"<p>Project schema version {PROJECT_SCHEMA_VERSION}.</p>"
            "<p>SI-only SCPT arrival-time picking and analyst-defined layered shear-wave velocity inversion.</p>"
>>>>>>> Stashed changes
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

    depths = np.asarray([2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 25.0])
    layer_bottoms = np.asarray([6.0, 16.0, 25.0])
    expected_vs = np.asarray([180.0, 270.0, 390.0])
    times, rays, ray_z_nodes = forward_layered_model(depths, layer_bottoms, expected_vs, 2.4)
    assert times.shape == depths.shape
    assert all(abs(np.sum(ray.horizontal_segments_m) - 2.4) < 1.0e-8 for ray in rays)
<<<<<<< Updated upstream
    result = invert_velocity_profile(depths, times, 2.4, regularization=0.0)
    assert result.rmse_s < 1.0e-6, result.rmse_s
=======
    assert all(nodes[-1] == depth for nodes, depth in zip(ray_z_nodes, depths))
    result = invert_layered_velocity_profile(
        depths,
        times,
        2.4,
        layer_bottoms,
        observation_std_s=np.full(depths.size, 0.0005),
    )
    assert result.rmse_s < 1.0e-6, result.rmse_s
    assert np.allclose(result.model_layer_velocities_mps, expected_vs, rtol=2.0e-4)
    assert result.regularization_cost == 0.0
    first_ensemble = generate_layered_velocity_uncertainty_ensemble(
        result, 2.4, ensemble_size=4, random_seed=11705
    )
    second_ensemble = generate_layered_velocity_uncertainty_ensemble(
        result, 2.4, ensemble_size=4, random_seed=11705
    )
    assert np.allclose(first_ensemble.velocity_lower_mps, second_ensemble.velocity_lower_mps)
    assert np.allclose(first_ensemble.velocity_median_mps, second_ensemble.velocity_median_mps)
    assert np.allclose(first_ensemble.velocity_upper_mps, second_ensemble.velocity_upper_mps)
    assert np.allclose(first_ensemble.vs30_samples_mps, second_ensemble.vs30_samples_mps)
>>>>>>> Stashed changes
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
