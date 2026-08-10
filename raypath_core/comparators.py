"""Independent comparator interpretations for SCPT velocity profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .constants import VELOCITY_MAX, VELOCITY_MIN
from .models import RayPathError, RaySolution
from .rays import solve_direct_ray


@dataclass(frozen=True)
class SlopeLayerResult:
    """One fitted straight segment on a corrected-time versus depth plot."""

    top_depth_m: float
    bottom_depth_m: float
    velocity_mps: float
    slope_s_per_m: float
    intercept_s: float
    observation_indices: np.ndarray
    rmse_s: float


@dataclass(frozen=True)
class SlopeMethodResult:
    """Piecewise-linear slope-method interpretation and its residuals."""

    depths_m: np.ndarray
    measured_times_s: np.ndarray
    corrected_times_s: np.ndarray
    calculated_times_s: np.ndarray
    residuals_s: np.ndarray
    layer_boundaries_m: np.ndarray
    layers: tuple[SlopeLayerResult, ...]


@dataclass(frozen=True)
class GeologicalInversionResult:
    """Reduced-parameter RayPath inversion using interpreted layer boundaries."""

    receiver_depths_m: np.ndarray
    layer_tops_m: np.ndarray
    layer_bottoms_m: np.ndarray
    velocities_mps: np.ndarray
    observed_times_s: np.ndarray
    calculated_times_s: np.ndarray
    residuals_s: np.ndarray
    receiver_offsets_m: np.ndarray
    ray_parameters: np.ndarray
    ray_x_segments: tuple[np.ndarray, ...]
    rmse_s: float
    success: bool
    message: str
    iterations: int
    bound_active_flags: np.ndarray
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CrossCorrelationResult:
    """Lag and normalised correlation for a pair of uniformly sampled traces."""

    lag_samples: int
    lag_ms: float
    correlation: float


def _validated_observations(
    depths_m: Sequence[float],
    times_s: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    depths = np.asarray(depths_m, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if depths.ndim != 1 or times.ndim != 1 or depths.size < 2 or depths.size != times.size:
        raise ValueError("Comparator methods require at least two paired depth and arrival-time observations.")
    if not np.all(np.isfinite(depths)) or not np.all(depths > 0.0) or not np.all(np.diff(depths) > 0.0):
        raise ValueError("Receiver depths must be finite, positive, and strictly increasing.")
    if not np.all(np.isfinite(times)) or not np.all(times > 0.0):
        raise ValueError("Arrival times must be finite and positive.")
    return depths, times


def _receiver_offsets(offsets_m: float | Sequence[float], count: int) -> np.ndarray:
    if np.isscalar(offsets_m):
        offsets = np.full(count, float(offsets_m), dtype=float)
    else:
        offsets = np.asarray(offsets_m, dtype=float)
    if offsets.shape != (count,) or not np.all(np.isfinite(offsets)) or np.any(offsets < 0.0):
        raise ValueError("Receiver offsets must be one finite non-negative value per observation.")
    return offsets


def corrected_vertical_travel_times(
    depths_m: Sequence[float],
    measured_times_s: Sequence[float],
    receiver_offsets_m: float | Sequence[float],
) -> np.ndarray:
    """Apply the conventional straight-ray cosine correction to travel times.

    This comparator uses ``t_vertical = t_measured * z / hypot(z, offset)``.
    It is intentionally independent of the refracted RayPath forward model and
    should be reported as an approximate interpretation when layering causes a
    materially curved ray path.
    """

    depths, times = _validated_observations(depths_m, measured_times_s)
    offsets = _receiver_offsets(receiver_offsets_m, depths.size)
    return times * depths / np.hypot(depths, offsets)


def _validated_boundaries(boundaries_m: Sequence[float], deepest_m: float) -> np.ndarray:
    boundaries = np.asarray(boundaries_m, dtype=float)
    if boundaries.ndim != 1:
        raise ValueError("Layer boundaries must be a one-dimensional sequence.")
    if boundaries.size and (
        not np.all(np.isfinite(boundaries))
        or np.any(boundaries <= 0.0)
        or np.any(boundaries >= deepest_m)
        or np.any(np.diff(boundaries) <= 0.0)
    ):
        raise ValueError("Layer boundaries must be finite, strictly increasing, and inside the observed depth range.")
    return boundaries


def fit_slope_method(
    depths_m: Sequence[float],
    measured_times_s: Sequence[float],
    receiver_offsets_m: float | Sequence[float],
    layer_boundaries_m: Sequence[float] = (),
) -> SlopeMethodResult:
    """Fit independent linear segments to corrected vertical travel time."""

    depths, measured = _validated_observations(depths_m, measured_times_s)
    corrected = corrected_vertical_travel_times(depths, measured, receiver_offsets_m)
    boundaries = _validated_boundaries(layer_boundaries_m, float(depths[-1]))
    edges = np.concatenate(([0.0], boundaries, [float(depths[-1])]))
    calculated = np.full(depths.size, np.nan, dtype=float)
    layers: list[SlopeLayerResult] = []
    for layer_index, (top, bottom) in enumerate(zip(edges[:-1], edges[1:])):
        if layer_index == 0:
            indices = np.flatnonzero((depths >= top) & (depths <= bottom))
        else:
            indices = np.flatnonzero((depths > top) & (depths <= bottom))
        if indices.size < 2:
            raise ValueError(
                f"Slope layer {top:g}–{bottom:g} m contains {indices.size} observation(s); at least two are required."
            )
        design = np.column_stack((depths[indices], np.ones(indices.size)))
        slope, intercept = np.linalg.lstsq(design, corrected[indices], rcond=None)[0]
        if not math.isfinite(float(slope)) or slope <= 0.0:
            raise ValueError(f"Slope layer {top:g}–{bottom:g} m does not have a positive travel-time gradient.")
        fitted = design @ np.asarray([slope, intercept])
        calculated[indices] = fitted
        residuals = corrected[indices] - fitted
        layers.append(
            SlopeLayerResult(
                top_depth_m=float(top),
                bottom_depth_m=float(bottom),
                velocity_mps=1.0 / float(slope),
                slope_s_per_m=float(slope),
                intercept_s=float(intercept),
                observation_indices=indices,
                rmse_s=float(np.sqrt(np.mean(residuals * residuals))),
            )
        )
    if np.any(~np.isfinite(calculated)):
        raise RuntimeError("Slope segmentation did not assign every observation.")
    return SlopeMethodResult(
        depths_m=depths.copy(),
        measured_times_s=measured.copy(),
        corrected_times_s=corrected,
        calculated_times_s=calculated,
        residuals_s=corrected - calculated,
        layer_boundaries_m=boundaries.copy(),
        layers=tuple(layers),
    )


def _geological_layers_at_depth(
    receiver_depth_m: float,
    layer_tops_m: np.ndarray,
    layer_bottoms_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    thicknesses = np.maximum(np.minimum(layer_bottoms_m, receiver_depth_m) - layer_tops_m, 0.0)
    active = np.flatnonzero(thicknesses > 1.0e-12)
    if active.size == 0:
        raise ValueError("A receiver does not intersect any geological layer.")
    return thicknesses[active], active


def forward_geological_model(
    receiver_depths_m: Sequence[float],
    layer_velocities_mps: Sequence[float],
    layer_boundaries_m: Sequence[float],
    receiver_offsets_m: float | Sequence[float],
) -> tuple[np.ndarray, tuple[RaySolution, ...]]:
    """Calculate direct-ray times for a piecewise-constant geological model."""

    depths = np.asarray(receiver_depths_m, dtype=float)
    if depths.ndim != 1 or depths.size == 0 or not np.all(np.isfinite(depths)):
        raise ValueError("At least one finite receiver depth is required.")
    if np.any(depths <= 0.0) or np.any(np.diff(depths) <= 0.0):
        raise ValueError("Receiver depths must be positive and strictly increasing.")
    boundaries = _validated_boundaries(layer_boundaries_m, float(depths[-1]))
    velocities = np.asarray(layer_velocities_mps, dtype=float)
    layer_bottoms = np.concatenate((boundaries, [float(depths[-1])]))
    layer_tops = np.concatenate(([0.0], boundaries))
    if velocities.shape != layer_bottoms.shape or not np.all(np.isfinite(velocities)) or np.any(velocities <= 0.0):
        raise ValueError("Provide one finite positive velocity for every geological layer.")
    offsets = _receiver_offsets(receiver_offsets_m, depths.size)
    rays: list[RaySolution] = []
    for depth, offset in zip(depths, offsets):
        thicknesses, active = _geological_layers_at_depth(float(depth), layer_tops, layer_bottoms)
        rays.append(solve_direct_ray(thicknesses, velocities[active], float(offset)))
    return np.asarray([ray.travel_time_s for ray in rays]), tuple(rays)


def invert_geological_layer_profile(
    receiver_depths_m: Sequence[float],
    observed_times_s: Sequence[float],
    receiver_offsets_m: float | Sequence[float],
    layer_boundaries_m: Sequence[float],
    velocity_bounds_mps: tuple[float, float] = (VELOCITY_MIN, VELOCITY_MAX),
) -> GeologicalInversionResult:
    """Invert fewer geological-layer velocities than arrival observations."""

    depths, observed = _validated_observations(receiver_depths_m, observed_times_s)
    boundaries = _validated_boundaries(layer_boundaries_m, float(depths[-1]))
    layer_bottoms = np.concatenate((boundaries, [float(depths[-1])]))
    layer_tops = np.concatenate(([0.0], boundaries))
    layer_count = layer_bottoms.size
    if layer_count >= depths.size:
        raise ValueError(
            "A geological-layer inversion must have fewer velocity parameters than arrival observations."
        )
    lower, upper = (float(value) for value in velocity_bounds_mps)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("Velocity bounds must be finite, positive, and increasing.")
    offsets = _receiver_offsets(receiver_offsets_m, depths.size)
    apparent = depths / observed
    initial_value = float(np.clip(np.median(apparent), lower, upper))
    initial = np.full(layer_count, initial_value, dtype=float)

    def objective(log_velocities: np.ndarray) -> float:
        velocities = np.exp(log_velocities)
        try:
            calculated, _ = forward_geological_model(depths, velocities, boundaries, offsets)
        except (ValueError, RayPathError, FloatingPointError):
            return 1.0e30
        residual_ms = (calculated - observed) * 1000.0
        return float(np.mean(residual_ms * residual_ms))

    log_lower = math.log(lower)
    log_upper = math.log(upper)
    result = minimize(
        objective,
        np.log(initial),
        method="L-BFGS-B",
        bounds=[(log_lower, log_upper)] * layer_count,
        options={"maxiter": 1000, "ftol": 1.0e-14, "gtol": 1.0e-9},
    )
    velocities = np.exp(result.x)
    calculated, rays = forward_geological_model(depths, velocities, boundaries, offsets)
    residuals = calculated - observed
    if not np.all(np.isfinite(calculated)) or not np.all(np.isfinite(velocities)):
        raise RayPathError(f"Geological-layer inversion did not produce a finite model: {result.message}")
    tolerance = 1.0e-3
    bound_active = (result.x <= log_lower + tolerance) | (result.x >= log_upper - tolerance)
    warnings: list[str] = []
    if np.any(bound_active):
        warnings.append(f"{int(np.count_nonzero(bound_active))} geological velocity parameter(s) are at bounds.")
    if not result.success:
        warnings.append(f"Optimiser status: {result.message}")
    return GeologicalInversionResult(
        receiver_depths_m=depths.copy(),
        layer_tops_m=layer_tops,
        layer_bottoms_m=layer_bottoms,
        velocities_mps=velocities,
        observed_times_s=observed.copy(),
        calculated_times_s=calculated,
        residuals_s=residuals,
        receiver_offsets_m=offsets.copy(),
        ray_parameters=np.asarray([ray.ray_parameter for ray in rays]),
        ray_x_segments=tuple(ray.horizontal_segments_m.copy() for ray in rays),
        rmse_s=float(np.sqrt(np.mean(residuals * residuals))),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        bound_active_flags=bound_active,
        warnings=tuple(warnings),
    )


def cross_correlation_interval_time(
    shallow_trace: Sequence[float],
    deeper_trace: Sequence[float],
    sample_interval_ms: float,
    maximum_lag_ms: float | None = None,
) -> CrossCorrelationResult:
    """Estimate the deeper-trace delay from normalised full cross-correlation."""

    shallow = np.asarray(shallow_trace, dtype=float)
    deeper = np.asarray(deeper_trace, dtype=float)
    if shallow.ndim != 1 or deeper.ndim != 1 or shallow.size < 3 or shallow.shape != deeper.shape:
        raise ValueError("Cross-correlation requires equal one-dimensional traces with at least three samples.")
    if not np.all(np.isfinite(shallow)) or not np.all(np.isfinite(deeper)):
        raise ValueError("Cross-correlation traces must contain only finite values.")
    interval = float(sample_interval_ms)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("Sample interval must be finite and positive.")
    shallow = shallow - float(np.mean(shallow))
    deeper = deeper - float(np.mean(deeper))
    shallow_norm = float(np.linalg.norm(shallow))
    deeper_norm = float(np.linalg.norm(deeper))
    if shallow_norm <= np.finfo(float).eps or deeper_norm <= np.finfo(float).eps:
        raise ValueError("Cross-correlation is undefined for a constant trace.")
    correlation = np.correlate(deeper, shallow, mode="full") / (shallow_norm * deeper_norm)
    lags = np.arange(-shallow.size + 1, shallow.size, dtype=int)
    allowed = lags >= 0
    if maximum_lag_ms is not None:
        maximum = float(maximum_lag_ms)
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError("Maximum lag must be finite and non-negative.")
        allowed &= lags * interval <= maximum + 1.0e-12
    if not np.any(allowed):
        raise ValueError("No cross-correlation lag lies inside the requested search range.")
    allowed_indices = np.flatnonzero(allowed)
    peak_index = int(allowed_indices[np.argmax(correlation[allowed])])
    lag_samples = int(lags[peak_index])
    return CrossCorrelationResult(
        lag_samples=lag_samples,
        lag_ms=lag_samples * interval,
        correlation=float(correlation[peak_index]),
    )
