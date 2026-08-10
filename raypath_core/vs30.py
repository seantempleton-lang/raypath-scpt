"""Vs30 calculations, including TS 1170.5:2025 Method 1."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .models import Ts1170Method1Vs30Result, Vs30Result

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

    depths, velocities = _validated_vs_profile(depths_m, velocities_mps, "Vs30")
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


