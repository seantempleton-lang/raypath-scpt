"""Data models shared by the headless engineering modules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import REPORT_QUALITY_ENSEMBLE_MINIMUM

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
    receiver_offsets_m: np.ndarray
    rmse_s: float
    success: bool
    message: str
    iterations: int
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


@dataclass(frozen=True)
class Vs30Result:
    """Experimental travel-time average through the upper 30 metres."""

    value_mps: float
    vertical_travel_time_s: float
    included_thicknesses_m: np.ndarray
    included_velocities_mps: np.ndarray
    measured_depth_m: float
    extrapolated_thickness_m: float
    extrapolated_velocity_mps: float | None
    extrapolation_weight_factor: float


@dataclass(frozen=True)
class Ts1170Method1Vs30Result:
    """Auditable TS 1170.5:2025 Method 1 result for a direct SCPT Vs profile.

    The standards-adjusted profile replaces the upper 3 m with the
    depth-average Vs measured from 2.5 m to 3.5 m.  If direct measurements
    terminate from 25 m to 30 m, the velocity of the last measured layer is
    extended to 30 m.  Site-class bands are Vs30-only screening bands; final
    classification still requires the additional geological and geotechnical
    criteria in TS 1170.5.
    """

    method: str
    value_mps: float
    lower_bound_mps: float
    upper_bound_mps: float
    uncertainty_factor: float
    raw_value_mps: float
    vertical_travel_time_s: float
    raw_vertical_travel_time_s: float
    included_thicknesses_m: np.ndarray
    included_velocities_mps: np.ndarray
    raw_included_velocities_mps: np.ndarray
    measured_depth_m: float
    adjusted_shallow_thickness_m: float
    shallow_reference_velocity_mps: float
    extrapolated_thickness_m: float
    extrapolated_velocity_mps: float | None
    slow_velocity_thickness_top20_m: float
    indicative_vs30_bands: tuple[str, ...]
    spans_multiple_vs30_bands: bool
    notes: tuple[str, ...]


DEPTH_BASIS_LABELS = {
    "along_rods": "Measured along rods",
    "vertical": "Already corrected vertical depth",
}


@dataclass(frozen=True)
class DeviationPoint:
    """Cone deviation applying to the interval ending at one recorded depth."""

    recorded_depth_m: float
    inclination_deg: float = 0.0
    azimuth_deg: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "recorded_depth_m": self.recorded_depth_m,
            "inclination_deg": self.inclination_deg,
            "azimuth_deg": self.azimuth_deg,
        }


@dataclass
class SurveyGeometry:
    """Auditable source, receiver, datum, orientation, and deviation metadata."""

    source_offset_uncertainty_m: float | None = None
    coordinate_system: str = ""
    vertical_datum: str = ""
    elevations_enabled: bool = False
    source_elevation_m: float = 0.0
    receiver_reference_elevation_m: float = 0.0
    receiver_depth_reference: str = "Borehole/direct-push ground reference"
    depth_basis: str = "along_rods"
    source_to_receiver_bearing_deg: float | None = None
    source_block_axis_bearing_deg: float | None = None
    channel_17_strike_bearing_deg: float | None = None
    channel_18_strike_bearing_deg: float | None = None
    receiver_orientation_bearing_deg: float | None = None
    deviation_points: list[DeviationPoint] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_offset_uncertainty_m": self.source_offset_uncertainty_m,
            "coordinate_system": self.coordinate_system,
            "vertical_datum": self.vertical_datum,
            "elevations_enabled": self.elevations_enabled,
            "source_elevation_m": self.source_elevation_m,
            "receiver_reference_elevation_m": self.receiver_reference_elevation_m,
            "receiver_depth_reference": self.receiver_depth_reference,
            "depth_basis": self.depth_basis,
            "source_to_receiver_bearing_deg": self.source_to_receiver_bearing_deg,
            "source_block_axis_bearing_deg": self.source_block_axis_bearing_deg,
            "channel_17_strike_bearing_deg": self.channel_17_strike_bearing_deg,
            "channel_18_strike_bearing_deg": self.channel_18_strike_bearing_deg,
            "receiver_orientation_bearing_deg": self.receiver_orientation_bearing_deg,
            "deviation_points": [point.to_dict() for point in self.deviation_points],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SurveyGeometry":
        if not payload:
            return cls()

        def optional_float(key: str) -> float | None:
            value = payload.get(key)
            return None if value in (None, "") else float(value)

        points = [
            DeviationPoint(
                recorded_depth_m=float(item["recorded_depth_m"]),
                inclination_deg=float(item.get("inclination_deg", 0.0)),
                azimuth_deg=(
                    None if item.get("azimuth_deg") in (None, "") else float(item["azimuth_deg"])
                ),
            )
            for item in payload.get("deviation_points", [])
            if "recorded_depth_m" in item
        ]
        depth_basis = str(payload.get("depth_basis", "along_rods"))
        if depth_basis not in DEPTH_BASIS_LABELS:
            depth_basis = "along_rods"
        return cls(
            source_offset_uncertainty_m=optional_float("source_offset_uncertainty_m"),
            coordinate_system=str(payload.get("coordinate_system", "")).strip(),
            vertical_datum=str(payload.get("vertical_datum", "")).strip(),
            elevations_enabled=bool(payload.get("elevations_enabled", False)),
            source_elevation_m=float(payload.get("source_elevation_m", 0.0)),
            receiver_reference_elevation_m=float(payload.get("receiver_reference_elevation_m", 0.0)),
            receiver_depth_reference=str(
                payload.get("receiver_depth_reference", "Borehole/direct-push ground reference")
            ).strip(),
            depth_basis=depth_basis,
            source_to_receiver_bearing_deg=optional_float("source_to_receiver_bearing_deg"),
            source_block_axis_bearing_deg=optional_float("source_block_axis_bearing_deg"),
            channel_17_strike_bearing_deg=optional_float("channel_17_strike_bearing_deg"),
            channel_18_strike_bearing_deg=optional_float("channel_18_strike_bearing_deg"),
            receiver_orientation_bearing_deg=optional_float("receiver_orientation_bearing_deg"),
            deviation_points=points,
            notes=str(payload.get("notes", "")).strip(),
        )


@dataclass(frozen=True)
class CorrectedGeometry:
    """Calculated receiver coordinates used by the one-dimensional ray model."""

    recorded_depths_m: np.ndarray
    source_offset_m: float
    vertical_depths_m: np.ndarray
    receiver_offsets_m: np.ndarray
    receiver_east_m: np.ndarray
    receiver_north_m: np.ndarray
    warnings: tuple[str, ...]

    @property
    def correction_applied(self) -> bool:
        return bool(
            np.any(np.abs(self.vertical_depths_m - self.recorded_depths_m) > 1.0e-9)
            or np.any(np.abs(self.receiver_offsets_m - self.source_offset_m) > 1.0e-9)
        )


