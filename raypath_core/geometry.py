"""Survey-geometry validation and receiver-coordinate correction."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .models import CorrectedGeometry, DEPTH_BASIS_LABELS, SurveyGeometry

def _bearing_vector(bearing_deg: float) -> tuple[float, float]:
    """Return east/north unit components for a clockwise-from-north bearing."""

    radians = math.radians(float(bearing_deg) % 360.0)
    return math.sin(radians), math.cos(radians)


def _bearing_separation(first_deg: float, second_deg: float) -> float:
    """Return the smallest absolute bearing separation in degrees."""

    return abs((float(first_deg) - float(second_deg) + 180.0) % 360.0 - 180.0)


def calculate_corrected_geometry(
    recorded_depths_m: Sequence[float],
    source_offset_m: float,
    geometry: SurveyGeometry,
) -> CorrectedGeometry:
    """Calculate receiver coordinates from depth, elevation, inclination, and azimuth data."""

    depths = np.asarray(recorded_depths_m, dtype=float)
    if depths.ndim != 1 or depths.size == 0:
        raise ValueError("Geometry correction requires at least one receiver depth.")
    if not np.all(np.isfinite(depths)) or not np.all(depths > 0.0) or not np.all(np.diff(depths) > 0.0):
        raise ValueError("Recorded receiver depths must be finite, positive, and strictly increasing.")
    offset = float(source_offset_m)
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError("Source offset must be finite and non-negative.")
    if geometry.depth_basis not in DEPTH_BASIS_LABELS:
        raise ValueError("Receiver depth basis is not recognized.")

    warnings: list[str] = []
    if not 1.0 <= offset <= 3.0:
        warnings.append("source offset is outside the typically recommended 1–3 m range")
    if geometry.source_offset_uncertainty_m is None or geometry.source_offset_uncertainty_m <= 0.0:
        warnings.append("source-offset measurement uncertainty is not recorded")
    if not geometry.receiver_depth_reference:
        warnings.append("receiver depth reference is not recorded")
    if not geometry.coordinate_system:
        warnings.append("coordinate system is not recorded")
    if geometry.source_to_receiver_bearing_deg is None:
        warnings.append("source-to-sounding bearing is not recorded")
    if geometry.source_block_axis_bearing_deg is None:
        warnings.append("source-block axis bearing is not recorded")
    if geometry.channel_17_strike_bearing_deg is None or geometry.channel_18_strike_bearing_deg is None:
        warnings.append("channel 17/18 strike bearings are incomplete")
    if geometry.receiver_orientation_bearing_deg is None:
        warnings.append("receiver orientation bearing is not recorded")
    if not geometry.elevations_enabled:
        warnings.append("source and receiver-reference elevations are not recorded")
    elif not geometry.vertical_datum:
        warnings.append("vertical datum is not recorded")

    bearing = geometry.source_to_receiver_bearing_deg or 0.0
    collar_east_unit, collar_north_unit = _bearing_vector(bearing)
    current_east = offset * collar_east_unit
    current_north = offset * collar_north_unit
    point_by_depth = {round(point.recorded_depth_m, 6): point for point in geometry.deviation_points}
    vertical_from_reference = 0.0
    previous_depth = 0.0
    vertical_depths: list[float] = []
    eastings: list[float] = []
    northings: list[float] = []
    missing_deviation_depths: list[float] = []
    missing_azimuth_depths: list[float] = []
    for depth in depths:
        interval = float(depth - previous_depth)
        point = point_by_depth.get(round(float(depth), 6))
        if point is None:
            missing_deviation_depths.append(float(depth))
            inclination = 0.0
            azimuth = None
        else:
            inclination = float(point.inclination_deg)
            azimuth = point.azimuth_deg
        if not math.isfinite(inclination) or not 0.0 <= inclination < 90.0:
            raise ValueError(f"Inclination at {depth:.3f} m must be between 0 and 90 degrees from vertical.")
        inclination_rad = math.radians(inclination)
        if geometry.depth_basis == "along_rods":
            vertical_increment = interval * math.cos(inclination_rad)
            horizontal_increment = interval * math.sin(inclination_rad)
        else:
            vertical_increment = interval
            horizontal_increment = interval * math.tan(inclination_rad)
        vertical_from_reference += vertical_increment
        if horizontal_increment > 1.0e-12:
            if azimuth is None or not math.isfinite(float(azimuth)):
                missing_azimuth_depths.append(float(depth))
            else:
                east_unit, north_unit = _bearing_vector(float(azimuth))
                current_east += horizontal_increment * east_unit
                current_north += horizontal_increment * north_unit
        source_elevation_delta = (
            geometry.source_elevation_m - geometry.receiver_reference_elevation_m
            if geometry.elevations_enabled
            else 0.0
        )
        vertical_depths.append(source_elevation_delta + vertical_from_reference)
        eastings.append(current_east)
        northings.append(current_north)
        previous_depth = float(depth)

    if missing_deviation_depths:
        warnings.append(
            f"inclination is not recorded at {len(missing_deviation_depths)} receiver interval(s); zero assumed"
        )
    if missing_azimuth_depths:
        warnings.append(
            f"deviation azimuth is missing at {len(missing_azimuth_depths)} inclined interval(s); "
            "horizontal deviation omitted"
        )
    if (
        geometry.source_to_receiver_bearing_deg is not None
        and geometry.source_block_axis_bearing_deg is not None
    ):
        expected_axis = (geometry.source_to_receiver_bearing_deg + 90.0) % 180.0
        axis_difference = abs(
            (geometry.source_block_axis_bearing_deg % 180.0) - expected_axis
        )
        axis_difference = min(axis_difference, 180.0 - axis_difference)
        if axis_difference > 10.0:
            warnings.append("source-block axis is not within 10° of perpendicular to the source–sounding line")
    if (
        geometry.channel_17_strike_bearing_deg is not None
        and geometry.channel_18_strike_bearing_deg is not None
        and abs(_bearing_separation(
            geometry.channel_17_strike_bearing_deg,
            geometry.channel_18_strike_bearing_deg,
        ) - 180.0) > 10.0
    ):
        warnings.append("channel 17 and 18 strike bearings are not opposite within 10°")

    vertical = np.asarray(vertical_depths, dtype=float)
    if not np.all(np.isfinite(vertical)) or not np.all(vertical > 0.0) or not np.all(np.diff(vertical) > 0.0):
        raise ValueError(
            "Corrected receiver depths are not positive and strictly increasing; check elevations, depth basis, "
            "and inclination data."
        )
    east = np.asarray(eastings, dtype=float)
    north = np.asarray(northings, dtype=float)
    offsets = np.hypot(east, north)
    return CorrectedGeometry(
        recorded_depths_m=depths.copy(),
        source_offset_m=offset,
        vertical_depths_m=vertical,
        receiver_offsets_m=offsets,
        receiver_east_m=east,
        receiver_north_m=north,
        warnings=tuple(warnings),
    )


