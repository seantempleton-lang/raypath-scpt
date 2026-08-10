"""Headless engineering API for RayPath SCPT.

This package intentionally has no Qt or Matplotlib dependency.  The desktop
entry point re-exports these names while legacy imports are migrated.
"""

from . import constants as _constants
from .constants import *
from .comparators import (
    CrossCorrelationResult,
    GeologicalInversionResult,
    SlopeLayerResult,
    SlopeMethodResult,
    corrected_vertical_travel_times,
    cross_correlation_interval_time,
    fit_slope_method,
    forward_geological_model,
    invert_geological_layer_profile,
)
from .geometry import calculate_corrected_geometry
from .inversion import (
    depth_aware_regularization_operator,
    generate_velocity_uncertainty_ensemble,
    invert_velocity_profile,
    select_regularization_lcurve,
)
from .models import (
    DEPTH_BASIS_LABELS,
    CorrectedGeometry,
    DeviationPoint,
    InversionResult,
    RayPathError,
    RaySolution,
    RegularizationSelectionResult,
    SurveyGeometry,
    Ts1170Method1Vs30Result,
    VelocityUncertaintyResult,
    Vs30Result,
    uncertainty_ensemble_classification,
)
from .project_io import ProjectDocument, project_schema_version, read_project_file, write_project_file
from .rays import forward_model, pseudo_interval_velocities, solve_direct_ray
from .vs30 import calculate_ts1170_5_method1_vs30, calculate_vs30, ts1170_5_vs30_band
from .waveforms import (
    GruFormatError,
    WaveformQcMetrics,
    WaveformRecord,
    add_suggested_picks,
    calculate_waveform_qc,
    gru_deviation_points,
    parse_gru,
    suggest_pair_crossover,
    suggest_pick_uncertainty_ms,
    suggest_trace_picks,
)

__all__ = [name for name in vars(_constants) if name.isupper()]
__all__ += [
    "DEPTH_BASIS_LABELS",
    "CorrectedGeometry",
    "CrossCorrelationResult",
    "DeviationPoint",
    "GruFormatError",
    "GeologicalInversionResult",
    "InversionResult",
    "RayPathError",
    "RaySolution",
    "ProjectDocument",
    "RegularizationSelectionResult",
    "SurveyGeometry",
    "SlopeLayerResult",
    "SlopeMethodResult",
    "Ts1170Method1Vs30Result",
    "VelocityUncertaintyResult",
    "Vs30Result",
    "WaveformQcMetrics",
    "WaveformRecord",
    "add_suggested_picks",
    "calculate_corrected_geometry",
    "corrected_vertical_travel_times",
    "calculate_ts1170_5_method1_vs30",
    "calculate_vs30",
    "calculate_waveform_qc",
    "depth_aware_regularization_operator",
    "cross_correlation_interval_time",
    "fit_slope_method",
    "forward_model",
    "forward_geological_model",
    "generate_velocity_uncertainty_ensemble",
    "gru_deviation_points",
    "invert_velocity_profile",
    "invert_geological_layer_profile",
    "parse_gru",
    "project_schema_version",
    "pseudo_interval_velocities",
    "select_regularization_lcurve",
    "read_project_file",
    "solve_direct_ray",
    "suggest_pair_crossover",
    "suggest_pick_uncertainty_ms",
    "suggest_trace_picks",
    "ts1170_5_vs30_band",
    "uncertainty_ensemble_classification",
    "write_project_file",
]
