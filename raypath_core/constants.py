"""Shared application and engineering constants."""


APP_NAME = "RayPath SCPT"
APP_VERSION = "0.8.0-alpha.1"
PROJECT_SUFFIX = ".rpscpt"
PROJECT_SCHEMA_VERSION = 10
SUPPORTED_PROJECT_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
GRU_PRE_TRIGGER_MS = 50.0
DEFAULT_PICKER_HALF_WIDTH_MS = 20.0
VELOCITY_MIN = 50.0
VELOCITY_MAX = 2000.0
VELOCITY_PROFILE_DISPLAY_MAX = 600.0
TRACE_PICK_KINDS = ("first_peak", "zero_cross", "max_peak")
PAIR_PICK_KINDS = ("crossover",)
PICK_KINDS = ("first_peak", "crossover", "zero_cross", "max_peak")
PICK_SEQUENCE: tuple[tuple[str, int | None], ...] = (
    ("first_peak", 17),
    ("first_peak", 18),
    ("crossover", None),
    ("zero_cross", 17),
    ("zero_cross", 18),
    ("max_peak", 17),
    ("max_peak", 18),
)
PICK_COLUMNS = {kind: index + 1 for index, kind in enumerate(PICK_KINDS)}
PICK_LABELS = {
    "first_peak": "First peak/trough",
    "crossover": "Pair crossover",
    "zero_cross": "Individual zero crossing (experimental)",
    "max_peak": "Maximum peak (experimental)",
}
MODEL_COLORS = {
    "first_peak": "#ff9b54",
    "crossover": "#2fb7a8",
    "zero_cross": "#8b949e",
    "max_peak": "#d97cff",
}
CHANNEL_LABELS = {17: "Left", 18: "Right"}
REVIEW_STATES = ("not_reviewed", "accepted", "accepted_with_comment", "rejected")
REVIEW_LABELS = {
    "not_reviewed": "Not reviewed",
    "accepted": "Accepted",
    "accepted_with_comment": "Accepted with comment",
    "rejected": "Rejected - excluded from inversion",
}
QC_SNR_WARNING_DB = 10.0
QC_CORRELATION_WARNING = 0.60
QC_MAX_CORRELATION_LAG_MS = 5.0
QC_PICK_DISAGREEMENT_WARNING_MS = 2.0
DEFAULT_MANUAL_PICK_UNCERTAINTY_MS = 0.5
DEFAULT_ENSEMBLE_SIZE = 0
QUICK_ENSEMBLE_SIZE = 20
FINAL_ENSEMBLE_SIZE = 200
REPORT_QUALITY_ENSEMBLE_MINIMUM = 100
DEFAULT_UNCERTAINTY_SEED = 11705
