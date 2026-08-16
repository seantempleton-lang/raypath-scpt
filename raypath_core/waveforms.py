"""Waveform imports, arrival suggestions, and advisory quality control."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .constants import *
from .models import DeviationPoint


@dataclass
class AcquisitionHitRecord:
    """One selected-channel hit contributing to an acquisition shear stack."""

    wave_type: str
    hit_number: int
    time_ms: np.ndarray
    values: np.ndarray
    selected_channel: str
    raw_file: str
    included: bool = True
    correlation_coefficient: float | None = None
    correlation_lag_samples: int | None = None
    signal_to_noise_ratio: float | None = None
    gain_1: int | None = None
    gain_2: int | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.wave_type, self.hit_number


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
    review_state: str = "not_reviewed"
    review_comment: str = ""
    pick_uncertainty_ms: float | None = None
    pick_uncertainty_source: str = "automatic_qc_default"
    pick_uncertainty_basis: str = "Half the waveform sample interval."
    acquisition_hits: list[AcquisitionHitRecord] = field(default_factory=list)

    @staticmethod
    def pick_key(kind: str, channel: int) -> str:
        return f"{kind}_{channel}"

    def get_pick(self, kind: str, channel: int) -> float | None:
        value = self.picks_ms.get(self.pick_key(kind, channel))
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def set_pick(self, kind: str, channel: int, time_ms: float | None) -> None:
        self.picks_ms[self.pick_key(kind, channel)] = None if time_ms is None else float(time_ms)

    def get_pair_pick(self, kind: str) -> float | None:
        value = self.picks_ms.get(kind)
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def set_pair_pick(self, kind: str, time_ms: float | None) -> None:
        self.picks_ms[kind] = None if time_ms is None else float(time_ms)

    def arrival_ms(self, kind: str) -> float | None:
        if kind in PAIR_PICK_KINDS:
            return self.get_pair_pick(kind)
        values = [self.get_pick(kind, channel) for channel in (17, 18)]
        finite = [value for value in values if value is not None]
        return float(np.mean(finite)) if finite else None

    @property
    def is_excluded(self) -> bool:
        """Return whether the analyst explicitly rejected this observation."""

        return self.review_state == "rejected"

    @property
    def recorded_time_ms(self) -> np.ndarray:
        """Return the source sample clock before any trigger correction."""

        return self.time_ms + self.pre_trigger_ms

    def recorded_picks_ms(self) -> dict[str, float | None]:
        """Return current picks expressed on the original source sample clock."""

        return {
            key: None if value is None else float(value) + self.pre_trigger_ms
            for key, value in self.picks_ms.items()
        }

    def rebuild_acquisition_stacks(self) -> None:
        """Rebuild left/right direct sums from currently included hits."""

        if not self.acquisition_hits:
            return
        rebuilt: dict[str, np.ndarray] = {}
        for wave_type in ("shear_left", "shear_right"):
            hits = [
                hit
                for hit in self.acquisition_hits
                if hit.wave_type == wave_type and hit.included
            ]
            if not hits:
                raise ValueError(f"At least one {wave_type} hit must remain included.")
            for hit in hits:
                if hit.time_ms.shape != self.time_ms.shape or not np.allclose(
                    hit.time_ms, self.time_ms, rtol=0.0, atol=1.0e-6
                ):
                    raise ValueError(
                        f"Acquisition {wave_type} hit {hit.hit_number} does not share the stack sample clock."
                    )
            rebuilt[wave_type] = np.sum(
                np.stack([hit.values for hit in hits], axis=0), axis=0, dtype=float
            )
        self.left = rebuilt["shear_left"]
        self.right = rebuilt["shear_right"]

    def acquisition_contributor_counts(self) -> tuple[int, int]:
        """Return included left/right acquisition hit counts."""

        left_count = sum(hit.included and hit.wave_type == "shear_left" for hit in self.acquisition_hits)
        right_count = sum(hit.included and hit.wave_type == "shear_right" for hit in self.acquisition_hits)
        return left_count, right_count


@dataclass(frozen=True)
class WaveformQcMetrics:
    """Deterministic signal-quality measurements for one opposing trace pair."""

    sample_interval_ms: float
    sample_interval_deviation_pct: float
    sample_interval_consistent: bool
    noise_rms_left: float
    noise_rms_right: float
    snr_left_db: float
    snr_right_db: float
    sign_reversed_correlation: float
    correlation_lag_ms: float
    polarity_reversed: bool
    first_peak_disagreement_ms: float | None
    zero_cross_disagreement_ms: float | None
    max_peak_disagreement_ms: float | None
    clipped_left: bool
    clipped_right: bool
    constant_left: bool
    constant_right: bool
    warnings: tuple[str, ...]

    @property
    def passes_minimum(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable audit snapshot."""

        def finite_or_none(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "sample_interval_ms": self.sample_interval_ms,
            "sample_interval_deviation_pct": self.sample_interval_deviation_pct,
            "sample_interval_consistent": self.sample_interval_consistent,
            "noise_rms_left": finite_or_none(self.noise_rms_left),
            "noise_rms_right": finite_or_none(self.noise_rms_right),
            "snr_left_db": finite_or_none(self.snr_left_db),
            "snr_right_db": finite_or_none(self.snr_right_db),
            "sign_reversed_correlation": finite_or_none(self.sign_reversed_correlation),
            "correlation_lag_ms": self.correlation_lag_ms,
            "polarity_reversed": self.polarity_reversed,
            "first_peak_disagreement_ms": self.first_peak_disagreement_ms,
            "zero_cross_disagreement_ms": self.zero_cross_disagreement_ms,
            "max_peak_disagreement_ms": self.max_peak_disagreement_ms,
            "clipped_left": self.clipped_left,
            "clipped_right": self.clipped_right,
            "constant_left": self.constant_left,
            "constant_right": self.constant_right,
            "warnings": list(self.warnings),
        }


class GruFormatError(ValueError):
    """Raised for malformed or unsupported GRU content."""


class ScptAcquisitionFormatError(ValueError):
    """Raised for malformed or unsupported SCPT acquisition output."""


SEISMIC_HEADER_RE = re.compile(
    r"^\[SEISMIC\s+TEST\s*=\s*(?P<test>\d+)\s+DEPTH\s*=\s*(?P<depth>[-+0-9.eE]+)\s*\]$",
    re.IGNORECASE,
)
E_RECORD_RE = re.compile(
    r"^\*E:(?P<time>[-+0-9.eE]+)#17:(?P<left>[-+0-9.eE]+)#18:(?P<right>[-+0-9.eE]+)!$",
    re.IGNORECASE,
)
E_SINGLE_CHANNEL_RECORD_RE = re.compile(
    r"^\*E:(?P<time>[-+0-9.eE]+)#(?P<channel>17|18):(?P<value>[-+0-9.eE]+)!$",
    re.IGNORECASE,
)
D_RECORD_RE = re.compile(
    r"^\*D:(?P<depth>[-+0-9.eE]+)(?P<channels>(?:#\d+:[-+0-9.eE]+)+)!$",
    re.IGNORECASE,
)
D_CHANNEL_RE = re.compile(r"#(?P<channel>\d+):(?P<value>[-+0-9.eE]+)", re.IGNORECASE)


def parse_scpt_acquisition(
    path: str | Path,
    import_messages: list[str] | None = None,
) -> list[WaveformRecord]:
    """Import paired shear stacks from an SCPT acquisition sounding.

    ``path`` may name the sounding directory, its ``sounding.json`` file, or
    its ``stacked_waveforms.csv`` file.  The acquisition application's
    baseline-corrected direct sums are retained unchanged.  P-wave rows are
    intentionally ignored because RayPath SCPT currently interprets shear-wave
    velocity only.
    """

    source = Path(path)
    root = source if source.is_dir() else source.parent
    sounding_path = root / "sounding.json"
    waveform_path = root / "stacked_waveforms.csv"
    if not sounding_path.is_file():
        raise FileNotFoundError(f"SCPT acquisition metadata not found: {sounding_path}")
    if not waveform_path.is_file():
        raise FileNotFoundError(f"SCPT stacked waveforms not found: {waveform_path}")

    try:
        metadata = json.loads(sounding_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ScptAcquisitionFormatError(f"Invalid sounding.json: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != "scpt-acquisition":
        raise ScptAcquisitionFormatError(
            "sounding.json is not an SCPT acquisition sounding (expected schema 'scpt-acquisition')."
        )
    try:
        schema_version = int(metadata.get("schema_version"))
    except (TypeError, ValueError) as exc:
        raise ScptAcquisitionFormatError("sounding.json has no valid schema_version.") from exc
    if schema_version != 1:
        raise ScptAcquisitionFormatError(
            f"Unsupported SCPT acquisition schema version {schema_version}; this reader supports version 1."
        )
    try:
        sample_rate_hz = float(metadata["sample_rate_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScptAcquisitionFormatError("sounding.json has no valid sample_rate_hz.") from exc
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ScptAcquisitionFormatError("sounding.json sample_rate_hz must be finite and positive.")

    required_fields = {
        "depth_m",
        "wave_type",
        "sample_index",
        "time_from_trigger_s",
        "sum_counts",
        "stack_count",
    }
    groups: dict[tuple[float, str], list[tuple[int, float, float, int]]] = {}
    p_wave_rows = 0
    with waveform_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ScptAcquisitionFormatError(
                "stacked_waveforms.csv is missing required column(s): "
                + ", ".join(sorted(missing_fields))
            )
        for line_number, row in enumerate(reader, 2):
            wave_type = str(row["wave_type"]).strip().lower()
            if wave_type == "p_wave":
                p_wave_rows += 1
                continue
            if wave_type not in {"shear_left", "shear_right"}:
                raise ScptAcquisitionFormatError(
                    f"Unsupported wave_type {row['wave_type']!r} on line {line_number}."
                )
            try:
                depth_m = float(row["depth_m"])
                sample_index = int(row["sample_index"])
                time_s = float(row["time_from_trigger_s"])
                sum_counts = float(row["sum_counts"])
                stack_count = int(row["stack_count"])
            except (TypeError, ValueError) as exc:
                raise ScptAcquisitionFormatError(
                    f"Invalid numeric value in stacked_waveforms.csv on line {line_number}."
                ) from exc
            if not all(math.isfinite(value) for value in (depth_m, time_s, sum_counts)):
                raise ScptAcquisitionFormatError(
                    f"Non-finite value in stacked_waveforms.csv on line {line_number}."
                )
            if depth_m <= 0.0 or sample_index < 0 or stack_count < 1:
                raise ScptAcquisitionFormatError(
                    f"Invalid depth, sample index, or stack count on line {line_number}."
                )
            groups.setdefault((depth_m, wave_type), []).append(
                (sample_index, time_s * 1000.0, sum_counts, stack_count)
            )

    if not groups:
        raise ScptAcquisitionFormatError("No shear_left or shear_right stacks were found.")
    if p_wave_rows and import_messages is not None:
        import_messages.append(
            f"Ignored {p_wave_rows} P-wave stack row(s); RayPath SCPT currently processes shear waves only."
        )

    root_resolved = root.resolve()

    def load_hit(
        wave_type: str,
        item: dict[str, Any],
        stack_time_ms: np.ndarray,
    ) -> AcquisitionHitRecord:
        try:
            hit_number = int(item["hit_number"])
            selected_channel = str(item["selected_channel"]).strip().lower()
            raw_file = str(item["raw_file"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ScptAcquisitionFormatError(
                f"Invalid {wave_type} hit metadata at {depth_m:g} m."
            ) from exc
        if hit_number < 1 or selected_channel not in {"x", "y"} or not raw_file:
            raise ScptAcquisitionFormatError(
                f"Invalid {wave_type} hit identity at {depth_m:g} m."
            )
        hit_path = (root / Path(raw_file)).resolve()
        try:
            hit_path.relative_to(root_resolved)
        except ValueError as exc:
            raise ScptAcquisitionFormatError(
                f"Acquisition hit path escapes the sounding directory: {raw_file}"
            ) from exc
        if not hit_path.is_file():
            raise FileNotFoundError(f"Acquisition hit CSV not found: {hit_path}")
        time_values: list[float] = []
        signal_values: list[float] = []
        with hit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            needed = {"time_from_trigger_s", f"{selected_channel}_counts"}
            missing = needed - set(reader.fieldnames or ())
            if missing:
                raise ScptAcquisitionFormatError(
                    f"{raw_file} is missing required column(s): {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, 2):
                try:
                    time_values.append(float(row["time_from_trigger_s"]) * 1000.0)
                    signal_values.append(float(row[f"{selected_channel}_counts"]))
                except (TypeError, ValueError) as exc:
                    raise ScptAcquisitionFormatError(
                        f"Invalid selected-channel sample in {raw_file} on line {line_number}."
                    ) from exc
        hit_time_ms = np.asarray(time_values, dtype=float)
        raw_values = np.asarray(signal_values, dtype=float)
        if hit_time_ms.shape != stack_time_ms.shape or not np.allclose(
            hit_time_ms, stack_time_ms, rtol=0.0, atol=1.0e-6
        ):
            raise ScptAcquisitionFormatError(
                f"{wave_type} hit {hit_number} at {depth_m:g} m does not share the stack sample clock."
            )
        pre_trigger = hit_time_ms < 0.0
        if not np.any(pre_trigger):
            raise ScptAcquisitionFormatError(
                f"{wave_type} hit {hit_number} at {depth_m:g} m has no pre-trigger baseline samples."
            )
        values = raw_values - float(np.mean(raw_values[pre_trigger]))
        correlation = item.get("correlation") if isinstance(item.get("correlation"), dict) else {}
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        selected_metrics = metrics.get(selected_channel) if isinstance(metrics.get(selected_channel), dict) else {}

        def optional_float(value: Any) -> float | None:
            if value in (None, ""):
                return None
            converted = float(value)
            return converted if math.isfinite(converted) else None

        def optional_int(value: Any) -> int | None:
            return None if value in (None, "") else int(value)

        return AcquisitionHitRecord(
            wave_type=wave_type,
            hit_number=hit_number,
            time_ms=hit_time_ms,
            values=values,
            selected_channel=selected_channel,
            raw_file=raw_file,
            correlation_coefficient=optional_float(correlation.get("coefficient")),
            correlation_lag_samples=optional_int(correlation.get("lag_samples")),
            signal_to_noise_ratio=optional_float(selected_metrics.get("signal_to_noise_ratio")),
            gain_1=optional_int(item.get("gain_1")),
            gain_2=optional_int(item.get("gain_2")),
        )

    depths = sorted({depth for depth, _wave_type in groups})
    records: list[WaveformRecord] = []
    expected_interval_ms = 1000.0 / sample_rate_hz
    for test_number, depth_m in enumerate(depths, 1):
        missing = [
            wave_type
            for wave_type in ("shear_left", "shear_right")
            if (depth_m, wave_type) not in groups
        ]
        if missing:
            raise ScptAcquisitionFormatError(
                f"Depth {depth_m:g} m is missing required stack(s): {', '.join(missing)}."
            )

        parsed: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for wave_type in ("shear_left", "shear_right"):
            rows = sorted(groups[(depth_m, wave_type)], key=lambda item: item[0])
            indices = [item[0] for item in rows]
            if indices != list(range(len(rows))):
                raise ScptAcquisitionFormatError(
                    f"{wave_type} at {depth_m:g} m has duplicate, missing, or non-zero-based sample indices."
                )
            if len(rows) < 3:
                raise ScptAcquisitionFormatError(
                    f"{wave_type} at {depth_m:g} m contains too few waveform samples."
                )
            times_ms = np.asarray([item[1] for item in rows], dtype=float)
            values = np.asarray([item[2] for item in rows], dtype=float)
            stack_counts = {item[3] for item in rows}
            if len(stack_counts) != 1:
                raise ScptAcquisitionFormatError(
                    f"{wave_type} at {depth_m:g} m changes stack_count within one trace."
                )
            intervals = np.diff(times_ms)
            if np.any(intervals <= 0.0):
                raise ScptAcquisitionFormatError(
                    f"{wave_type} at {depth_m:g} m has non-increasing sample times."
                )
            if not np.allclose(intervals, expected_interval_ms, rtol=1.0e-6, atol=1.0e-6):
                raise ScptAcquisitionFormatError(
                    f"{wave_type} at {depth_m:g} m sample times do not match the "
                    f"{sample_rate_hz:g} Hz sounding configuration."
                )
            if not (times_ms[0] < 0.0 <= times_ms[-1]):
                raise ScptAcquisitionFormatError(
                    f"{wave_type} at {depth_m:g} m does not span the trigger at 0 ms."
                )
            parsed[wave_type] = (times_ms, values, stack_counts.pop())

        left_time, left_values, left_count = parsed["shear_left"]
        right_time, right_values, right_count = parsed["shear_right"]
        if left_time.shape != right_time.shape or not np.allclose(
            left_time, right_time, rtol=0.0, atol=1.0e-6
        ):
            raise ScptAcquisitionFormatError(
                f"Left and right stacks at {depth_m:g} m do not share the same sample clock."
            )
        if left_count != right_count:
            raise ScptAcquisitionFormatError(
                f"Left and right stacks at {depth_m:g} m have different contributor counts "
                f"({left_count} and {right_count})."
            )
        record = WaveformRecord(
            test_number=test_number,
            depth_m=depth_m,
            time_ms=left_time,
            left=left_values,
            right=right_values,
            # The acquisition CSV clock is already relative to the
            # physical trigger, so no GRU-style clock correction applies.
            pre_trigger_ms=0.0,
            pick_uncertainty_ms=expected_interval_ms / 2.0,
        )
        depth_path = root / "depths" / f"{depth_m:07.3f}m" / "depth.json"
        if depth_path.is_file():
            try:
                depth_payload = json.loads(depth_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                raise ScptAcquisitionFormatError(f"Invalid {depth_path}: {exc}") from exc
            depth_groups = depth_payload.get("groups") if isinstance(depth_payload, dict) else None
            if not isinstance(depth_groups, dict):
                raise ScptAcquisitionFormatError(f"{depth_path} has no valid groups object.")
            acquisition_hits: list[AcquisitionHitRecord] = []
            for wave_type, stack_time, stack_values, expected_count in (
                ("shear_left", left_time, left_values, left_count),
                ("shear_right", right_time, right_values, right_count),
            ):
                group = depth_groups.get(wave_type)
                items = group.get("hits") if isinstance(group, dict) else None
                if not isinstance(items, list):
                    raise ScptAcquisitionFormatError(
                        f"{depth_path} has no valid {wave_type} hit list."
                    )
                hits = [
                    load_hit(wave_type, item, stack_time)
                    for item in items
                    if isinstance(item, dict) and str(item.get("status", "")).lower() == "accepted"
                ]
                if len(hits) != expected_count:
                    raise ScptAcquisitionFormatError(
                        f"{wave_type} at {depth_m:g} m has {len(hits)} accepted hit file(s), "
                        f"but stacked_waveforms.csv records {expected_count} contributors."
                    )
                rebuilt = np.sum(np.stack([hit.values for hit in hits], axis=0), axis=0, dtype=float)
                if not np.allclose(rebuilt, stack_values, rtol=0.0, atol=1.0e-6):
                    raise ScptAcquisitionFormatError(
                        f"Accepted {wave_type} hits at {depth_m:g} m do not reproduce the persisted stack."
                    )
                acquisition_hits.extend(hits)
            record.acquisition_hits = acquisition_hits
        elif import_messages is not None:
            import_messages.append(
                f"Individual hit metadata was unavailable at {depth_m:g} m; imported the persisted stacks only."
            )
        records.append(record)

    if import_messages is not None:
        hit_count = sum(len(record.acquisition_hits) for record in records)
        if hit_count:
            import_messages.append(f"Loaded {hit_count} individual shear-hit contributor(s) for review.")
        import_messages.append(
            "Imported baseline-corrected direct-sum shear stacks without normalization or polarity changes."
        )
    return records


def parse_gru(
    path: str | Path,
    pre_trigger_ms: float = GRU_PRE_TRIGGER_MS,
    import_messages: list[str] | None = None,
) -> list[WaveformRecord]:
    """Stream a GRU file and return paired traces on a trigger-relative clock.

    GOnsite/GORILLA GRU files do not declare their 50 ms pre-trigger recording
    period.  The sample times in the file are therefore shifted by
    ``pre_trigger_ms`` during import: a recorded time of 50.0 ms becomes the
    physical trigger at 0.0 ms, and only post-trigger pick times feed the
    velocity inversion. Completely empty seismic acquisition blocks are
    skipped and described in ``import_messages`` when supplied. Complete
    single-channel acquisitions are also skipped because they cannot form a
    paired trace; truncated paired records remain an error.
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
    unpaired_channels: set[int] = set()
    unpaired_sample_count = 0

    def finish_record() -> None:
        nonlocal current, times, left, right, unpaired_channels, unpaired_sample_count
        if current is None:
            return
        if not times:
            if import_messages is not None:
                if unpaired_sample_count:
                    missing_channels = sorted({17, 18} - unpaired_channels)
                    if len(missing_channels) == 1:
                        detail = (
                            f"channel #{missing_channels[0]} missing from "
                            f"{unpaired_sample_count} waveform samples"
                        )
                    else:
                        detail = f"{unpaired_sample_count} waveform samples were not paired by channel"
                    import_messages.append(
                        f"Skipped incomplete seismic test {current[0]} at {current[1]:g} m ({detail})."
                    )
                else:
                    import_messages.append(
                        f"Skipped empty seismic test {current[0]} at {current[1]:g} m (no waveform samples)."
                    )
            current = None
            times, left, right = [], [], []
            unpaired_channels, unpaired_sample_count = set(), 0
            return
        if unpaired_sample_count:
            raise GruFormatError(
                f"Seismic test {current[0]} mixes paired samples with {unpaired_sample_count} "
                "single-channel samples."
            )
        if len(times) < 3:
            raise GruFormatError(f"Seismic test {current[0]} at {current[1]:g} m contains too few samples.")
        recorded_time = np.asarray(times, dtype=float)
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if not (
            np.all(np.isfinite(recorded_time))
            and np.all(np.isfinite(left_array))
            and np.all(np.isfinite(right_array))
        ):
            raise GruFormatError(f"Seismic test {current[0]} contains non-finite samples.")
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
                left=left_array,
                right=right_array,
                pre_trigger_ms=pre_trigger,
                pick_uncertainty_ms=float(np.median(np.diff(recorded_time))) / 2.0,
            )
        )
        current = None
        times, left, right = [], [], []
        unpaired_channels, unpaired_sample_count = set(), 0

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
                    single_channel_match = E_SINGLE_CHANNEL_RECORD_RE.match(line)
                    if single_channel_match:
                        unpaired_channels.add(int(single_channel_match.group("channel")))
                        unpaired_sample_count += 1
                        continue
                    raise GruFormatError(f"Malformed seismic sample on line {line_number}.")
                times.append(float(sample_match.group("time")))
                left.append(float(sample_match.group("left")))
                right.append(float(sample_match.group("right")))
    finish_record()
    if not records:
        raise GruFormatError("No paired #17/#18 seismic waveform blocks were found in this GRU file.")
    records.sort(key=lambda record: record.depth_m)
    return records


def gru_deviation_points(
    path: str | Path,
    receiver_depths_m: Sequence[float],
    import_messages: list[str] | None = None,
) -> list[DeviationPoint]:
    """Derive receiver-interval cone inclination from GRU channels #30/#31.

    The two GRU channels are signed X/Y inclinations in degrees from vertical.
    Their resultant is ``hypot(x, y)``. Dense CPT scan samples are integrated
    over each seismic receiver interval and reduced to one equivalent
    inclination that preserves the interval's vertical projection. No global
    deviation azimuth is inferred because the GRU file does not establish how
    the inclinometer axes were oriented relative to the project coordinate
    system.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"GRU file not found: {source}")
    receivers = np.asarray(receiver_depths_m, dtype=float)
    if (
        receivers.ndim != 1
        or receivers.size == 0
        or not np.all(np.isfinite(receivers))
        or not np.all(receivers > 0.0)
        or not np.all(np.diff(receivers) > 0.0)
    ):
        raise ValueError("Receiver depths for GRU inclination import must be finite, positive, and increasing.")

    samples_by_depth: dict[float, list[tuple[float, float]]] = {}
    incomplete_axis_rows = 0
    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("*D:"):
                continue
            match = D_RECORD_RE.match(line)
            if match is None:
                continue
            channels = {
                int(item.group("channel")): float(item.group("value"))
                for item in D_CHANNEL_RE.finditer(match.group("channels"))
            }
            if (30 in channels) != (31 in channels):
                incomplete_axis_rows += 1
                continue
            if 30 not in channels:
                continue
            depth = float(match.group("depth"))
            x_tilt = channels[30]
            y_tilt = channels[31]
            if not all(math.isfinite(value) for value in (depth, x_tilt, y_tilt)) or depth <= 0.0:
                continue
            samples_by_depth.setdefault(depth, []).append((x_tilt, y_tilt))

    if incomplete_axis_rows and import_messages is not None:
        import_messages.append(
            f"Ignored {incomplete_axis_rows} CPT inclination row(s) missing either GRU channel #30 or #31."
        )
    if not samples_by_depth:
        if import_messages is not None:
            import_messages.append("No paired GRU #30/#31 cone-inclination samples were available.")
        return []

    sample_depths = np.asarray(sorted(samples_by_depth), dtype=float)
    sample_inclinations = np.asarray(
        [
            math.hypot(
                float(np.mean([pair[0] for pair in samples_by_depth[depth]])),
                float(np.mean([pair[1] for pair in samples_by_depth[depth]])),
            )
            for depth in sample_depths
        ],
        dtype=float,
    )
    valid_sample = np.isfinite(sample_inclinations) & (sample_inclinations >= 0.0) & (sample_inclinations < 90.0)
    sample_depths = sample_depths[valid_sample]
    sample_inclinations = sample_inclinations[valid_sample]
    if sample_depths.size == 0:
        if import_messages is not None:
            import_messages.append("GRU #30/#31 cone-inclination samples were outside the valid 0-90 degree range.")
        return []

    points: list[DeviationPoint] = []
    unavailable_intervals = 0
    previous_depth = 0.0
    for receiver_depth in receivers:
        mask = (sample_depths > previous_depth + 1.0e-9) & (sample_depths <= receiver_depth + 1.0e-9)
        interval_depths = sample_depths[mask]
        interval_inclinations = sample_inclinations[mask]
        if interval_depths.size == 0:
            unavailable_intervals += 1
            previous_depth = float(receiver_depth)
            continue
        segment_lengths = np.diff(np.r_[previous_depth, interval_depths])
        if interval_depths[-1] < receiver_depth - 1.0e-9:
            segment_lengths = np.r_[segment_lengths, receiver_depth - interval_depths[-1]]
            interval_inclinations = np.r_[interval_inclinations, interval_inclinations[-1]]
        interval_length = float(receiver_depth - previous_depth)
        mean_vertical_ratio = float(
            np.sum(segment_lengths * np.cos(np.radians(interval_inclinations))) / interval_length
        )
        equivalent_inclination = math.degrees(math.acos(float(np.clip(mean_vertical_ratio, 0.0, 1.0))))
        points.append(DeviationPoint(float(receiver_depth), equivalent_inclination, None))
        previous_depth = float(receiver_depth)

    if unavailable_intervals and import_messages is not None:
        import_messages.append(
            f"GRU cone inclination was unavailable for {unavailable_intervals} receiver interval(s); "
            "those intervals remain uncorrected."
        )
    return points


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
    if pre_trigger_indices.size:
        baseline_indices = pre_trigger_indices
        post_trigger_start = min(int(np.searchsorted(t, 0.0, side="right")), t.size - 1)
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
        "zero_cross": cross_time,
        "max_peak": float(t[max_peak]),
    }


def suggest_pair_crossover(
    time_ms: np.ndarray,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> float:
    """Suggest the first post-arrival intersection of a reversed trace pair.

    The two traces are independently baseline-corrected and amplitude-normalised
    for the suggestion only.  Searching begins after both suggested first
    peak/trough times, which avoids selecting a noise intersection before the
    shear-wave arrival.  The result remains a review aid and can be moved by the
    analyst in the picker.
    """

    t = np.asarray(time_ms, dtype=float)
    left = _smoothed_trace(np.asarray(left_values, dtype=float))
    right = _smoothed_trace(np.asarray(right_values, dtype=float))
    if t.ndim != 1 or left.shape != t.shape or right.shape != t.shape or t.size < 3:
        raise ValueError("Pair crossover requires matching one-dimensional waveform arrays.")

    pre_trigger = t < 0.0

    def normalized(values: np.ndarray) -> np.ndarray:
        baseline_values = values[pre_trigger]
        baseline = float(np.median(baseline_values)) if baseline_values.size else float(np.median(values))
        centred = values - baseline
        post_trigger = np.abs(centred[t >= 0.0])
        scale = float(np.max(post_trigger)) if post_trigger.size else float(np.max(np.abs(centred)))
        return centred / scale if math.isfinite(scale) and scale > 0.0 else centred

    left_normalized = normalized(left)
    right_normalized = normalized(right)
    difference = left_normalized - right_normalized
    left_suggestions = suggest_trace_picks(t, left)
    right_suggestions = suggest_trace_picks(t, right)
    search_time = max(left_suggestions["first_peak"], right_suggestions["first_peak"], 0.0)
    start = min(int(np.searchsorted(t, search_time, side="left")), t.size - 2)
    crossings = np.flatnonzero(difference[start:-1] * difference[start + 1 :] <= 0.0) + start
    if crossings.size:
        index = int(crossings[0])
        delta = difference[index + 1] - difference[index]
        fraction = 0.0 if delta == 0.0 else float(np.clip(-difference[index] / delta, 0.0, 1.0))
        return float(t[index] + fraction * (t[index + 1] - t[index]))

    return float(np.mean([left_suggestions["zero_cross"], right_suggestions["zero_cross"]]))


def add_suggested_picks(records: Iterable[WaveformRecord], overwrite: bool = False) -> None:
    """Populate missing manual-pick fields with automatic trace suggestions."""

    for record in records:
        for channel, values in ((17, record.left), (18, record.right)):
            suggestions = suggest_trace_picks(record.time_ms, values)
            for kind, time_ms in suggestions.items():
                if overwrite or record.get_pick(kind, channel) is None:
                    record.set_pick(kind, channel, time_ms)
        if overwrite or record.get_pair_pick("crossover") is None:
            record.set_pair_pick(
                "crossover",
                suggest_pair_crossover(record.time_ms, record.left, record.right),
            )
        if record.pick_uncertainty_source != "analyst_override":
            suggestion, basis = suggest_pick_uncertainty_ms(record)
            record.pick_uncertainty_ms = suggestion
            record.pick_uncertainty_source = "automatic_qc_default"
            record.pick_uncertainty_basis = basis


def calculate_waveform_qc(record: WaveformRecord) -> WaveformQcMetrics:
    """Calculate transparent, deterministic QC metrics for a trace pair.

    The metrics are advisory.  They never change a pick or automatically reject
    a record.  Rejection remains an explicit analyst decision stored in the
    project audit data.
    """

    t = np.asarray(record.time_ms, dtype=float)
    left = np.asarray(record.left, dtype=float)
    right = np.asarray(record.right, dtype=float)
    if t.ndim != 1 or t.size < 3 or left.shape != t.shape or right.shape != t.shape:
        raise ValueError("Waveform QC requires matching one-dimensional trace arrays.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("Waveform QC cannot evaluate non-finite samples.")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("Waveform QC requires strictly increasing sample times.")
    sample_interval = float(np.median(dt))
    interval_deviation_pct = 100.0 * float(np.max(np.abs(dt - sample_interval))) / sample_interval
    interval_consistent = interval_deviation_pct <= 1.0
    pre_trigger = t < 0.0

    def centred(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        baseline_values = values[pre_trigger]
        if baseline_values.size < 3:
            baseline_values = values[: max(3, min(values.size, values.size // 8))]
        baseline = float(np.median(baseline_values))
        return values - baseline, baseline_values - baseline

    left_centred, left_noise = centred(left)
    right_centred, right_noise = centred(right)
    candidate_arrivals = [
        value
        for kind in PICK_KINDS
        if (value := record.arrival_ms(kind)) is not None
    ]
    arrival = min(candidate_arrivals) if candidate_arrivals else 0.0
    signal_mask = (t >= max(0.0, arrival - 2.0)) & (t <= arrival + 80.0)
    if not np.any(signal_mask):
        signal_mask = t >= 0.0

    def signal_metrics(values: np.ndarray, noise: np.ndarray) -> tuple[float, float]:
        noise_rms = float(np.sqrt(np.mean(noise * noise))) if noise.size else 0.0
        signal_peak = float(np.max(np.abs(values[signal_mask]))) if np.any(signal_mask) else 0.0
        if signal_peak <= 0.0:
            return noise_rms, float("-inf")
        if noise_rms <= np.finfo(float).eps * max(signal_peak, 1.0):
            return noise_rms, float("inf")
        return noise_rms, 20.0 * math.log10(signal_peak / noise_rms)

    def constant_or_clipped(values: np.ndarray) -> tuple[bool, bool]:
        amplitude_range = float(np.ptp(values))
        scale = max(float(np.max(np.abs(values))), 1.0)
        is_constant = amplitude_range <= np.finfo(float).eps * scale * 32.0
        if is_constant:
            return True, False
        absolute = np.abs(values - float(np.median(values[pre_trigger])))
        extreme = absolute >= float(np.max(absolute)) * (1.0 - 1.0e-9)
        longest_run = run = 0
        for is_extreme in extreme:
            run = run + 1 if is_extreme else 0
            longest_run = max(longest_run, run)
        return False, longest_run >= 3

    constant_left, clipped_left = constant_or_clipped(left)
    constant_right, clipped_right = constant_or_clipped(right)
    noise_rms_left, snr_left = signal_metrics(left_centred, left_noise)
    noise_rms_right, snr_right = signal_metrics(right_centred, right_noise)

    correlation_anchor = record.get_pair_pick("crossover")
    if correlation_anchor is None:
        correlation_anchor = arrival
    correlation_mask = (t >= max(0.0, correlation_anchor - 15.0)) & (t <= correlation_anchor + 40.0)
    left_window = left_centred[correlation_mask]
    right_window = -right_centred[correlation_mask]
    max_lag_samples = max(1, round(QC_MAX_CORRELATION_LAG_MS / sample_interval))
    best_correlation = -1.0
    best_lag = 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            a, b = left_window[-lag:], right_window[:lag]
        elif lag > 0:
            a, b = left_window[:-lag], right_window[lag:]
        else:
            a, b = left_window, right_window
        if a.size < 3:
            continue
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        correlation = float(np.dot(a, b) / denominator) if denominator > 0.0 else -1.0
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    best_correlation = float(np.clip(best_correlation, -1.0, 1.0))

    def trace_disagreement(kind: str) -> float | None:
        left_pick = record.get_pick(kind, 17)
        right_pick = record.get_pick(kind, 18)
        return None if left_pick is None or right_pick is None else abs(left_pick - right_pick)

    first_peak_left = record.get_pick("first_peak", 17)
    first_peak_right = record.get_pick("first_peak", 18)
    if first_peak_left is not None and first_peak_right is not None:
        left_amplitude = float(np.interp(first_peak_left, t, left_centred))
        right_amplitude = float(np.interp(first_peak_right, t, right_centred))
        polarity_reversed = left_amplitude * right_amplitude < 0.0
        peak_disagreement = trace_disagreement("first_peak")
    else:
        polarity_reversed = False
        peak_disagreement = None
    zero_cross_disagreement = trace_disagreement("zero_cross")
    max_peak_disagreement = trace_disagreement("max_peak")

    warnings: list[str] = []
    if min(snr_left, snr_right) < QC_SNR_WARNING_DB:
        warnings.append(f"SNR below {QC_SNR_WARNING_DB:g} dB")
    if best_correlation < QC_CORRELATION_WARNING:
        warnings.append(f"sign-reversed correlation below {QC_CORRELATION_WARNING:.2f}")
    if not polarity_reversed:
        warnings.append("first peak/trough polarity is not reversed")
    disagreement_limit = max(QC_PICK_DISAGREEMENT_WARNING_MS, 2.0 * sample_interval)
    if peak_disagreement is not None and peak_disagreement > disagreement_limit:
        warnings.append(f"first peak/trough disagreement exceeds {disagreement_limit:g} ms")
    if zero_cross_disagreement is not None and zero_cross_disagreement > disagreement_limit:
        warnings.append(f"experimental zero-cross disagreement exceeds {disagreement_limit:g} ms")
    if max_peak_disagreement is not None and max_peak_disagreement > disagreement_limit:
        warnings.append(f"experimental max-peak disagreement exceeds {disagreement_limit:g} ms")
    if clipped_left or clipped_right:
        warnings.append("possible clipping or flat-topped extreme")
    if constant_left or constant_right:
        warnings.append("constant or near-constant trace")
    if not interval_consistent:
        warnings.append("sample interval varies by more than 1%")

    return WaveformQcMetrics(
        sample_interval_ms=sample_interval,
        sample_interval_deviation_pct=interval_deviation_pct,
        sample_interval_consistent=interval_consistent,
        noise_rms_left=noise_rms_left,
        noise_rms_right=noise_rms_right,
        snr_left_db=snr_left,
        snr_right_db=snr_right,
        sign_reversed_correlation=best_correlation,
        correlation_lag_ms=best_lag * sample_interval,
        polarity_reversed=polarity_reversed,
        first_peak_disagreement_ms=peak_disagreement,
        zero_cross_disagreement_ms=zero_cross_disagreement,
        max_peak_disagreement_ms=max_peak_disagreement,
        clipped_left=clipped_left,
        clipped_right=clipped_right,
        constant_left=constant_left,
        constant_right=constant_right,
        warnings=tuple(warnings),
    )


def suggest_pick_uncertainty_ms(record: WaveformRecord) -> tuple[float, str]:
    """Return a transparent one-sigma arrival uncertainty suggestion from QC.

    The half-sample interval is the lower bound.  Opposing-trace disagreement,
    correlation lag, low SNR, weak reversed-pair correlation, clipping, and
    constant traces can only increase the suggestion.  The analyst may replace
    the result in the picker; the basis remains in the project audit trail.
    """

    qc = calculate_waveform_qc(record)
    base = 0.5 * qc.sample_interval_ms
    candidates: list[tuple[float, str]] = [(base, "half sample interval")]
    disagreements = [
        value
        for value in (
            qc.first_peak_disagreement_ms,
            qc.zero_cross_disagreement_ms,
            qc.max_peak_disagreement_ms,
        )
        if value is not None and math.isfinite(value)
    ]
    if disagreements:
        candidates.append((0.5 * max(disagreements), "half the maximum opposing-trace pick disagreement"))
    candidates.append((0.5 * abs(qc.correlation_lag_ms), "half the reversed-pair correlation lag"))
    finite_snr = [value for value in (qc.snr_left_db, qc.snr_right_db) if math.isfinite(value)]
    if finite_snr and min(finite_snr) < QC_SNR_WARNING_DB:
        multiplier = min(3.0, 1.0 + (QC_SNR_WARNING_DB - min(finite_snr)) / QC_SNR_WARNING_DB)
        candidates.append((base * multiplier, "low-SNR allowance"))
    if qc.sign_reversed_correlation < QC_CORRELATION_WARNING:
        deficit = QC_CORRELATION_WARNING - qc.sign_reversed_correlation
        multiplier = min(3.0, 1.0 + 2.0 * deficit / max(QC_CORRELATION_WARNING, 1.0e-6))
        candidates.append((base * multiplier, "weak reversed-pair correlation allowance"))
    if qc.clipped_left or qc.clipped_right:
        candidates.append((2.0 * qc.sample_interval_ms, "clipped-trace allowance"))
    if qc.constant_left or qc.constant_right:
        candidates.append((3.0 * qc.sample_interval_ms, "constant-trace allowance"))
    value, controlling_basis = max(candidates, key=lambda item: item[0])
    return float(max(value, np.finfo(float).eps)), (
        f"Automatic QC suggestion controlled by {controlling_basis}; interpreted as one standard deviation."
    )
