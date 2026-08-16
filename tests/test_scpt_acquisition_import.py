"""Regression tests for SCPT acquisition sounding imports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raypath_core import ScptAcquisitionFormatError, parse_scpt_acquisition


class ScptAcquisitionImportTests(unittest.TestCase):
    def _write_sounding(
        self,
        root: Path,
        rows: list[dict[str, object]],
        *,
        sample_rate_hz: int = 1000,
    ) -> None:
        (root / "sounding.json").write_text(
            json.dumps(
                {
                    "schema": "scpt-acquisition",
                    "schema_version": 1,
                    "sample_rate_hz": sample_rate_hz,
                }
            ),
            encoding="utf-8",
        )
        with (root / "stacked_waveforms.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "depth_m",
                    "wave_type",
                    "sample_index",
                    "time_from_trigger_s",
                    "sum_counts",
                    "stack_count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _paired_rows(depth_m: float = 0.5) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for wave_type, values in (
            ("shear_left", (1.5, 4.5, 7.5, 3.5)),
            ("shear_right", (-2.5, -5.5, -8.5, -4.5)),
            ("p_wave", (10.0, 11.0, 12.0, 13.0)),
        ):
            for index, (time_s, value) in enumerate(zip((-0.001, 0.0, 0.001, 0.002), values)):
                rows.append(
                    {
                        "depth_m": depth_m,
                        "wave_type": wave_type,
                        "sample_index": index,
                        "time_from_trigger_s": time_s,
                        "sum_counts": value,
                        "stack_count": 3,
                    }
                )
        return rows

    def test_imports_paired_direct_sum_stacks_on_trigger_relative_clock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sounding(root, self._paired_rows())
            messages: list[str] = []

            records = parse_scpt_acquisition(root, import_messages=messages)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].depth_m, 0.5)
        self.assertEqual(records[0].pre_trigger_ms, 0.0)
        np.testing.assert_allclose(records[0].time_ms, [-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(records[0].left, [1.5, 4.5, 7.5, 3.5])
        np.testing.assert_allclose(records[0].right, [-2.5, -5.5, -8.5, -4.5])
        self.assertTrue(any("P-wave" in message for message in messages))
        self.assertTrue(any("without normalization" in message for message in messages))

    def test_accepts_sounding_json_or_stacked_csv_as_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sounding(root, self._paired_rows())

            from_json = parse_scpt_acquisition(root / "sounding.json")
            from_csv = parse_scpt_acquisition(root / "stacked_waveforms.csv")

        np.testing.assert_array_equal(from_json[0].left, from_csv[0].left)

    def test_rejects_a_depth_without_both_opposing_shear_stacks(self) -> None:
        rows = [row for row in self._paired_rows() if row["wave_type"] != "shear_right"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sounding(root, rows)

            with self.assertRaisesRegex(ScptAcquisitionFormatError, "missing required stack"):
                parse_scpt_acquisition(root)

    def test_rejects_sample_clock_inconsistent_with_sounding_configuration(self) -> None:
        rows = self._paired_rows()
        rows[2]["time_from_trigger_s"] = 0.0015
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sounding(root, rows)

            with self.assertRaisesRegex(ScptAcquisitionFormatError, "sample times do not match"):
                parse_scpt_acquisition(root)

    def test_loads_individual_hits_and_rebuilds_stack_after_exclusion(self) -> None:
        times = (-0.001, 0.0, 0.001, 0.002)
        hit_values = {
            ("shear_left", 1): (10.0, 11.0, 14.0, 12.0),
            ("shear_left", 2): (20.0, 22.0, 26.0, 24.0),
            ("shear_right", 1): (-10.0, -11.0, -14.0, -12.0),
            ("shear_right", 2): (-20.0, -22.0, -26.0, -24.0),
        }
        rows: list[dict[str, object]] = []
        for wave_type in ("shear_left", "shear_right"):
            corrected = [
                np.asarray(hit_values[(wave_type, hit_number)], dtype=float)
                - hit_values[(wave_type, hit_number)][0]
                for hit_number in (1, 2)
            ]
            stack = np.sum(np.stack(corrected), axis=0)
            for index, (time_s, value) in enumerate(zip(times, stack)):
                rows.append(
                    {
                        "depth_m": 1.0,
                        "wave_type": wave_type,
                        "sample_index": index,
                        "time_from_trigger_s": time_s,
                        "sum_counts": value,
                        "stack_count": 2,
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sounding(root, rows)
            depth_root = root / "depths" / "001.000m"
            hit_payloads: dict[str, list[dict[str, object]]] = {}
            for wave_type in ("shear_left", "shear_right"):
                group_root = depth_root / wave_type
                group_root.mkdir(parents=True)
                hit_payloads[wave_type] = []
                for hit_number in (1, 2):
                    raw_file = f"depths/001.000m/{wave_type}/hit_{hit_number:03}.csv"
                    with (root / raw_file).open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(
                            [
                                "sample_index",
                                "time_from_trigger_s",
                                "x_raw_u16",
                                "x_counts",
                                "y_raw_u16",
                                "y_counts",
                                "trigger_marker",
                            ]
                        )
                        for index, (time_s, value) in enumerate(
                            zip(times, hit_values[(wave_type, hit_number)])
                        ):
                            writer.writerow((index, time_s, 0, 0, 0, value, int(time_s == 0.0)))
                    hit_payloads[wave_type].append(
                        {
                            "hit_number": hit_number,
                            "selected_channel": "y",
                            "raw_file": raw_file,
                            "status": "accepted",
                            "gain_1": 4,
                            "gain_2": 2,
                            "correlation": (
                                None
                                if hit_number == 1
                                else {"coefficient": 0.9, "lag_samples": 0}
                            ),
                            "metrics": {"y": {"signal_to_noise_ratio": 12.0}},
                        }
                    )
            (depth_root / "depth.json").write_text(
                json.dumps(
                    {
                        "groups": {
                            wave_type: {"hits": hit_payloads[wave_type]}
                            for wave_type in ("shear_left", "shear_right")
                        }
                    }
                ),
                encoding="utf-8",
            )

            records = parse_scpt_acquisition(root)

        record = records[0]
        self.assertEqual(len(record.acquisition_hits), 4)
        self.assertEqual(record.acquisition_contributor_counts(), (2, 2))
        excluded = next(
            hit
            for hit in record.acquisition_hits
            if hit.wave_type == "shear_right" and hit.hit_number == 2
        )
        excluded.included = False
        record.rebuild_acquisition_stacks()
        self.assertEqual(record.acquisition_contributor_counts(), (2, 1))
        np.testing.assert_allclose(record.right, [0.0, -1.0, -4.0, -2.0])


if __name__ == "__main__":
    unittest.main()
