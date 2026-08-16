"""Desktop project round-trip coverage for acquisition sounding sources."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from raypath_core import add_suggested_picks, parse_scpt_acquisition
from raypath_scpt import RayPathMainWindow, WaveformPickerDialog


class ScptAcquisitionProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_sounding(root: Path) -> None:
        (root / "sounding.json").write_text(
            json.dumps(
                {
                    "schema": "scpt-acquisition",
                    "schema_version": 1,
                    "sample_rate_hz": 1000,
                }
            ),
            encoding="utf-8",
        )
        with (root / "stacked_waveforms.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "depth_m",
                    "wave_type",
                    "sample_index",
                    "time_from_trigger_s",
                    "sum_counts",
                    "stack_count",
                ]
            )
            times = (-0.002, -0.001, 0.0, 0.001, 0.002, 0.003)
            for wave_type, values in (
                ("shear_left", (0.0, 0.0, 1.0, 4.0, 1.5, 0.2)),
                ("shear_right", (0.0, 0.0, -1.0, -4.0, -1.5, -0.2)),
            ):
                for index, (time_s, value) in enumerate(zip(times, values)):
                    writer.writerow((1.0, wave_type, index, time_s, value, 2))
        depth_root = root / "depths" / "001.000m"
        groups: dict[str, dict[str, object]] = {}
        for wave_type, stack_values in (
            ("shear_left", (0.0, 0.0, 1.0, 4.0, 1.5, 0.2)),
            ("shear_right", (0.0, 0.0, -1.0, -4.0, -1.5, -0.2)),
        ):
            group_root = depth_root / wave_type
            group_root.mkdir(parents=True)
            hits = []
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
                    for index, (time_s, stack_value) in enumerate(zip(times, stack_values)):
                        writer.writerow((index, time_s, 0, 0, 0, 10.0 + stack_value / 2.0, int(time_s == 0.0)))
                hits.append(
                    {
                        "hit_number": hit_number,
                        "selected_channel": "y",
                        "raw_file": raw_file,
                        "status": "accepted",
                        "correlation": None if hit_number == 1 else {"coefficient": 1.0, "lag_samples": 0},
                        "metrics": {"y": {"signal_to_noise_ratio": 20.0}},
                    }
                )
            groups[wave_type] = {"hits": hits}
        (depth_root / "depth.json").write_text(json.dumps({"groups": groups}), encoding="utf-8")

    def test_saved_project_reloads_acquisition_directory(self) -> None:
        source = RayPathMainWindow()
        loaded = RayPathMainWindow()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "SCPT_SAMPLE"
                root.mkdir()
                self._write_sounding(root)
                source.waveform_records = parse_scpt_acquisition(root)
                excluded = next(
                    hit
                    for hit in source.waveform_records[0].acquisition_hits
                    if hit.wave_type == "shear_right" and hit.hit_number == 2
                )
                excluded.included = False
                source.waveform_records[0].rebuild_acquisition_stacks()
                add_suggested_picks(source.waveform_records)
                source.gru_path = root
                source.gru_pre_trigger_ms = 0.0
                source.waveform_source_format = "scpt_acquisition"
                source._populate_table_from_picks()
                project_path = Path(directory) / "acquisition_round_trip.rpscpt"
                source.project_path = project_path

                self.assertTrue(source.save_project())
                payload = json.loads(project_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["waveform_source_format"], "scpt_acquisition")
                self.assertEqual(payload["timing_audit"]["correction"], "none; acquisition source clock is already trigger-relative")
                self.assertFalse(payload["picks"][0]["acquisition_hit_review"][3]["included"])

                loaded._load_project(project_path)

                self.assertEqual(loaded.waveform_source_format, "scpt_acquisition")
                self.assertEqual(loaded.gru_path, root)
                self.assertEqual(len(loaded.waveform_records), 1)
                self.assertEqual(loaded.waveform_records[0].acquisition_contributor_counts(), (2, 1))
                np.testing.assert_array_equal(
                    loaded.waveform_records[0].left,
                    source.waveform_records[0].left,
                )
                self.assertEqual(
                    loaded.waveform_records[0].picks_ms,
                    source.waveform_records[0].picks_ms,
                )
        finally:
            source._set_dirty(False)
            loaded._set_dirty(False)
            source.close()
            loaded.close()

    def test_picker_excludes_hit_and_rebuilds_visible_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "SCPT_SAMPLE"
            root.mkdir()
            self._write_sounding(root)
            records = parse_scpt_acquisition(root)
            add_suggested_picks(records)
            original_right = records[0].right.copy()
            dialog = WaveformPickerDialog(records, root.name)
            try:
                self.assertTrue(dialog.hit_review_box.isVisibleTo(dialog))
                self.assertEqual(dialog.hit_table.rowCount(), 4)

                dialog.hit_table.item(3, 0).setCheckState(Qt.CheckState.Unchecked)

                self.assertEqual(records[0].acquisition_contributor_counts(), (2, 1))
                self.assertEqual(records[0].review_state, "not_reviewed")
                self.assertFalse(np.array_equal(records[0].right, original_right))
                self.assertIn("Right 1/2", dialog.hit_review_box.title())
            finally:
                dialog.close()


if __name__ == "__main__":
    unittest.main()
