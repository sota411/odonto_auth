from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quality_gate import (  # noqa: E402
    QualityThresholds,
    evaluate_quality,
    load_thresholds,
    measure_image_quality,
)


def thresholds() -> QualityThresholds:
    return QualityThresholds(
        min_detected_teeth=4,
        min_mean_detection_confidence=0.25,
        min_laplacian_variance=20.0,
        min_mean_luminance=40.0,
        max_mean_luminance=220.0,
        max_dark_pixel_fraction=0.25,
        max_bright_pixel_fraction=0.25,
        dark_pixel_threshold=10,
        bright_pixel_threshold=245,
    )


class QualityGateTest(unittest.TestCase):
    def test_rejects_an_uncalibrated_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "quality_gate.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "uncalibrated",
                        "thresholds": {
                            "min_detected_teeth": 4,
                            "min_mean_detection_confidence": 0.25,
                            "min_laplacian_variance": 20.0,
                            "min_mean_luminance": 40.0,
                            "max_mean_luminance": 220.0,
                            "max_dark_pixel_fraction": 0.25,
                            "max_bright_pixel_fraction": 0.25,
                            "dark_pixel_threshold": 10,
                            "bright_pixel_threshold": 245,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "not calibrated"):
                load_thresholds(path)

    def test_rejects_non_integer_threshold_fields(self) -> None:
        invalid = thresholds().__dict__ | {"min_detected_teeth": 4.5}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "quality_gate.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "calibrated",
                        "thresholds": invalid,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "min_detected_teeth must be an integer"):
                load_thresholds(path)

    def test_measures_image_and_accepts_a_sharp_well_exposed_input(self) -> None:
        checkerboard = np.indices((64, 64)).sum(axis=0) % 2
        grayscale = np.where(checkerboard == 0, 80, 180).astype(np.uint8)
        image = np.repeat(grayscale[:, :, None], 3, axis=2)

        measurement = measure_image_quality(
            image,
            detected_teeth=6,
            mean_detection_confidence=0.8,
            thresholds=thresholds(),
        )
        decision = evaluate_quality(measurement, thresholds())

        self.assertGreater(measurement.laplacian_variance, 20.0)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reasons, ())

    def test_reports_every_failed_quality_rule(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        measurement = measure_image_quality(
            image,
            detected_teeth=2,
            mean_detection_confidence=0.1,
            thresholds=thresholds(),
        )
        decision = evaluate_quality(measurement, thresholds())

        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.reasons,
            (
                "insufficient_detected_teeth",
                "low_detection_confidence",
                "blurry",
                "underexposed",
                "excessive_dark_clipping",
            ),
        )

    def test_rejects_invalid_detection_inputs(self) -> None:
        image = np.full((16, 16, 3), 128, dtype=np.uint8)

        with self.assertRaisesRegex(RuntimeError, "detected_teeth"):
            measure_image_quality(
                image,
                detected_teeth=7,
                mean_detection_confidence=0.5,
                thresholds=thresholds(),
            )

        with self.assertRaisesRegex(RuntimeError, "mean_detection_confidence"):
            measure_image_quality(
                image,
                detected_teeth=4,
                mean_detection_confidence=float("nan"),
                thresholds=thresholds(),
            )

    def test_rejects_an_empty_image_before_opencv(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "image must not be empty"):
            measure_image_quality(
                np.empty((0, 0, 3), dtype=np.uint8),
                detected_teeth=4,
                mean_detection_confidence=0.5,
                thresholds=thresholds(),
            )
