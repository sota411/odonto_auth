from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_rendered_matching  # noqa: E402


FDI_LABELS = (11, 12, 13, 21, 22, 23)


class RenderedMatchingEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.render_root = self.root / "rendered"
        self.output_dir = self.root / "evaluation"
        self.render_root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            render_root=self.render_root,
            output_dir=self.output_dir,
            crop_size=64,
            crop_padding=0.12,
            min_common_teeth=1,
        )

    def write_view(
        self,
        case_id: str,
        view_id: str,
        *,
        patient_id: str | None = None,
        variant: int = 0,
    ) -> dict[str, str]:
        image_relative = Path("images") / case_id / f"{view_id}.png"
        label_relative = Path("labels") / case_id / f"{view_id}.png"
        image_path = self.render_root / image_relative
        label_path = self.render_root / label_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = 96, 152
        rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        labels = np.zeros((height, width), dtype=np.uint8)
        case_variant = sum(case_id.encode("utf-8")) % 5
        for index, fdi_label in enumerate(FDI_LABELS):
            x0 = 5 + index * 24
            tooth_width = 11 + (index + case_variant) % 4
            y0 = 12 + (index * 3 + variant) % 8
            y1 = 82 - (index + case_variant) % 6
            labels[y0:y1, x0 : x0 + tooth_width] = fdi_label
            if case_variant % 2:
                labels[y0 + 8 : y0 + 16, x0 + tooth_width // 2 : x0 + tooth_width] = 0
            mask = labels == fdi_label
            rgb[mask] = (
                30 + index * 20,
                60 + case_variant * 25,
                90 + variant * 15,
            )
            stripe = mask & ((np.indices(mask.shape)[0] + variant + index) % 7 == 0)
            rgb[stripe] = (220, 210 - case_variant * 10, 80 + index * 10)

        Image.fromarray(rgb, mode="RGB").save(image_path)
        Image.fromarray(labels, mode="L").save(label_path)
        digest = (f"{case_variant + 1:x}" * 64)[:64]
        return {
            "patient_id": patient_id or case_id,
            "case_id": case_id,
            "jaw": "upper",
            "view_id": view_id,
            "azimuth_deg": str(variant * 15.0),
            "elevation_deg": str(variant * 3.0),
            "camera_position": "[0.0,-10.0,0.0]",
            "focal_point": "[0.0,0.0,0.0]",
            "view_up": "[0.0,0.0,1.0]",
            "parallel_scale": "10.0",
            "image_width": str(width),
            "image_height": str(height),
            "image_path": image_relative.as_posix(),
            "label_path": label_relative.as_posix(),
            "source_path": f"/synthetic/{case_id}.npz",
            "source_sha256": digest,
        }

    def valid_rows(self) -> list[dict[str, str]]:
        return [
            self.write_view("case-b", "right", variant=1),
            self.write_view("case-a", "right", variant=1),
            self.write_view("case-b", "front", variant=0),
            self.write_view("case-a", "front", variant=0),
        ]

    def write_manifest(self, rows: list[dict[str, str]]) -> None:
        with (self.render_root / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=evaluate_rendered_matching.REQUIRED_MANIFEST_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_scores_all_deterministic_pairs_and_writes_metrics_atomically(self) -> None:
        self.write_manifest(self.valid_rows())

        with mock.patch(
            "evaluate_rendered_matching.score_pair",
            wraps=evaluate_rendered_matching.score_pair,
        ) as score_pair_spy:
            summary = evaluate_rendered_matching.run(self.args())

        self.assertEqual(score_pair_spy.call_count, 6)
        self.assertEqual(summary["manifest_rows"], 4)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(
            summary["pair_counts"]["generated"],
            {"total": 6, "genuine": 2, "impostor": 4},
        )
        self.assertEqual(summary["pair_counts"]["scored"], summary["pair_counts"]["generated"])
        self.assertEqual(
            summary["pair_counts"]["skipped"],
            {"total": 0, "genuine": 0, "impostor": 0},
        )
        self.assertTrue(math.isfinite(summary["metrics"]["fused"]["roc_auc"]))
        self.assertEqual(
            tuple(summary["metrics"]["per_tooth"]),
            evaluate_rendered_matching.TOOTH_NAMES,
        )

        expected_files = {
            "metrics.csv",
            "score_distribution.png",
            "scores.csv",
            "skipped_pairs.csv",
            "summary.json",
        }
        self.assertEqual({path.name for path in self.output_dir.iterdir()}, expected_files)
        self.assertGreater((self.output_dir / "score_distribution.png").stat().st_size, 0)

        with (self.output_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        self.assertEqual([row["scope"] for row in metrics], ["fused", *evaluate_rendered_matching.TOOTH_NAMES])
        self.assertEqual({row["feature"] for row in metrics}, {"hog"})

        with (self.output_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
            scores = list(csv.DictReader(handle))
        self.assertEqual(len(scores), 6)
        self.assertEqual([row["pair_id"] for row in scores], [f"pair-{index:06d}" for index in range(1, 7)])
        self.assertEqual([row["is_genuine"] for row in scores], ["1", "0", "0", "0", "0", "1"])
        self.assertEqual(
            tuple(json.loads(scores[0]["per_tooth_scores"])),
            evaluate_rendered_matching.TOOTH_NAMES,
        )
        self.assertEqual(
            (self.output_dir / "skipped_pairs.csv").read_text(encoding="utf-8").count("\n"),
            1,
        )
        written_summary = json.loads(
            (self.output_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(written_summary, summary)

        second_output = self.root / "second-evaluation"
        second_args = self.args()
        second_args.output_dir = second_output
        evaluate_rendered_matching.run(second_args)
        self.assertEqual(
            (self.output_dir / "scores.csv").read_bytes(),
            (second_output / "scores.csv").read_bytes(),
        )

    def test_rejects_manifest_and_png_contract_violations_fail_fast(self) -> None:
        def assert_invalid(
            name: str,
            expected_message: str,
            rows: list[dict[str, str]],
        ) -> None:
            with self.subTest(name=name):
                self.write_manifest(rows)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    evaluate_rendered_matching.run(self.args())
                self.assertFalse(self.output_dir.exists())

        rows = self.valid_rows()
        rows[0]["image_path"] = "../escape.png"
        assert_invalid("path traversal", "remain within render root", rows)

        rows = self.valid_rows()
        rows.append(dict(rows[0]))
        assert_invalid("duplicate case/view", "duplicate case_id/view_id", rows)

        rows = self.valid_rows()
        rows = [
            row
            for row in rows
            if not (row["case_id"] == "case-a" and row["view_id"] == "right")
        ]
        assert_invalid("fewer than two views", "at least two views", rows)

        rows = self.valid_rows()
        label_path = self.render_root / rows[0]["label_path"]
        Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L").save(
            label_path
        )
        assert_invalid("shape mismatch", "shape mismatch", rows)

    def test_rejects_rgb_label_foreground_mismatch(self) -> None:
        rows = self.valid_rows()
        image_path = self.render_root / rows[0]["image_path"]
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        label = np.asarray(Image.open(self.render_root / rows[0]["label_path"]))
        y, x = np.argwhere(label != 0)[0]
        image[y, x] = 255
        Image.fromarray(image, mode="RGB").save(image_path)
        self.write_manifest(rows)

        with self.assertRaisesRegex(RuntimeError, "foreground mismatch"):
            evaluate_rendered_matching.run(self.args())

    def test_requires_both_genuine_and_impostor_pair_classes(self) -> None:
        rows = [
            self.write_view("only-case", "front", variant=0),
            self.write_view("only-case", "right", variant=1),
        ]
        self.write_manifest(rows)

        with self.assertRaisesRegex(RuntimeError, "both genuine and impostor"):
            evaluate_rendered_matching.run(self.args())

    def test_rejects_multiple_cases_for_one_patient(self) -> None:
        rows = self.valid_rows()
        for row in rows:
            row["patient_id"] = "patient-shared"
        self.write_manifest(rows)

        with self.assertRaisesRegex(RuntimeError, "one case per patient"):
            evaluate_rendered_matching.run(self.args())
        self.assertFalse(self.output_dir.exists())

    def test_records_pairs_skipped_for_insufficient_common_teeth(self) -> None:
        rows = self.valid_rows()
        retained_label_by_view = {
            ("case-a", "front"): 11,
            ("case-a", "right"): 12,
        }
        for row in rows:
            retained_label = retained_label_by_view.get(
                (row["case_id"], row["view_id"])
            )
            if retained_label is None:
                continue
            label_path = self.render_root / row["label_path"]
            image_path = self.render_root / row["image_path"]
            labels = np.asarray(Image.open(label_path)).copy()
            image = np.asarray(Image.open(image_path).convert("RGB")).copy()
            removed = (labels != 0) & (labels != retained_label)
            labels[removed] = 0
            image[removed] = 255
            Image.fromarray(labels, mode="L").save(label_path)
            Image.fromarray(image, mode="RGB").save(image_path)
        self.write_manifest(rows)

        summary = evaluate_rendered_matching.run(self.args())

        self.assertEqual(
            summary["pair_counts"]["scored"],
            {"total": 5, "genuine": 1, "impostor": 4},
        )
        self.assertEqual(
            summary["pair_counts"]["skipped"],
            {"total": 1, "genuine": 1, "impostor": 0},
        )
        self.assertEqual(
            summary["skipped_by_reason"], {"insufficient_common_teeth": 1}
        )
        with (self.output_dir / "skipped_pairs.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            skipped = list(csv.DictReader(handle))
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["is_genuine"], "1")
        self.assertEqual(skipped[0]["common_teeth"], "0")
        self.assertEqual(skipped[0]["reason"], "insufficient_common_teeth")

    def test_rejects_an_output_directory_that_would_replace_the_inputs(self) -> None:
        self.write_manifest(self.valid_rows())
        args = self.args()
        args.output_dir = self.render_root

        with self.assertRaisesRegex(RuntimeError, "must not be the render root"):
            evaluate_rendered_matching.run(args)

        self.assertTrue((self.render_root / "manifest.csv").is_file())
        self.assertTrue((self.render_root / "images").is_dir())

    def test_preserves_previous_output_when_atomic_generation_fails(self) -> None:
        self.write_manifest(self.valid_rows())
        self.output_dir.mkdir()
        stable_path = self.output_dir / "stable.txt"
        stable_path.write_text("previous", encoding="utf-8")

        with mock.patch(
            "evaluate_rendered_matching.write_score_distribution_plot",
            side_effect=RuntimeError("injected plot failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected plot failure"):
                evaluate_rendered_matching.run(self.args())

        self.assertEqual(stable_path.read_text(encoding="utf-8"), "previous")
        self.assertEqual(list(self.output_dir.iterdir()), [stable_path])
        self.assertEqual(list(self.root.glob(".evaluation.generation_*")), [])


if __name__ == "__main__":
    unittest.main()
