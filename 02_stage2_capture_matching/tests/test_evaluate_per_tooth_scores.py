from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_per_tooth_scores import (  # noqa: E402
    METRICS_COLUMNS,
    REQUIRED_SCORE_COLUMNS,
    load_score_records,
    run,
)


def score_row(
    *,
    query_subject_id: str,
    template_subject_id: str,
    query_session_id: str,
    template_session_id: str,
    is_genuine: str,
    per_tooth_scores: str,
) -> dict[str, str]:
    return {
        "query_subject_id": query_subject_id,
        "template_subject_id": template_subject_id,
        "query_session_id": query_session_id,
        "template_session_id": template_session_id,
        "is_genuine": is_genuine,
        "per_tooth_scores": per_tooth_scores,
    }


class PerToothEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.scores_csv = self.root / "scores.csv"
        self.output_dir = self.root / "per_tooth_eval"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_scores(
        self,
        rows: list[dict[str, str]],
        fieldnames: tuple[str, ...] = REQUIRED_SCORE_COLUMNS,
    ) -> None:
        with self.scores_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(scores_csv=self.scores_csv, output_dir=self.output_dir)

    def test_exports_metrics_and_explicit_statuses_after_session_filtering(self) -> None:
        self.write_scores(
            [
                score_row(
                    query_subject_id="P0",
                    template_subject_id="P0",
                    query_session_id="S0",
                    template_session_id="S0",
                    is_genuine="1",
                    per_tooth_scores='{"R1":1.0,"R2":0.9}',
                ),
                score_row(
                    query_subject_id="P1",
                    template_subject_id="P1",
                    query_session_id="S2",
                    template_session_id="S1",
                    is_genuine="1",
                    per_tooth_scores='{"R1":0.9,"R2":0.8,"L1":0.7}',
                ),
                score_row(
                    query_subject_id="P2",
                    template_subject_id="P2",
                    query_session_id="S4",
                    template_session_id="S3",
                    is_genuine="true",
                    per_tooth_scores='{"R1":0.7,"R2":0.6}',
                ),
                score_row(
                    query_subject_id="P3",
                    template_subject_id="P1",
                    query_session_id="S6",
                    template_session_id="S1",
                    is_genuine="0",
                    per_tooth_scores='{"R1":0.3,"R2":0.4,"L1":0.2,"L2":0.2}',
                ),
                score_row(
                    query_subject_id="P4",
                    template_subject_id="P2",
                    query_session_id="S7",
                    template_session_id="S3",
                    is_genuine="false",
                    per_tooth_scores='{"R1":0.1,"R2":0.2,"L2":0.1}',
                ),
            ]
        )

        summary = run(self.args())

        self.assertEqual(summary["total_records"], 5)
        self.assertEqual(summary["used_records"], 4)
        self.assertEqual(summary["excluded_same_session_genuine"], 1)
        self.assertEqual(summary["evaluated_teeth"], 3)

        r1 = summary["per_tooth"]["R1"]
        self.assertEqual(r1["status"], "evaluated")
        self.assertEqual(r1["genuine_count"], 2)
        self.assertEqual(r1["impostor_count"], 2)
        self.assertEqual(r1["missing_score_records"], 0)
        self.assertAlmostEqual(r1["roc_auc"], 1.0)
        self.assertAlmostEqual(r1["genuine_mean"], 0.8)
        self.assertAlmostEqual(r1["impostor_mean"], 0.2)
        self.assertAlmostEqual(r1["d_prime"], 6.0)

        self.assertEqual(summary["per_tooth"]["L1"]["status"], "evaluated")
        self.assertIsNone(summary["per_tooth"]["L1"]["d_prime"])
        self.assertEqual(summary["per_tooth"]["L2"]["status"], "not_evaluated")
        self.assertEqual(
            summary["per_tooth"]["L2"]["reason"],
            "missing_genuine_class",
        )
        self.assertEqual(summary["per_tooth"]["L2"]["impostor_count"], 2)
        self.assertEqual(summary["per_tooth"]["R3"]["reason"], "no_scores")
        self.assertEqual(summary["per_tooth"]["L3"]["reason"], "no_scores")

        with (self.output_dir / "per_tooth_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(tuple(rows[0]), METRICS_COLUMNS)
        self.assertEqual([row["tooth"] for row in rows], ["R1", "R2", "L1"])
        self.assertEqual(rows[0]["genuine_count"], "2")
        self.assertEqual(rows[0]["impostor_count"], "2")
        self.assertEqual(rows[2]["d_prime"], "")

        written_summary = json.loads(
            (self.output_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(written_summary, summary)

    def test_reports_a_tooth_with_only_genuine_scores(self) -> None:
        self.write_scores(
            [
                score_row(
                    query_subject_id="P1",
                    template_subject_id="P1",
                    query_session_id="S2",
                    template_session_id="S1",
                    is_genuine="1",
                    per_tooth_scores='{"R1":0.9}',
                ),
                score_row(
                    query_subject_id="P2",
                    template_subject_id="P3",
                    query_session_id="S4",
                    template_session_id="S3",
                    is_genuine="0",
                    per_tooth_scores='{"R2":0.1}',
                ),
            ]
        )

        summary = run(self.args())

        self.assertEqual(summary["evaluated_teeth"], 0)
        self.assertEqual(
            summary["per_tooth"]["R1"]["reason"],
            "missing_impostor_class",
        )
        self.assertEqual(
            summary["per_tooth"]["R2"]["reason"],
            "missing_genuine_class",
        )
        self.assertEqual(
            (self.output_dir / "per_tooth_metrics.csv").read_text(encoding="utf-8"),
            ",".join(METRICS_COLUMNS) + "\n",
        )

    def test_rejects_invalid_per_tooth_json_contracts(self) -> None:
        invalid_values = {
            "invalid JSON": "{",
            "JSON object": "[]",
            "duplicate tooth": '{"R1":0.8,"R1":0.7}',
            "unknown tooth": '{"R4":0.8}',
            "non-empty": "{}",
        }
        for expected_message, value in invalid_values.items():
            with self.subTest(value=value):
                self.write_scores(
                    [
                        score_row(
                            query_subject_id="P1",
                            template_subject_id="P1",
                            query_session_id="S2",
                            template_session_id="S1",
                            is_genuine="1",
                            per_tooth_scores=value,
                        )
                    ]
                )
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    load_score_records(self.scores_csv)

    def test_rejects_non_finite_non_numeric_and_out_of_range_scores(self) -> None:
        invalid_values = [
            ("real number", '{"R1":true}'),
            ("finite", '{"R1":NaN}'),
            ("from 0 to 1", '{"R1":1.01}'),
            ("from 0 to 1", '{"R1":-0.01}'),
        ]
        for expected_message, value in invalid_values:
            with self.subTest(value=value):
                self.write_scores(
                    [
                        score_row(
                            query_subject_id="P1",
                            template_subject_id="P1",
                            query_session_id="S2",
                            template_session_id="S1",
                            is_genuine="1",
                            per_tooth_scores=value,
                        )
                    ]
                )
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    load_score_records(self.scores_csv)

    def test_rejects_label_inconsistency(self) -> None:
        invalid_labels = [
            ("P1", "P2", "1", "genuine record has different subjects"),
            ("P1", "P1", "0", "impostor record has the same subject"),
        ]
        for query_subject, template_subject, label, expected_message in invalid_labels:
            with self.subTest(label=label):
                self.write_scores(
                    [
                        score_row(
                            query_subject_id=query_subject,
                            template_subject_id=template_subject,
                            query_session_id="S2",
                            template_session_id="S1",
                            is_genuine=label,
                            per_tooth_scores='{"R1":0.9}',
                        )
                    ]
                )
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    load_score_records(self.scores_csv)

    def test_rejects_missing_columns_before_replacing_existing_output(self) -> None:
        self.output_dir.mkdir()
        sentinel = self.output_dir / "existing.txt"
        sentinel.write_text("keep\n", encoding="utf-8")
        missing_header = tuple(
            column for column in REQUIRED_SCORE_COLUMNS if column != "per_tooth_scores"
        )
        self.write_scores([], fieldnames=missing_header)

        with self.assertRaisesRegex(RuntimeError, "missing required columns"):
            run(self.args())

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
        self.assertEqual(list(self.output_dir.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
