from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_authentication import (  # noqa: E402
    ScoreRecord,
    bootstrap_metric_intervals,
    build_curve,
    compute_auc,
    compute_eer,
    compute_score_distribution,
    enforce_session_separation,
    evaluate_conditions,
    load_condition_metadata,
    main,
    write_det_plot,
)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def score_record(score: float, is_genuine: bool, index: int) -> ScoreRecord:
    subject_id = "genuine" if is_genuine else f"impostor-{index}"
    return ScoreRecord(
        query_id=f"query-{index}",
        template_id=f"template-{index}",
        query_subject_id=subject_id,
        template_subject_id="genuine",
        query_session_id=f"query-session-{index}",
        template_session_id=f"template-session-{index}",
        is_genuine=is_genuine,
        fused_score=score,
        source_row_number=index + 2,
    )


class ScoreDistributionTest(unittest.TestCase):
    def test_computes_population_statistics_and_d_prime(self) -> None:
        records = [
            score_record(0.8, True, 0),
            score_record(1.0, True, 1),
            score_record(0.2, False, 2),
            score_record(0.4, False, 3),
        ]

        result = compute_score_distribution(records)

        self.assertAlmostEqual(result.genuine_mean, 0.9)
        self.assertAlmostEqual(result.genuine_std, 0.1)
        self.assertAlmostEqual(result.impostor_mean, 0.3)
        self.assertAlmostEqual(result.impostor_std, 0.1)
        self.assertAlmostEqual(result.d_prime, 6.0)

    def test_marks_d_prime_undefined_when_both_variances_are_zero(self) -> None:
        records = [score_record(0.9, True, 0), score_record(0.1, False, 1)]

        result = compute_score_distribution(records)

        self.assertIsNone(result.d_prime)


class AuthenticationCurveTest(unittest.TestCase):
    def test_computes_eer_and_auc_for_known_scores(self) -> None:
        records = [
            score_record(0.8, True, 0),
            score_record(0.4, True, 1),
            score_record(0.6, False, 2),
            score_record(0.2, False, 3),
        ]

        points = build_curve(records)

        self.assertAlmostEqual(compute_eer(points).eer, 0.5)
        self.assertAlmostEqual(compute_auc(points), 0.75)

    def test_excludes_only_same_session_genuine_records(self) -> None:
        excluded_genuine = replace(
            score_record(0.9, True, 0),
            query_session_id="same-session",
            template_session_id="same-session",
        )
        kept_genuine = score_record(0.8, True, 1)
        same_session_impostor = replace(
            score_record(0.1, False, 2),
            query_session_id="same-session",
            template_session_id="same-session",
        )

        kept, excluded_count = enforce_session_separation(
            [excluded_genuine, kept_genuine, same_session_impostor]
        )

        self.assertEqual(kept, [kept_genuine, same_session_impostor])
        self.assertEqual(excluded_count, 1)

    def test_bootstrap_is_reproducible_and_returns_ordered_percentiles(self) -> None:
        records = [
            score_record(0.9, True, 0),
            score_record(0.7, True, 1),
            score_record(0.6, True, 2),
            score_record(0.5, False, 3),
            score_record(0.3, False, 4),
            score_record(0.1, False, 5),
        ]

        first = bootstrap_metric_intervals(records, samples=200, seed=17)
        second = bootstrap_metric_intervals(records, samples=200, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first.samples, 200)
        self.assertEqual(first.seed, 17)
        self.assertEqual({interval.metric for interval in first.intervals}, {"eer", "roc_auc", "d_prime"})
        for interval in first.intervals:
            self.assertLessEqual(interval.lower, interval.upper)

    def test_bootstrap_omits_d_prime_when_the_point_estimate_is_undefined(self) -> None:
        records = [score_record(0.9, True, 0), score_record(0.1, False, 1)]

        result = bootstrap_metric_intervals(records, samples=20, seed=3)

        self.assertEqual({interval.metric for interval in result.intervals}, {"eer", "roc_auc"})

    def test_writes_det_plot_from_finite_probit_coordinates(self) -> None:
        records = [
            score_record(0.8, True, 0),
            score_record(0.4, True, 1),
            score_record(0.6, False, 2),
            score_record(0.2, False, 3),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "det.png"

            write_det_plot(output, build_curve(records))

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


class ConditionEvaluationTest(unittest.TestCase):
    def test_joins_subject_and_session_and_evaluates_each_value(self) -> None:
        records = [
            score_record(0.9, True, 0),
            score_record(0.8, False, 1),
            score_record(0.7, True, 2),
            score_record(0.2, False, 3),
        ]
        records = [
            replace(record, query_subject_id=f"subject-{index}", query_session_id="session")
            for index, record in enumerate(records)
        ]
        metadata = {
            ("subject-0", "session"): {"lighting": "normal"},
            ("subject-1", "session"): {"lighting": "normal"},
            ("subject-2", "session"): {"lighting": "reflection"},
            ("subject-3", "session"): {"lighting": "reflection"},
        }

        results = evaluate_conditions(records, metadata, ("lighting",))

        self.assertEqual([(result.value, result.status) for result in results], [("normal", "ok"), ("reflection", "ok")])
        self.assertEqual([result.total_records for result in results], [2, 2])

    def test_condition_metadata_rejects_conflicting_duplicate_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conditions.csv"
            path.write_text(
                "subject_id,session_id,lighting\n"
                "p1,s1,normal\n"
                "p1,s1,reflection\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "conflicting condition metadata"):
                load_condition_metadata(path, "subject_id", "session_id", ("lighting",))

    def test_condition_evaluation_rejects_missing_query_metadata(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "condition metadata is missing"):
            evaluate_conditions([score_record(0.9, True, 0)], {}, ("lighting",))


class AuthenticationOutputTest(unittest.TestCase):
    def test_main_writes_bootstrap_det_and_anonymous_condition_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "evaluation"
            args = SimpleNamespace(
                scores_csv=FIXTURES_DIR / "auth_scores_smoke.csv",
                output_dir=output_dir,
                bootstrap_samples=50,
                bootstrap_seed=42,
                confidence_level=0.95,
                conditions_csv=FIXTURES_DIR / "auth_conditions_smoke.csv",
                condition_column=["lighting", "oral_condition"],
                condition_subject_column="subject_id",
                condition_session_column="session_id",
            )

            with mock.patch("evaluate_authentication.parse_args", return_value=args):
                self.assertEqual(main(), 0)

            for name in (
                "bootstrap_intervals.csv",
                "condition_metrics.csv",
                "det_curve.png",
                "summary.json",
            ):
                self.assertGreater((output_dir / name).stat().st_size, 0)
            public_summary = (output_dir / "summary.json").read_text(encoding="utf-8")
            self.assertNotIn('"p1"', public_summary)
            self.assertNotIn('"c2"', public_summary)


if __name__ == "__main__":
    unittest.main()
