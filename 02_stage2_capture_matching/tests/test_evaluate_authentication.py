from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_authentication import (  # noqa: E402
    ScoreRecord,
    build_curve,
    compute_auc,
    compute_eer,
    compute_score_distribution,
    enforce_session_separation,
)


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


if __name__ == "__main__":
    unittest.main()
