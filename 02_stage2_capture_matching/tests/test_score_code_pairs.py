from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


TOOTH_NAMES = ("R1", "R2", "R3", "L1", "L2", "L3")
PAIR_COLUMNS = (
    "split",
    "pair_id",
    "is_genuine",
    "template_id",
    "query_id",
    "template_patient_id",
    "query_patient_id",
    "template_checkup_id",
    "query_checkup_id",
    "template_photographs",
    "query_photographs",
)
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "score_code_pairs.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))

import score_code_pairs  # noqa: E402


class ScoreCodePairsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = score_code_pairs

    def setUp(self) -> None:
        self.temp_dir_context = TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_context.name)
        self.pairs_csv = self.temp_dir / "pairs.csv"
        self.features_npz = self.temp_dir / "features.npz"
        self.output_dir = self.temp_dir / "output"
        self.images_root = self.temp_dir / "images"
        self.photo_dir = self.images_root / "Images" / "Photographs"
        self.photo_dir.mkdir(parents=True)
        self.references_by_checkup: dict[str, tuple[str, ...]] = {}

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    def base_pairs(self) -> list[dict[str, object]]:
        return [
            {
                "split": "test",
                "pair_id": "test-000001",
                "is_genuine": 1,
                "template_id": "p1:c1",
                "query_id": "p1:c2",
                "template_patient_id": "p1",
                "query_patient_id": "p1",
                "template_checkup_id": "c1",
                "query_checkup_id": "c2",
                "template_photographs": "p1-c1.jpg",
                "query_photographs": "p1-c2.jpg",
            },
            {
                "split": "test",
                "pair_id": "test-000002",
                "is_genuine": 0,
                "template_id": "p1:c1",
                "query_id": "p2:c3",
                "template_patient_id": "p1",
                "query_patient_id": "p2",
                "template_checkup_id": "c1",
                "query_checkup_id": "c3",
                "template_photographs": "p1-c1.jpg",
                "query_photographs": "p2-c3.jpg",
            },
        ]

    def write_pairs(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            for prefix in ("template", "query"):
                checkup_id = str(row[f"{prefix}_id"])
                references = tuple(str(row[f"{prefix}_photographs"]).split("|"))
                if (
                    checkup_id in self.references_by_checkup
                    and self.references_by_checkup[checkup_id] != references
                ):
                    raise AssertionError(f"inconsistent test references for {checkup_id}")
                self.references_by_checkup[checkup_id] = references
                for reference in references:
                    path = self.photo_dir / reference
                    if not path.exists():
                        path.write_bytes(reference.encode("utf-8"))
        with self.pairs_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PAIR_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def write_features(
        self,
        checkup_ids: tuple[str, ...] = ("p1:c1", "p1:c2", "p2:c3"),
        patient_ids: tuple[str, ...] = ("p1", "p1", "p2"),
        embeddings: np.ndarray | None = None,
        present: np.ndarray | None = None,
    ) -> None:
        count = len(checkup_ids)
        if embeddings is None:
            embeddings = np.zeros((count, len(TOOTH_NAMES), 2), dtype=np.float32)
            embeddings[..., 0] = 1.0
            if "p2:c3" in checkup_ids:
                embeddings[checkup_ids.index("p2:c3"), :, :] = (0.0, 1.0)
        if present is None:
            present = np.ones((count, len(TOOTH_NAMES)), dtype=np.bool_)
        photo_manifest_json = []
        for checkup_id in checkup_ids:
            references = self.references_by_checkup[checkup_id]
            manifest = []
            for reference in references:
                content = (self.photo_dir / reference).read_bytes()
                manifest.append(
                    {
                        "reference": reference,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            photo_manifest_json.append(json.dumps(manifest, separators=(",", ":")))
        np.savez(
            self.features_npz,
            checkup_ids=np.asarray(checkup_ids, dtype="U16"),
            patient_ids=np.asarray(patient_ids, dtype="U16"),
            tooth_names=np.asarray(TOOTH_NAMES, dtype="U2"),
            embeddings=embeddings,
            present=present,
            photo_manifest_json=np.asarray(photo_manifest_json),
            ignored_metadata=np.asarray(["extra"]),
        )

    def run_cli(self, *extra_args: str) -> int:
        argv = [
            "--pairs-csv",
            str(self.pairs_csv),
            "--features-npz",
            str(self.features_npz),
            "--output-dir",
            str(self.output_dir),
            "--images-root",
            str(self.images_root),
            *extra_args,
        ]
        with redirect_stdout(io.StringIO()):
            return self.module.main(argv)

    def read_csv(self, name: str) -> list[dict[str, str]]:
        with (self.output_dir / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_writes_canonical_scores_and_summary(self) -> None:
        self.write_pairs(self.base_pairs())
        self.write_features()

        self.assertEqual(self.run_cli(), 0)

        scores = self.read_csv("scores.csv")
        self.assertEqual(len(scores), 2)
        self.assertEqual(
            list(scores[0]),
            [
                "query_id",
                "template_id",
                "query_subject_id",
                "template_subject_id",
                "query_session_id",
                "template_session_id",
                "is_genuine",
                "fused_score",
                "pair_id",
                "split",
                "common_teeth",
                "per_tooth_scores",
            ],
        )
        self.assertEqual(scores[0]["query_subject_id"], "p1")
        self.assertEqual(scores[0]["template_session_id"], "c1")
        self.assertEqual(scores[0]["common_teeth"], "6")
        self.assertEqual(set(json.loads(scores[0]["per_tooth_scores"])), set(TOOTH_NAMES))
        self.assertAlmostEqual(float(scores[0]["fused_score"]), 1.0)
        self.assertAlmostEqual(float(scores[1]["fused_score"]), 0.5)

        self.assertEqual(self.read_csv("skipped_pairs.csv"), [])
        summary = json.loads((self.output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["scored_pairs"], 2)
        self.assertEqual(summary["skipped_pairs"], 0)
        self.assertEqual(summary["genuine_pairs"], 1)
        self.assertEqual(summary["impostor_pairs"], 1)

    def test_rejects_invalid_embedding_shape(self) -> None:
        self.write_pairs(self.base_pairs())
        embeddings = np.ones((3, 5, 2), dtype=np.float32)
        self.write_features(embeddings=embeddings)

        with self.assertRaisesRegex(RuntimeError, "embeddings.*shape"):
            self.run_cli()
        self.assertFalse((self.output_dir / "scores.csv").exists())

    def test_records_pair_with_missing_features(self) -> None:
        rows = self.base_pairs()
        rows.append(
            {
                "split": "test",
                "pair_id": "test-000003",
                "is_genuine": 0,
                "template_id": "p1:c1",
                "query_id": "p3:missing",
                "template_patient_id": "p1",
                "query_patient_id": "p3",
                "template_checkup_id": "c1",
                "query_checkup_id": "missing",
                "template_photographs": "p1-c1.jpg",
                "query_photographs": "p3-missing.jpg",
            }
        )
        self.write_pairs(rows)
        self.write_features()

        self.assertEqual(self.run_cli(), 0)

        skipped = self.read_csv("skipped_pairs.csv")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["pair_id"], "test-000003")
        self.assertEqual(skipped[0]["reason"], "missing_query_feature")
        summary = json.loads((self.output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["scored_pairs"], 2)
        self.assertEqual(summary["skipped_pairs"], 1)

    def test_records_pair_with_insufficient_common_teeth(self) -> None:
        rows = self.base_pairs()
        rows.append(
            {
                "split": "test",
                "pair_id": "test-000003",
                "is_genuine": 0,
                "template_id": "p3:c4",
                "query_id": "p4:c5",
                "template_patient_id": "p3",
                "query_patient_id": "p4",
                "template_checkup_id": "c4",
                "query_checkup_id": "c5",
                "template_photographs": "p3-c4.jpg",
                "query_photographs": "p4-c5.jpg",
            }
        )
        self.write_pairs(rows)
        checkup_ids = ("p1:c1", "p1:c2", "p2:c3", "p3:c4", "p4:c5")
        patient_ids = ("p1", "p1", "p2", "p3", "p4")
        present = np.ones((5, len(TOOTH_NAMES)), dtype=np.bool_)
        present[3] = False
        present[3, 0] = True
        present[4] = False
        present[4, 1] = True
        self.write_features(checkup_ids, patient_ids, present=present)

        self.assertEqual(self.run_cli(), 0)

        skipped = self.read_csv("skipped_pairs.csv")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "insufficient_common_teeth")
        self.assertEqual(skipped[0]["common_teeth"], "0")

    def test_records_pair_with_shared_photo_content(self) -> None:
        rows = self.base_pairs()
        rows.append(
            {
                "split": "test",
                "pair_id": "test-000003",
                "is_genuine": 0,
                "template_id": "p3:c4",
                "query_id": "p4:c5",
                "template_patient_id": "p3",
                "query_patient_id": "p4",
                "template_checkup_id": "c4",
                "query_checkup_id": "c5",
                "template_photographs": "p3-c4.jpg",
                "query_photographs": "p4-c5.jpg",
            }
        )
        self.write_pairs(rows)
        duplicate_bytes = b"same-photo-content"
        (self.photo_dir / "p3-c4.jpg").write_bytes(duplicate_bytes)
        (self.photo_dir / "p4-c5.jpg").write_bytes(duplicate_bytes)
        checkup_ids = ("p1:c1", "p1:c2", "p2:c3", "p3:c4", "p4:c5")
        patient_ids = ("p1", "p1", "p2", "p3", "p4")
        self.write_features(checkup_ids, patient_ids)

        self.assertEqual(self.run_cli(), 0)

        skipped = self.read_csv("skipped_pairs.csv")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["pair_id"], "test-000003")
        self.assertEqual(skipped[0]["reason"], "shared_photo_content")
        summary = json.loads((self.output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["skipped_by_reason"], {"shared_photo_content": 1})
        self.assertEqual(
            summary["skipped_by_reason_and_label"],
            {"shared_photo_content": {"genuine": 0, "impostor": 1}},
        )

    def test_rejects_photo_changed_after_feature_extraction(self) -> None:
        self.write_pairs(self.base_pairs())
        self.write_features()
        (self.photo_dir / "p1-c2.jpg").write_bytes(b"changed-after-extraction")

        with self.assertRaisesRegex(RuntimeError, "changed since feature extraction"):
            self.run_cli()

        self.assertFalse(self.output_dir.exists())

    def test_shared_path_across_checkups_is_treated_as_shared_content(self) -> None:
        rows = self.base_pairs()
        rows[0]["query_photographs"] = "p1-c1.jpg"
        self.write_pairs(rows)
        self.write_features()

        pairs = self.module.load_selected_pairs(self.pairs_csv, "test", 0)
        features = self.module.build_feature_store(self.features_npz)
        content_hashes = self.module.verify_feature_photo_manifests(
            pairs,
            features,
            self.images_root,
        )
        scored, skipped = self.module.score_selected_pairs(pairs, features, content_hashes, 1)

        self.assertEqual(len(scored), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, "shared_photo_content")

    def test_rejects_output_when_only_one_label_is_scored(self) -> None:
        self.write_pairs(self.base_pairs())
        self.write_features(checkup_ids=("p1:c1", "p1:c2"), patient_ids=("p1", "p1"))

        with self.assertRaisesRegex(RuntimeError, "no impostor pairs were scored"):
            self.run_cli()

        self.assertFalse((self.output_dir / "scores.csv").exists())
        self.assertFalse((self.output_dir / "skipped_pairs.csv").exists())
        self.assertFalse((self.output_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
