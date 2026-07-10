from __future__ import annotations

import csv
import io
import json
import random
import subprocess
import sys
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_code_pair_manifest  # noqa: E402
import prepare_code_annotation_batch  # noqa: E402


CHECKUP_COLUMNS = (
    "split",
    "patient_id",
    "checkup_id",
    "checkup_uid",
    "photograph_count",
    "photographs",
    "source_row_number",
)
EXPECTED_NAMES = (
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
)


class PrepareCodeAnnotationBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkups_csv = self.root / "checkups.csv"
        self.source_summary = self.root / "summary.json"
        self.images_root = self.root / "COde"
        self.photo_root = self.images_root / "Images" / "Photographs"
        self.photo_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_checkups(self, rows: list[dict[str, object]]) -> None:
        with self.checkups_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CHECKUP_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self.write_source_summary()

    def write_source_summary(
        self,
        *,
        seed: int = 42,
        checkups_csv_sha256: str | None = None,
    ) -> None:
        if checkups_csv_sha256 is None:
            checkups_csv_sha256 = prepare_code_annotation_batch.sha256_file(self.checkups_csv)
        self.source_summary.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
                    "checkups_csv_sha256": checkups_csv_sha256,
                }
            ),
            encoding="utf-8",
        )

    def row(
        self,
        split: str,
        patient_id: str,
        checkup_id: str,
        references: tuple[str, ...],
        source_row_number: int,
    ) -> dict[str, object]:
        for reference in references:
            path = self.photo_root / reference
            if not path.exists():
                path.write_bytes(f"content:{reference}".encode())
        return {
            "split": split,
            "patient_id": patient_id,
            "checkup_id": checkup_id,
            "checkup_uid": f"{patient_id}:{checkup_id}",
            "photograph_count": len(references),
            "photographs": "|".join(references),
            "source_row_number": source_row_number,
        }

    def base_rows(self) -> list[dict[str, object]]:
        patient_ids = [f"patient-secret-{index:02d}" for index in range(1, 15)]
        split_by_patient = build_code_pair_manifest.split_patients(
            patient_ids,
            train_ratio=0.70,
            val_ratio=0.15,
            rng=random.Random(42),
        )
        return [
            self.row(
                split_by_patient[patient_id],
                patient_id,
                f"checkup-{index:02d}",
                (f"secret-{index:02d}-1.jpg", f"secret-{index:02d}-2.jpg"),
                index + 1,
            )
            for index, patient_id in enumerate(patient_ids, start=1)
        ]

    def run_cli(self, output_dir: Path, *extra_args: str) -> int:
        argv = [
            "--checkups-csv",
            str(self.checkups_csv),
            "--source-summary",
            str(self.source_summary),
            "--images-root",
            str(self.images_root),
            "--output-dir",
            str(output_dir),
            "--train-checkups",
            "2",
            "--val-checkups",
            "1",
            "--photos-per-checkup",
            "2",
            "--seed",
            "42",
            *extra_args,
        ]
        with redirect_stdout(io.StringIO()):
            return prepare_code_annotation_batch.main(argv)

    def read_manifest(self, output_dir: Path) -> list[dict[str, str]]:
        with (output_dir / "annotation_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            return list(csv.DictReader(handle))

    def test_writes_a_deterministic_patient_separated_cvat_batch(self) -> None:
        self.write_checkups(self.base_rows())
        first_output = self.root / "first"
        second_output = self.root / "second"

        self.assertEqual(self.run_cli(first_output), 0)
        self.assertEqual(self.run_cli(second_output), 0)

        self.assertEqual(
            (first_output / "annotation_manifest.csv").read_bytes(),
            (second_output / "annotation_manifest.csv").read_bytes(),
        )
        for split in ("train", "val"):
            self.assertEqual(
                (first_output / f"cvat_{split}_images.zip").read_bytes(),
                (second_output / f"cvat_{split}_images.zip").read_bytes(),
            )
        self.assertFalse((first_output / "cvat_images.zip").exists())
        manifest = self.read_manifest(first_output)
        self.assertEqual(len(manifest), 6)
        train_patients = {row["source_patient_id"] for row in manifest if row["split"] == "train"}
        val_patients = {row["source_patient_id"] for row in manifest if row["split"] == "val"}
        self.assertEqual(len(train_patients), 2)
        self.assertEqual(len(val_patients), 1)
        self.assertTrue(train_patients.isdisjoint(val_patients))
        self.assertTrue(all(row["annotation_status"] == "pending" for row in manifest))
        self.assertEqual(
            len({(row["source_patient_id"], row["source_checkup_id"]) for row in manifest}),
            3,
        )

        labels = json.loads((first_output / "cvat_labels.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(label["name"] for label in labels), EXPECTED_NAMES)
        self.assertTrue(all(label["type"] == "polygon" for label in labels))
        for split, expected_count in (("train", 4), ("val", 2)):
            with zipfile.ZipFile(first_output / f"cvat_{split}_images.zip") as archive:
                names = archive.namelist()
                entries = archive.infolist()
            self.assertEqual(len(names), expected_count)
            self.assertTrue(all(name.startswith(f"images/code_{split}_") for name in names))
            self.assertFalse(any("secret" in name for name in names))
            self.assertTrue(all(entry.create_system == 3 for entry in entries))
            self.assertTrue(all(entry.compress_type == zipfile.ZIP_STORED for entry in entries))

        summary = json.loads((first_output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["selected_images"], 6)
        self.assertEqual(summary["selected_checkups"], {"train": 2, "val": 1})
        self.assertEqual(summary["selected_images_by_split"], {"train": 4, "val": 2})
        self.assertEqual(set(summary["cvat_archive_sha256_by_split"]), {"train", "val"})
        self.assertEqual(
            summary["checkups_csv_sha256"],
            prepare_code_annotation_batch.sha256_file(self.checkups_csv),
        )
        self.assertEqual(
            summary["source_summary_sha256"],
            prepare_code_annotation_batch.sha256_file(self.source_summary),
        )
        self.assertEqual(
            summary["source_split"],
            {"seed": 42, "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
        )

    def test_uses_the_documented_selection_defaults(self) -> None:
        args = prepare_code_annotation_batch.parse_args(
            [
                "--checkups-csv",
                str(self.checkups_csv),
                "--source-summary",
                str(self.source_summary),
                "--images-root",
                str(self.images_root),
            ]
        )

        self.assertEqual(args.seed, 42)
        self.assertEqual(args.expected_source_seed, 42)
        self.assertEqual(args.train_checkups, 10)
        self.assertEqual(args.val_checkups, 5)
        self.assertEqual(args.photos_per_checkup, 4)

    def test_rejects_a_patient_assigned_to_multiple_splits(self) -> None:
        rows = self.base_rows()
        train_row = next(row for row in rows if row["split"] == "train")
        rows.append(
            self.row(
                "val",
                str(train_row["patient_id"]),
                "checkup-cross-split",
                ("cross-1.jpg", "cross-2.jpg"),
                999,
            )
        )
        self.write_checkups(rows)

        with self.assertRaisesRegex(RuntimeError, "multiple splits"):
            self.run_cli(self.root / "output")

    def test_rejects_unexpected_source_split_settings(self) -> None:
        self.write_checkups(self.base_rows())
        self.write_source_summary(seed=7)

        with self.assertRaisesRegex(RuntimeError, "source split seed"):
            self.run_cli(self.root / "output")

    def test_rejects_a_split_assignment_that_does_not_match_the_source_seed(self) -> None:
        rows = self.base_rows()
        train_row = next(row for row in rows if row["split"] == "train")
        train_row["split"] = "test"
        self.write_checkups(rows)

        with self.assertRaisesRegex(RuntimeError, "patient split does not match"):
            self.run_cli(self.root / "output")

    def test_rejects_a_source_summary_for_different_checkups(self) -> None:
        self.write_checkups(self.base_rows())
        self.write_source_summary(checkups_csv_sha256="0" * 64)

        with self.assertRaisesRegex(RuntimeError, "checkups CSV SHA-256"):
            self.run_cli(self.root / "output")

    def test_skips_duplicate_photo_content(self) -> None:
        rows = self.base_rows()
        train_row = next(row for row in rows if row["split"] == "train")
        references = str(train_row["photographs"]).split("|")
        (self.photo_root / references[1]).write_bytes((self.photo_root / references[0]).read_bytes())
        replacement = "unique-train-replacement.jpg"
        (self.photo_root / replacement).write_bytes(b"unique replacement")
        train_row["photograph_count"] = 3
        train_row["photographs"] = f"{references[0]}|{references[1]}|{replacement}"
        self.write_checkups(rows)

        self.assertEqual(self.run_cli(self.root / "output"), 0)

        manifest = self.read_manifest(self.root / "output")
        hashes = [row["source_sha256"] for row in manifest]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_excludes_content_that_exists_in_the_test_split(self) -> None:
        rows = self.base_rows()
        train_references = [
            reference
            for row in rows
            if row["split"] == "train"
            for reference in str(row["photographs"]).split("|")
        ]
        test_references: list[str] = []
        for index, train_reference in enumerate(train_references, start=1):
            test_reference = f"test-duplicate-{index}.jpg"
            (self.photo_root / test_reference).write_bytes(
                (self.photo_root / train_reference).read_bytes()
            )
            test_references.append(test_reference)
        test_row = next(row for row in rows if row["split"] == "test")
        test_row["photograph_count"] = len(test_references)
        test_row["photographs"] = "|".join(test_references)
        self.write_checkups(rows)

        with self.assertRaisesRegex(RuntimeError, "not enough eligible train"):
            self.run_cli(self.root / "output")

    def test_keeps_previous_output_when_selection_cannot_be_satisfied(self) -> None:
        self.write_checkups(self.base_rows())
        output_dir = self.root / "output"
        output_dir.mkdir()
        marker = output_dir / "previous.txt"
        marker.write_text("stable", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "not enough eligible"):
            self.run_cli(output_dir, "--train-checkups", "99")

        self.assertEqual(marker.read_text(encoding="utf-8"), "stable")
        self.assertEqual(list(output_dir.iterdir()), [marker])

    def test_rejects_a_nonignored_output_inside_the_repository(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        unsafe_output = repository_root / "annotation-output-must-not-be-tracked"

        with self.assertRaisesRegex(RuntimeError, "Git-ignored"):
            prepare_code_annotation_batch.validate_output_location(
                unsafe_output,
                repository_root,
            )

    def test_allows_an_ignored_output_inside_the_repository(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        ignored_output = (
            repository_root
            / "01_stage1_real_image_extraction"
            / "datasets"
            / "dataset_real"
            / "test-output"
        )

        prepare_code_annotation_batch.validate_output_location(
            ignored_output,
            repository_root,
        )

    def test_rejects_an_ignored_output_that_contains_tracked_files(self) -> None:
        repository_root = self.root / "repository"
        output_dir = repository_root / "private"
        output_dir.mkdir(parents=True)
        (repository_root / ".gitignore").write_text("private/\n", encoding="utf-8")
        (output_dir / "tracked.txt").write_text("tracked", encoding="utf-8")
        subprocess.run(
            ["git", "init", "--quiet", str(repository_root)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "-C", str(repository_root), "add", "-f", "private/tracked.txt"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        with self.assertRaisesRegex(RuntimeError, "tracked"):
            prepare_code_annotation_batch.validate_output_location(
                output_dir,
                repository_root,
            )


if __name__ == "__main__":
    unittest.main()
