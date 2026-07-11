from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
import struct
import sys
import unittest
import zipfile
from contextlib import chdir, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from PIL import Image
from ultralytics.data.utils import check_det_dataset


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import finalize_code_annotation_dataset  # noqa: E402


CLASS_NAMES = (
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
TARGET_CLASS_IDS = (0, 1, 2, 7, 8, 9)
MANIFEST_COLUMNS = (
    "split",
    "image_name",
    "patient_token",
    "checkup_token",
    "source_patient_id",
    "source_checkup_id",
    "source_row_number",
    "source_photo_reference",
    "source_sha256",
    "annotation_status",
    "view_tag",
    "lighting_tag",
    "oral_condition_tag",
    "notes",
)


def polygon_lines(class_ids: tuple[int, ...] = TARGET_CLASS_IDS) -> str:
    return "\n".join(
        f"{class_id} 0.100000 0.100000 0.300000 0.100000 0.200000 0.300000"
        for class_id in class_ids
    )


def image_bytes(image_name: str) -> bytes:
    digest = hashlib.sha256(image_name.encode()).digest()
    image = Image.new("RGB", (4, 4), color=tuple(digest[:3]))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FinalizeCodeAnnotationDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.batch_dir = self.root / "batch"
        self.batch_dir.mkdir()
        self.train_export = self.root / "train_export.zip"
        self.val_export = self.root / "val_export.zip"
        self.output_dir = self.root / "dataset"
        self.rows = self.base_manifest_rows()
        self.write_labels_json()
        self.write_manifest(self.rows)
        self.write_batch_zip("train", self.rows)
        self.write_batch_zip("val", self.rows)
        self.write_batch_summary()
        self.write_export(self.train_export, "train", self.rows)
        self.write_export(self.val_export, "val", self.rows)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def base_manifest_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for split in ("train", "val"):
            for index, status in enumerate(("complete", "negative", "excluded"), start=1):
                image_name = f"code_{split}_{index:04d}_01.jpg"
                content = image_bytes(image_name)
                rows.append(
                    {
                        "split": split,
                        "image_name": image_name,
                        "patient_token": f"{split}_patient_{index:04d}",
                        "checkup_token": f"{split}_checkup_{index:04d}",
                        "source_patient_id": f"private-{split}-{index}",
                        "source_checkup_id": f"private-checkup-{split}-{index}",
                        "source_row_number": str(index + 1),
                        "source_photo_reference": f"private/{image_name}",
                        "source_sha256": hashlib.sha256(content).hexdigest(),
                        "annotation_status": status,
                        "view_tag": "frontal",
                        "lighting_tag": "normal",
                        "oral_condition_tag": "none",
                        "notes": "",
                    }
                )
        return rows

    def write_labels_json(self, names: tuple[str, ...] = CLASS_NAMES) -> None:
        labels = [{"name": name, "type": "polygon", "attributes": []} for name in names]
        (self.batch_dir / "cvat_labels.json").write_text(
            json.dumps(labels),
            encoding="utf-8",
        )

    def write_manifest(self, rows: list[dict[str, str]]) -> None:
        with (self.batch_dir / "annotation_manifest.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def write_batch_summary(self) -> None:
        summary = {
            "manifest_sha256": finalize_code_annotation_dataset.sha256_file(
                self.batch_dir / "annotation_manifest.csv"
            ),
            "manifest_identity_sha256": (
                finalize_code_annotation_dataset.manifest_identity_sha256(
                    self.batch_dir / "annotation_manifest.csv"
                )
            ),
            "cvat_archive_sha256_by_split": {
                split: finalize_code_annotation_dataset.sha256_file(
                    self.batch_dir / f"cvat_{split}_images.zip"
                )
                for split in ("train", "val")
            },
        }
        (self.batch_dir / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )

    def write_zip(self, path: Path, entries: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, content in entries.items():
                archive.writestr(name, content)

    def write_batch_zip(self, split: str, rows: list[dict[str, str]]) -> None:
        entries = {
            f"images/{row['image_name']}": image_bytes(row["image_name"])
            for row in rows
            if row["split"] == split
        }
        self.write_zip(self.batch_dir / f"cvat_{split}_images.zip", entries)

    def write_export(
        self,
        path: Path,
        split: str,
        rows: list[dict[str, str]],
        *,
        label_override: dict[str, str] | None = None,
    ) -> None:
        labels = {
            row["image_name"]: polygon_lines()
            for row in rows
            if row["split"] == split and row["annotation_status"] == "complete"
        }
        if label_override is not None:
            labels.update(label_override)
        image_list = "\n".join(
            f"./images/train/{row['image_name']}" for row in rows if row["split"] == split
        )
        entries: dict[str, bytes] = {
            "data.yaml": yaml.safe_dump(
                {
                    "path": ".",
                    "train": "train.txt",
                    "names": dict(enumerate(CLASS_NAMES)),
                }
            ).encode(),
            "train.txt": image_list.encode(),
        }
        for image_name, content in labels.items():
            entries[f"labels/train/{Path(image_name).stem}.txt"] = content.encode()
        for row in rows:
            if row["split"] == split:
                entries[f"images/train/{row['image_name']}"] = image_bytes(row["image_name"])
        self.write_zip(path, entries)

    def run_cli(self, *extra_args: str) -> int:
        argv = [
            "--batch-dir",
            str(self.batch_dir),
            "--train-export",
            str(self.train_export),
            "--val-export",
            str(self.val_export),
            "--output-dir",
            str(self.output_dir),
            *extra_args,
        ]
        with redirect_stdout(io.StringIO()):
            return finalize_code_annotation_dataset.main(argv)

    def test_builds_a_split_preserving_dataset(self) -> None:
        self.assertEqual(self.run_cli(), 0)

        for split in ("train", "val"):
            image_names = sorted(path.name for path in (self.output_dir / "images" / split).iterdir())
            label_names = sorted(path.name for path in (self.output_dir / "labels" / split).iterdir())
            self.assertEqual(
                image_names,
                [f"code_{split}_0001_01.jpg", f"code_{split}_0002_01.jpg"],
            )
            self.assertEqual(
                label_names,
                [f"code_{split}_0001_01.txt", f"code_{split}_0002_01.txt"],
            )
            self.assertEqual(
                (self.output_dir / "labels" / split / f"code_{split}_0002_01.txt").read_text(),
                "",
            )

        dataset_yaml = yaml.safe_load(
            (self.output_dir / "dataset_code_real.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(dataset_yaml["train"], "images/train")
        self.assertEqual(dataset_yaml["val"], "images/val")
        self.assertEqual(dataset_yaml["path"], self.output_dir.resolve().as_posix())
        self.assertEqual(tuple(dataset_yaml["names"].values()), CLASS_NAMES)
        self.assertNotIn("test", dataset_yaml)
        with chdir(self.root):
            checked_dataset = check_det_dataset(
                self.output_dir / "dataset_code_real.yaml",
                autodownload=False,
            )
        self.assertEqual(
            Path(checked_dataset["train"]).resolve(),
            (self.output_dir / "images" / "train").resolve(),
        )
        self.assertEqual(
            Path(checked_dataset["val"]).resolve(),
            (self.output_dir / "images" / "val").resolve(),
        )

        summary = json.loads((self.output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["images_by_split"], {"train": 2, "val": 2})
        self.assertEqual(summary["negative_images_by_split"], {"train": 1, "val": 1})
        self.assertEqual(summary["excluded_images_by_split"], {"train": 1, "val": 1})
        self.assertEqual(
            summary["instances_by_class"],
            {name: 2 for name in ("R1", "R2", "R3", "L1", "L2", "L3")},
        )
        first_summary = (self.output_dir / "summary.json").read_bytes()
        first_metadata = (self.output_dir / "metadata.csv").read_bytes()

        self.assertEqual(self.run_cli(), 0)
        self.assertEqual((self.output_dir / "summary.json").read_bytes(), first_summary)
        self.assertEqual((self.output_dir / "metadata.csv").read_bytes(), first_metadata)

    def test_rejects_pending_status_and_keeps_previous_output(self) -> None:
        self.rows[0]["annotation_status"] = "pending"
        self.write_manifest(self.rows)
        self.output_dir.mkdir()
        marker = self.output_dir / "previous.txt"
        marker.write_text("stable", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.run_cli()

        self.assertEqual(marker.read_text(encoding="utf-8"), "stable")
        self.assertEqual(list(self.output_dir.iterdir()), [marker])

    def test_rejects_an_export_with_wrong_class_order(self) -> None:
        wrong_names = ("R2", "R1", *CLASS_NAMES[2:])
        entries = {
            "data.yaml": yaml.safe_dump(
                {"path": ".", "train": "train.txt", "names": dict(enumerate(wrong_names))}
            ).encode(),
            "train.txt": "\n".join(
                f"./images/train/{row['image_name']}"
                for row in self.rows
                if row["split"] == "train"
            ).encode(),
            "labels/train/code_train_0001_01.txt": polygon_lines().encode(),
        }
        self.write_zip(self.train_export, entries)

        with self.assertRaisesRegex(RuntimeError, "class names"):
            self.run_cli()

    def test_rejects_invalid_polygon_or_non_target_class(self) -> None:
        for invalid_line, message in (
            ("3 0.1 0.1 0.3 0.1 0.2 0.3", "class ID"),
            ("0 0.1 0.1 1.2 0.1 0.2 0.3", "coordinate"),
            ("0 0.1 0.1 0.2 0.2", "polygon"),
        ):
            with self.subTest(invalid_line=invalid_line):
                self.write_export(
                    self.train_export,
                    "train",
                    self.rows,
                    label_override={"code_train_0001_01.jpg": invalid_line},
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_cli()

    def test_preserves_a_valid_polygon_with_close_coordinates(self) -> None:
        label_text = (
            "0 0.1000001 0.1000000 0.1000004 0.1000000 "
            "0.1000004 0.3000000 0.1000001 0.3000000"
        )

        normalized, _ = finalize_code_annotation_dataset.validate_label_text(
            label_text,
            "close.jpg",
        )
        coordinates = [float(token) for token in normalized.split()[1:]]

        self.assertGreater(finalize_code_annotation_dataset.polygon_area(coordinates), 0.0)

    def test_rejects_a_nonempty_negative_annotation(self) -> None:
        self.write_export(
            self.train_export,
            "train",
            self.rows,
            label_override={"code_train_0002_01.jpg": polygon_lines()},
        )

        with self.assertRaisesRegex(RuntimeError, "negative image"):
            self.run_cli()

    def test_rejects_missing_or_cross_split_annotations(self) -> None:
        self.write_export(
            self.train_export,
            "train",
            self.rows,
            label_override={"code_val_0001_01.jpg": polygon_lines()},
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected annotation"):
            self.run_cli()

        self.write_zip(
            self.train_export,
            {
                "data.yaml": yaml.safe_dump(
                    {"path": ".", "train": "train.txt", "names": dict(enumerate(CLASS_NAMES))}
                ).encode(),
                "train.txt": "\n".join(
                    f"./images/train/{row['image_name']}"
                    for row in self.rows
                    if row["split"] == "train"
                ).encode(),
                **{
                    f"images/train/{row['image_name']}": image_bytes(row["image_name"])
                    for row in self.rows
                    if row["split"] == "train"
                },
            },
        )
        with self.assertRaisesRegex(RuntimeError, "missing annotation"):
            self.run_cli()

    def test_rejects_a_batch_image_that_does_not_match_the_manifest(self) -> None:
        entries = {
            f"images/{row['image_name']}": (
                b"tampered"
                if row["image_name"] == "code_train_0001_01.jpg"
                else image_bytes(row["image_name"])
            )
            for row in self.rows
            if row["split"] == "train"
        }
        self.write_zip(self.batch_dir / "cvat_train_images.zip", entries)

        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.run_cli()

    def test_rejects_an_undecodable_batch_image(self) -> None:
        target = next(row for row in self.rows if row["image_name"] == "code_train_0001_01.jpg")
        target["source_sha256"] = hashlib.sha256(b"not an image").hexdigest()
        self.write_manifest(self.rows)
        entries = {
            f"images/{row['image_name']}": (
                b"not an image" if row is target else image_bytes(row["image_name"])
            )
            for row in self.rows
            if row["split"] == "train"
        }
        self.write_zip(self.batch_dir / "cvat_train_images.zip", entries)
        self.write_batch_summary()

        with self.assertRaisesRegex(RuntimeError, "decode"):
            self.run_cli()

    def test_converts_missing_optional_decoder_for_invalid_input(self) -> None:
        missing_decoder = ModuleNotFoundError(
            "No module named 'pi_heif'",
            name="pi_heif",
        )

        with patch.object(
            finalize_code_annotation_dataset.Image,
            "open",
            side_effect=missing_decoder,
        ):
            with self.assertRaisesRegex(RuntimeError, "decode"):
                self.run_cli()

    def test_rejects_a_truncated_jpeg_that_passes_header_parsing(self) -> None:
        target = next(row for row in self.rows if row["image_name"] == "code_train_0001_01.jpg")
        truncated = image_bytes(target["image_name"])[:-2]
        target["source_sha256"] = hashlib.sha256(truncated).hexdigest()
        self.write_manifest(self.rows)
        entries = {
            f"images/{row['image_name']}": (
                truncated if row is target else image_bytes(row["image_name"])
            )
            for row in self.rows
            if row["split"] == "train"
        }
        self.write_zip(self.batch_dir / "cvat_train_images.zip", entries)
        self.write_batch_summary()

        with self.assertRaisesRegex(RuntimeError, "decode"):
            self.run_cli()

    def test_rejects_missing_condition_tags(self) -> None:
        self.rows[0]["view_tag"] = ""
        self.write_manifest(self.rows)

        with self.assertRaisesRegex(RuntimeError, "view_tag"):
            self.run_cli()

    def test_rejects_an_unknown_condition_tag(self) -> None:
        self.rows[0]["lighting_tag"] = "bright-ish"
        self.write_manifest(self.rows)

        with self.assertRaisesRegex(RuntimeError, "unsupported value"):
            self.run_cli()

    def test_rejects_an_unsafe_zip_path(self) -> None:
        entries = {
            "data.yaml": yaml.safe_dump(
                {"path": ".", "train": "train.txt", "names": dict(enumerate(CLASS_NAMES))}
            ).encode(),
            "train.txt": "\n".join(
                f"./images/train/{row['image_name']}"
                for row in self.rows
                if row["split"] == "train"
            ).encode(),
            "labels/train/code_train_0001_01.txt": polygon_lines().encode(),
            "../outside.txt": b"unsafe",
        }
        self.write_zip(self.train_export, entries)

        with self.assertRaisesRegex(RuntimeError, "unsafe ZIP entry"):
            self.run_cli()

    def test_rejects_a_zip_with_too_many_entries_before_processing(self) -> None:
        entries = {f"empty/{index}.txt": b"" for index in range(6)}
        self.write_zip(self.train_export, entries)

        with self.assertRaisesRegex(RuntimeError, "entry count"):
            self.run_cli("--max-zip-entries", "5")

    def test_rejects_a_forged_eocd_entry_count_before_processing(self) -> None:
        entries = {f"empty/{index}.txt": b"" for index in range(6)}
        self.write_zip(self.train_export, entries)
        archive_bytes = bytearray(self.train_export.read_bytes())
        eocd_offset = archive_bytes.rfind(b"PK\x05\x06")
        struct.pack_into("<HH", archive_bytes, eocd_offset + 8, 1, 1)
        self.train_export.write_bytes(archive_bytes)

        with self.assertRaisesRegex(RuntimeError, "entry count"):
            self.run_cli("--max-zip-entries", "5")

    def test_rejects_a_zip_symlink(self) -> None:
        link = zipfile.ZipInfo("data.yaml")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.train_export, "w") as archive:
            archive.writestr(link, b"target")

        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.run_cli()

    def test_rejects_a_missing_or_extra_export_image(self) -> None:
        train_rows = [row for row in self.rows if row["split"] == "train"]
        entries = {
            "data.yaml": yaml.safe_dump(
                {"path": ".", "train": "train.txt", "names": dict(enumerate(CLASS_NAMES))}
            ).encode(),
            "train.txt": f"./images/train/{train_rows[0]['image_name']}".encode(),
            "labels/train/code_train_0001_01.txt": polygon_lines().encode(),
        }
        self.write_zip(self.train_export, entries)

        with self.assertRaisesRegex(RuntimeError, "export image set"):
            self.run_cli()

        entries["train.txt"] = (
            "\n".join(
                [
                    *(f"./images/train/{row['image_name']}" for row in train_rows),
                    "./images/train/extra.jpg",
                ]
            ).encode()
        )
        self.write_zip(self.train_export, entries)

        with self.assertRaisesRegex(RuntimeError, "export image set"):
            self.run_cli()

    def test_rejects_an_export_from_an_older_batch_generation(self) -> None:
        target = next(row for row in self.rows if row["image_name"] == "code_train_0001_01.jpg")
        replacement = image_bytes(f"replacement:{target['image_name']}")
        target["source_sha256"] = hashlib.sha256(replacement).hexdigest()
        self.write_manifest(self.rows)
        entries = {
            f"images/{row['image_name']}": (
                replacement if row is target else image_bytes(row["image_name"])
            )
            for row in self.rows
            if row["split"] == "train"
        }
        self.write_zip(self.batch_dir / "cvat_train_images.zip", entries)
        self.write_batch_summary()

        with self.assertRaisesRegex(RuntimeError, "export image SHA-256"):
            self.run_cli()

    def test_rejects_a_manifest_from_another_batch_generation(self) -> None:
        self.rows[0]["patient_token"] = "another-generation"
        self.write_manifest(self.rows)

        with self.assertRaisesRegex(RuntimeError, "manifest identity SHA-256"):
            self.run_cli()

    def test_keeps_previous_output_when_writing_the_generation_fails(self) -> None:
        self.output_dir.mkdir()
        marker = self.output_dir / "previous.txt"
        marker.write_text("stable", encoding="utf-8")

        with patch.object(Path, "write_bytes", side_effect=OSError("injected write failure")):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                self.run_cli()

        self.assertEqual(marker.read_text(encoding="utf-8"), "stable")
        self.assertEqual(list(self.output_dir.iterdir()), [marker])
        generation_paths = list(self.output_dir.parent.glob(f".{self.output_dir.name}.generation_*"))
        self.assertEqual(generation_paths, [])


if __name__ == "__main__":
    unittest.main()
