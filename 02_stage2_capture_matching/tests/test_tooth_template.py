from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np


TOOTH_NAMES = ("R1", "R2", "R3", "L1", "L2", "L3")
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tooth_template.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))

import tooth_template  # noqa: E402


class ToothTemplateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_context.name)
        self.features_path = self.temp_dir / "features_hog.npz"
        self.template_path = self.temp_dir / "templates" / "subject-1.npz"

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    def feature_arrays(self) -> dict[str, np.ndarray]:
        checkup_ids = np.asarray(("subject-1:c1", "subject-1:c2", "subject-2:c3"))
        patient_ids = np.asarray(("subject-1", "subject-1", "subject-2"))
        embeddings = np.zeros((3, len(TOOTH_NAMES), 2), dtype=np.float32)
        embeddings[0] = np.asarray(
            ((1, 0), (0, 2), (9, 9), (2, 0), (2, 0), (2, 0)), dtype=np.float32
        )
        embeddings[1] = np.asarray(
            ((0, 1), (8, 8), (7, 7), (0, 2), (2, 0), (2, 0)), dtype=np.float32
        )
        embeddings[2, :, 1] = 1.0
        present = np.ones((3, len(TOOTH_NAMES)), dtype=np.bool_)
        present[0, 2] = False
        present[1, 1:3] = False
        photo_hashes = tuple(f"{index:064x}" for index in range(1, 4))
        photo_manifest_json = np.asarray(
            [
                json.dumps(
                    [{"reference": f"{checkup_id}.jpg", "sha256": photo_hash}],
                    separators=(",", ":"),
                )
                for checkup_id, photo_hash in zip(
                    checkup_ids,
                    photo_hashes,
                    strict=True,
                )
            ]
        )
        return {
            "checkup_ids": checkup_ids,
            "patient_ids": patient_ids,
            "tooth_names": np.asarray(TOOTH_NAMES),
            "embeddings": embeddings,
            "present": present,
            "photo_manifest_json": photo_manifest_json,
            "feature_name": np.asarray("hog"),
            "segmentation_weights_sha256": np.asarray("b" * 64),
            "segmentation_imgsz": np.asarray(832, dtype=np.int64),
            "segmentation_conf": np.asarray(0.1, dtype=np.float64),
            "segmentation_iou": np.asarray(0.7, dtype=np.float64),
            "crop_size": np.asarray(224, dtype=np.int64),
            "crop_padding": np.asarray(0.12, dtype=np.float64),
            "preprocessing_format_version": np.asarray(
                "tooth-axis-normalized-crop-v1"
            ),
        }

    def write_features(self, **overrides: np.ndarray) -> None:
        arrays = self.feature_arrays()
        arrays.update(overrides)
        np.savez(self.features_path, **arrays)

    def create_template(self) -> tooth_template.ToothTemplate:
        return tooth_template.create_template(
            features_path=self.features_path,
            output_path=self.template_path,
            subject_id="subject-1",
            checkup_ids=("subject-1:c1", "subject-1:c2"),
            feature_name="hog",
            created_at="2026-07-11T00:00:00Z",
        )

    def test_creates_averaged_normalized_template_with_required_metadata(self) -> None:
        self.write_features()

        template = self.create_template()

        self.assertEqual(template.format_version, tooth_template.FORMAT_VERSION)
        self.assertEqual(template.subject_id, "subject-1")
        self.assertEqual(template.created_at, "2026-07-11T00:00:00Z")
        self.assertEqual(template.feature_name, "hog")
        self.assertEqual(template.segmentation_weights_sha256, "b" * 64)
        self.assertEqual(template.segmentation_imgsz, 832)
        self.assertEqual(template.segmentation_conf, 0.1)
        self.assertEqual(template.segmentation_iou, 0.7)
        self.assertEqual(template.crop_size, 224)
        self.assertEqual(template.crop_padding, 0.12)
        self.assertEqual(
            template.preprocessing_format_version,
            "tooth-axis-normalized-crop-v1",
        )
        self.assertEqual(
            template.registration_image_sha256,
            (f"{1:064x}", f"{2:064x}"),
        )
        self.assertEqual(template.tooth_names, TOOTH_NAMES)
        self.assertEqual(template.source_checkup_ids, ("subject-1:c1", "subject-1:c2"))
        np.testing.assert_array_equal(
            template.present,
            np.asarray((True, True, False, True, True, True)),
        )
        np.testing.assert_allclose(
            template.embeddings[0],
            np.asarray((1.0, 1.0)) / np.sqrt(2.0),
            atol=1e-7,
        )
        np.testing.assert_allclose(template.embeddings[1], (0.0, 1.0), atol=1e-7)
        np.testing.assert_array_equal(template.embeddings[2], (0.0, 0.0))
        np.testing.assert_allclose(
            template.embeddings[3],
            np.asarray((1.0, 1.0)) / np.sqrt(2.0),
            atol=1e-7,
        )
        expected_hash = hashlib.sha256(self.features_path.read_bytes()).hexdigest()
        self.assertEqual(template.source_features_sha256, expected_hash)

        with np.load(self.template_path, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(tooth_template.REQUIRED_TEMPLATE_ARRAYS))
            self.assertEqual(archive["embeddings"].shape, (6, 2))
            self.assertEqual(archive["present"].shape, (6,))

        loaded = tooth_template.load_template(
            self.template_path,
            source_features_path=self.features_path,
        )
        np.testing.assert_array_equal(loaded.embeddings, template.embeddings)

    def test_cli_creates_template_and_prints_summary(self) -> None:
        self.write_features()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            result = tooth_template.main(
                [
                    "--features-npz",
                    str(self.features_path),
                    "--output-npz",
                    str(self.template_path),
                    "--subject-id",
                    "subject-1",
                    "--checkup-id",
                    "subject-1:c1",
                    "--checkup-id",
                    "subject-1:c2",
                    "--feature-name",
                    "hog",
                ]
            )

        self.assertEqual(result, 0)
        self.assertTrue(self.template_path.is_file())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["subject_id"], "subject-1")
        self.assertEqual(summary["source_checkup_ids"], ["subject-1:c1", "subject-1:c2"])
        self.assertEqual(summary["present_teeth"], 5)

    def test_rejects_nonexistent_checkup(self) -> None:
        self.write_features()

        with self.assertRaisesRegex(RuntimeError, "checkup.*not found"):
            tooth_template.create_template(
                self.features_path,
                self.template_path,
                "subject-1",
                ("subject-1:c1", "subject-1:missing"),
                "hog",
            )

        self.assertFalse(self.template_path.exists())

    def test_rejects_checkup_from_another_subject(self) -> None:
        self.write_features()

        with self.assertRaisesRegex(RuntimeError, "subject ID mismatch"):
            tooth_template.create_template(
                self.features_path,
                self.template_path,
                "subject-1",
                ("subject-1:c1", "subject-2:c3"),
                "hog",
            )

    def test_rejects_feature_name_mismatch(self) -> None:
        self.write_features()

        with self.assertRaisesRegex(RuntimeError, "feature_name mismatch"):
            tooth_template.create_template(
                self.features_path,
                self.template_path,
                "subject-1",
                ("subject-1:c1", "subject-1:c2"),
                "resnet18",
            )

    def test_rejects_less_than_two_unique_checkups(self) -> None:
        self.write_features()

        for checkup_ids in (("subject-1:c1",), ("subject-1:c1", "subject-1:c1")):
            with self.subTest(checkup_ids=checkup_ids):
                with self.assertRaisesRegex(RuntimeError, "at least two unique"):
                    tooth_template.create_template(
                        self.features_path,
                        self.template_path,
                        "subject-1",
                        checkup_ids,
                        "hog",
                    )

    def test_rejects_invalid_source_feature_shapes_and_metadata(self) -> None:
        cases = {
            "embedding shape": {
                "embeddings": np.ones((3, 5, 2), dtype=np.float32),
            },
            "present dtype": {
                "present": np.ones((3, 6), dtype=np.uint8),
            },
            "feature scalar": {
                "feature_name": np.asarray(("hog",)),
            },
        }
        for label, overrides in cases.items():
            with self.subTest(label=label):
                self.write_features(**overrides)
                with self.assertRaises(RuntimeError):
                    self.create_template()

    def test_rejects_missing_source_extraction_contract(self) -> None:
        arrays = self.feature_arrays()
        del arrays["segmentation_weights_sha256"]
        np.savez(self.features_path, **arrays)

        with self.assertRaisesRegex(RuntimeError, "missing required extraction contract"):
            self.create_template()

    def test_rejects_present_tooth_with_zero_mean_embedding(self) -> None:
        arrays = self.feature_arrays()
        arrays["embeddings"][0, 0] = (1.0, 0.0)
        arrays["embeddings"][1, 0] = (-1.0, 0.0)
        self.write_features(embeddings=arrays["embeddings"])

        with self.assertRaisesRegex(RuntimeError, "zero-norm mean embedding.*R1"):
            self.create_template()

    def test_strict_loader_rejects_shape_extra_array_and_non_normalized_value(self) -> None:
        self.template_path.parent.mkdir(parents=True)
        valid = {
            "format_version": np.asarray(tooth_template.FORMAT_VERSION),
            "subject_id": np.asarray("subject-1"),
            "created_at": np.asarray("2026-07-11T00:00:00Z"),
            "feature_name": np.asarray("hog"),
            "tooth_names": np.asarray(TOOTH_NAMES),
            "embeddings": np.tile(np.asarray((1.0, 0.0), dtype=np.float32), (6, 1)),
            "present": np.ones(6, dtype=np.bool_),
            "source_checkup_ids": np.asarray(("subject-1:c1", "subject-1:c2")),
            "source_features_sha256": np.asarray("a" * 64),
            "segmentation_weights_sha256": np.asarray("b" * 64),
            "segmentation_imgsz": np.asarray(832, dtype=np.int64),
            "segmentation_conf": np.asarray(0.1, dtype=np.float64),
            "segmentation_iou": np.asarray(0.7, dtype=np.float64),
            "crop_size": np.asarray(224, dtype=np.int64),
            "crop_padding": np.asarray(0.12, dtype=np.float64),
            "preprocessing_format_version": np.asarray(
                "tooth-axis-normalized-crop-v1"
            ),
            "registration_image_sha256": np.asarray(("c" * 64, "d" * 64)),
        }
        invalid_cases = {
            "old version": {**valid, "format_version": np.asarray(1, dtype=np.int64)},
            "shape": {**valid, "embeddings": np.ones((5, 2), dtype=np.float32)},
            "extra": {**valid, "unexpected": np.asarray(1)},
            "normalized": {
                **valid,
                "embeddings": np.tile(np.asarray((2.0, 0.0), dtype=np.float32), (6, 1)),
            },
            "contract type": {
                **valid,
                "segmentation_conf": np.asarray(1, dtype=np.int64),
            },
            "registration shape": {
                **valid,
                "registration_image_sha256": np.asarray("c" * 64),
            },
            "registration hash": {
                **valid,
                "registration_image_sha256": np.asarray(("invalid",)),
            },
        }
        for label, arrays in invalid_cases.items():
            with self.subTest(label=label):
                np.savez(self.template_path.parent / f"{label}.npz", **arrays)
                with self.assertRaises(RuntimeError):
                    tooth_template.load_template(self.template_path.parent / f"{label}.npz")

    def test_loader_detects_changed_source_feature_file(self) -> None:
        self.write_features()
        self.create_template()
        arrays = self.feature_arrays()
        arrays["embeddings"][0, 0] = (0.0, 1.0)
        np.savez(self.features_path, **arrays)

        with self.assertRaisesRegex(RuntimeError, "source feature SHA-256 mismatch"):
            tooth_template.load_template(
                self.template_path,
                source_features_path=self.features_path,
            )

    def test_loader_detects_semantically_valid_template_tampering(self) -> None:
        self.write_features()
        self.create_template()
        with np.load(self.template_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["embeddings"][0] = (1.0, 0.0)
        np.savez(self.template_path, **arrays)

        with self.assertRaisesRegex(RuntimeError, "template content mismatch"):
            tooth_template.load_template(
                self.template_path,
                source_features_path=self.features_path,
            )

    def test_rejects_explicit_empty_created_at(self) -> None:
        self.write_features()

        with self.assertRaisesRegex(RuntimeError, "created_at must not be empty"):
            tooth_template.create_template(
                self.features_path,
                self.template_path,
                "subject-1",
                ("subject-1:c1", "subject-1:c2"),
                "hog",
                created_at="",
            )

    def test_atomic_save_preserves_existing_output_on_write_failure(self) -> None:
        self.write_features()
        self.template_path.parent.mkdir(parents=True)
        self.template_path.write_bytes(b"existing-template")

        with mock.patch.object(
            tooth_template.np,
            "savez_compressed",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to save template NPZ"):
                self.create_template()

        self.assertEqual(self.template_path.read_bytes(), b"existing-template")
        self.assertEqual(list(self.template_path.parent.glob("*.tmp.npz")), [])

    def test_atomic_save_preserves_existing_output_when_temporary_npz_is_invalid(self) -> None:
        self.write_features()
        self.template_path.parent.mkdir(parents=True)
        self.template_path.write_bytes(b"existing-template")
        real_template_arrays = tooth_template.template_arrays

        def invalid_arrays(template: tooth_template.ToothTemplate) -> dict[str, np.ndarray]:
            arrays = real_template_arrays(template)
            del arrays["registration_image_sha256"]
            return arrays

        with mock.patch.object(
            tooth_template,
            "template_arrays",
            side_effect=invalid_arrays,
        ):
            with self.assertRaisesRegex(RuntimeError, "template NPZ arrays mismatch"):
                self.create_template()

        self.assertEqual(self.template_path.read_bytes(), b"existing-template")

    def test_atomic_save_rechecks_source_before_replacing_existing_output(self) -> None:
        self.write_features()
        self.template_path.parent.mkdir(parents=True)
        self.template_path.write_bytes(b"existing-template")
        real_load_template = tooth_template.load_template

        def load_then_change_source(
            path: Path,
            source_features_path: Path | None = None,
        ) -> tooth_template.ToothTemplate:
            loaded = real_load_template(path, source_features_path)
            arrays = self.feature_arrays()
            arrays["embeddings"][0, 0] = (0.0, 1.0)
            np.savez(self.features_path, **arrays)
            return loaded

        with mock.patch.object(
            tooth_template,
            "load_template",
            side_effect=load_then_change_source,
        ):
            with self.assertRaisesRegex(RuntimeError, "source feature NPZ changed before publish"):
                self.create_template()

        self.assertEqual(self.template_path.read_bytes(), b"existing-template")


if __name__ == "__main__":
    unittest.main()
