from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from code_photos import sha256_file  # noqa: E402
from extract_code_features import (  # noqa: E402
    PREPROCESSING_FORMAT_VERSION,
    Checkup,
    CropRecord,
    Photo,
    build_normalized_crop,
    load_verified_resnet_state_dict,
    iter_photo_chunks,
    main,
    process_crop_chunk,
    retain_feature_view,
    write_audit_crops,
    write_feature_file,
)
from score_code_pairs import build_feature_store  # noqa: E402


class IterPhotoChunksTest(unittest.TestCase):
    def test_bounds_each_chunk_and_preserves_order(self) -> None:
        values = list(range(130))

        chunks = list(iter_photo_chunks(values, 64))

        self.assertEqual([len(chunk) for chunk in chunks], [64, 64, 2])
        self.assertEqual([value for chunk in chunks for value in chunk], values)

    def test_empty_input_has_no_chunks(self) -> None:
        self.assertEqual(list(iter_photo_chunks([], 64)), [])

    def test_rejects_non_positive_chunk_size(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "chunk_size must be positive"):
            list(iter_photo_chunks([1], 0))


class RetainFeatureViewTest(unittest.TestCase):
    def test_keeps_only_highest_confidence_views_as_float32(self) -> None:
        views: list[tuple[np.ndarray, float, str]] = []

        retain_feature_view(views, np.asarray([1.0], dtype=np.float64), 0.2, "low.jpg", 2)
        retain_feature_view(views, np.asarray([2.0], dtype=np.float64), 0.9, "high.jpg", 2)
        retain_feature_view(views, np.asarray([3.0], dtype=np.float64), 0.7, "mid.jpg", 2)

        self.assertEqual([view[1] for view in views], [0.9, 0.7])
        self.assertTrue(all(view[0].dtype == np.float32 for view in views))


class FeatureExtractionBoundaryTest(unittest.TestCase):
    def test_builds_crop_after_tooth_axis_normalization(self) -> None:
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        mask = np.zeros((8, 9), dtype=np.bool_)
        mask[2:6, 3:5] = True
        normalized_image = np.ones_like(image)
        normalized_mask = np.ones_like(mask)
        expected_crop = np.full((16, 16, 3), 7, dtype=np.uint8)

        with mock.patch(
            "extract_code_features.normalize_tooth_axis",
            return_value=(normalized_image, normalized_mask),
        ) as normalize_axis:
            with mock.patch(
                "extract_code_features.masked_square_crop",
                return_value=expected_crop,
            ) as square_crop:
                actual = build_normalized_crop(image, mask, 16, 0.12)

        normalize_axis.assert_called_once_with(image, mask)
        square_crop.assert_called_once_with(
            normalized_image,
            normalized_mask,
            output_size=16,
            padding_ratio=0.12,
        )
        self.assertIs(actual, expected_crop)

    def test_releases_yolo_predictor_before_bounded_feature_batches(self) -> None:
        predictor = SimpleNamespace(results=[object()], batch=object(), dataset=object())
        model = SimpleNamespace(predictor=predictor)
        records = [
            CropRecord(
                checkup_uid="patient:checkup",
                tooth_index=0,
                confidence=0.9,
                photo_reference=f"photo-{index}.jpg",
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )
            for index in range(33)
        ]
        observed_batch_sizes: list[int] = []

        def observe_flush(batch: list[CropRecord], *_: object, **__: object) -> None:
            self.assertIsNone(predictor.results)
            self.assertIsNone(predictor.batch)
            self.assertIsNone(predictor.dataset)
            observed_batch_sizes.append(len(batch))
            batch.clear()

        with mock.patch(
            "extract_code_features.flush_crop_buffer",
            side_effect=observe_flush,
        ):
            process_crop_chunk(
                model,
                records,
                {},
                feature_batch_size=16,
                max_views_per_tooth=3,
                resnet_model=None,
                resnet_transform=None,
                hog=None,
                device=SimpleNamespace(),
            )

        self.assertEqual(observed_batch_sizes, [16, 16, 1])
        self.assertEqual(records, [])

    def test_rejects_resnet_checkpoint_before_torch_load(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "resnet50.pth"
            checkpoint.write_bytes(b"tampered")

            with mock.patch("extract_code_features.torch.load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "ResNet50 weight SHA-256 mismatch"):
                    load_verified_resnet_state_dict(checkpoint)

            torch_load.assert_not_called()

    def test_rejects_yolo_checkpoint_before_feature_collection(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pair_csv = root / "pairs.csv"
            pair_csv.write_text("header\n", encoding="utf-8")
            weights = root / "best.pt"
            weights.write_bytes(b"checkpoint")
            args = SimpleNamespace(
                pairs_csv=pair_csv,
                images_root=root,
                weights=weights,
                expected_weights_sha256="0" * 64,
                output_dir=root / "output",
                split="test",
                max_pairs=0,
                feature_types=["hog"],
                device="cpu",
                imgsz=832,
                conf=0.05,
                iou=0.7,
                source_chunk_size=64,
                feature_batch_size=16,
                crop_size=224,
                crop_padding=0.12,
                max_views_per_tooth=3,
                audit_crops=0,
                validate_only=False,
            )

            with mock.patch("extract_code_features.parse_args", return_value=args):
                with mock.patch("extract_code_features.collect_features") as collect:
                    with self.assertRaisesRegex(RuntimeError, "segmentation weight SHA-256 mismatch"):
                        main()

            collect.assert_not_called()

    def test_feature_file_round_trips_photo_fingerprints(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            photo_path = root / "photo.jpg"
            photo_path.write_bytes(b"photo-content")
            checkup = Checkup(
                split="test",
                uid="patient:checkup",
                patient_id="patient",
                checkup_id="checkup",
                photographs=("photo.jpg",),
            )
            photo = Photo(
                checkup_uid=checkup.uid,
                path=photo_path,
                reference="photo.jpg",
                content_sha256=sha256_file(photo_path),
            )
            feature_path = root / "features.npz"

            write_feature_file(
                feature_path,
                "test-feature",
                {checkup.uid: checkup},
                [photo],
                {checkup.uid: {0: [(np.asarray([1.0, 0.0]), 0.9, "photo.jpg")]}},
                3,
                segmentation_weights_sha256="b" * 64,
                imgsz=832,
                conf=0.1,
                iou=0.7,
                crop_size=224,
                crop_padding=0.12,
                preprocessing_format_version=PREPROCESSING_FORMAT_VERSION,
            )
            store = build_feature_store(feature_path)

            self.assertEqual(store.embeddings.shape, (1, 6, 2))
            self.assertTrue(store.present[0, 0])
            self.assertEqual(store.photo_fingerprints[0][0].reference, "photo.jpg")
            self.assertEqual(store.photo_fingerprints[0][0].sha256, sha256_file(photo_path))
            with np.load(feature_path, allow_pickle=False) as archive:
                self.assertEqual(archive["segmentation_weights_sha256"].item(), "b" * 64)
                self.assertEqual(archive["segmentation_imgsz"].item(), 832)
                self.assertEqual(archive["segmentation_conf"].item(), 0.1)
                self.assertEqual(archive["segmentation_iou"].item(), 0.7)
                self.assertEqual(archive["crop_size"].item(), 224)
                self.assertEqual(archive["crop_padding"].item(), 0.12)
                self.assertEqual(
                    archive["preprocessing_format_version"].item(),
                    PREPROCESSING_FORMAT_VERSION,
                )
                self.assertEqual(archive["feature_name"].item(), "test-feature")

    def test_audit_outputs_do_not_include_source_identifiers(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "audit"
            secret_checkup = "patient-secret:checkup-secret"
            secret_photo = "photo-secret.jpg"
            result = write_audit_crops(
                output_dir,
                [
                    CropRecord(
                        checkup_uid=secret_checkup,
                        tooth_index=0,
                        confidence=0.9,
                        photo_reference=secret_photo,
                        image=np.zeros((8, 8, 3), dtype=np.uint8),
                    )
                ],
            )

            self.assertIsNotNone(result)
            manifest = (output_dir / "manifest.csv").read_text(encoding="utf-8")
            generated_names = "\n".join(path.name for path in output_dir.iterdir())
            self.assertNotIn(secret_checkup, manifest + generated_names)
            self.assertNotIn(secret_photo, manifest + generated_names)


if __name__ == "__main__":
    unittest.main()
