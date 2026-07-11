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

import cv2
import numpy as np


TOOTH_NAMES = ("R1", "R2", "R3", "L1", "L2", "L3")
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "match_teeth.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))

import match_teeth  # noqa: E402
from tooth_template import FORMAT_VERSION  # noqa: E402


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeBoxes:
    def __init__(self, classes: list[int], confidences: list[float]) -> None:
        self.cls = FakeTensor(np.asarray(classes, dtype=np.float32))
        self.conf = FakeTensor(np.asarray(confidences, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.cls.value)


class FakeMasks:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.data = FakeTensor(np.asarray(masks, dtype=np.float32))


class FakeResult:
    def __init__(
        self,
        classes: list[int],
        confidences: list[float],
        masks: list[np.ndarray],
    ) -> None:
        self.boxes = FakeBoxes(classes, confidences)
        self.masks = FakeMasks(masks)


class MatchTeethTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_context.name)
        self.query_path = self.temp_dir / "query-001.png"
        self.template_path = self.temp_dir / "template.npz"
        self.weights_path = self.temp_dir / "best.pt"
        self.resnet_weights_path = self.temp_dir / "resnet50.pth"

        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[:, :, 1] = 127
        self.assertTrue(cv2.imwrite(str(self.query_path), image))
        self.weights_path.write_bytes(b"verified-yolo-weights")
        self.resnet_weights_path.write_bytes(b"verified-resnet-weights")
        self.write_template()

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_template(
        self,
        *,
        feature_name: str = "hog",
        tooth_names: tuple[str, ...] = TOOTH_NAMES,
        present: np.ndarray | None = None,
        segmentation_weights_sha256: str | None = None,
        segmentation_imgsz: int = 832,
        segmentation_conf: float = 0.1,
        segmentation_iou: float = 0.7,
        crop_size: int = 224,
        crop_padding: float = 0.12,
        preprocessing_format_version: str = "tooth-axis-normalized-crop-v1",
        registration_image_sha256: tuple[str, ...] = ("c" * 64,),
    ) -> None:
        if present is None:
            present = np.asarray((True, True, False, False, False, False))
        embeddings = np.zeros((6, 2), dtype=np.float32)
        embeddings[0] = (1.0, 0.0)
        embeddings[1] = (0.0, 1.0)
        if segmentation_weights_sha256 is None:
            segmentation_weights_sha256 = self.sha256(self.weights_path)
        np.savez(
            self.template_path,
            format_version=np.asarray(FORMAT_VERSION, dtype=np.int64),
            subject_id=np.asarray("subject-001"),
            created_at=np.asarray("2026-07-11T00:00:00Z"),
            feature_name=np.asarray(feature_name),
            tooth_names=np.asarray(tooth_names),
            embeddings=embeddings,
            present=present,
            source_checkup_ids=np.asarray(("subject-001:c1", "subject-001:c2")),
            source_features_sha256=np.asarray("a" * 64),
            segmentation_weights_sha256=np.asarray(segmentation_weights_sha256),
            segmentation_imgsz=np.asarray(segmentation_imgsz, dtype=np.int64),
            segmentation_conf=np.asarray(segmentation_conf, dtype=np.float64),
            segmentation_iou=np.asarray(segmentation_iou, dtype=np.float64),
            crop_size=np.asarray(crop_size, dtype=np.int64),
            crop_padding=np.asarray(crop_padding, dtype=np.float64),
            preprocessing_format_version=np.asarray(preprocessing_format_version),
            registration_image_sha256=np.asarray(registration_image_sha256),
        )

    def cli_args(self, *extra: str, feature_name: str = "hog") -> list[str]:
        return [
            "--query",
            str(self.query_path),
            "--template",
            str(self.template_path),
            "--query-id",
            "query-001",
            "--weights",
            str(self.weights_path),
            "--expected-weights-sha256",
            self.sha256(self.weights_path),
            "--expected-query-sha256",
            self.sha256(self.query_path),
            "--feature-name",
            feature_name,
            "--device",
            "cpu",
            *extra,
        ]

    @staticmethod
    def expected_model_names() -> dict[int, str]:
        return {0: "R1", 1: "R2", 2: "R3", 7: "L1", 8: "L2", 9: "L3"}

    def test_validate_only_checks_inputs_without_loading_models(self) -> None:
        stdout = io.StringIO()

        with (
            mock.patch.object(match_teeth, "YOLO") as yolo,
            mock.patch.object(match_teeth, "build_hog") as build_hog,
            mock.patch.object(match_teeth, "build_resnet") as build_resnet,
            redirect_stdout(stdout),
        ):
            result = match_teeth.main(self.cli_args("--validate-only"))

        self.assertEqual(result, 0)
        yolo.assert_not_called()
        build_hog.assert_not_called()
        build_resnet.assert_not_called()
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["query_id"], "query-001")
        self.assertEqual(summary["template_subject_id"], "subject-001")
        self.assertEqual(summary["feature_name"], "hog")
        self.assertTrue(summary["validation_only"])

    def test_hog_match_selects_highest_confidence_mask_and_prints_json(self) -> None:
        self.write_template(crop_size=8)
        low_r1 = np.zeros((12, 12), dtype=np.float32)
        low_r1[1:3, 1:3] = 1.0
        high_r1 = np.zeros((12, 12), dtype=np.float32)
        high_r1[3:9, 4:7] = 1.0
        r2 = np.zeros((12, 12), dtype=np.float32)
        r2[2:10, 8:10] = 1.0
        ignored = np.ones((12, 12), dtype=np.float32)
        prediction = FakeResult(
            [0, 0, 1, 6],
            [0.20, 0.90, 0.80, 0.99],
            [low_r1, high_r1, r2, ignored],
        )
        model = mock.Mock()
        model.names = self.expected_model_names()
        model.predict.return_value = [prediction]
        crop = np.full((8, 8, 3), 11, dtype=np.uint8)
        stdout = io.StringIO()

        with (
            mock.patch.object(match_teeth, "YOLO", return_value=model),
            mock.patch.object(match_teeth, "build_hog", return_value=mock.sentinel.hog),
            mock.patch.object(
                match_teeth,
                "build_normalized_crop",
                return_value=crop,
            ) as build_crop,
            mock.patch.object(
                match_teeth,
                "extract_hog_batch",
                return_value=[
                    np.asarray((1.0, 0.0), dtype=np.float32),
                    np.asarray((0.0, 1.0), dtype=np.float32),
                ],
            ) as extract_hog,
            redirect_stdout(stdout),
        ):
            result = match_teeth.main(self.cli_args("--crop-size", "8"))

        self.assertEqual(result, 0)
        self.assertEqual(build_crop.call_count, 2)
        np.testing.assert_array_equal(build_crop.call_args_list[0].args[1], high_r1 > 0.5)
        np.testing.assert_array_equal(build_crop.call_args_list[1].args[1], r2 > 0.5)
        records = extract_hog.call_args.args[0]
        self.assertEqual([record.tooth_index for record in records], [0, 1])
        np.testing.assert_allclose(
            [record.confidence for record in records],
            [0.9, 0.8],
        )
        prediction_source = model.predict.call_args.kwargs["source"]
        self.assertIsInstance(prediction_source, np.ndarray)

        output = json.loads(stdout.getvalue())
        self.assertEqual(output["query_id"], "query-001")
        self.assertEqual(output["template_subject_id"], "subject-001")
        self.assertEqual(output["per_tooth_scores"], {"R1": 1.0, "R2": 1.0})
        self.assertEqual(output["fused_score"], 1.0)

    def test_resnet50_match_reuses_existing_extractor(self) -> None:
        self.write_template(feature_name="resnet50")
        mask = np.ones((12, 12), dtype=np.float32)
        prediction = FakeResult([0], [0.75], [mask])
        model = mock.Mock(names=self.expected_model_names())
        model.predict.return_value = [prediction]
        fake_resnet = mock.Mock()
        fake_transform = mock.sentinel.transform

        with (
            mock.patch.object(match_teeth, "YOLO", return_value=model),
            mock.patch.object(match_teeth, "prepare_resnet_checkpoint"),
            mock.patch.object(
                match_teeth,
                "build_resnet",
                return_value=(fake_resnet, fake_transform),
            ) as build_resnet,
            mock.patch.object(
                match_teeth,
                "build_normalized_crop",
                return_value=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            mock.patch.object(
                match_teeth,
                "extract_resnet_batch",
                return_value=[np.asarray((1.0, 0.0), dtype=np.float32)],
            ) as extract_resnet,
            redirect_stdout(io.StringIO()),
        ):
            result = match_teeth.main(
                self.cli_args(
                    "--resnet-weights",
                    str(self.resnet_weights_path),
                    feature_name="resnet50",
                )
            )

        self.assertEqual(result, 0)
        build_resnet.assert_called_once()
        self.assertIs(extract_resnet.call_args.args[1], fake_resnet)
        self.assertIs(extract_resnet.call_args.args[2], fake_transform)

    def test_rejects_weight_and_query_sha256_mismatches_before_model_load(self) -> None:
        cases = (
            ("--expected-weights-sha256", "segmentation weight SHA-256 mismatch"),
            ("--expected-query-sha256", "query image SHA-256 mismatch"),
        )
        for option, message in cases:
            with self.subTest(option=option):
                args = self.cli_args()
                args[args.index(option) + 1] = "0" * 64
                with mock.patch.object(match_teeth, "YOLO") as yolo:
                    with self.assertRaisesRegex(RuntimeError, message):
                        match_teeth.main(args)
                yolo.assert_not_called()

    def test_rejects_feature_name_mismatch_before_model_load(self) -> None:
        with mock.patch.object(match_teeth, "YOLO") as yolo:
            with self.assertRaisesRegex(RuntimeError, "feature_name mismatch"):
                match_teeth.main(self.cli_args(feature_name="resnet50"))

        yolo.assert_not_called()

    def test_rejects_registered_image_reuse_before_model_load(self) -> None:
        self.write_template(
            registration_image_sha256=(self.sha256(self.query_path),),
        )

        with mock.patch.object(match_teeth, "YOLO") as yolo:
            with self.assertRaisesRegex(RuntimeError, "registered image reuse"):
                match_teeth.main(self.cli_args())

        yolo.assert_not_called()

    def test_rejects_extraction_contract_mismatches_before_model_load(self) -> None:
        cases = (
            (
                {"segmentation_weights_sha256": "d" * 64},
                "segmentation_weights_sha256",
            ),
            ({"segmentation_imgsz": 640}, "segmentation_imgsz"),
            ({"segmentation_conf": 0.2}, "segmentation_conf"),
            ({"segmentation_iou": 0.6}, "segmentation_iou"),
            ({"crop_size": 128}, "crop_size"),
            ({"crop_padding": 0.2}, "crop_padding"),
            (
                {"preprocessing_format_version": "tooth-axis-normalized-crop-v2"},
                "preprocessing_format_version",
            ),
        )
        for template_overrides, field_name in cases:
            with self.subTest(field_name=field_name):
                self.write_template(**template_overrides)
                with mock.patch.object(match_teeth, "YOLO") as yolo:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        f"extraction contract mismatch.*{field_name}",
                    ):
                        match_teeth.main(self.cli_args())
                yolo.assert_not_called()

    def test_rejects_unexpected_template_tooth_order(self) -> None:
        self.write_template(tooth_names=("L1", "R2", "R3", "R1", "L2", "L3"))

        with self.assertRaisesRegex(RuntimeError, "unexpected order"):
            match_teeth.main(self.cli_args("--validate-only"))

    def test_rejects_missing_resnet_checkpoint(self) -> None:
        self.write_template(feature_name="resnet50")

        with self.assertRaisesRegex(RuntimeError, "--resnet-weights is required"):
            match_teeth.main(
                self.cli_args("--validate-only", feature_name="resnet50")
            )

    def test_fails_when_common_teeth_are_below_minimum(self) -> None:
        prediction = FakeResult([], [], [])
        model = mock.Mock(names=self.expected_model_names())
        model.predict.return_value = [prediction]

        with (
            mock.patch.object(match_teeth, "YOLO", return_value=model),
            mock.patch.object(match_teeth, "build_hog") as build_hog,
        ):
            with self.assertRaisesRegex(RuntimeError, "common teeth count"):
                match_teeth.main(self.cli_args("--min-common-teeth", "2"))

        build_hog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
