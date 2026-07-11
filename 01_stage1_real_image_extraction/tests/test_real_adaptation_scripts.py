from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "01_stage1_real_image_extraction" / "scripts"
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


def load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_v8 = load_script("train_tooth_seg_flont_v8")
validate_real = load_script("validate_tooth_seg_real")


class DatasetFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.train_dir = root / "images" / "train"
        self.val_dir = root / "images" / "val"
        self.train_dir.mkdir(parents=True)
        self.val_dir.mkdir(parents=True)
        self.yaml_path = root / "dataset_real.yaml"
        self.write_yaml()

    def add_val_image(self, name: str) -> Path:
        path = self.val_dir / name
        path.write_bytes(f"{self.root}:val:{name}".encode())
        return path

    def add_train_image(self, name: str) -> Path:
        path = self.train_dir / name
        path.write_bytes(f"{self.root}:train:{name}".encode())
        return path

    def write_yaml(self, **overrides: object) -> None:
        payload: dict[str, object] = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(CLASS_NAMES)),
        }
        payload.update(overrides)
        self.yaml_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


class TrainV8Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.synthetic = DatasetFixture(self.root / "synthetic")
        self.real = DatasetFixture(self.root / "real")
        self.synthetic.add_train_image("synthetic_train.jpg")
        self.synthetic.add_val_image("synthetic_val.jpg")
        self.real.add_train_image("real_train.jpg")
        self.real.add_val_image("real_val.jpg")
        self.weights = self.root / "v7_best.pt"
        self.weights.write_bytes(b"checkpoint")
        self.project = self.root / "runs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def argv(self, *extra: str) -> list[str]:
        return [
            "--synthetic-data",
            str(self.synthetic.yaml_path),
            "--real-data",
            str(self.real.yaml_path),
            "--weights",
            str(self.weights),
            "--project",
            str(self.project),
            *extra,
        ]

    def test_data_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            train_v8.parse_args([])

    def test_prepare_validates_dataset_classes_and_split_paths(self) -> None:
        args = train_v8.parse_args(self.argv("--prepare-only"))
        prepared = train_v8.prepare_training(args, device_resolver=lambda _: "cpu")

        self.assertEqual(prepared["kwargs"]["classes"], [0, 1, 2, 7, 8, 9])
        self.assertLess(prepared["kwargs"]["lr0"], 5e-4)
        self.assertGreater(prepared["kwargs"]["freeze"], 0)
        self.assertGreaterEqual(prepared["kwargs"]["patience"], 1)
        self.assertEqual(prepared["kwargs"]["fliplr"], 0.0)
        self.assertEqual(prepared["mixed_dataset"]["val"], str(self.real.val_dir))
        self.assertEqual(
            prepared["mixed_dataset"]["train"],
            [
                str(self.synthetic.train_dir),
                str(self.real.train_dir),
                str(self.real.train_dir),
            ],
        )
        self.assertEqual(prepared["mixing"]["real_repeat"], 2)
        self.assertEqual(
            [item.__class__.__name__ for item in prepared["kwargs"]["augmentations"]],
            ["RandomBrightnessContrast", "Blur", "ImageCompression"],
        )

    def test_prepare_rejects_wrong_target_class_name(self) -> None:
        names = dict(enumerate(CLASS_NAMES))
        names[7] = "wrong"
        self.real.write_yaml(names=names)

        with self.assertRaisesRegex(RuntimeError, "class 7"):
            train_v8.prepare_training(
                train_v8.parse_args(self.argv()), device_resolver=lambda _: "cpu"
            )

    def test_prepare_rejects_domain_class_order_mismatch(self) -> None:
        names = dict(enumerate(CLASS_NAMES))
        names[3], names[4] = names[4], names[3]
        self.synthetic.write_yaml(names=names)

        with self.assertRaisesRegex(RuntimeError, "class order"):
            train_v8.prepare_training(
                train_v8.parse_args(self.argv()), device_resolver=lambda _: "cpu"
            )

    def test_prepare_rejects_noncontiguous_class_ids(self) -> None:
        names = {
            class_id: CLASS_NAMES[class_id]
            for class_id in (0, 1, 2, 7, 8, 9)
        }
        self.real.write_yaml(names=names)

        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            train_v8.prepare_training(
                train_v8.parse_args(self.argv()), device_resolver=lambda _: "cpu"
            )
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            metadata_stub = self.root / "metadata.csv"
            metadata_stub.write_text("", encoding="utf-8")
            validate_real.prepare_validation(
                validate_real.parse_args(
                    [
                        "--data",
                        str(self.real.yaml_path),
                        "--metadata",
                        str(metadata_stub),
                        "--model",
                        f"v7_zero_shot={self.weights}",
                        "--project",
                        str(self.project),
                    ]
                )
            )

    def test_prepare_rejects_missing_train_split_and_output_collision(self) -> None:
        (self.real.train_dir / "real_train.jpg").unlink()
        self.real.train_dir.rmdir()
        args = train_v8.parse_args(self.argv())
        with self.assertRaisesRegex(RuntimeError, "train"):
            train_v8.prepare_training(args, device_resolver=lambda _: "cpu")

        self.real.train_dir.mkdir()
        self.real.add_train_image("real_train.jpg")
        (self.project / args.name).mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "output already exists"):
            train_v8.prepare_training(args, device_resolver=lambda _: "cpu")

    def test_prepare_only_prints_json_without_constructing_yolo(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            train_v8.main(
                self.argv("--prepare-only"),
                yolo_factory=mock.Mock(side_effect=AssertionError("YOLO must not run")),
                device_resolver=lambda _: "cpu",
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "prepare-only")
        self.assertEqual(payload["train_kwargs"]["device"], "cpu")
        self.assertEqual(len(payload["train_kwargs"]["augmentations"]), 3)
        self.assertEqual(payload["mixed_dataset"]["val"], str(self.real.val_dir))
        self.assertEqual(payload["mixing"]["effective_real_train_images"], 2)

    def test_prepare_only_and_dry_run_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            train_v8.parse_args(self.argv("--prepare-only", "--dry-run"))

    def test_dry_run_uses_explicit_domain_lists_one_epoch_and_cpu(self) -> None:
        model = mock.Mock()

        def train(**kwargs: object) -> None:
            mixed = yaml.safe_load(
                Path(str(kwargs["data"])).read_text(encoding="utf-8")
            )
            train_list = Path(mixed["train"])
            val_list = Path(mixed["val"])
            train_images = train_list.read_text(encoding="utf-8").splitlines()
            val_images = val_list.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(train_images), 2)
            self.assertTrue(any("synthetic" in path for path in train_images))
            self.assertTrue(any("real" in path for path in train_images))
            self.assertEqual(val_images, [str(self.real.val_dir / "real_val.jpg")])
            output_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
            (output_dir / "weights").mkdir(parents=True)
            (output_dir / "results.csv").write_text("epoch\n0\n", encoding="utf-8")
            (output_dir / "weights" / "best.pt").write_bytes(b"best")

        model.train.side_effect = train
        with redirect_stdout(io.StringIO()):
            train_v8.main(
                self.argv("--dry-run", "--device", "cpu"),
                yolo_factory=mock.Mock(return_value=model),
                device_resolver=lambda _: "cpu",
            )

        kwargs = model.train.call_args.kwargs
        self.assertEqual(kwargs["epochs"], 1)
        self.assertEqual(kwargs["fraction"], 1.0)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["fliplr"], 0.0)
        output_dir = self.project / "v8_real_finetune_dry_run"
        persisted = yaml.safe_load(
            (output_dir / "mixed_dataset.yaml").read_text(encoding="utf-8")
        )
        persisted_train = Path(persisted["train"])
        persisted_val = Path(persisted["val"])
        self.assertTrue(persisted_train.is_file())
        self.assertTrue(persisted_val.is_file())
        self.assertTrue(
            all(Path(path).is_file() for path in persisted_train.read_text().splitlines())
        )
        self.assertEqual(
            persisted_val.read_text(encoding="utf-8").splitlines(),
            [str(self.real.val_dir / "real_val.jpg")],
        )

    def test_training_passes_prepared_kwargs_to_ultralytics(self) -> None:
        model = mock.Mock()
        factory = mock.Mock(return_value=model)

        def train(**kwargs: object) -> None:
            mixed = yaml.safe_load(
                Path(str(kwargs["data"])).read_text(encoding="utf-8")
            )
            self.assertEqual(mixed["val"], str(self.real.val_dir))
            output_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
            (output_dir / "weights").mkdir(parents=True)
            (output_dir / "results.csv").write_text("epoch\n0\n", encoding="utf-8")
            (output_dir / "weights" / "best.pt").write_bytes(b"best")

        model.train.side_effect = train

        with redirect_stdout(io.StringIO()):
            train_v8.main(
                self.argv(), yolo_factory=factory, device_resolver=lambda _: "cpu"
            )

        factory.assert_called_once_with(str(self.weights.resolve()))
        model.train.assert_called_once()
        kwargs = model.train.call_args.kwargs
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["fliplr"], 0.0)
        self.assertFalse(kwargs["exist_ok"])
        output_dir = self.project / "v8_real_finetune"
        self.assertTrue((output_dir / "mixed_dataset.yaml").is_file())
        manifest = json.loads(
            (output_dir / "mixed_dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["mixing"]["real_repeat"], 2)
        self.assertEqual(
            manifest["train_kwargs"]["data"],
            str(output_dir / "mixed_dataset.yaml"),
        )
        self.assertTrue((output_dir / "weights" / "best.pt").is_file())

    def test_training_rejects_missing_required_outputs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "results.csv"):
            with redirect_stdout(io.StringIO()):
                train_v8.main(
                    self.argv(),
                    yolo_factory=mock.Mock(return_value=mock.Mock()),
                    device_resolver=lambda _: "cpu",
                )


class ValidateRealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset = DatasetFixture(self.root / "dataset")
        self.dataset.add_train_image("train.jpg")
        self.dataset.add_val_image("val.jpg")
        self.v7_weights = self.root / "v7.pt"
        self.v8_weights = self.root / "v8.pt"
        self.v7_weights.write_bytes(b"v7")
        self.v8_weights.write_bytes(b"v8")
        self.project = self.root / "validation"
        self.metadata = self.write_metadata(
            [
                self.metadata_row("train", "train.jpg", "p1", "c1"),
                self.metadata_row("val", "val.jpg", "p2", "c2"),
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def argv(self, *extra: str) -> list[str]:
        return [
            "--data",
            str(self.dataset.yaml_path),
            "--metadata",
            str(self.metadata),
            "--model",
            f"v7_zero_shot={self.v7_weights}",
            "--model",
            f"v8={self.v8_weights}",
            "--project",
            str(self.project),
            *extra,
        ]

    def write_metadata(self, rows: list[dict[str, str]]) -> Path:
        path = self.root / "metadata.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "split",
                    "image_name",
                    "patient_token",
                    "checkup_token",
                    "source_sha256",
                    "view_tag",
                    "lighting_tag",
                    "oral_condition_tag",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def metadata_row(
        self,
        split: str,
        image_name: str,
        patient_token: str,
        checkup_token: str,
        *,
        source_sha256: str | None = None,
        view_tag: str = "front",
        lighting_tag: str = "bright",
        oral_condition_tag: str = "dry",
    ) -> dict[str, str]:
        image_path = (
            self.dataset.train_dir if split == "train" else self.dataset.val_dir
        ) / image_name
        actual_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        return {
            "split": split,
            "image_name": image_name,
            "patient_token": patient_token,
            "checkup_token": checkup_token,
            "source_sha256": source_sha256
            if source_sha256 is not None
            else actual_sha256,
            "view_tag": view_tag,
            "lighting_tag": lighting_tag,
            "oral_condition_tag": oral_condition_tag,
        }

    def fake_result(self, save_dir: Path, mask_offset: float) -> SimpleNamespace:
        class FakeMetric:
            def __init__(self, offset: float) -> None:
                self.ap_class_index = list(TARGET_CLASS_IDS)
                self.offset = offset

            def class_result(self, index: int) -> tuple[float, float, float, float]:
                return (
                    0.7 + self.offset,
                    0.6 + self.offset,
                    0.5 + index / 100.0 + self.offset,
                    0.4 + index / 100.0 + self.offset,
                )

        metrics = {
            key: float(index) / 10.0 + mask_offset
            for index, key in enumerate(validate_real.METRIC_KEYS, start=1)
        }
        return SimpleNamespace(
            save_dir=save_dir,
            results_dict=metrics,
            names=dict(enumerate(CLASS_NAMES)),
            box=FakeMetric(mask_offset / 2.0),
            seg=FakeMetric(mask_offset),
        )

    def test_prepare_only_prints_shared_val_kwargs_without_yolo(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            validate_real.main(
                self.argv("--prepare-only"),
                yolo_factory=mock.Mock(side_effect=AssertionError("YOLO must not run")),
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "prepare-only")
        self.assertEqual(
            [model["label"] for model in payload["models"]],
            ["v7_zero_shot", "v8"],
        )
        for model in payload["models"]:
            self.assertEqual(model["val_kwargs"]["classes"], [0, 1, 2, 7, 8, 9])
            self.assertEqual(model["val_kwargs"]["split"], "val")

    def test_metadata_argument_is_required(self) -> None:
        argv = self.argv()
        metadata_index = argv.index("--metadata")
        del argv[metadata_index : metadata_index + 2]
        with self.assertRaises(SystemExit):
            validate_real.parse_args(argv)

    def test_prepare_rejects_duplicate_labels_and_unsafe_output(self) -> None:
        duplicate = self.argv()
        duplicate[duplicate.index(f"v8={self.v8_weights}")] = (
            f"v7_zero_shot={self.v8_weights}"
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate model label"):
            validate_real.prepare_validation(validate_real.parse_args(duplicate))

        args = validate_real.parse_args(self.argv())
        self.project.mkdir(parents=True)
        (self.project / args.name).write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "regular directory"):
            validate_real.prepare_validation(args)

    def test_full_val_requires_all_classes_but_condition_reports_missing(self) -> None:
        result = self.fake_result(self.root / "save", 0.0)
        result.box.ap_class_index = list(TARGET_CLASS_IDS[:-1])
        result.seg.ap_class_index = list(TARGET_CLASS_IDS[:-1])

        with self.assertRaisesRegex(RuntimeError, "lacks target classes"):
            validate_real.extract_per_class_metrics(result, require_all=True)
        rows = validate_real.extract_per_class_metrics(result, require_all=False)
        self.assertFalse(rows[-1]["has_ground_truth"])
        self.assertIsNone(rows[-1]["mask"])

    def test_metadata_must_exactly_match_real_val_images(self) -> None:
        self.dataset.add_val_image("a.jpg")
        self.dataset.add_val_image("b.jpg")
        metadata = self.write_metadata(
            [
                self.metadata_row("train", "train.jpg", "p1", "c1"),
                self.metadata_row("val", "val.jpg", "p2", "c2"),
                self.metadata_row("val", "a.jpg", "p3", "c3"),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "metadata val image set"):
            validate_real.prepare_validation(
                validate_real.parse_args(self.argv("--metadata", str(metadata)))
            )

    def test_metadata_must_exactly_match_real_train_images(self) -> None:
        self.dataset.add_train_image("train-a.jpg")
        self.dataset.add_train_image("train-b.jpg")
        metadata = self.write_metadata(
            [
                self.metadata_row("train", "train.jpg", "p1", "c1"),
                self.metadata_row("train", "train-a.jpg", "p2", "c2"),
                self.metadata_row("val", "val.jpg", "p3", "c3"),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "metadata train image set"):
            validate_real.prepare_validation(
                validate_real.parse_args(self.argv("--metadata", str(metadata)))
            )

    def test_metadata_requires_finalizer_identity_columns(self) -> None:
        metadata = self.root / "metadata.csv"
        metadata.write_text(
            "split,image_name,view_tag,lighting_tag,oral_condition_tag\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "missing columns"):
            validate_real.prepare_validation(
                validate_real.parse_args(self.argv("--metadata", str(metadata)))
            )

    def test_metadata_rejects_cross_split_identity_leakage(self) -> None:
        cases = (
            ("patient_token", "same-patient", "c1", "same-patient", "c2"),
            ("checkup_token", "p1", "same-checkup", "p2", "same-checkup"),
        )
        for field, train_patient, train_checkup, val_patient, val_checkup in cases:
            with self.subTest(field=field):
                metadata = self.write_metadata(
                    [
                        self.metadata_row(
                            "train",
                            "train.jpg",
                            train_patient,
                            train_checkup,
                        ),
                        self.metadata_row(
                            "val",
                            "val.jpg",
                            val_patient,
                            val_checkup,
                        ),
                    ]
                )
                with self.assertRaisesRegex(RuntimeError, field):
                    validate_real.prepare_validation(
                        validate_real.parse_args(
                            self.argv("--metadata", str(metadata))
                        )
                    )

    def test_metadata_rejects_tampered_source_sha256(self) -> None:
        metadata = self.write_metadata(
            [
                self.metadata_row(
                    "train",
                    "train.jpg",
                    "p1",
                    "c1",
                    source_sha256="0" * 64,
                ),
                self.metadata_row("val", "val.jpg", "p2", "c2"),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "does not match image"):
            validate_real.prepare_validation(
                validate_real.parse_args(self.argv("--metadata", str(metadata)))
            )

    def test_metadata_rejects_duplicate_image_content_across_splits(self) -> None:
        train_image = self.dataset.train_dir / "train.jpg"
        val_image = self.dataset.val_dir / "val.jpg"
        val_image.write_bytes(train_image.read_bytes())
        metadata = self.write_metadata(
            [
                self.metadata_row("train", "train.jpg", "p1", "c1"),
                self.metadata_row("val", "val.jpg", "p2", "c2"),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "source_sha256"):
            validate_real.prepare_validation(
                validate_real.parse_args(self.argv("--metadata", str(metadata)))
            )

    def test_metadata_prepare_only_outputs_counts_and_planned_conditions(self) -> None:
        self.dataset.add_val_image("b.jpg")
        metadata = self.write_metadata(
            [
                self.metadata_row("train", "train.jpg", "p1", "c1"),
                self.metadata_row("val", "val.jpg", "p2", "c2"),
                self.metadata_row(
                    "val",
                    "b.jpg",
                    "p3",
                    "c3",
                    view_tag="oblique",
                    lighting_tag="dim",
                    oral_condition_tag="wet",
                ),
            ]
        )

        output = io.StringIO()
        with redirect_stdout(output):
            validate_real.main(
                self.argv("--metadata", str(metadata), "--prepare-only"),
                yolo_factory=mock.Mock(side_effect=AssertionError("YOLO must not run")),
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["metadata"]["train_image_count"], 1)
        self.assertEqual(payload["metadata"]["val_image_count"], 2)
        self.assertEqual(len(payload["conditions"]), 6)
        self.assertFalse((self.project / "real_val_comparison").exists())

    def test_validation_writes_overall_metrics_to_json_and_csv(self) -> None:
        created_models: list[mock.Mock] = []

        def factory(weights: str) -> mock.Mock:
            model = mock.Mock()

            def val(**kwargs: object) -> SimpleNamespace:
                save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                save_dir.mkdir(parents=True)
                offset = 0.1 if weights.endswith("v8.pt") else 0.0
                return self.fake_result(save_dir, offset)

            model.val.side_effect = val
            created_models.append(model)
            return model

        with redirect_stdout(io.StringIO()):
            validate_real.main(self.argv(), yolo_factory=factory)

        output_dir = self.project / "real_val_comparison"
        payload = json.loads(
            (output_dir / "overall_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual([row["model"] for row in payload["results"]], ["v7_zero_shot", "v8"])
        self.assertEqual(payload["profile"]["classes"], [0, 1, 2, 7, 8, 9])
        self.assertTrue(payload["comparison"]["all_six_mask_map50_improved"])
        self.assertEqual(len(payload["results"][0]["per_class"]), 6)
        for result in payload["results"]:
            self.assertTrue(result["save_dir"].startswith(str(output_dir)))
            self.assertNotIn(".generation_", result["save_dir"])

        with (output_dir / "overall_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["model"] for row in rows], ["v7_zero_shot", "v8"])
        with (output_dir / "per_class_comparison.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            comparison_rows = list(csv.DictReader(handle))
        self.assertEqual(len(comparison_rows), 6)
        self.assertTrue(all(row["mask_map50_improved"] == "True" for row in comparison_rows))
        self.assertEqual(len(created_models), 2)
        for model in created_models:
            kwargs = model.val.call_args_list[0].kwargs
            self.assertEqual(kwargs["data"], str(self.dataset.yaml_path.resolve()))
            self.assertEqual(kwargs["classes"], [0, 1, 2, 7, 8, 9])
            self.assertFalse(kwargs["exist_ok"])

    def test_validation_failure_preserves_existing_output_and_discards_staging(self) -> None:
        output_dir = self.project / "real_val_comparison"
        output_dir.mkdir(parents=True)
        marker = output_dir / "marker.txt"
        marker.write_text("existing", encoding="utf-8")

        def factory(weights: str) -> mock.Mock:
            if weights.endswith("v8.pt"):
                raise RuntimeError("candidate failed")
            model = mock.Mock()

            def val(**kwargs: object) -> SimpleNamespace:
                save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                save_dir.mkdir(parents=True)
                return self.fake_result(save_dir, 0.0)

            model.val.side_effect = val
            return model

        with self.assertRaisesRegex(RuntimeError, "candidate failed"):
            with redirect_stdout(io.StringIO()):
                validate_real.main(self.argv(), yolo_factory=factory)

        self.assertEqual(marker.read_text(encoding="utf-8"), "existing")
        self.assertEqual(
            list(self.project.glob(".real_val_comparison.generation_*")), []
        )

        def successful_factory(weights: str) -> mock.Mock:
            model = mock.Mock()

            def val(**kwargs: object) -> SimpleNamespace:
                save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                save_dir.mkdir(parents=True)
                offset = 0.1 if weights.endswith("v8.pt") else 0.0
                return self.fake_result(save_dir, offset)

            model.val.side_effect = val
            return model

        with redirect_stdout(io.StringIO()):
            validate_real.main(self.argv(), yolo_factory=successful_factory)
        self.assertFalse(marker.exists())
        self.assertTrue((output_dir / "overall_metrics.json").is_file())

    def test_metadata_conditions_generate_yaml_and_run_each_subset(self) -> None:
        metadata = self.metadata
        models: list[mock.Mock] = []

        def factory(weights: str) -> mock.Mock:
            model = mock.Mock()

            def val(**kwargs: object) -> SimpleNamespace:
                save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                save_dir.mkdir(parents=True)
                offset = 0.1 if weights.endswith("v8.pt") else 0.0
                return self.fake_result(save_dir, offset)

            model.val.side_effect = val
            models.append(model)
            return model

        with redirect_stdout(io.StringIO()):
            validate_real.main(
                self.argv("--metadata", str(metadata)), yolo_factory=factory
            )

        output_dir = self.project / "real_val_comparison"
        condition_payload = json.loads(
            (output_dir / "condition_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(condition_payload["conditions"]), 3)
        self.assertEqual([model.val.call_count for model in models], [4, 4])
        self.assertEqual(
            len(list((output_dir / "condition_datasets").glob("*/dataset.yaml"))),
            3,
        )
        for dataset_yaml in (output_dir / "condition_datasets").glob(
            "*/dataset.yaml"
        ):
            payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
            self.assertEqual(payload["path"], str(dataset_yaml.parent))
            self.assertEqual(payload["val"], str(dataset_yaml.parent / "images.txt"))
        self.assertTrue((output_dir / "metadata_groups.csv").is_file())


if __name__ == "__main__":
    unittest.main()
