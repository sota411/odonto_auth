from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import render_teeth3ds_views  # noqa: E402


UPPER_FRONT_LABELS = (13, 12, 11, 21, 22, 23)


class RenderTeeth3dsViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "case-a_upper_multitooth.npz"
        self.write_synthetic_mesh(self.source)
        self.views = (
            render_teeth3ds_views.ViewSpec("front", 0.0, 0.0),
            render_teeth3ds_views.ViewSpec("oblique", 20.0, 8.0),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_synthetic_mesh(
        self,
        path: Path,
        *,
        jaw: str = "upper",
        invalid_face: bool = False,
        repeated_vertex_face: bool = False,
        near_degenerate_face: bool = False,
        x_offset: float = 0.0,
        include_far_background: bool = False,
        scale: float = 1.0,
    ) -> None:
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        labels: list[int] = []
        for index, label in enumerate(UPPER_FRONT_LABELS):
            x0 = x_offset + (index - 3) * 2.2
            x1 = x0 + 1.8
            z0 = -1.0
            z1 = 1.0
            first = len(vertices)
            vertices.extend(
                (
                    (x0, 0.0, z0),
                    (x1, 0.0, z0),
                    (x1, 0.0, z1),
                    (x0, 0.0, z1),
                )
            )
            faces.extend(((first, first + 2, first + 1), (first, first + 3, first + 2)))
            labels.extend((label,) * 4)
        if include_far_background:
            first = len(vertices)
            vertices.extend(((-100.0, 20.0, -50.0), (100.0, 20.0, -50.0), (0.0, 20.0, 50.0)))
            faces.append((first, first + 1, first + 2))
            labels.extend((0, 0, 0))
        if invalid_face:
            faces[0] = (0, 1, len(vertices))
        if repeated_vertex_face:
            faces[0] = (0, 0, 1)
        if near_degenerate_face:
            first = len(vertices)
            vertices.extend(((0.0, 10.0, 0.0), (1.0, 10.0, 0.0), (1.0, 10.0, 1e-16)))
            faces.append((first, first + 1, first + 2))
            labels.extend((11, 11, 11))
        np.savez(
            path,
            vertices=(np.asarray(vertices, dtype=np.float64) * scale).astype(np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            vertex_labels=np.asarray(labels, dtype=np.int32),
            jaw=np.asarray(jaw),
            patient_id=np.asarray("case-a"),
        )

    def render(self, output_dir: Path) -> list[dict[str, object]]:
        return render_teeth3ds_views.render_sources(
            (self.source,),
            output_dir,
            self.views,
            image_size=(192, 128),
        )

    def test_renders_two_deterministic_views_and_provenance_manifests(self) -> None:
        first_output = self.root / "first"
        second_output = self.root / "second"

        first_rows = self.render(first_output)
        second_rows = self.render(second_output)

        self.assertEqual(first_rows, second_rows)
        self.assertEqual(len(first_rows), 2)
        self.assertEqual(
            sorted(
                path.relative_to(first_output)
                for path in first_output.rglob("*")
                if path.is_file()
            ),
            sorted(
                path.relative_to(second_output)
                for path in second_output.rglob("*")
                if path.is_file()
            ),
        )
        for first_path in sorted(path for path in first_output.rglob("*") if path.is_file()):
            relative_path = first_path.relative_to(first_output)
            self.assertEqual(first_path.read_bytes(), (second_output / relative_path).read_bytes())

        with (first_output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        json_manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
        expected_sha256 = hashlib.sha256(self.source.read_bytes()).hexdigest()

        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(len(json_manifest["views"]), 2)
        self.assertEqual(json_manifest["label_encoding"], "uint8_fdi")
        self.assertEqual({row["patient_id"] for row in csv_rows}, {"case-a"})
        self.assertEqual({row["case_id"] for row in csv_rows}, {"case-a_upper_multitooth"})
        self.assertEqual({row["view_id"] for row in csv_rows}, {"front", "oblique"})
        self.assertEqual({row["source_sha256"] for row in csv_rows}, {expected_sha256})
        self.assertTrue(all(float(row["degenerate_area_tolerance"]) > 0.0 for row in csv_rows))
        self.assertEqual({row["rgb_scope"] for row in csv_rows}, {"front_teeth_only"})
        self.assertEqual(
            {tuple(json.loads(row["rendered_fdi_labels"])) for row in csv_rows},
            {(11, 12, 13, 21, 22, 23)},
        )
        self.assertTrue(all(json.loads(row["camera_position"]) for row in csv_rows))
        self.assertTrue(all((first_output / row["image_path"]).is_file() for row in csv_rows))
        self.assertTrue(all((first_output / row["label_path"]).is_file() for row in csv_rows))

    def test_writes_a_deterministic_public_source_manifest_without_identifiers(self) -> None:
        second_source = self.root / "private-case-b_upper_multitooth.npz"
        self.write_synthetic_mesh(second_source, x_offset=0.5)
        expected_hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.source, second_source)
        }
        first_output = self.root / "public-first"
        second_output = self.root / "public-second"

        for output_dir in (first_output, second_output):
            render_teeth3ds_views.render_sources(
                (second_source, self.source),
                output_dir,
                self.views,
                image_size=(64, 64),
            )

        first_manifest = first_output / "source_manifest.csv"
        second_manifest = second_output / "source_manifest.csv"
        self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
        with first_manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(
            reader.fieldnames,
            [
                "source_index",
                "source_sha256",
                "source_faces_total",
                "dropped_degenerate_faces",
                "degenerate_area_tolerance",
            ],
        )
        self.assertEqual([row["source_index"] for row in rows], ["1", "2"])
        self.assertEqual({row["source_sha256"] for row in rows}, expected_hashes)
        self.assertEqual({row["source_faces_total"] for row in rows}, {"12"})
        self.assertEqual({row["dropped_degenerate_faces"] for row in rows}, {"0"})
        self.assertTrue(all(float(row["degenerate_area_tolerance"]) > 0.0 for row in rows))
        manifest_text = first_manifest.read_text(encoding="utf-8")
        self.assertNotIn("patient_id", manifest_text)
        self.assertNotIn("case_id", manifest_text)
        self.assertNotIn("source_path", manifest_text)
        self.assertNotIn("case-a", manifest_text)
        self.assertNotIn("private-case-b", manifest_text)

    def test_rgb_and_fdi_label_pixels_are_aligned(self) -> None:
        output_dir = self.root / "aligned"
        rows = self.render(output_dir)
        front = next(row for row in rows if row["view_id"] == "front")

        rgb = np.asarray(Image.open(output_dir / str(front["image_path"])).convert("RGB"))
        labels = np.asarray(Image.open(output_dir / str(front["label_path"])))
        rgb_foreground = np.any(rgb != 255, axis=2)
        label_foreground = labels != 0

        self.assertEqual(rgb.shape[:2], labels.shape)
        np.testing.assert_array_equal(rgb_foreground, label_foreground)
        self.assertEqual(set(np.unique(labels)), {0, *UPPER_FRONT_LABELS})

    def test_camera_framing_uses_front_teeth_instead_of_non_target_geometry(self) -> None:
        self.write_synthetic_mesh(self.source, include_far_background=True)
        output_dir = self.root / "front-framing"

        rows = self.render(output_dir)
        front = next(row for row in rows if row["view_id"] == "front")
        labels = np.asarray(Image.open(output_dir / str(front["label_path"])))
        rgb = np.asarray(Image.open(output_dir / str(front["image_path"])).convert("RGB"))
        occupied_columns = np.flatnonzero(np.any(labels != 0, axis=0))

        self.assertGreaterEqual(occupied_columns.size, 150)
        np.testing.assert_array_equal(np.any(rgb != 255, axis=2), labels != 0)

    def test_rejects_fewer_than_two_distinct_camera_angles(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two distinct views"):
            render_teeth3ds_views.render_sources(
                (self.source,),
                self.root / "one-view",
                (render_teeth3ds_views.ViewSpec("front", 0.0, 0.0),),
                image_size=(64, 64),
            )

        with self.assertRaisesRegex(ValueError, "camera angles must be unique"):
            render_teeth3ds_views.render_sources(
                (self.source,),
                self.root / "duplicate-angle",
                (
                    render_teeth3ds_views.ViewSpec("front", 0.0, 0.0),
                    render_teeth3ds_views.ViewSpec("same-camera", 0.0, 0.0),
                ),
                image_size=(64, 64),
            )

    def test_rejects_unknown_jaw(self) -> None:
        self.write_synthetic_mesh(self.source, jaw="sideways")

        with self.assertRaisesRegex(ValueError, "unknown jaw"):
            self.render(self.root / "unknown-jaw")

    def test_rejects_lower_jaw_for_the_upper_only_evaluator_contract(self) -> None:
        self.write_synthetic_mesh(self.source, jaw="lower")

        with self.assertRaisesRegex(ValueError, "upper jaw only"):
            self.render(self.root / "lower-jaw")

    def test_rejects_an_invalid_mesh(self) -> None:
        self.write_synthetic_mesh(self.source, invalid_face=True)

        with self.assertRaisesRegex(ValueError, "face index out of bounds"):
            self.render(self.root / "invalid-mesh")

    def test_explicitly_drops_degenerate_faces_and_records_the_count(self) -> None:
        self.write_synthetic_mesh(self.source, repeated_vertex_face=True)

        with self.assertRaisesRegex(ValueError, "repeat a vertex index"):
            self.render(self.root / "strict-invalid-mesh")

        output_dir = self.root / "repaired-mesh"
        rows = render_teeth3ds_views.render_sources(
            (self.source,),
            output_dir,
            self.views,
            image_size=(192, 128),
            drop_degenerate_faces=True,
        )

        self.assertEqual({row["source_faces_total"] for row in rows}, {12})
        self.assertEqual({row["dropped_degenerate_faces"] for row in rows}, {1})
        self.assertTrue(all(float(row["degenerate_area_tolerance"]) > 0.0 for row in rows))
        document = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {row["dropped_degenerate_faces"] for row in document["views"]},
            {1},
        )
        self.assertTrue(
            all(row["degenerate_area_tolerance"] > 0.0 for row in document["views"])
        )
        with (output_dir / "source_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            public_rows = list(csv.DictReader(handle))
        self.assertEqual(len(public_rows), 1)
        self.assertEqual(public_rows[0]["source_faces_total"], "12")
        self.assertEqual(public_rows[0]["dropped_degenerate_faces"], "1")
        self.assertGreater(float(public_rows[0]["degenerate_area_tolerance"]), 0.0)

    def test_keeps_small_but_valid_triangles(self) -> None:
        self.write_synthetic_mesh(self.source, scale=1e-6)

        source = render_teeth3ds_views.load_source_mesh(self.source)

        self.assertEqual(source.source_faces_total, 12)
        self.assertEqual(source.dropped_degenerate_faces, 0)
        self.assertGreater(source.degenerate_area_tolerance, 0.0)

    def test_degenerate_judgment_is_invariant_under_uniform_scaling(self) -> None:
        normalized_tolerances: list[float] = []
        for index, scale in enumerate((1.0, 1e6)):
            source_path = self.root / f"scaled-{index}_upper_multitooth.npz"
            self.write_synthetic_mesh(
                source_path,
                near_degenerate_face=True,
                scale=scale,
            )

            with self.assertRaisesRegex(ValueError, "degenerate face"):
                render_teeth3ds_views.load_source_mesh(source_path)
            source = render_teeth3ds_views.load_source_mesh(
                source_path,
                drop_degenerate_faces=True,
            )

            self.assertEqual(source.source_faces_total, 13)
            self.assertEqual(source.dropped_degenerate_faces, 1)
            normalized_tolerances.append(source.degenerate_area_tolerance / scale**2)

        np.testing.assert_allclose(
            normalized_tolerances[0],
            normalized_tolerances[1],
            rtol=1e-6,
            atol=0.0,
        )

    def test_rejects_overwriting_the_same_views(self) -> None:
        output_dir = self.root / "existing"
        self.render(output_dir)

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite view"):
            self.render(output_dir)

    def test_rejects_a_changed_source_for_an_existing_case(self) -> None:
        output_dir = self.root / "source-mismatch"
        self.render(output_dir)
        self.write_synthetic_mesh(self.source, x_offset=0.5)

        with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
            self.render(output_dir)

    def test_view_cli_parser_is_explicit_and_strict(self) -> None:
        view = render_teeth3ds_views.parse_view("right:-15:5")

        self.assertEqual(view, render_teeth3ds_views.ViewSpec("right", -15.0, 5.0))
        with self.assertRaisesRegex(ValueError, "VIEW_ID:AZIMUTH:ELEVATION"):
            render_teeth3ds_views.parse_view("front")
        with self.assertRaisesRegex(ValueError, "elevation"):
            render_teeth3ds_views.parse_view("front:0:90")


if __name__ == "__main__":
    unittest.main()
