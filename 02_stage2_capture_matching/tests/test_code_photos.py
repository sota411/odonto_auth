from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from code_photos import resolve_photo_reference  # noqa: E402


class ResolvePhotoReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = TemporaryDirectory()
        self.root = Path(self.temp_dir_context.name) / "COde"
        self.photo_dir = self.root / "Images" / "Photographs"
        self.photo_dir.mkdir(parents=True)
        self.photo = self.photo_dir / "sample.jpg"
        self.photo.write_bytes(b"image")

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    def test_resolves_bare_and_prefixed_references(self) -> None:
        self.assertEqual(resolve_photo_reference(self.root, "sample.jpg"), self.photo)
        self.assertEqual(
            resolve_photo_reference(self.root, "Images/Photographs/sample.jpg"),
            self.photo,
        )

    def test_rejects_a_symlink_that_escapes_the_root(self) -> None:
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes(b"outside")
        (self.photo_dir / "escape.jpg").symlink_to(outside)

        with self.assertRaisesRegex(RuntimeError, "escapes"):
            resolve_photo_reference(self.root, "escape.jpg")

    def test_rejects_a_symlink_to_a_sibling_image_directory(self) -> None:
        radiograph_dir = self.root / "Images" / "Radiographs"
        radiograph_dir.mkdir()
        radiograph = radiograph_dir / "radiograph.jpg"
        radiograph.write_bytes(b"radiograph")
        (self.photo_dir / "sibling.jpg").symlink_to(radiograph)

        with self.assertRaisesRegex(RuntimeError, "photograph directory"):
            resolve_photo_reference(self.root, "sibling.jpg")


if __name__ == "__main__":
    unittest.main()
