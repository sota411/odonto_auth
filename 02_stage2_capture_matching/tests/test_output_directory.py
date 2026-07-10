from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from output_directory import (  # noqa: E402
    backup_path,
    create_generation_directory,
    publish_generation,
)


class OutputDirectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_context = TemporaryDirectory()
        self.parent = Path(self.temp_dir_context.name)
        self.output_dir = self.parent / "result"

    def tearDown(self) -> None:
        self.temp_dir_context.cleanup()

    def test_replaces_the_complete_previous_generation(self) -> None:
        self.output_dir.mkdir()
        (self.output_dir / "stale.txt").write_text("old", encoding="utf-8")
        generation = create_generation_directory(self.output_dir)
        (generation / "current.txt").write_text("new", encoding="utf-8")

        publish_generation(generation, self.output_dir)

        self.assertEqual((self.output_dir / "current.txt").read_text(encoding="utf-8"), "new")
        self.assertFalse((self.output_dir / "stale.txt").exists())
        self.assertFalse(backup_path(self.output_dir).exists())

    def test_restores_the_previous_generation_when_publish_fails(self) -> None:
        self.output_dir.mkdir()
        (self.output_dir / "stable.txt").write_text("old", encoding="utf-8")
        generation = create_generation_directory(self.output_dir)
        (generation / "candidate.txt").write_text("new", encoding="utf-8")
        real_replace = os.replace
        call_count = 0

        def fail_second_replace(source: Path, destination: Path) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("injected publish failure")
            real_replace(source, destination)

        with mock.patch("output_directory.os.replace", side_effect=fail_second_replace):
            with self.assertRaisesRegex(RuntimeError, "failed to publish output generation"):
                publish_generation(generation, self.output_dir)

        self.assertEqual((self.output_dir / "stable.txt").read_text(encoding="utf-8"), "old")
        self.assertFalse((self.output_dir / "candidate.txt").exists())
        self.assertTrue(generation.is_dir())
        self.assertFalse(backup_path(self.output_dir).exists())

    def test_rejects_a_stale_backup(self) -> None:
        backup_path(self.output_dir).mkdir()

        with self.assertRaisesRegex(RuntimeError, "manual recovery"):
            create_generation_directory(self.output_dir)

    def test_publishes_a_relative_output_path(self) -> None:
        previous_working_directory = Path.cwd()
        try:
            os.chdir(self.parent)
            output_dir = Path("result")
            generation = create_generation_directory(output_dir)
            (generation / "current.txt").write_text("new", encoding="utf-8")

            publish_generation(generation, output_dir)

            self.assertEqual(
                (output_dir / "current.txt").read_text(encoding="utf-8"),
                "new",
            )
        finally:
            os.chdir(previous_working_directory)

    def test_rejects_a_dangling_output_symlink(self) -> None:
        self.output_dir.symlink_to(self.parent / "missing-target")

        with self.assertRaisesRegex(RuntimeError, "regular directory"):
            create_generation_directory(self.output_dir)


if __name__ == "__main__":
    unittest.main()
