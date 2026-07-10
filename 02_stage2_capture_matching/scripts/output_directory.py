from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def create_generation_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise RuntimeError(f"output path must be a regular directory: {output_dir}")
    backup = backup_path(output_dir)
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(
            f"stale output backup requires manual recovery before rerun: {backup}"
        )
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.generation_",
            dir=output_dir.parent,
        )
    )


def publish_generation(generation_dir: Path, output_dir: Path) -> None:
    if not generation_dir.is_dir() or generation_dir.is_symlink():
        raise RuntimeError(f"generation path must be a regular directory: {generation_dir}")
    if generation_dir.parent.resolve() != output_dir.parent.resolve():
        raise RuntimeError("generation and output directories must share the same parent.")

    backup = backup_path(output_dir)
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"output backup already exists: {backup}")
    had_previous = output_dir.exists() or output_dir.is_symlink()
    if had_previous:
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise RuntimeError(f"output path must be a regular directory: {output_dir}")
        os.replace(output_dir, backup)

    try:
        os.replace(generation_dir, output_dir)
    except OSError as publish_error:
        if had_previous:
            try:
                os.replace(backup, output_dir)
            except OSError as restore_error:
                raise RuntimeError(
                    f"failed to publish {generation_dir} and restore previous output from {backup}"
                ) from restore_error
        raise RuntimeError(f"failed to publish output generation: {generation_dir}") from publish_error

    if had_previous:
        shutil.rmtree(backup)


def discard_generation(generation_dir: Path) -> None:
    if generation_dir.exists():
        shutil.rmtree(generation_dir)


def backup_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.previous"
