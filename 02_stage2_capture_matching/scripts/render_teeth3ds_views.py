from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv
from PIL import Image


UPPER_FRONT_LABELS = (11, 12, 13, 21, 22, 23)
DEGENERATE_AREA_RELATIVE_TOLERANCE = 64.0 * np.finfo(np.float64).eps
MANIFEST_FIELDS = (
    "patient_id",
    "case_id",
    "jaw",
    "rgb_scope",
    "rendered_fdi_labels",
    "view_id",
    "azimuth_deg",
    "elevation_deg",
    "camera_position",
    "focal_point",
    "view_up",
    "parallel_scale",
    "image_width",
    "image_height",
    "image_path",
    "label_path",
    "source_path",
    "source_sha256",
    "source_faces_total",
    "dropped_degenerate_faces",
    "degenerate_area_tolerance",
)
SOURCE_MANIFEST_FIELDS = (
    "source_index",
    "source_sha256",
    "source_faces_total",
    "dropped_degenerate_faces",
    "degenerate_area_tolerance",
)
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class CameraParameters:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_scale: float


@dataclass(frozen=True)
class SourceMesh:
    source_path: Path
    source_sha256: str
    source_faces_total: int
    dropped_degenerate_faces: int
    degenerate_area_tolerance: float
    patient_id: str
    case_id: str
    jaw: str
    vertices: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    framing_vertices: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_view(value: str) -> ViewSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("view must use VIEW_ID:AZIMUTH:ELEVATION")
    view_id, azimuth_text, elevation_text = parts
    if not SAFE_ID_PATTERN.fullmatch(view_id):
        raise ValueError(f"invalid view ID: {view_id!r}")
    try:
        azimuth = float(azimuth_text)
        elevation = float(elevation_text)
    except ValueError as error:
        raise ValueError("azimuth and elevation must be finite numbers") from error
    if not math.isfinite(azimuth) or not -180.0 <= azimuth <= 180.0:
        raise ValueError("azimuth must be finite and within [-180, 180]")
    if not math.isfinite(elevation) or not -80.0 <= elevation <= 80.0:
        raise ValueError("elevation must be finite and within [-80, 80]")
    return ViewSpec(view_id, azimuth, elevation)


def parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("image size must use WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("image width and height must be integers") from error
    if width < 32 or height < 32:
        raise ValueError("image width and height must each be at least 32 pixels")
    return width, height


def _read_scalar_text(data: np.lib.npyio.NpzFile, key: str, source_path: Path) -> str:
    if key not in data.files:
        raise ValueError(f"{source_path}: missing required array {key!r}")
    value = np.asarray(data[key])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{source_path}: {key!r} must be a scalar string")
    scalar = value.item()
    try:
        text = (
            scalar.decode("utf-8").strip()
            if isinstance(scalar, bytes)
            else str(scalar).strip()
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{source_path}: {key!r} must use UTF-8") from error
    if not text or any(character in text for character in "\r\n\0"):
        raise ValueError(f"{source_path}: invalid {key!r}")
    return text


def _validate_mesh_arrays(
    source_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    *,
    drop_degenerate_faces: bool,
) -> tuple[np.ndarray, int, int, float]:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ValueError(f"{source_path}: vertices must have shape [N, 3] with N >= 3")
    if not (
        np.issubdtype(vertices.dtype, np.floating)
        or np.issubdtype(vertices.dtype, np.integer)
    ) or not np.all(np.isfinite(vertices)):
        raise ValueError(f"{source_path}: vertices must contain only finite numbers")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
        raise ValueError(f"{source_path}: faces must have shape [M, 3] with M >= 1")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{source_path}: faces must contain integer indices")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError(f"{source_path}: face index out of bounds")
    if vertex_labels.shape != (vertices.shape[0],):
        raise ValueError(f"{source_path}: vertex_labels must have shape [N]")
    if not np.issubdtype(vertex_labels.dtype, np.integer):
        raise ValueError(f"{source_path}: vertex_labels must contain integers")
    if np.any(vertex_labels < 0) or np.any(vertex_labels > 255):
        raise ValueError(f"{source_path}: vertex_labels must fit uint8 label images")

    repeated_vertex = np.any(
        np.diff(np.sort(faces, axis=1), axis=1) == 0,
        axis=1,
    )
    vertices_float64 = np.asarray(vertices, dtype=np.float64)
    triangle_vertices = vertices_float64[faces]
    edge_vectors = np.concatenate(
        (
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 1],
            triangle_vertices[:, 0] - triangle_vertices[:, 2],
        ),
        axis=0,
    )
    edge_lengths_squared = np.einsum("ij,ij->i", edge_vectors, edge_vectors)
    max_edge_squared = float(edge_lengths_squared.max())
    if not math.isfinite(max_edge_squared):
        raise ValueError(f"{source_path}: mesh edge length exceeds float64 range")

    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    if not np.all(np.isfinite(triangle_areas)):
        raise ValueError(f"{source_path}: triangle area exceeds float64 range")
    # Area and squared edge length scale identically under uniform scaling.
    degenerate_area_tolerance = (
        max_edge_squared * DEGENERATE_AREA_RELATIVE_TOLERANCE
    )
    degenerate = repeated_vertex | (triangle_areas <= degenerate_area_tolerance)
    dropped_count = int(degenerate.sum())
    if dropped_count and not drop_degenerate_faces:
        if np.any(repeated_vertex):
            raise ValueError(f"{source_path}: faces must not repeat a vertex index")
        raise ValueError(f"{source_path}: mesh contains a degenerate face")
    filtered_faces = faces[~degenerate]
    if filtered_faces.shape[0] == 0:
        raise ValueError(f"{source_path}: mesh has no non-degenerate faces")
    return (
        filtered_faces,
        int(faces.shape[0]),
        dropped_count,
        degenerate_area_tolerance,
    )


def load_source_mesh(
    source_path: Path,
    *,
    drop_degenerate_faces: bool = False,
) -> SourceMesh:
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() != ".npz":
        raise ValueError(f"source must be an existing .npz file: {source_path}")
    source_sha256 = sha256_file(source_path)
    try:
        with np.load(source_path, allow_pickle=False) as data:
            required_arrays = {"vertices", "faces", "vertex_labels"}
            missing = required_arrays.difference(data.files)
            if missing:
                raise ValueError(
                    f"{source_path}: missing required arrays: {', '.join(sorted(missing))}"
                )
            vertices = np.asarray(data["vertices"])
            faces = np.asarray(data["faces"])
            vertex_labels = np.asarray(data["vertex_labels"])
            jaw = _read_scalar_text(data, "jaw", source_path)
            patient_id = _read_scalar_text(data, "patient_id", source_path)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(str(source_path)):
            raise
        raise ValueError(f"failed to read NPZ source {source_path}: {error}") from error
    if sha256_file(source_path) != source_sha256:
        raise RuntimeError(f"source changed while being read: {source_path}")

    if jaw == "lower":
        raise ValueError(f"{source_path}: lower jaw is unsupported; renderer is upper jaw only")
    if jaw != "upper":
        raise ValueError(f"{source_path}: unknown jaw {jaw!r}; expected 'upper'")
    case_id = source_path.stem
    if not SAFE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"{source_path}: filename stem is not a safe case ID")
    (
        faces,
        source_faces_total,
        dropped_degenerate_faces,
        degenerate_area_tolerance,
    ) = _validate_mesh_arrays(
        source_path,
        vertices,
        faces,
        vertex_labels,
        drop_degenerate_faces=drop_degenerate_faces,
    )

    labels_per_face = vertex_labels[faces]
    unanimous = np.all(labels_per_face == labels_per_face[:, :1], axis=1)
    face_labels = np.where(unanimous, labels_per_face[:, 0], 0)
    front_labels = np.asarray(UPPER_FRONT_LABELS)
    face_labels = np.where(np.isin(face_labels, front_labels), face_labels, 0).astype(np.uint8)
    framing_indices = np.unique(faces[face_labels != 0])
    if framing_indices.size == 0:
        raise ValueError(f"{source_path}: mesh has no labeled front-tooth faces for jaw {jaw!r}")

    return SourceMesh(
        source_path=source_path,
        source_sha256=source_sha256,
        source_faces_total=source_faces_total,
        dropped_degenerate_faces=dropped_degenerate_faces,
        degenerate_area_tolerance=degenerate_area_tolerance,
        patient_id=patient_id,
        case_id=case_id,
        jaw=jaw,
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        face_labels=face_labels,
        framing_vertices=np.asarray(vertices[framing_indices], dtype=np.float64),
    )


def _validate_views(views: Sequence[ViewSpec]) -> None:
    if len(views) < 2:
        raise ValueError("at least two distinct views are required")
    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("view IDs must be unique")
    angle_pairs = [(view.azimuth_deg, view.elevation_deg) for view in views]
    if len(set(angle_pairs)) != len(angle_pairs):
        raise ValueError("camera angles must be unique")
    for view in views:
        parse_view(f"{view.view_id}:{view.azimuth_deg}:{view.elevation_deg}")


def _camera_parameters(
    vertices: np.ndarray,
    view: ViewSpec,
    image_size: tuple[int, int],
) -> CameraParameters:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    focal_point = (minimum + maximum) / 2.0
    azimuth = math.radians(view.azimuth_deg)
    elevation = math.radians(view.elevation_deg)
    direction = np.asarray(
        (
            math.cos(elevation) * math.sin(azimuth),
            -math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        )
    )
    world_up = np.asarray((0.0, 0.0, 1.0))
    view_up = world_up - np.dot(world_up, direction) * direction
    view_up /= np.linalg.norm(view_up)
    view_right = np.cross(direction, view_up)
    view_right /= np.linalg.norm(view_right)

    centered = vertices - focal_point
    vertical_span = np.ptp(centered @ view_up)
    horizontal_span = np.ptp(centered @ view_right)
    aspect_ratio = image_size[0] / image_size[1]
    parallel_scale = max(vertical_span / 2.0, horizontal_span / (2.0 * aspect_ratio)) * 1.08
    if not math.isfinite(parallel_scale) or parallel_scale <= 0.0:
        raise ValueError("mesh bounds cannot define a valid camera scale")
    diagonal = float(np.linalg.norm(maximum - minimum))
    position = focal_point + direction * max(diagonal * 2.0, 1.0)

    return CameraParameters(
        position=tuple(float(value) for value in position),
        focal_point=tuple(float(value) for value in focal_point),
        view_up=tuple(float(value) for value in view_up),
        parallel_scale=float(parallel_scale),
    )


def _poly_data(source: SourceMesh) -> pv.PolyData:
    front_face_mask = source.face_labels != 0
    front_faces = source.faces[front_face_mask]
    padded_faces = np.column_stack(
        (np.full(front_faces.shape[0], 3, dtype=np.int64), front_faces)
    ).ravel()
    mesh = pv.PolyData(source.vertices, padded_faces)
    mesh.cell_data["fdi_rgb"] = np.repeat(
        source.face_labels[front_face_mask, None],
        3,
        axis=1,
    )
    return mesh


def _configure_plotter(
    mesh: pv.PolyData,
    camera: CameraParameters,
    image_size: tuple[int, int],
    *,
    label_render: bool,
) -> pv.Plotter:
    plotter = pv.Plotter(
        off_screen=True,
        window_size=image_size,
        lighting="none" if label_render else "light kit",
    )
    plotter.render_window.SetMultiSamples(0)
    plotter.set_background("black" if label_render else "white")
    if label_render:
        plotter.add_mesh(
            mesh,
            scalars="fdi_rgb",
            preference="cell",
            rgb=True,
            lighting=False,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(
            mesh,
            color=(0.70, 0.75, 0.80),
            ambient=0.35,
            diffuse=0.55,
            specular=0.05,
            smooth_shading=False,
            show_scalar_bar=False,
        )
    plotter.camera_position = [camera.position, camera.focal_point, camera.view_up]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = camera.parallel_scale
    plotter.reset_camera_clipping_range()
    return plotter


def _screenshot(
    mesh: pv.PolyData,
    camera: CameraParameters,
    image_size: tuple[int, int],
    *,
    label_render: bool,
) -> np.ndarray:
    plotter = _configure_plotter(
        mesh,
        camera,
        image_size,
        label_render=label_render,
    )
    try:
        image = np.asarray(plotter.screenshot(return_img=True))
    finally:
        plotter.close()
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise RuntimeError(f"VTK returned an unexpected screenshot shape: {image.shape}")
    return np.asarray(image[:, :, :3], dtype=np.uint8)


def _render_view(
    source: SourceMesh,
    view: ViewSpec,
    image_size: tuple[int, int],
    image_path: Path,
    label_path: Path,
) -> CameraParameters:
    camera = _camera_parameters(source.framing_vertices, view, image_size)
    mesh = _poly_data(source)
    rgb_image = _screenshot(mesh, camera, image_size, label_render=False)
    encoded_labels = _screenshot(mesh, camera, image_size, label_render=True)
    if not np.array_equal(encoded_labels[:, :, 0], encoded_labels[:, :, 1]) or not np.array_equal(
        encoded_labels[:, :, 0], encoded_labels[:, :, 2]
    ):
        raise RuntimeError("VTK changed direct FDI label colors during rendering")
    label_image = encoded_labels[:, :, 0]
    allowed_labels = np.asarray((0, *UPPER_FRONT_LABELS), dtype=np.uint8)
    if not np.all(np.isin(label_image, allowed_labels)):
        unexpected = np.unique(label_image[~np.isin(label_image, allowed_labels)])
        raise RuntimeError(f"VTK produced unexpected FDI label values: {unexpected.tolist()}")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_image, mode="RGB").save(image_path)
    Image.fromarray(label_image, mode="L").save(label_path)
    return camera


def _json_vector(values: tuple[float, float, float]) -> str:
    return json.dumps([round(value, 12) for value in values], separators=(",", ":"))


def _manifest_row(
    source: SourceMesh,
    view: ViewSpec,
    camera: CameraParameters,
    image_size: tuple[int, int],
    image_path: Path,
    label_path: Path,
) -> dict[str, object]:
    return {
        "patient_id": source.patient_id,
        "case_id": source.case_id,
        "jaw": source.jaw,
        "rgb_scope": "front_teeth_only",
        "rendered_fdi_labels": list(UPPER_FRONT_LABELS),
        "view_id": view.view_id,
        "azimuth_deg": view.azimuth_deg,
        "elevation_deg": view.elevation_deg,
        "camera_position": [round(value, 12) for value in camera.position],
        "focal_point": [round(value, 12) for value in camera.focal_point],
        "view_up": [round(value, 12) for value in camera.view_up],
        "parallel_scale": round(camera.parallel_scale, 12),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "image_path": image_path.as_posix(),
        "label_path": label_path.as_posix(),
        "source_path": str(source.source_path),
        "source_sha256": source.source_sha256,
        "source_faces_total": source.source_faces_total,
        "dropped_degenerate_faces": source.dropped_degenerate_faces,
        "degenerate_area_tolerance": source.degenerate_area_tolerance,
    }


def _write_manifests(output_dir: Path, rows: Sequence[dict[str, object]]) -> None:
    json_document = {
        "schema_version": 1,
        "label_encoding": "uint8_fdi",
        "rgb_scope": "front_teeth_only",
        "degenerate_area_relative_tolerance": DEGENERATE_AREA_RELATIVE_TOLERANCE,
        "views": list(rows),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_document, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["camera_position"] = _json_vector(tuple(row["camera_position"]))
            csv_row["focal_point"] = _json_vector(tuple(row["focal_point"]))
            csv_row["view_up"] = _json_vector(tuple(row["view_up"]))
            csv_row["rendered_fdi_labels"] = json.dumps(
                row["rendered_fdi_labels"],
                separators=(",", ":"),
            )
            writer.writerow(csv_row)


def _write_source_manifest(output_dir: Path, sources: Sequence[SourceMesh]) -> None:
    public_sources = sorted(sources, key=lambda source: source.source_sha256)
    with (output_dir / "source_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SOURCE_MANIFEST_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for source_index, source in enumerate(public_sources, start=1):
            writer.writerow(
                {
                    "source_index": source_index,
                    "source_sha256": source.source_sha256,
                    "source_faces_total": source.source_faces_total,
                    "dropped_degenerate_faces": source.dropped_degenerate_faces,
                    "degenerate_area_tolerance": format(
                        source.degenerate_area_tolerance,
                        ".17g",
                    ),
                }
            )


def _reject_existing_output(
    output_dir: Path,
    sources: Sequence[SourceMesh],
    views: Sequence[ViewSpec],
) -> None:
    if not output_dir.exists() and not output_dir.is_symlink():
        return
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir() or not manifest_path.is_file():
        raise FileExistsError(
            f"output path already exists and is not a renderer output: {output_dir}"
        )
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_rows = document["views"]
        if not isinstance(existing_rows, list):
            raise TypeError("views is not a list")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid existing manifest {manifest_path}: {error}") from error

    existing_by_case: dict[str, list[dict[str, object]]] = {}
    for row in existing_rows:
        if not isinstance(row, dict) or "case_id" not in row or "source_sha256" not in row:
            raise RuntimeError(f"invalid existing manifest row in {manifest_path}")
        existing_by_case.setdefault(str(row["case_id"]), []).append(row)
    for source in sources:
        case_rows = existing_by_case.get(source.case_id, [])
        hashes = {str(row["source_sha256"]) for row in case_rows}
        if hashes and hashes != {source.source_sha256}:
            raise RuntimeError(f"source SHA-256 mismatch for existing case {source.case_id!r}")
        existing_view_ids = {str(row.get("view_id")) for row in case_rows}
        duplicate_view_ids = existing_view_ids.intersection(view.view_id for view in views)
        if duplicate_view_ids:
            duplicate = sorted(duplicate_view_ids)[0]
            raise FileExistsError(
                f"refusing to overwrite view {duplicate!r} for case {source.case_id!r}"
            )
    raise FileExistsError(f"output directory already exists: {output_dir}")


def render_sources(
    source_paths: Sequence[Path],
    output_dir: Path,
    views: Sequence[ViewSpec],
    *,
    image_size: tuple[int, int] = (512, 512),
    drop_degenerate_faces: bool = False,
) -> list[dict[str, object]]:
    _validate_views(views)
    if (
        len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in image_size)
        or image_size[0] < 32
        or image_size[1] < 32
    ):
        raise ValueError("image width and height must each be at least 32 pixels")
    if not source_paths:
        raise ValueError("at least one NPZ source is required")
    sources = [
        load_source_mesh(
            path,
            drop_degenerate_faces=drop_degenerate_faces,
        )
        for path in source_paths
    ]
    case_ids = [source.case_id for source in sources]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("source case IDs must be unique")

    output_dir = output_dir.expanduser().resolve()
    _reject_existing_output(output_dir, sources, views)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.render-",
            dir=output_dir.parent,
        )
    )
    rows: list[dict[str, object]] = []
    try:
        for source in sorted(sources, key=lambda item: item.case_id):
            for view in views:
                image_relative = Path("images") / f"{source.case_id}__{view.view_id}.png"
                label_relative = Path("labels") / f"{source.case_id}__{view.view_id}.png"
                camera = _render_view(
                    source,
                    view,
                    image_size,
                    staging_dir / image_relative,
                    staging_dir / label_relative,
                )
                rows.append(
                    _manifest_row(
                        source,
                        view,
                        camera,
                        image_size,
                        image_relative,
                        label_relative,
                    )
                )
        _write_manifests(staging_dir, rows)
        _write_source_manifest(staging_dir, sources)
        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return rows


def collect_sources(paths: Sequence[Path]) -> tuple[Path, ...]:
    collected: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            directory_sources = sorted(resolved.glob("*.npz"))
            if not directory_sources:
                raise ValueError(f"source directory contains no NPZ files: {resolved}")
            collected.extend(directory_sources)
        elif resolved.is_file() and resolved.suffix.lower() == ".npz":
            collected.append(resolved)
        else:
            raise ValueError(f"source must be an NPZ file or directory: {resolved}")
    if len(set(collected)) != len(collected):
        raise ValueError("duplicate NPZ source path")
    return tuple(collected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic, pixel-aligned RGB and FDI label views "
            "from Teeth3DS NPZ meshes."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        help="NPZ file or a directory of NPZ files; repeat for multiple inputs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        type=parse_view,
        help="Explicit camera as VIEW_ID:AZIMUTH:ELEVATION; provide at least two.",
    )
    parser.add_argument("--image-size", type=parse_image_size, default=(512, 512))
    parser.add_argument(
        "--drop-degenerate-faces",
        action="store_true",
        help=(
            "Explicitly remove faces below the scale-relative area tolerance "
            "and record their count in the manifest."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = collect_sources(args.source)
    rows = render_sources(
        sources,
        args.output_dir,
        args.view,
        image_size=args.image_size,
        drop_degenerate_faces=args.drop_degenerate_faces,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "sources": len(sources),
                "views": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
