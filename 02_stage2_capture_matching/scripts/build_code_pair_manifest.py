from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from code_photos import sha256_file


REQUIRED_COLUMNS = ("patient_id", "checkup_id", "photographs")
IMAGE_SUFFIX_PATTERN = re.compile(r"\.(?:jpg|jpeg|png)\Z", re.IGNORECASE)
PHOTO_REF_PATTERN = re.compile(r"Images/Photographs/[^,\s'\"\]\[]+\.(?:jpg|jpeg|png)", re.IGNORECASE)
FALLBACK_IMAGE_PATTERN = re.compile(r"[^,\s'\"\]\[]+\.(?:jpg|jpeg|png)", re.IGNORECASE)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Checkup:
    patient_id: str
    checkup_id: str
    photographs: tuple[str, ...]
    source_row_number: int

    @property
    def uid(self) -> str:
        return f"{self.patient_id}:{self.checkup_id}"


@dataclass(frozen=True)
class Pair:
    split: str
    pair_id: str
    label: str
    template: Checkup
    query: Checkup

    @property
    def is_genuine(self) -> bool:
        return self.label == "genuine"


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description=(
            "Build patient-level splits and genuine/impostor checkup pairs from "
            "COde complete_dataset.csv without downloading images or running ML."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to COde complete_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "02_stage2_capture_matching" / "logs" / "code_pair_manifest",
        help="Directory for checkups.csv, pairs.csv, and summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient shuffling and impostor sampling.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Patient-level train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Patient-level validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Patient-level test split ratio.",
    )
    parser.add_argument(
        "--impostors-per-genuine",
        type=int,
        default=1,
        help="Number of sampled impostor pairs for each genuine pair.",
    )
    parser.add_argument(
        "--min-photos",
        type=int,
        default=1,
        help="Skip checkups with fewer photograph references than this value.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.impostors_per_genuine < 0:
        raise RuntimeError("--impostors-per-genuine must be non-negative.")
    if args.min_photos < 0:
        raise RuntimeError("--min-photos must be non-negative.")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio < 0.0 for ratio in ratios):
        raise RuntimeError("split ratios must be non-negative.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "--train-ratio, --val-ratio, and --test-ratio must sum to 1.0; "
            f"got {sum(ratios):.12f}."
        )


def clean_required(value: str, column: str, row_number: int) -> str:
    cleaned = value.strip()
    if cleaned == "":
        raise RuntimeError(f"{column} is empty at CSV row {row_number}.")
    return cleaned


def flatten_literal_photos(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        flattened: list[str] = []
        for item in value.values():
            flattened.extend(flatten_literal_photos(item))
        return flattened
    if isinstance(value, list | tuple):
        flattened = []
        for item in value:
            flattened.extend(flatten_literal_photos(item))
        return flattened
    raise RuntimeError(f"unsupported photograph literal value: {value!r}")


def extract_image_refs(raw_value: str) -> tuple[str, ...]:
    raw = raw_value.strip()
    if raw == "":
        return ()

    refs: list[str]
    if raw[0] in "[{(":
        try:
            literal_value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            refs = PHOTO_REF_PATTERN.findall(raw)
            if not refs:
                refs = FALLBACK_IMAGE_PATTERN.findall(raw)
        else:
            refs = []
            for item in flatten_literal_photos(literal_value):
                cleaned_item = item.strip()
                if IMAGE_SUFFIX_PATTERN.search(cleaned_item):
                    refs.append(cleaned_item)
                    continue

                embedded_refs = PHOTO_REF_PATTERN.findall(cleaned_item)
                if not embedded_refs:
                    embedded_refs = FALLBACK_IMAGE_PATTERN.findall(cleaned_item)
                if not embedded_refs:
                    raise RuntimeError(f"photograph literal does not contain an image path: {item!r}")
                refs.extend(embedded_refs)
    else:
        refs = PHOTO_REF_PATTERN.findall(raw)
        if not refs:
            refs = FALLBACK_IMAGE_PATTERN.findall(raw)

    normalized: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        cleaned = ref.strip().strip("'\"")
        if cleaned == "":
            continue
        if cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return tuple(normalized)


def validate_fieldnames(fieldnames: list[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise RuntimeError(f"CSV has no header: {path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise RuntimeError(f"CSV is missing required columns {missing}: {path}")


def load_checkups(path: Path, min_photos: int) -> tuple[list[Checkup], int, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        validate_fieldnames(reader.fieldnames, path)
        checkups: list[Checkup] = []
        seen_keys: dict[tuple[str, str], int] = {}
        source_rows = 0
        skipped_min_photos = 0

        for row_number, row in enumerate(reader, start=2):
            source_rows += 1
            patient_id = clean_required(row["patient_id"], "patient_id", row_number)
            checkup_id = clean_required(row["checkup_id"], "checkup_id", row_number)
            photographs = extract_image_refs(row["photographs"])
            if len(photographs) < min_photos:
                skipped_min_photos += 1
                continue

            key = (patient_id, checkup_id)
            if key in seen_keys:
                previous_row_number = seen_keys[key]
                raise RuntimeError(
                    f"duplicate patient_id/checkup_id at CSV row {row_number}; "
                    f"first seen at row {previous_row_number}: {patient_id}/{checkup_id}"
                )
            seen_keys[key] = row_number
            checkups.append(
                Checkup(
                    patient_id=patient_id,
                    checkup_id=checkup_id,
                    photographs=photographs,
                    source_row_number=row_number,
                )
            )

    if not checkups:
        raise RuntimeError(f"no checkups remained after filtering: {path}")
    return checkups, source_rows, skipped_min_photos


def group_checkups_by_patient(checkups: list[Checkup]) -> dict[str, list[Checkup]]:
    grouped: dict[str, list[Checkup]] = defaultdict(list)
    for checkup in checkups:
        grouped[checkup.patient_id].append(checkup)
    return {patient_id: sorted(items, key=lambda item: item.checkup_id) for patient_id, items in grouped.items()}


def split_patients(
    patient_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, str]:
    shuffled = sorted(patient_ids)
    rng.shuffle(shuffled)

    patient_count = len(shuffled)
    train_count = math.floor(patient_count * train_ratio)
    val_count = math.floor(patient_count * val_ratio)
    train_end = train_count
    val_end = train_end + val_count

    split_by_patient: dict[str, str] = {}
    for patient_id in shuffled[:train_end]:
        split_by_patient[patient_id] = "train"
    for patient_id in shuffled[train_end:val_end]:
        split_by_patient[patient_id] = "val"
    for patient_id in shuffled[val_end:]:
        split_by_patient[patient_id] = "test"
    return split_by_patient


def choose_impostors(
    grouped: dict[str, list[Checkup]],
    split_patient_ids: list[str],
    excluded_patient_id: str,
    template: Checkup,
    used_pair_keys: set[tuple[str, str]],
    count: int,
    rng: random.Random,
) -> list[Checkup]:
    candidates = [
        checkup
        for patient_id in split_patient_ids
        if patient_id != excluded_patient_id
        for checkup in grouped[patient_id]
        if (checkup.uid, template.uid) not in used_pair_keys
    ]
    if len(candidates) < count:
        raise RuntimeError(
            "cannot sample the requested unique impostor pairs because this split has too few other-patient checkups: "
            f"excluded_patient_id={excluded_patient_id}, requested={count}, available={len(candidates)}"
        )
    return rng.sample(candidates, count)


def build_pairs(
    grouped: dict[str, list[Checkup]],
    split_by_patient: dict[str, str],
    impostors_per_genuine: int,
    rng: random.Random,
) -> tuple[list[Pair], dict[str, int]]:
    patients_by_split: dict[str, list[str]] = {split: [] for split in SPLIT_NAMES}
    for patient_id, split in split_by_patient.items():
        patients_by_split[split].append(patient_id)
    for split in SPLIT_NAMES:
        patients_by_split[split].sort()

    pairs: list[Pair] = []
    used_pair_keys: set[tuple[str, str]] = set()
    skipped_impostors = 0
    genuine_pair_count = 0

    for split in SPLIT_NAMES:
        for patient_id in patients_by_split[split]:
            checkups = grouped[patient_id]
            if len(checkups) < 2:
                continue

            for template, query in combinations(checkups, 2):
                genuine_pair_count += 1
                pair_index = len(pairs) + 1
                used_pair_keys.add((query.uid, template.uid))
                pairs.append(
                    Pair(
                        split=split,
                        pair_id=f"{split}-{pair_index:06d}",
                        label="genuine",
                        template=template,
                        query=query,
                    )
                )
                for impostor in choose_impostors(
                    grouped,
                    patients_by_split[split],
                    patient_id,
                    template,
                    used_pair_keys,
                    impostors_per_genuine,
                    rng,
                ):
                    pair_index = len(pairs) + 1
                    used_pair_keys.add((impostor.uid, template.uid))
                    pairs.append(
                        Pair(
                            split=split,
                            pair_id=f"{split}-{pair_index:06d}",
                            label="impostor",
                            template=template,
                            query=impostor,
                        )
                    )

    if genuine_pair_count == 0:
        raise RuntimeError("no genuine pairs were generated; COde requires patients with at least two checkups.")
    return pairs, {"skipped_impostors": skipped_impostors, "genuine_pair_count": genuine_pair_count}


def write_checkups_csv(path: Path, checkups: list[Checkup], split_by_patient: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "patient_id",
                "checkup_id",
                "checkup_uid",
                "photograph_count",
                "photographs",
                "source_row_number",
            ]
        )
        for checkup in sorted(checkups, key=lambda item: (split_by_patient[item.patient_id], item.patient_id, item.checkup_id)):
            writer.writerow(
                [
                    split_by_patient[checkup.patient_id],
                    checkup.patient_id,
                    checkup.checkup_id,
                    checkup.uid,
                    len(checkup.photographs),
                    "|".join(checkup.photographs),
                    checkup.source_row_number,
                ]
            )


def write_pairs_csv(path: Path, pairs: list[Pair]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "pair_id",
                "label",
                "is_genuine",
                "template_id",
                "query_id",
                "template_patient_id",
                "query_patient_id",
                "template_checkup_id",
                "query_checkup_id",
                "template_photograph_count",
                "query_photograph_count",
                "template_photographs",
                "query_photographs",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    pair.split,
                    pair.pair_id,
                    pair.label,
                    int(pair.is_genuine),
                    pair.template.uid,
                    pair.query.uid,
                    pair.template.patient_id,
                    pair.query.patient_id,
                    pair.template.checkup_id,
                    pair.query.checkup_id,
                    len(pair.template.photographs),
                    len(pair.query.photographs),
                    "|".join(pair.template.photographs),
                    "|".join(pair.query.photographs),
                ]
            )


def summarize(
    args: argparse.Namespace,
    checkups: list[Checkup],
    source_rows: int,
    skipped_min_photos: int,
    split_by_patient: dict[str, str],
    pairs: list[Pair],
    pair_stats: dict[str, int],
) -> dict[str, object]:
    split_patient_counts = {split: 0 for split in SPLIT_NAMES}
    split_checkup_counts = {split: 0 for split in SPLIT_NAMES}
    split_pair_counts = {split: {"genuine": 0, "impostor": 0} for split in SPLIT_NAMES}

    for patient_split in split_by_patient.values():
        split_patient_counts[patient_split] += 1
    for checkup in checkups:
        split_checkup_counts[split_by_patient[checkup.patient_id]] += 1
    for pair in pairs:
        split_pair_counts[pair.split][pair.label] += 1

    return {
        "input": str(args.input),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "source_rows": source_rows,
        "skipped_min_photos": skipped_min_photos,
        "patients": len(split_by_patient),
        "checkups": len(checkups),
        "pairs": len(pairs),
        "genuine_pairs": sum(1 for pair in pairs if pair.is_genuine),
        "impostor_pairs": sum(1 for pair in pairs if not pair.is_genuine),
        "skipped_impostors": pair_stats["skipped_impostors"],
        "split_patient_counts": split_patient_counts,
        "split_checkup_counts": split_checkup_counts,
        "split_pair_counts": split_pair_counts,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)

    checkups, source_rows, skipped_min_photos = load_checkups(args.input, args.min_photos)
    grouped = group_checkups_by_patient(checkups)
    rng = random.Random(args.seed)
    split_by_patient = split_patients(
        list(grouped),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        rng=rng,
    )
    pairs, pair_stats = build_pairs(grouped, split_by_patient, args.impostors_per_genuine, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkups_csv = args.output_dir / "checkups.csv"
    pairs_csv = args.output_dir / "pairs.csv"
    summary_json = args.output_dir / "summary.json"

    write_checkups_csv(checkups_csv, checkups, split_by_patient)
    write_pairs_csv(pairs_csv, pairs)
    summary = summarize(args, checkups, source_rows, skipped_min_photos, split_by_patient, pairs, pair_stats)
    summary["checkups_csv_sha256"] = sha256_file(checkups_csv)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"checkups: {checkups_csv}")
    print(f"pairs: {pairs_csv}")
    print(f"summary: {summary_json}")
    print(f"patients={summary['patients']} checkups={summary['checkups']} pairs={summary['pairs']}")
    print(f"genuine_pairs={summary['genuine_pairs']} impostor_pairs={summary['impostor_pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
