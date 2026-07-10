from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from output_directory import create_generation_directory, discard_generation, publish_generation


COLUMN_ALIASES = {
    "query_id": ("query_id",),
    "template_id": ("template_id",),
    "query_subject_id": ("query_subject_id", "query_patient_id"),
    "template_subject_id": ("template_subject_id", "template_patient_id"),
    "query_session_id": ("query_session_id", "query_checkup_id"),
    "template_session_id": ("template_session_id", "template_checkup_id"),
    "is_genuine": ("is_genuine",),
    "fused_score": ("fused_score",),
}
OPERATING_FAR_TARGETS = (0.01, 0.001)


@dataclass(frozen=True)
class ScoreRecord:
    query_id: str
    template_id: str
    query_subject_id: str
    template_subject_id: str
    query_session_id: str
    template_session_id: str
    is_genuine: bool
    fused_score: float
    source_row_number: int


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    false_accept_rate: float
    false_reject_rate: float
    true_positive_rate: float
    false_positive_rate: float
    true_positive_count: int
    false_positive_count: int
    true_negative_count: int
    false_negative_count: int


@dataclass(frozen=True)
class EerResult:
    eer: float
    threshold: float


@dataclass(frozen=True)
class OperatingPoint:
    target_far: float
    threshold: float
    false_accept_rate: float
    false_reject_rate: float


@dataclass(frozen=True)
class ScoreDistribution:
    genuine_mean: float
    genuine_std: float
    impostor_mean: float
    impostor_std: float
    d_prime: float | None


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def parse_args() -> argparse.Namespace:
    repo_root = find_project_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Evaluate 1:1 authentication scores and export FAR/FRR/EER metrics."
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        required=True,
        help="Score CSV with query/template subject, session, is_genuine, and fused_score columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "02_stage2_capture_matching" / "logs" / "auth_eval",
        help="Directory for summary.json, curves.csv, operating_points.csv, and plots.",
    )
    return parser.parse_args()


def resolve_column(fieldnames: list[str], canonical: str) -> str:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in fieldnames:
            return alias
    raise RuntimeError(f"score CSV is missing required column for {canonical}: accepted={COLUMN_ALIASES[canonical]}")


def clean_required(value: str, column: str, row_number: int) -> str:
    cleaned = value.strip()
    if cleaned == "":
        raise RuntimeError(f"{column} is empty at CSV row {row_number}.")
    return cleaned


def parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "genuine"}:
        return True
    if normalized in {"0", "false", "no", "impostor"}:
        return False
    raise RuntimeError(f"is_genuine must be boolean-like at CSV row {row_number}: {value!r}")


def parse_score(value: str, row_number: int) -> float:
    try:
        score = float(value)
    except ValueError as exc:
        raise RuntimeError(f"fused_score must be numeric at CSV row {row_number}: {value!r}") from exc
    if not math.isfinite(score):
        raise RuntimeError(f"fused_score must be finite at CSV row {row_number}: {value!r}")
    return score


def load_score_records(path: Path) -> list[ScoreRecord]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"score CSV has no header: {path}")
        fieldnames = reader.fieldnames
        columns = {canonical: resolve_column(fieldnames, canonical) for canonical in COLUMN_ALIASES}

        records: list[ScoreRecord] = []
        seen_ids: dict[tuple[str, str], int] = {}
        for row_number, row in enumerate(reader, start=2):
            query_id = clean_required(row[columns["query_id"]], columns["query_id"], row_number)
            template_id = clean_required(row[columns["template_id"]], columns["template_id"], row_number)
            key = (query_id, template_id)
            if key in seen_ids:
                raise RuntimeError(
                    f"duplicate query_id/template_id at CSV row {row_number}; first seen at row {seen_ids[key]}"
                )
            seen_ids[key] = row_number

            record = ScoreRecord(
                query_id=query_id,
                template_id=template_id,
                query_subject_id=clean_required(
                    row[columns["query_subject_id"]], columns["query_subject_id"], row_number
                ),
                template_subject_id=clean_required(
                    row[columns["template_subject_id"]], columns["template_subject_id"], row_number
                ),
                query_session_id=clean_required(
                    row[columns["query_session_id"]], columns["query_session_id"], row_number
                ),
                template_session_id=clean_required(
                    row[columns["template_session_id"]], columns["template_session_id"], row_number
                ),
                is_genuine=parse_bool(row[columns["is_genuine"]], row_number),
                fused_score=parse_score(row[columns["fused_score"]], row_number),
                source_row_number=row_number,
            )
            validate_record_label(record)
            records.append(record)

    if not records:
        raise RuntimeError(f"score CSV has no records: {path}")
    return records


def validate_record_label(record: ScoreRecord) -> None:
    same_subject = record.query_subject_id == record.template_subject_id
    if record.is_genuine and not same_subject:
        raise RuntimeError(
            "genuine record has different subjects: "
            f"row={record.source_row_number}, query={record.query_subject_id}, template={record.template_subject_id}"
        )
    if not record.is_genuine and same_subject:
        raise RuntimeError(
            "impostor record has the same subject: "
            f"row={record.source_row_number}, subject={record.query_subject_id}"
        )


def enforce_session_separation(records: list[ScoreRecord]) -> tuple[list[ScoreRecord], int]:
    kept: list[ScoreRecord] = []
    excluded = 0
    for record in records:
        same_session = record.query_session_id == record.template_session_id
        if record.is_genuine and same_session:
            excluded += 1
            continue
        kept.append(record)
    if not kept:
        raise RuntimeError("all score records were excluded by session separation.")
    return kept, excluded


def build_curve(records: list[ScoreRecord]) -> list[CurvePoint]:
    genuine_scores = [record.fused_score for record in records if record.is_genuine]
    impostor_scores = [record.fused_score for record in records if not record.is_genuine]
    if not genuine_scores:
        raise RuntimeError("no genuine scores remained after filtering.")
    if not impostor_scores:
        raise RuntimeError("no impostor scores remained after filtering.")

    unique_scores = sorted({record.fused_score for record in records}, reverse=True)
    epsilon = max((max(unique_scores) - min(unique_scores)) * 1e-9, 1e-12)
    thresholds = [max(unique_scores) + epsilon, *unique_scores, min(unique_scores) - epsilon]

    points: list[CurvePoint] = []
    genuine_count = len(genuine_scores)
    impostor_count = len(impostor_scores)
    for threshold in thresholds:
        true_positive_count = sum(score >= threshold for score in genuine_scores)
        false_negative_count = genuine_count - true_positive_count
        false_positive_count = sum(score >= threshold for score in impostor_scores)
        true_negative_count = impostor_count - false_positive_count
        far = false_positive_count / impostor_count
        frr = false_negative_count / genuine_count
        points.append(
            CurvePoint(
                threshold=threshold,
                false_accept_rate=far,
                false_reject_rate=frr,
                true_positive_rate=true_positive_count / genuine_count,
                false_positive_rate=far,
                true_positive_count=true_positive_count,
                false_positive_count=false_positive_count,
                true_negative_count=true_negative_count,
                false_negative_count=false_negative_count,
            )
        )
    return points


def compute_eer(points: list[CurvePoint]) -> EerResult:
    previous = points[0]
    previous_diff = previous.false_accept_rate - previous.false_reject_rate
    if previous_diff == 0.0:
        return EerResult(eer=previous.false_accept_rate, threshold=previous.threshold)

    for point in points[1:]:
        diff = point.false_accept_rate - point.false_reject_rate
        if diff == 0.0:
            return EerResult(eer=point.false_accept_rate, threshold=point.threshold)
        if previous_diff * diff < 0.0:
            ratio = abs(previous_diff) / (abs(previous_diff) + abs(diff))
            eer = previous.false_accept_rate + ratio * (point.false_accept_rate - previous.false_accept_rate)
            threshold = previous.threshold + ratio * (point.threshold - previous.threshold)
            return EerResult(eer=eer, threshold=threshold)
        previous = point
        previous_diff = diff

    closest = min(points, key=lambda point: abs(point.false_accept_rate - point.false_reject_rate))
    return EerResult(eer=(closest.false_accept_rate + closest.false_reject_rate) / 2.0, threshold=closest.threshold)


def compute_auc(points: list[CurvePoint]) -> float:
    roc_points = sorted((point.false_positive_rate, point.true_positive_rate) for point in points)
    auc = 0.0
    previous_fpr, previous_tpr = roc_points[0]
    for fpr, tpr in roc_points[1:]:
        auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        previous_fpr = fpr
        previous_tpr = tpr
    return auc


def compute_score_distribution(records: list[ScoreRecord]) -> ScoreDistribution:
    genuine_scores = [record.fused_score for record in records if record.is_genuine]
    impostor_scores = [record.fused_score for record in records if not record.is_genuine]
    if not genuine_scores:
        raise RuntimeError("no genuine scores remained after filtering.")
    if not impostor_scores:
        raise RuntimeError("no impostor scores remained after filtering.")

    genuine_mean = sum(genuine_scores) / len(genuine_scores)
    impostor_mean = sum(impostor_scores) / len(impostor_scores)
    genuine_variance = sum((score - genuine_mean) ** 2 for score in genuine_scores) / len(genuine_scores)
    impostor_variance = sum((score - impostor_mean) ** 2 for score in impostor_scores) / len(impostor_scores)
    genuine_std = math.sqrt(genuine_variance)
    impostor_std = math.sqrt(impostor_variance)
    pooled_std = math.sqrt((genuine_variance + impostor_variance) / 2.0)
    d_prime = None if pooled_std == 0.0 else (genuine_mean - impostor_mean) / pooled_std
    return ScoreDistribution(
        genuine_mean=genuine_mean,
        genuine_std=genuine_std,
        impostor_mean=impostor_mean,
        impostor_std=impostor_std,
        d_prime=d_prime,
    )


def choose_operating_point(points: list[CurvePoint], target_far: float) -> OperatingPoint:
    candidates = [point for point in points if point.false_accept_rate <= target_far]
    if not candidates:
        raise RuntimeError(f"no operating point found at FAR <= {target_far}.")
    best = min(candidates, key=lambda point: (point.false_reject_rate, -point.false_accept_rate))
    return OperatingPoint(
        target_far=target_far,
        threshold=best.threshold,
        false_accept_rate=best.false_accept_rate,
        false_reject_rate=best.false_reject_rate,
    )


def write_curves_csv(path: Path, points: list[CurvePoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "threshold",
                "far",
                "frr",
                "fpr",
                "tpr",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    f"{point.threshold:.12g}",
                    f"{point.false_accept_rate:.12g}",
                    f"{point.false_reject_rate:.12g}",
                    f"{point.false_positive_rate:.12g}",
                    f"{point.true_positive_rate:.12g}",
                    point.true_positive_count,
                    point.false_positive_count,
                    point.true_negative_count,
                    point.false_negative_count,
                ]
            )


def write_operating_points_csv(path: Path, operating_points: list[OperatingPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_far", "threshold", "far", "frr"])
        for point in operating_points:
            writer.writerow(
                [
                    f"{point.target_far:.12g}",
                    f"{point.threshold:.12g}",
                    f"{point.false_accept_rate:.12g}",
                    f"{point.false_reject_rate:.12g}",
                ]
            )


def write_roc_plot(path: Path, points: list[CurvePoint], auc: float) -> None:
    roc_points = sorted((point.false_positive_rate, point.true_positive_rate) for point in points)
    fpr_values = [point[0] for point in roc_points]
    tpr_values = [point[1] for point in roc_points]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.plot(fpr_values, tpr_values, color="#2563eb", linewidth=2.0, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_far_frr_plot(path: Path, points: list[CurvePoint], eer: EerResult) -> None:
    thresholds = [point.threshold for point in points]
    far_values = [point.false_accept_rate for point in points]
    frr_values = [point.false_reject_rate for point in points]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(thresholds, far_values, color="#dc2626", linewidth=2.0, label="FAR")
    ax.plot(thresholds, frr_values, color="#16a34a", linewidth=2.0, label="FRR")
    ax.scatter([eer.threshold], [eer.eer], color="#111827", s=28, label=f"EER={eer.eer:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("FAR / FRR")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_score_distribution_plot(path: Path, records: list[ScoreRecord]) -> None:
    genuine_scores = [record.fused_score for record in records if record.is_genuine]
    impostor_scores = [record.fused_score for record in records if not record.is_genuine]
    all_scores = genuine_scores + impostor_scores
    minimum = min(all_scores)
    maximum = max(all_scores)
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    bin_count = min(30, max(5, math.ceil(math.sqrt(len(all_scores)))))
    bin_width = (maximum - minimum) / bin_count
    bins = [minimum + index * bin_width for index in range(bin_count + 1)]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.hist(
        impostor_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#dc2626",
        label="Impostor",
    )
    ax.hist(
        genuine_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#2563eb",
        label="Genuine",
    )
    ax.set_xlabel("Fused Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize(
    args: argparse.Namespace,
    raw_records: list[ScoreRecord],
    records: list[ScoreRecord],
    excluded_same_session_genuine: int,
    eer: EerResult,
    auc: float,
    distribution: ScoreDistribution,
    operating_points: list[OperatingPoint],
) -> dict[str, object]:
    return {
        "scores_csv_name": args.scores_csv.name,
        "total_records": len(raw_records),
        "used_records": len(records),
        "excluded_same_session_genuine": excluded_same_session_genuine,
        "genuine_scores": sum(1 for record in records if record.is_genuine),
        "impostor_scores": sum(1 for record in records if not record.is_genuine),
        "eer": eer.eer,
        "eer_threshold": eer.threshold,
        "roc_auc": auc,
        "score_distribution": {
            "genuine_mean": distribution.genuine_mean,
            "genuine_std": distribution.genuine_std,
            "impostor_mean": distribution.impostor_mean,
            "impostor_std": distribution.impostor_std,
            "d_prime": distribution.d_prime,
        },
        "operating_points": [
            {
                "target_far": point.target_far,
                "threshold": point.threshold,
                "far": point.false_accept_rate,
                "frr": point.false_reject_rate,
            }
            for point in operating_points
        ],
    }


def main() -> int:
    args = parse_args()
    raw_records = load_score_records(args.scores_csv)
    records, excluded_same_session_genuine = enforce_session_separation(raw_records)
    points = build_curve(records)
    eer = compute_eer(points)
    auc = compute_auc(points)
    distribution = compute_score_distribution(records)
    operating_points = [choose_operating_point(points, target) for target in OPERATING_FAR_TARGETS]

    generation_dir = create_generation_directory(args.output_dir)
    try:
        write_curves_csv(generation_dir / "curves.csv", points)
        write_operating_points_csv(generation_dir / "operating_points.csv", operating_points)
        write_roc_plot(generation_dir / "roc_curve.png", points, auc)
        write_far_frr_plot(generation_dir / "far_frr_curve.png", points, eer)
        write_score_distribution_plot(generation_dir / "score_distribution.png", records)
        summary = summarize(
            args,
            raw_records,
            records,
            excluded_same_session_genuine,
            eer,
            auc,
            distribution,
            operating_points,
        )
        (generation_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        publish_generation(generation_dir, args.output_dir)
    finally:
        discard_generation(generation_dir)

    curves_csv = args.output_dir / "curves.csv"
    operating_points_csv = args.output_dir / "operating_points.csv"
    summary_json = args.output_dir / "summary.json"
    roc_png = args.output_dir / "roc_curve.png"
    far_frr_png = args.output_dir / "far_frr_curve.png"
    score_distribution_png = args.output_dir / "score_distribution.png"

    print(f"summary: {summary_json}")
    print(f"curves: {curves_csv}")
    print(f"operating_points: {operating_points_csv}")
    print(f"roc: {roc_png}")
    print(f"far_frr: {far_frr_png}")
    print(f"score_distribution: {score_distribution_png}")
    print(f"used_records={summary['used_records']} excluded_same_session_genuine={excluded_same_session_genuine}")
    print(f"eer={eer.eer:.6f} threshold={eer.threshold:.6f} roc_auc={auc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
