from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BestScore:
    version: str
    epoch: int
    fitness: float
    box_map50: float
    box_map: float
    mask_map50: float
    mask_map: float


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError(f"project root not found from: {start}")


def compute_best_score(results_csv: Path, version: str) -> BestScore:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"results.csv is empty: {results_csv}")

    best_row: dict[str, str] | None = None
    best_fitness = -1.0
    for row in rows:
        box = float(row["metrics/mAP50(B)"]) * 0.1 + float(row["metrics/mAP50-95(B)"]) * 0.9
        mask = float(row["metrics/mAP50(M)"]) * 0.1 + float(row["metrics/mAP50-95(M)"]) * 0.9
        fitness = (box + mask) / 2.0
        if fitness > best_fitness:
            best_fitness = fitness
            best_row = row

    assert best_row is not None
    return BestScore(
        version=version,
        epoch=int(float(best_row["epoch"])),
        fitness=best_fitness,
        box_map50=float(best_row["metrics/mAP50(B)"]),
        box_map=float(best_row["metrics/mAP50-95(B)"]),
        mask_map50=float(best_row["metrics/mAP50(M)"]),
        mask_map=float(best_row["metrics/mAP50-95(M)"]),
    )


def write_csv(path: Path, scores: list[BestScore]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "version",
                "best_epoch",
                "fitness",
                "box_map50",
                "box_map50_95",
                "mask_map50",
                "mask_map50_95",
            ]
        )
        for score in scores:
            writer.writerow(
                [
                    score.version,
                    score.epoch,
                    f"{score.fitness:.6f}",
                    f"{score.box_map50:.5f}",
                    f"{score.box_map:.5f}",
                    f"{score.mask_map50:.5f}",
                    f"{score.mask_map:.5f}",
                ]
            )


def write_markdown(path: Path, scores: list[BestScore]) -> None:
    best_score = max(scores, key=lambda score: score.fitness)
    worst_score = min(scores, key=lambda score: score.fitness)
    lines = [
        "# tooth_seg_flont v1-v7 best score summary",
        "",
        "| version | best epoch | fitness | box mAP50-95 | mask mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for score in scores:
        lines.append(
            f"| {score.version} | {score.epoch} | {score.fitness:.6f} | "
            f"{score.box_map:.5f} | {score.mask_map:.5f} |"
        )
    lines.extend(
        [
            "",
            f"- best: `{best_score.version}` ({best_score.fitness:.6f})",
            f"- worst: `{worst_score.version}` ({worst_score.fitness:.6f})",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, scores: list[BestScore]) -> None:
    versions = [score.version for score in scores]
    fitness_values = [score.fitness for score in scores]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot(versions, fitness_values, color="#0f766e", marker="o", linewidth=2.2, markersize=6)
    ax.set_title("tooth_seg_flont Best Fitness (v1-v7)")
    ax.set_xlabel("Version")
    ax.set_ylabel("Fitness")
    ax.set_ylim(min(fitness_values) - 0.03, max(fitness_values) + 0.01)
    ax.grid(True, axis="y", alpha=0.25)

    for score in scores:
        ax.annotate(
            f"{score.fitness:.3f}",
            (score.version, score.fitness),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = find_project_root(Path(__file__).resolve())
    result_root = repo_root / "01_stage1_real_image_extraction" / "experiments"
    evaluation_root = repo_root / "02_stage2_capture_matching" / "evaluation"
    note_root = repo_root / "02_stage2_capture_matching" / "notes"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    note_root.mkdir(parents=True, exist_ok=True)

    scores = [
        compute_best_score(
            repo_root / "90_archive" / "legacy_preliminary" / "experiments" / f"tooth_seg_flont_v{i}" / "results.csv",
            f"v{i}",
        )
        for i in range(1, 4)
    ]
    scores.extend(
        [
            compute_best_score(result_root / "v4_baseline" / "results.csv", "v4"),
            compute_best_score(result_root / "v5" / "results.csv", "v5"),
            compute_best_score(result_root / "v6_best" / "results.csv", "v6"),
            compute_best_score(result_root / "v7_best" / "results.csv", "v7"),
        ]
    )

    csv_path = evaluation_root / "tooth_seg_flont_v1_v7_best_scores.csv"
    plot_path = evaluation_root / "tooth_seg_flont_v1_v7_best_scores.png"
    markdown_path = note_root / "tooth_seg_flont_v1_v7_summary.md"

    write_csv(csv_path, scores)
    write_plot(plot_path, scores)
    write_markdown(markdown_path, scores)

    print(f"csv: {csv_path}")
    print(f"plot: {plot_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
