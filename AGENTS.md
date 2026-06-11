# Repository Guidelines

## Project Structure & Module Organization
- `01_stage1_real_image_extraction/` contains first-stage real-image extraction scripts and public experiment summaries.
- `02_stage2_capture_matching/` contains second-stage evaluation scripts, score summaries, and notes for capture and matching analysis.
- `90_archive/` contains older public preliminary result summaries and auxiliary scripts.
- Private/local-only materials such as proposal drafts, raw datasets, notebooks, logs, and model weights are intentionally excluded by `.gitignore`.

## Build, Test, and Development Commands
- `uv sync`: install Python dependencies defined in `pyproject.toml`.
- `uv run python 01_stage1_real_image_extraction/scripts/train_tooth_seg_flont_v7.py --dry-run --device cpu --workers 0 --project /tmp/mitou_clean_v7_smoke`: smoke-test the main v7 training entry point.
- `uv run python 02_stage2_capture_matching/scripts/score_experiment.py`: compare the default v4 baseline and v7 best fitness values.
- `uv run python 02_stage2_capture_matching/scripts/compare_all_metrics.py`: compare all available metrics for the default baseline and candidate.
- `uv run python 02_stage2_capture_matching/scripts/summarize_tooth_seg_scores.py`: regenerate the v1-v7 score summary under `02_stage2_capture_matching/`.
- Use `uv` commands for reproducibility instead of ad-hoc `pip install`.

## Coding Style & Naming Conventions
- Use 4-space indentation in Python cells and scripts.
- Follow `snake_case` for Python variable and function names.
- Keep archived notebook names aligned with the existing convention: `Tooth_detection_model_<index>.ipynb`.
- Preserve existing dataset naming (`dataset_flont`, `dataset_flont_min1`, `dataset_flont_min5`) and output version labels (`v4_baseline`, `v6_best`, `v7_best`).
- Keep archived notebook edits small and deterministic; avoid hidden state between distant cells.

## Testing Guidelines
- There is no configured automated test suite yet (`pytest`/CI config is not present).
- Validate path and script changes with the representative `uv run` commands listed above.
- For model-impacting changes, check `results.csv` metric deltas, confusion matrix outputs (`confusion_matrix*.png`), and expected checkpoint outputs in `weights/`.
- If you add scripts, include a minimal repeatable smoke-test command in `README.md`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- PRs should include a concise summary, changed paths, reproduction commands, and model/data version used.
- For model-impacting changes, attach before/after metrics or screenshots of key plots.

## Data & Security Notes
- Do not commit patient-identifiable raw data or secrets.
- Avoid adding large binary artifacts unless they are explicitly required for reproducibility.
- Keep exploratory and historical artifacts under `90_archive/` unless they are part of the current first-stage or second-stage workflow.
