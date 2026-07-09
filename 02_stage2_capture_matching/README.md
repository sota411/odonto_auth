# 第二段階: 撮影条件と照合評価

第二段階は、撮影距離、開口角度、唾液ノイズ、照明条件などを整理し、再撮影しても同一人物の特徴が安定するかを確認する段階です。このフォルダには、比較・評価用のスクリプト、主要な評価結果、判断ログを置いています。

## 構成

```text
02_stage2_capture_matching/
├── scripts/
├── evaluation/
│   └── figures/
├── notes/
└── logs/
```

主要な再生成済み成果物:

- `evaluation/tooth_seg_flont_v1_v7_best_scores.csv`
- `evaluation/tooth_seg_flont_v1_v7_best_scores.png`
- `notes/tooth_seg_flont_v1_v7_summary.md`

## 代表コマンド

v4 baseline と v7 best の fitness 比較:

```bash
uv run python 02_stage2_capture_matching/scripts/score_experiment.py
```

全指標比較:

```bash
uv run python 02_stage2_capture_matching/scripts/compare_all_metrics.py
```

COde の匿名 ID 内照合ペア manifest 生成:

```bash
uv run python 02_stage2_capture_matching/scripts/build_code_pair_manifest.py --input 02_stage2_capture_matching/fixtures/code_complete_dataset_smoke.csv --output-dir /tmp/code_pair_manifest_smoke --train-ratio 1 --val-ratio 0 --test-ratio 0 --seed 7 --impostors-per-genuine 1
```

合成スコアでの認証評価 smoke test:

```bash
uv run python 02_stage2_capture_matching/scripts/evaluate_authentication.py --scores-csv 02_stage2_capture_matching/fixtures/auth_scores_smoke.csv --output-dir /tmp/auth_eval_smoke
```

v1 から v7 までのベストスコア集計:

```bash
uv run python 02_stage2_capture_matching/scripts/summarize_tooth_seg_scores.py
```

`build_code_pair_manifest.py` は画像をダウンロードせず、`complete_dataset.csv` の `patient_id`、`checkup_id`、`photographs` から患者単位 split を作ります。COde 本体で使う場合は `--input` を COde の `complete_dataset.csv` に差し替えます。`pairs.csv` は同一患者・別チェックアップを `genuine`、別患者を `impostor` とし、主な列は `split`, `pair_id`, `label`, `is_genuine`, `template_id`, `query_id`, `template_patient_id`, `query_patient_id`, `template_photographs`, `query_photographs` です。

`evaluate_authentication.py` は、照合スコア CSV から FAR、FRR、EER、ROC AUC を出力します。入力列と除外ルールは `notes/auth_evaluation_protocol.md` に記録しています。

## 判断

細かい trial を直接並べると比較対象が読みにくくなるため、第二段階の表側には主要な比較結果だけを残しています。全試行を確認する場合は、`90_archive/all_training_trials/` を見てください。
