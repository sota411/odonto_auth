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

v1 から v7 までのベストスコア集計:

```bash
uv run python 02_stage2_capture_matching/scripts/summarize_tooth_seg_scores.py
```

## 判断

細かい trial を直接並べると比較対象が読みにくくなるため、第二段階の表側には主要な比較結果だけを残しています。全試行を確認する場合は、`90_archive/all_training_trials/` を見てください。
