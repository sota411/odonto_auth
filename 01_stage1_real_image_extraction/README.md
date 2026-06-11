# 第一段階: 実写画像への抽出基盤

第一段階は、同意を得た人物の実写口腔内画像を収集し、擬似画像で学習した抽出モデルを実写へ適応させる段階です。このフォルダには、前歯6クラスの抽出基盤に必要なデータセット、学習スクリプト、主要成果だけを置いています。

## 構成

```text
01_stage1_real_image_extraction/
├── datasets/
│   ├── dataset_flont/
│   ├── dataset_flont_min1/
│   └── dataset_flont_min5/
├── scripts/
└── experiments/
    ├── v4_baseline/
    ├── v5/
    ├── v6_best/
    ├── v7_stage_a/
    └── v7_best/
```

## 代表コマンド

データセット再構築:

```bash
uv run python 01_stage1_real_image_extraction/scripts/build_dataset_flont_yolo.py
```

v7 疎通確認:

```bash
uv run python 01_stage1_real_image_extraction/scripts/train_tooth_seg_flont_v7.py --dry-run --device cpu --workers 0 --project /tmp/mitou_clean_v7_smoke
```

v7 検証:

```bash
uv run python 01_stage1_real_image_extraction/scripts/validate_tooth_seg_flont_v7.py --device cpu --project /tmp/mitou_clean_v7_validation
```

## 判断

主要成果として表に出す実験は、v4 baseline、v5、v6 best、v7 Stage A、v7 best に絞っています。細かい試行は、ルートの `90_archive/all_training_trials/` を見てください。

`Tooth_detection_model_5.ipynb` から `Tooth_detection_model_7.ipynb` までの notebook は、旧パスを含む履歴資料として `90_archive/stage1_notebook_history/` に移しました。現行の再実行には、このフォルダの `scripts/` を使います。
