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

COde アノテーション batch の準備:

```bash
uv run python 02_stage2_capture_matching/scripts/prepare_code_annotation_batch.py --checkups-csv /path/to/code_pair_manifest/checkups.csv --source-summary /path/to/code_pair_manifest/summary.json --images-root /path/to/COde --output-dir 01_stage1_real_image_extraction/datasets/dataset_real/code_annotation --seed 42 --train-checkups 10 --val-checkups 5 --photos-per-checkup 4
```

選定処理の smoke test:

```bash
uv run python -m unittest 02_stage2_capture_matching/tests/test_prepare_code_annotation_batch.py -v
```

このコマンドは、seed 42、患者単位 70/15/15 の元 split を検証し、train 40枚と val 20枚を選びます。1患者につき1 checkupだけを使い、test split は選びません。同じ画像内容が複数の checkup にある場合と、test split に同じ画像内容がある場合は SHA-256 で除外します。出力画像名には元の patient ID、checkup ID、写真名を含めません。

## CVAT アノテーション

`code_annotation/cvat_train_images.zip` と `code_annotation/cvat_val_images.zip` から、[CVAT standalone task](https://docs.cvat.ai/docs/manual/basics/create-annotation-task/) を一つずつ作ります。二つの ZIP を同じ task へ入れません。各 task の作成画面で Labels の Raw 入力を開き、`code_annotation/cvat_labels.json` の内容を貼り付けます。ラベルは polygon に固定されています。順序は `R1-R7, L1-L7` の14クラスから変えません。実際に polygon を付ける対象は `R1-R3, L1-L3` の6クラスです。ラベル順を6クラスへ詰めると v7 checkpoint の class ID と一致しなくなります。

歯冠の見えている境界を [polygon](https://docs.cvat.ai/docs/manual/advanced/annotation-with-polygons/manual-drawing/) で囲みます。画像ごとの作業後に、`annotation_manifest.csv` の `annotation_status` を `complete`、歯が写らない負例を `negative`、使用しない画像を `excluded` に変更します。`view_tag`、`lighting_tag`、`oral_condition_tag` には、後で条件別 mAP を集計できる値を記録します。

保存後は各 task を [Ultralytics YOLO Segmentation 1.0](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/) で別々に export します。train task の export だけを学習 split、val task の export だけを検証 split に割り当てます。standalone task の export 内にある subset 名だけを信用して両者を統合しません。実写画像、台帳、export ZIP は Git にコミットしません。export 後に14クラス順、空ラベル、各6クラスの instance 数を検査してから、ゼロショット評価と v8 fine-tune 用 dataset を構築します。

## 判断

主要成果として表に出す実験は、v4 baseline、v5、v6 best、v7 Stage A、v7 best に絞っています。細かい試行は、ルートの `90_archive/all_training_trials/` を見てください。

`Tooth_detection_model_5.ipynb` から `Tooth_detection_model_7.ipynb` までの notebook は、旧パスを含む履歴資料として `90_archive/stage1_notebook_history/` に移しました。現行の再実行には、このフォルダの `scripts/` を使います。
