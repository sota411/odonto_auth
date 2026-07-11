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
├── tests/
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

v8 fine-tune設定の非ML検証:

```bash
uv run python 01_stage1_real_image_extraction/scripts/train_tooth_seg_flont_v8.py --synthetic-data 01_stage1_real_image_extraction/datasets/dataset_flont/dataset_flont.yaml --real-data 01_stage1_real_image_extraction/datasets/dataset_real/dataset_code_real/dataset_code_real.yaml --real-repeat 2 --prepare-only --device cpu
```

このコマンドと下記の実写valコマンドは、annotation exportから`dataset_code_real`を確定した後に実行します。非ML smoke testは次のコマンドです。

```bash
uv run python -m unittest discover -s 01_stage1_real_image_extraction/tests -p 'test_*.py' -v
```

`--prepare-only` は擬似・実写dataset、class順、画像数、混合比、初期重み、出力先とaugmentation設定を検査し、YOLOを生成しません。trainには擬似trainを1回、実写trainを`--real-repeat`回だけ入れ、valには実写valだけを使います。実行時に生成した`mixed_dataset.yaml`と混合比JSONは実験ディレクトリへ保存されます。v8はv7 bestを初期重みに使い、低学習率、早期終了、backbone freeze、明度・コントラスト、blur、JPEG圧縮の拡張を設定します。左右のclass IDを維持するため、水平反転は無効です。

v8 実ML smoke設定:

```bash
uv run python 01_stage1_real_image_extraction/scripts/train_tooth_seg_flont_v8.py --synthetic-data 01_stage1_real_image_extraction/datasets/dataset_flont/dataset_flont.yaml --real-data 01_stage1_real_image_extraction/datasets/dataset_real/dataset_code_real/dataset_code_real.yaml --real-repeat 2 --dry-run --device cpu --workers 0 --project /tmp/odonto_v8_smoke
```

`--dry-run`は、擬似train、実写train、実写valからpath順の先頭画像を1枚ずつ選び、専用file listとYAMLを作って1 epochの学習を起動します。`fraction`は1.0です。成功後はfile list、`mixed_dataset.yaml`、`mixed_dataset.json`を実験ディレクトリへ保存するため、YAMLの参照は実行後も有効です。`--prepare-only`とは併用できません。このREADMEには再現用コマンドだけを記載しており、今回の実装確認では実ML smokeを実行していません。

実写valでのv7 zero-shotとv8比較設定の非ML検証:

```bash
uv run python 01_stage1_real_image_extraction/scripts/validate_tooth_seg_real.py --data 01_stage1_real_image_extraction/datasets/dataset_real/dataset_code_real/dataset_code_real.yaml --metadata 01_stage1_real_image_extraction/datasets/dataset_real/dataset_code_real/metadata.csv --model v7_zero_shot=01_stage1_real_image_extraction/experiments/v7_best/weights/best.pt --model v8=01_stage1_real_image_extraction/experiments/v8_best/weights/best.pt --project 01_stage1_real_image_extraction/experiments --name real_val_comparison --prepare-only
```

実評価では`--metadata`が必須です。対象6クラスごとのbox/mask mAP50とmAP50-95、v7からv8への差分、全6クラス改善の判定を出します。`metadata.csv`はtrain/val両方の画像集合と一致し、各行の`source_sha256`が対応画像の実SHA-256と一致しなければなりません。その確認後にpatient、checkup、SHA-256のsplit間重複を拒否します。3種類の条件タグごとの検証subsetも同じ重みで評価します。全モデル、全条件、集計を同一filesystem上の一時generationへ出力し、すべて成功した場合だけ最終出力をatomicに置き換えます。

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

歯冠の見えている境界を [polygon](https://docs.cvat.ai/docs/manual/advanced/annotation-with-polygons/manual-drawing/) で囲みます。画像ごとの作業後に、`annotation_manifest.csv` の `annotation_status` を `complete`、歯が写らない負例を `negative`、使用しない画像を `excluded` に変更します。採用する画像では、条件タグを次の値から一つずつ選びます。

- `view_tag`: `frontal`, `left_lateral`, `right_lateral`, `maxillary_occlusal`, `mandibular_occlusal`, `other`
- `lighting_tag`: `normal`, `dark`, `overexposed`, `reflection`
- `oral_condition_tag`: `none`, `orthodontic_appliance`, `restoration`, `missing_tooth`, `other`

`pending`、採用画像の空タグ、一覧外のタグが残っている場合、変換処理は停止します。`summary.json` に `manifest_identity_sha256` がない旧バッチは、全画像が `pending` であることを確認してから、上のアノテーションbatch準備コマンドを再実行します。アノテーション開始後はmanifestとZIPを上書きするため、batch準備コマンドを再実行しません。

保存後は各 task を [Ultralytics YOLO Segmentation 1.0](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/) で別々に export します。`Save images` を有効にし、`code_annotation/exports/train.zip` と `code_annotation/exports/val.zip` に置きます。変換処理では、export画像と元バッチZIPの画像が同一であることをSHA-256で確認します。

```bash
uv run python 02_stage2_capture_matching/scripts/finalize_code_annotation_dataset.py --batch-dir 01_stage1_real_image_extraction/datasets/dataset_real/code_annotation --train-export 01_stage1_real_image_extraction/datasets/dataset_real/code_annotation/exports/train.zip --val-export 01_stage1_real_image_extraction/datasets/dataset_real/code_annotation/exports/val.zip --output-dir 01_stage1_real_image_extraction/datasets/dataset_real/dataset_code_real
```

この処理では、元画像とmanifestのSHA-256、train/valの画像集合、14クラスの順序、対象6クラスのpolygonを検査します。train task のexportだけを学習split、val taskのexportだけを検証splitへ割り当て、CVAT内部のsubset名は使いません。実写画像、台帳、export ZIP、変換後datasetはGitにコミットしません。

`code_annotation/annotation_manifest.csv`の60枚は、polygon、status、3種類の条件タグを人手で確定してから変換します。`pending`が1件でも残る段階では、zero-shot評価とv8学習へ進みません。

2026年7月11日の実行では、60枚を確認し、正例26枚、負例2枚、除外32枚で確定した。SAM2.1 smallのbox promptでpolygon作成を補助し、全maskを画像へ重ねて確認した。trainは正例16枚と負例2枚、valは正例10枚である。v7 zero-shotのmask mAP50は0.000、v8全層fine-tuneは0.162で、6クラスすべてが改善した。重みと実画像はGitへ追加せず、公開可能な集計と制約を[実写適応レポート](../02_stage2_capture_matching/notes/v8_real_adaptation_report.md)に記録している。

## 判断

主要成果として表に出す実験は、v4 baseline、v5、v6 best、v7 Stage A、v7 best に絞っています。細かい試行は、ルートの `90_archive/all_training_trials/` を見てください。

`Tooth_detection_model_5.ipynb` から `Tooth_detection_model_7.ipynb` までの notebook は、旧パスを含む履歴資料として `90_archive/stage1_notebook_history/` に移しました。現行の再実行には、このフォルダの `scripts/` を使います。
