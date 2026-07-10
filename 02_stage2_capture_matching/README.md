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
- `evaluation/code_matching_baseline_metrics.csv`
- `evaluation/code_matching_resnet50_score_distribution.png`
- `evaluation/code_matching_hog_score_distribution.png`
- `notes/tooth_seg_flont_v1_v7_summary.md`
- `notes/matching_baseline_report.md`

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

照合ロジックとスコア生成の単体テスト:

```bash
uv run python -m unittest discover -s 02_stage2_capture_matching/tests -p 'test_*.py' -v
```

COde の実画像を使うゼロショット照合ベースライン:

```bash
uv run python 02_stage2_capture_matching/scripts/build_code_pair_manifest.py --input /path/to/COde/complete_dataset.csv --output-dir 02_stage2_capture_matching/logs/code_pair_manifest --seed 42 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --impostors-per-genuine 1 --min-photos 1
```

同じ患者 split から CVAT 用の実写アノテーション batch を作る:

```bash
uv run python 02_stage2_capture_matching/scripts/prepare_code_annotation_batch.py --checkups-csv 02_stage2_capture_matching/logs/code_pair_manifest/checkups.csv --source-summary 02_stage2_capture_matching/logs/code_pair_manifest/summary.json --images-root /path/to/COde --output-dir 01_stage1_real_image_extraction/datasets/dataset_real/code_annotation --seed 42 --train-checkups 10 --val-checkups 5 --photos-per-checkup 4
```

`summary.json` に `checkups_csv_sha256` がない場合は、直前の manifest 生成コマンドを再実行して3ファイルを同じ世代へ揃えます。アノテーション準備では、SHA-256、seed 42、患者単位 70/15/15 split を照合します。

```bash
uv run python 02_stage2_capture_matching/scripts/extract_code_features.py --pairs-csv 02_stage2_capture_matching/logs/code_pair_manifest/pairs.csv --images-root /path/to/COde --weights 01_stage1_real_image_extraction/experiments/v7_best/weights/best.pt --expected-weights-sha256 f945236eb2441dfbbd0c439a5cd1c3e4d94e97650f3d0429cff5ee6da7a90454 --output-dir 02_stage2_capture_matching/logs/code_features_test_conf005 --split test --feature-types resnet50 hog --device 0 --imgsz 832 --conf 0.05 --iou 0.70 --source-chunk-size 64 --feature-batch-size 16 --max-views-per-tooth 3 --audit-crops 60
```

```bash
uv run python 02_stage2_capture_matching/scripts/score_code_pairs.py --pairs-csv 02_stage2_capture_matching/logs/code_pair_manifest/pairs.csv --features-npz 02_stage2_capture_matching/logs/code_features_test_conf005/features_resnet50.npz --images-root /path/to/COde --output-dir 02_stage2_capture_matching/logs/code_scores_test_conf005_resnet50 --split test --min-common-teeth 1
```

```bash
uv run python 02_stage2_capture_matching/scripts/evaluate_authentication.py --scores-csv 02_stage2_capture_matching/logs/code_scores_test_conf005_resnet50/scores.csv --output-dir 02_stage2_capture_matching/logs/auth_eval_test_conf005_resnet50
```

HOG 特徴の採点と評価:

```bash
uv run python 02_stage2_capture_matching/scripts/score_code_pairs.py --pairs-csv 02_stage2_capture_matching/logs/code_pair_manifest/pairs.csv --features-npz 02_stage2_capture_matching/logs/code_features_test_conf005/features_hog.npz --images-root /path/to/COde --output-dir 02_stage2_capture_matching/logs/code_scores_test_conf005_hog --split test --min-common-teeth 1
```

```bash
uv run python 02_stage2_capture_matching/scripts/evaluate_authentication.py --scores-csv 02_stage2_capture_matching/logs/code_scores_test_conf005_hog/scores.csv --output-dir 02_stage2_capture_matching/logs/auth_eval_test_conf005_hog
```

v1 から v7 までのベストスコア集計:

```bash
uv run python 02_stage2_capture_matching/scripts/summarize_tooth_seg_scores.py
```

`build_code_pair_manifest.py` は画像をダウンロードせず、`complete_dataset.csv` の `patient_id`、`checkup_id`、`photographs` から患者単位 split を作ります。COde 本体で使う場合は `--input` を COde の `complete_dataset.csv` に差し替えます。`pairs.csv` は同一患者・別チェックアップを `genuine`、別患者を `impostor` とし、主な列は `split`, `pair_id`, `label`, `is_genuine`, `template_id`, `query_id`, `template_patient_id`, `query_patient_id`, `template_photographs`, `query_photographs` です。

`prepare_code_annotation_batch.py` は `checkups.csv` の train / val 患者だけから1患者1checkupで画像を選びます。選定順は seed と source ID の SHA-256 で固定し、内容が重複する写真は除外します。出力は匿名名の画像、CVAT用ZIP、14クラスのラベル定義、source IDと画像SHA-256を持つ ignored manifest です。test患者はアノテーションにも学習にも使いません。

`evaluate_authentication.py` は、照合スコア CSV から FAR、FRR、EER、ROC AUC、d-prime、genuine/impostor の分布図を出力します。入力列と除外ルールは `notes/auth_evaluation_protocol.md` に記録しています。

`extract_code_features.py` は、pair manifest で参照される checkup だけを読み、v7 の前歯6クラスを切り出します。`--images-root` には `Images/Photographs` を含む COde の展開ルートを指定します。v7 重みは `--expected-weights-sha256` と照合します。ResNet50 `IMAGENET1K_V2` は PyTorch の公式 URL から取得し、完全 SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` と照合してから `weights_only=True` で読み込みます。同一 checkup・歯種の上位3 viewを confidence 加重平均し、凍結 ResNet50 と HOG の特徴を別々の NPZ に保存します。NPZ の主な配列は `checkup_ids [N]`、`patient_ids [N]`、`tooth_names [6]`、`embeddings [N, 6, D]`、`present [N, 6]`、`photo_manifest_json [N]` です。`photo_manifest_json` には特徴抽出時の画像参照と SHA-256 を保存します。`--audit-crops` を指定すると、IDと元画像名を含まない masked crop とコンタクトシートを ignored output に保存します。

`score_code_pairs.py` は、`pairs.csv` と特徴 NPZ から共通歯種のコサイン類似度を計算し、歯種別スコアの単純平均を `fused_score` として出力します。採点前に現在の画像を再ハッシュし、抽出時の SHA-256 と一致しない場合は停止します。抽出時ハッシュを使って、別 checkup IDに同一内容の写真があるペアは `shared_photo_content` として除外します。特徴欠損や共通歯不足のペアも `skipped_pairs.csv` に理由付きで記録し、採点済みペアだけを `evaluate_authentication.py` 互換の `scores.csv` に保存します。

`extract_code_features.py`、`score_code_pairs.py`、`evaluate_authentication.py` は出力一式を同一世代として作成し、成功した場合だけ `--output-dir` を置き換えます。途中で失敗した場合は、前回の出力を残します。

COde は匿名 `patient_id` 内の研究評価にだけ使用します。実人物の特定、再識別、実運用の本人確認には使用しません。画像、特徴 NPZ、重み、監査 crop、評価ログは Git にコミットしません。

## 判断

細かい trial を直接並べると比較対象が読みにくくなるため、第二段階の表側には主要な比較結果だけを残しています。全試行を確認する場合は、`90_archive/all_training_trials/` を見てください。
