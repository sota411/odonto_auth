# odonto_auth

歯牙画像を用いた補助認証の研究用コードです。

スマートフォンなどで撮影した口腔内画像から歯牙領域を抽出し, 補助認証に必要な照合・評価へ接続するための実験コードを整理しています。現時点では, 前歯領域のセグメンテーション実験と, v1 から v7 までの評価結果を扱います。

口腔内画像は個人情報性の高い生体データであるため, このリポジトリでは実データと学習済み重みを配布していません。再実行には, 利用権限を確認したデータセットとモデル重みを別途用意してください。

## セットアップ

```bash
uv sync
```

## 代表コマンド

v7 学習スクリプトの dry-run:

```bash
uv run python 01_stage1_real_image_extraction/scripts/train_tooth_seg_flont_v7.py --dry-run --device cpu --workers 0 --project /tmp/mitou_clean_v7_smoke
```

v4 baseline と v7 best の fitness 比較:

```bash
uv run python 02_stage2_capture_matching/scripts/score_experiment.py
```

v1 から v7 までのスコア集計:

```bash
uv run python 02_stage2_capture_matching/scripts/summarize_tooth_seg_scores.py
```

## 公開範囲

- `01_stage1_real_image_extraction/`: 歯牙セグメンテーション用のスクリプトと主要な実験結果。
- `02_stage2_capture_matching/`: 評価スクリプト, score summary, 比較用の図表。
- `90_archive/`: 旧実験のうち, 公開してよい集計結果と補助スクリプト。

## 注意

このリポジトリ単体では, 実データやモデル重みを必要とする処理は再実行できません。データと重みの管理では, 被験者の同意, データの利用条件, プライバシー保護を優先してください。
