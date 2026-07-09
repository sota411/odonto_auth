# 認証評価プロトコル

## 目的

照合スコア CSV から FAR、FRR、EER、ROC AUC を同じ条件で再計算できるようにする。入力画像、歯牙抽出、特徴抽出、機械学習はこの手順には含めない。

## 入力 CSV

`evaluate_authentication.py` は次の列を読む。

| 列 | 内容 |
|---|---|
| `query_id` | 検証側チェックアップまたは画像の ID |
| `template_id` | 登録側テンプレートまたはチェックアップの ID |
| `query_subject_id` | 検証側の匿名被験者 ID |
| `template_subject_id` | 登録側の匿名被験者 ID |
| `query_session_id` | 検証側のセッション ID |
| `template_session_id` | 登録側のセッション ID |
| `is_genuine` | 同一被験者なら `1`、別被験者なら `0` |
| `fused_score` | スコア統合後の照合スコア。大きいほど本人らしい値にする |

COde の `pairs.csv` 由来の列名に合わせるため、`query_patient_id`、`template_patient_id`、`query_checkup_id`、`template_checkup_id` も同じ意味の列として受け付ける。

COde 由来のデータを使う場合は、匿名 `patient_id` 内の研究評価に限定する。再識別、実人物の特定、実運用の本人確認には使わない。

## 除外ルール

`is_genuine=1` で `query_session_id` と `template_session_id` が同じ行は、評価から除外する。同一セッション内の照合は FRR を低く見せるため、登録セッションと検証セッションを分ける。除外件数は `summary.json` の `excluded_same_session_genuine` に残す。

`is_genuine=1` なのに被験者 ID が違う行、または `is_genuine=0` なのに被験者 ID が同じ行は、入力ラベルの誤りとして停止する。

## 指標

閾値 `theta` で `fused_score >= theta` を受入とする。

- FAR: impostor を誤って受け入れた割合。
- FRR: genuine を誤って拒否した割合。
- EER: FAR と FRR が交差する点の誤り率。
- ROC AUC: FAR を横軸、True Accept Rate を縦軸にした面積。

追加認証では誤受入を抑える必要があるため、`FAR <= 1%` と `FAR <= 0.1%` で FRR が最小になる動作点も出力する。

## 出力

`--output-dir` に次のファイルを出力する。

- `summary.json`: 件数、除外件数、EER、ROC AUC、動作点。
- `curves.csv`: 閾値ごとの FAR、FRR、TPR、FPR。
- `operating_points.csv`: FAR 目標ごとの閾値と FRR。
- `roc_curve.png`: ROC 曲線。
- `far_frr_curve.png`: FAR と FRR の閾値曲線。
