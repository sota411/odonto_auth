# 認証評価プロトコル

## 目的

照合スコア CSV から FAR、FRR、EER、ROC AUC、DET曲線と信頼区間を同じ条件で再計算できるようにする。入力画像、歯牙抽出、特徴抽出、機械学習はこの手順には含めない。

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

`extract_code_features.py` は、各 checkup が参照する元画像の SHA-256 を特徴 NPZ の `photo_manifest_json` に保存する。`score_code_pairs.py` は採点前に現在の画像を再ハッシュし、抽出時の値と一致しなければ停止する。template と query が抽出時に同じ画像内容を1枚でも共有していたペアは、checkup IDが違っていても `shared_photo_content` として採点しない。COde には、別 checkup IDから同一内容の写真を参照する例が実在するためである。除外件数はスコア生成側の `summary.json` と `skipped_pairs.csv` に残す。

歯牙特徴がないペアと、共通する歯種が `--min-common-teeth` 未満のペアも採点しない。これらは `missing_*_feature` または `insufficient_common_teeth` として記録する。除外後に genuine と impostor のどちらかが0件になる場合は、認証指標を出さずに停止する。

`is_genuine=1` で `query_session_id` と `template_session_id` が同じ行は、評価から除外する。同一セッション内の照合は FRR を低く見せるため、登録セッションと検証セッションを分ける。除外件数は `summary.json` の `excluded_same_session_genuine` に残す。

`is_genuine=1` なのに被験者 ID が違う行、または `is_genuine=0` なのに被験者 ID が同じ行は、入力ラベルの誤りとして停止する。

## 歯種別識別力評価

`evaluate_per_tooth_scores.py` は、`score_code_pairs.py` が生成した `scores.csv` の `query_subject_id`、`template_subject_id`、`query_session_id`、`template_session_id`、`is_genuine`、`per_tooth_scores` を読む。`per_tooth_scores` は、`R1`、`R2`、`R3`、`L1`、`L2`、`L3` の一部をキーとし、`0` 以上 `1` 以下の有限な数値を値とする、空ではない JSON object でなければならない。不正 JSON、重複キー、未知歯種、真偽値などの非数値、非有限値、範囲外の値、必須列の欠損、被験者 ID と `is_genuine` の不整合があれば、出力を更新せずに停止する。

認証評価と同じく、same-session genuine は全入力の検証後に除外する。各行で JSON にない歯は、その歯の評価母集団からだけ除外する。除外後に genuine と impostor の双方がある歯について、ROC AUC、d-prime、genuine/impostor の平均と件数を計算する。片クラスだけの歯やスコアがない歯は指標を計算せず、理由を `summary.json` の `missing_genuine_class`、`missing_impostor_class`、`no_scores` として記録する。

歯種別評価の `--output-dir` には、次の2ファイルを atomic に出力する。

- `per_tooth_metrics.csv`: 両クラスがある歯だけの ROC AUC、d-prime、平均、件数。分散が両クラスとも0で d-prime が定義できない場合は空欄にする。
- `summary.json`: 入力件数、使用件数、same-session genuine 除外件数、および全6歯の評価状態、欠損件数、クラス別件数、指標または未評価理由。

## 指標

閾値 `theta` で `fused_score >= theta` を受入とする。

- FAR: impostor を誤って受け入れた割合。
- FRR: genuine を誤って拒否した割合。
- EER: FAR と FRR が交差する点の誤り率。
- ROC AUC: FAR を横軸、True Accept Rate を縦軸にした面積。
- d-prime: genuine と impostor の平均差を、両分布の分散から求めた pooled standard deviation で割った値。大きいほど分離している。

追加認証では誤受入を抑える必要があるため、`FAR <= 1%` と `FAR <= 0.1%` で FRR が最小になる動作点も出力する。

## 信頼区間

EER、ROC AUC、d-primeには、genuineとimpostorの件数を固定した層別bootstrapの95%区間を付ける。既定は2,000回、seed 42である。各再標本でEER、AUC、d-primeを再計算し、2.5 percentileと97.5 percentileを区間にする。標本内の分散が0になった回はd-primeだけ未定義として数え、`summary.json`の`undefined_d_prime_replicates`へ記録する。

このbootstrapはpair単位であり、同じ被験者が複数pairへ現れる依存を補正しない。現在の区間を最終的な母集団推定には使わない。被験者数が増えた段階で、被験者単位のcluster bootstrapへ切り替える。

## 条件別評価

`--conditions-csv`と`--condition-column`を指定すると、評価器はquery側の匿名被験者IDとセッションIDを条件メタデータへ結合する。列名の既定は`subject_id`と`session_id`で、異なる場合は`--condition-subject-column`と`--condition-session-column`で指定する。

同じ被験者・セッションに同じ条件列の異なる値がある場合、どの値へスコアを割り当てるか決められないため停止する。画像ごとに条件が違うデータは、画像単位のquery scoreへ分けるか、事前に一意なセッション条件を定義する。条件値ごとにgenuineとimpostorのどちらかしかない場合は、`condition_metrics.csv`へ`insufficient_classes`を記録し、EERやAUCを空欄にする。

## 品質ゲート

`quality_gate.py`は、対象6歯の検出歯数、平均segmentation confidence、Laplacian分散、平均輝度、暗部clip率、明部clip率を受け取り、不合格理由をJSONで返す。画像統計の計算に機械学習は使わない。歯数とconfidenceは抽出モデルの結果を呼び出し側から渡す。

閾値は`config/quality_gate.json`に置く。現在の設定は`status=uncalibrated`であり、CLIは実行を拒否する。数値欄は設定schemaを固定するための無効値で、運用閾値ではない。v8のquery単位品質値と照合スコアを同じ母集団で再生成し、FAR/FRRへの影響を測定してから`calibrated`へ変更する。`fixtures/quality_gate_smoke.json`はI/O確認専用であり、認証判定に使わない。

## 出力

`--output-dir` に次のファイルを出力する。

- `summary.json`: 件数、除外件数、EER、ROC AUC、動作点。
- `curves.csv`: 閾値ごとの FAR、FRR、TPR、FPR。
- `operating_points.csv`: FAR 目標ごとの閾値と FRR。
- `bootstrap_intervals.csv`: EER、ROC AUC、d-primeのbootstrap区間。
- `condition_metrics.csv`: 条件値ごとの件数、評価可否、EER、ROC AUC、d-prime。
- `roc_curve.png`: ROC 曲線。
- `det_curve.png`: FARとFRRを正規偏差尺度で表したDET曲線。
- `far_frr_curve.png`: FAR と FRR の閾値曲線。
- `score_distribution.png`: genuine と impostor のスコア分布。
