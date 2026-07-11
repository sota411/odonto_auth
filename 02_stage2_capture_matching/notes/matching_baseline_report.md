# COde ゼロショット照合ベースライン

## 結論

v7 の実写ゼロショット抽出と、学習なしの ResNet50/HOG 特徴では、COde の別 checkup 照合を十分に分離できなかった。歯軸回転正規化後の最良値は HOG の ROC AUC 0.605、EER 43.0%である。回転前の HOG から AUC は0.035、EERは2.6ポイント改善したが、実用精度には届いていない。

次に metric learning は行わない。1,998ペアのうち採点できたのは171件だけで、監査 crop にも歯肉や画像端の誤抽出が残った。ノイズの多い crop を使って埋め込みを学習する前に、Plan 03 の実写向け歯牙抽出を進める。

## データ

COde は [Hugging Face の一次配布元](https://huggingface.co/datasets/zirak-ai/COde)から取得した。ZIP は 963,785,879 bytes、SHA-256 は `f4e5664ef4d57caa7ecdf68963551dccf0053e380efeb2f198f1cd5d2d68781d` である。8,775行から写真参照がない3 checkup を除き、4,799患者、8,772 checkup を使った。

[Data Usage Agreement](https://huggingface.co/datasets/zirak-ai/COde/blob/main/Data%20Usage%20Agreement.pdf) に従い、匿名 `patient_id` 内の研究評価だけに使用した。再識別、実人物の特定、実運用の本人確認には使わない。画像、重み、特徴 NPZ、ペア別スコアは Git に保存していない。

患者単位で train/val/test を70/15/15に分け、seed は42に固定した。test split は721患者、1,314 checkup で、genuine 999件と impostor 999件を作った。

## 実行条件

- GPU: NVIDIA GeForce RTX 3080 10 GB
- segmentation: v7 best、`imgsz=832`、`conf=0.05`、`iou=0.70`
- v7 weight SHA-256: `f945236eb2441dfbbd0c439a5cd1c3e4d94e97650f3d0429cff5ee6da7a90454`
- ResNet50 weight SHA-256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- memory bounds: Ultralytics への入力は64枚ずつ、ResNet50 は最大16 cropずつ
- crop: mask外を0、歯牙maskの主軸をPCAで縦方向へ回転正規化、12%余白、224x224
- checkup集約: 歯種ごとに confidence 上位3 viewを加重平均
- deep feature: [ResNet50 IMAGENET1K_V2](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html) の分類層直前、2,048次元
- classical feature: HOG、1,764次元
- score: 共通歯種の cosine similarity を0から1へ写像し、単純平均
- leakage guard: 抽出時の元画像 SHA-256 を採点時に再検証し、template/query 間で1件でも一致するペアを除外

環境は PyTorch 2.10.0+cu128、Torchvision 0.25.0+cu128、Ultralytics 8.4.19、OpenCV 4.13.0、NumPy 2.4.2 だった。

## Pilot

先頭20ペア、21 checkup、141枚で segmentation confidence を比較した。

| confidence | 特徴あり checkup | 歯種スロット | 採点可能ペア |
|---:|---:|---:|---:|
| 0.10 | 10/21 | 18 | 3/20 |
| 0.05 | 10/21 | 26 | 5/20 |
| 0.01 | 16/21 | 49 | 13/20 |

0.01 は coverage が増えたが、監査 crop では0.05未満に歯肉や画像端が混ざった。認証指標を見て閾値を選ばず、crop の目視結果から0.05に固定した。

## 抽出結果

full test で参照したのは1,998ペア、1,124 checkup、6,496枚だった。v7 は1,267件を検出し、430 checkup に1歯以上の特徴を作った。checkup coverage は38.3%である。

歯種スロットは合計1,016件だった。内訳は R1 170、R2 160、R3 168、L1 263、L2 158、L3 97で、L1に偏った。

ペア採点前の SHA-256 照合では、151件が template/query 間で同一画像内容を共有していた。151件はすべて genuine で、genuine 全体の15.1%に当たる。これらを除外したうえで、1,676件は共通歯種がないため採点できなかった。最終的に残ったのは171件、内訳は genuine 92件、impostor 79件である。全ペアに対する coverage は8.6%だった。

## 回転正規化前の認証結果

| 特徴 | EER | ROC AUC | d-prime | genuine mean | impostor mean | FAR 1%以下の FRR |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 43.0% | 0.587 | 0.280 | 0.816 | 0.791 | 90.2% |
| HOG | 45.7% | 0.570 | 0.180 | 0.799 | 0.778 | 98.9% |

分布は [ResNet50](../evaluation/code_matching_resnet50_score_distribution.png) と [HOG](../evaluation/code_matching_hog_score_distribution.png) に保存した。数値の全桁は [code_matching_baseline_metrics.csv](../evaluation/code_matching_baseline_metrics.csv) に残している。

impostor が79件なので、観測可能な最小の非ゼロ FAR は `1/79 = 1.27%` である。FAR 1%と0.1%の動作点は、どちらも false accept が0件になる同じ閾値を返す。低 FAR 性能を測れたとは解釈しない。

## 回転正規化後の認証結果

2026年7月11日に RTX 3080 で6,496枚を再抽出した。segmentation重み、`imgsz/conf/iou`、cropサイズ、余白、最大view数、test pairは回転前と同一である。検出1,267件、特徴あり430 checkup、歯種スロット1,016件、採点171件も一致した。前処理formatだけを`tooth-axis-normalized-crop-v1`へ変更した比較である。

| 特徴 | EER | ROC AUC | d-prime | genuine mean | impostor mean | FAR 1%以下の FRR |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 42.4% | 0.592 | 0.292 | 0.812 | 0.788 | 94.6% |
| HOG | 43.0% | 0.605 | 0.290 | 0.828 | 0.802 | 98.9% |

回転前との比較では、ResNet50はAUCが0.005上がり、EERが0.6ポイント下がった。HOGはAUCが0.035上がり、EERが2.6ポイント下がった。一方、FAR 1%以下のFRRはResNet50で4.3ポイント悪化し、HOGでは変わらなかった。単一の動作点を含め、認証性能が安定して改善したとは判断しない。

分布は [ResNet50](../evaluation/code_matching_rotv2_resnet50_score_distribution.png) と [HOG](../evaluation/code_matching_rotv2_hog_score_distribution.png)、全桁は [code_matching_rotv2_baseline_metrics.csv](../evaluation/code_matching_rotv2_baseline_metrics.csv) に保存した。

## Teeth3DS別視点のHOG評価

実写評価とは別に、Teeth3DSの3D NPZ 135症例から正面0度と方位20度・仰角8度の2視点をPyVistaで生成した。各viewは前歯6本だけをRGBとFDI labelへ同じカメラで描画した。float64のmesh最大edgeに対する相対tolerance以下だったtriangleは25症例に439 faceあり、明示した`--drop-degenerate-faces`で除外した。症例別の除外数と判定toleranceはmanifestへ残した。

同一症例の別view 135組をgenuine、別症例の全view組合せ36,180組をimpostorとした。歯種maskから回転正規化済みcropを作り、HOG特徴のコサイン類似度を6歯で平均した結果、fused ROC AUCは0.757、d-primeは0.996だった。genuine平均は0.887、impostor平均は0.864で、分布は重なっている。

歯種別ではR1がROC AUC 0.830、d-prime 1.284で最も高かった。L2はAUC 0.559で、別視点でも識別力が弱い。全桁は [rendered_hog_matching_metrics.csv](../evaluation/rendered_hog_matching_metrics.csv)、分布図は [rendered_hog_score_distribution.png](../evaluation/rendered_hog_score_distribution.png) に保存した。入力135件の内容hashとface除外数は [teeth3ds_render_sources.csv](../fixtures/teeth3ds_render_sources.csv) に記録した。

この評価は同じ3D meshのカメラだけを変えており、唾液、照明、口唇、歯肉、撮影日差を含まない。そのため、実写性能の代替には使わない。Plan 04の擬似genuine/impostor実装確認と、歯種別傾向の予備値として扱う。

## 歯種別の識別力

既存の171件のスコアに対して、`evaluate_per_tooth_scores.py` で歯種ごとの ROC AUC と d-prime を算出した。次の表は歯軸回転正規化を導入する前に抽出した特徴の結果である。

| 歯種 | ResNet50 AUC | ResNet50 d-prime | HOG AUC | HOG d-prime |
|---|---:|---:|---:|---:|
| R1 | 0.739 | 0.711 | 0.731 | 0.649 |
| R2 | 0.605 | 0.377 | 0.649 | 0.472 |
| R3 | 0.643 | 0.505 | 0.601 | 0.315 |
| L1 | 0.522 | 0.032 | 0.540 | 0.091 |
| L2 | 0.583 | 0.273 | 0.622 | 0.301 |
| L3 | 0.556 | 0.220 | 0.540 | 0.140 |

R1が両特徴で最も高く、L1は検出件数が最多でも識別力がほぼランダムだった。ただし、R1でも genuine 25件、impostor 21件、L3では9件と7件しかなく、歯種の優劣を確定する規模ではない。全桁と歯種別件数は [ResNet50](../evaluation/code_matching_per_tooth_resnet50_metrics.csv) と [HOG](../evaluation/code_matching_per_tooth_hog_metrics.csv) に保存した。

回転正規化後の同じ171件では、歯種別結果は次のようになった。

| 歯種 | ResNet50 AUC | ResNet50 d-prime | HOG AUC | HOG d-prime |
|---|---:|---:|---:|---:|
| R1 | 0.716 | 0.674 | 0.737 | 0.640 |
| R2 | 0.632 | 0.627 | 0.679 | 0.646 |
| R3 | 0.657 | 0.459 | 0.661 | 0.491 |
| L1 | 0.551 | 0.184 | 0.555 | 0.182 |
| L2 | 0.694 | 0.549 | 0.656 | 0.389 |
| L3 | 0.460 | 0.013 | 0.730 | 0.603 |

HOGではR1、R2、R3、L1、L2、L3のうち5歯種でAUCが上がった。ResNet50ではR2、R3、L1、L2が上がり、R1とL3が下がった。L3は genuine 9件、impostor 7件だけなので、HOGの大幅上昇とResNet50の低下を一般化しない。全桁は [ResNet50](../evaluation/code_matching_rotv2_per_tooth_resnet50_metrics.csv) と [HOG](../evaluation/code_matching_rotv2_per_tooth_hog_metrics.csv) に保存した。

## 実画像CLIのend-to-end確認

2026年7月11日に、test splitの同一患者から2 checkup・18画像を登録し、特徴を3歯種のテンプレートへ集約した。照会には登録と異なる3番目のcheckupの画像を使い、画像SHA-256を固定した。CLIはR1 `0.8978243168`、L3 `0.8587185403`、融合スコア `0.8782714286`をJSONで返した。

同じ照会checkupの最初の3候補は登録側との共通歯が0本だったため、CLIは`min_common_teeth=1`の条件で停止した。これは入力欠損をスコア0などへ置き換えず、照合不能としてFail Fastする設計どおりの結果である。E2E成功はCLIの接続確認であり、融合スコア0.878を本人受入閾値として採用する根拠にはしない。

## 判断

回転正規化後はHOGのAUCがResNet50を0.013上回ったが、両方ともEERは42%以上である。現状のcropには照合へ使える情報が多少あるが、そのまま認証器にする根拠はない。

先に必要なのは実写向け segmentation の改善である。COde の前歯画像へ歯牙単位マスクを付け、v7 のゼロショット mAP と v8 fine-tune 後の mAPを人物またはセッション分離で比較する。抽出 coverage とクラス安定性が改善した後、同じ seed、同じ重複除外、同じ test pairを使って本ベースラインを再実行する。その結果でも分離が弱い場合に metric learning を検討する。

回転正規化だけではcoverageは改善しない。v8評価後は、同じseed、重複除外、test pair、`tooth-axis-normalized-crop-v1`を固定して再抽出し、今回の`rotv2`を比較対象にする。

今回の値には bootstrap 信頼区間を付けていない。採点対象が元のペアから大きく減った complete-case 評価でもあるため、最終性能としては扱わない。

## v8後の再評価

2026年7月11日に、採用したv8全層fine-tune重みで同じCOde test 6,496枚を再抽出した。回転正規化、confidence 0.05、IoU 0.70、入力832、pair manifestは`rotv2`から変えていない。検出は1,267件から14,965件、特徴ありcheckupは430件から1,006件、歯種スロットは1,016件から4,733件へ増えた。採点可能pairは171件から1,455件になった。

| 特徴 | 採点pair | ROC AUC | EER | d-prime |
|---|---:|---:|---:|---:|
| ResNet50 | 1,455 | 0.562 | 45.9% | 0.213 |
| HOG | 1,455 | 0.531 | 48.6% | 0.075 |

coverageは増えたが、分離は弱い。旧v7のAUCはResNet50 0.592、HOG 0.605だったものの、旧評価は採点できた171件だけなので単純な性能差としては扱わない。v8後の1,455件を次の基準集合とする。全桁は[統合指標](../evaluation/code_matching_rotv2_v8_baseline_metrics.csv)と、[ResNet50歯種別](../evaluation/code_matching_rotv2_v8_per_tooth_resnet50_metrics.csv)、[HOG歯種別](../evaluation/code_matching_rotv2_v8_per_tooth_hog_metrics.csv)に保存した。

v8重みでも照合CLIを実行し、2 checkup・18画像のHOG templateに対して別checkupの1画像を照合した。4歯のスコアと融合スコア0.850669をJSONで返した。CLIの接続は通ったが、この1件を受入閾値には使わない。

Plan 04の分岐条件に従うと、次はmetric learningの比較対象を作る段階である。ただし、実画像valは10枚しかなく、通常照明と過露光のmask mAP50は0だった。追加注釈でsegmentationの再現性を確認してから学習へ進む。

## 再現上の注意

Ultralytics 8.4.19 はパスの list を渡すと、list 全体を画像としてメモリへ展開する。6,496パスを一度に渡した初回 full run は終了コード137で停止した。`extract_code_features.py` は入力を64枚ずつ渡し、predictor が保持する結果、batch、dataset を解放してから最大16 cropずつ特徴を抽出する。レビュー修正後の141枚 pilot 実行中に確認した VRAM 使用量は9,898 / 10,240 MiBだった。容量が小さい GPU では、まず `--source-chunk-size` と `--feature-batch-size` を下げる。

再現コマンドと出力形式は [README.md](../README.md) にまとめた。
