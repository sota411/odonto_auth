# COde ゼロショット照合ベースライン

## 結論

v7 の実写ゼロショット抽出と、学習なしの ResNet50/HOG 特徴では、COde の別 checkup 照合を十分に分離できなかった。ResNet50 は HOG より良かったが、ROC AUC は 0.587、EER は 43.0% である。FAR を1%以下にすると FRR は90.2%まで上がった。

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
- crop: mask外を0、12%余白、224x224
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

## 認証結果

| 特徴 | EER | ROC AUC | d-prime | genuine mean | impostor mean | FAR 1%以下の FRR |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 43.0% | 0.587 | 0.280 | 0.816 | 0.791 | 90.2% |
| HOG | 45.7% | 0.570 | 0.180 | 0.799 | 0.778 | 98.9% |

分布は [ResNet50](../evaluation/code_matching_resnet50_score_distribution.png) と [HOG](../evaluation/code_matching_hog_score_distribution.png) に保存した。数値の全桁は [code_matching_baseline_metrics.csv](../evaluation/code_matching_baseline_metrics.csv) に残している。

impostor が79件なので、観測可能な最小の非ゼロ FAR は `1/79 = 1.27%` である。FAR 1%と0.1%の動作点は、どちらも false accept が0件になる同じ閾値を返す。低 FAR 性能を測れたとは解釈しない。

## 判断

凍結 ResNet50 は古典特徴より少し良い。ただし、genuine と impostor の平均差は0.025に留まり、分布は大きく重なった。現状の crop には照合へ使える情報が多少あるが、そのまま認証器にする根拠はない。

先に必要なのは実写向け segmentation の改善である。COde の前歯画像へ歯牙単位マスクを付け、v7 のゼロショット mAP と v8 fine-tune 後の mAPを人物またはセッション分離で比較する。抽出 coverage とクラス安定性が改善した後、同じ seed、同じ重複除外、同じ test pairを使って本ベースラインを再実行する。その結果でも分離が弱い場合に metric learning を検討する。

今回の値には bootstrap 信頼区間を付けていない。採点対象が元のペアから大きく減った complete-case 評価でもあるため、最終性能としては扱わない。

## 再現上の注意

Ultralytics 8.4.19 はパスの list を渡すと、list 全体を画像としてメモリへ展開する。6,496パスを一度に渡した初回 full run は終了コード137で停止した。`extract_code_features.py` は入力を64枚ずつ渡し、predictor が保持する結果、batch、dataset を解放してから最大16 cropずつ特徴を抽出する。レビュー修正後の141枚 pilot 実行中に確認した VRAM 使用量は9,898 / 10,240 MiBだった。容量が小さい GPU では、まず `--source-chunk-size` と `--feature-batch-size` を下げる。

再現コマンドと出力形式は [README.md](../README.md) にまとめた。
