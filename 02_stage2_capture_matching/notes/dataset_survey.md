# 公開口腔内写真データセット調査

調査日: 2026-06-12

## 目的

`plans/plan_01_dataset_research.md` に基づき、実写の口腔内カラー写真を含む公開データセットを探索し、前歯6本のセグメンテーションと個人照合に使えるかを一次情報で判定した。

判定は次の3用途に分けた。

- 照合用: 被写体 ID と複数撮影、または複数チェックアップが確認できること。
- 抽出用: 前歯が写るカラー写真があり、歯牙、歯肉、疾患、歯科所見などの注釈が確認できること。
- 対象外: X線、CBCT、3Dスキャン、口腔外写真、または前歯カラー写真の条件を満たさないこと。

## 結論

照合用には [COde: A benchmark multimodal oro-dental dataset](https://huggingface.co/datasets/zirak-ai/COde) を第一候補にする。Hugging Face の一次情報で、8,775 dental checkups、4,800 patients、50,000 intraoral photographs、CC BY 4.0 が確認できる。GitHub 公開分の `complete_dataset.csv` には `checkup_id`、`patient_id`、`photographs` 列があり、ローカル検証では 8,775 行、4,800 患者、複数チェックアップを持つ患者 2,339 名、写真参照 49,938 件が確認できた。COde の Data Usage Agreement は再識別を禁じているため、利用範囲は匿名 `patient_id` 内のペア評価に限定し、実人物の特定、再識別、または実運用の本人確認を目的にしない。補助認証用途や生体認証研究として進める前には、配布元の追加許諾または同意を得た自前データが必要である。COde には歯牙単位のセグメンテーションマスクが確認できないため、前歯6本の教師ありセグメンテーションには単独では不足する。

抽出用には [FDTooth](https://physionet.org/content/fdtooth/1.0.0/) と [AlphaDent](https://zenodo.org/records/16582489)、[SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth)、および [Annotated intraoral image dataset for dental caries detection](https://zenodo.org/records/14827784) を条件付きで検討する。FDTooth は前歯と匿名患者 ID を持つが申請制で、写真は 1 患者 1 枚である。AlphaDent と SegmentAnyTooth は歯牙セグメンテーションに近いが、照合用の複数セッションは確認できない。Zenodo のう蝕データは5ビューの口腔内写真と検出注釈を持つが、注釈対象はう蝕であり、歯牙単位マスクではない。

完全条件である「前歯が写る実写カラー口腔内写真、被写体 ID、複数セッション、歯牙単位セグメンテーション注釈、再配布可能ライセンス」を同時に満たす公開データセットは、下記の探索範囲では確認できなかった。したがって、Plan 02 では自前収集または COde を使った照合プロトタイプの検証と、別データによる抽出モデルの fine-tune を分けて設計する。

## 候補一覧

| 名称 | 一次情報 URL | 画像種別 | 規模 | ID・複数セッション | アノテーション | ライセンス・利用条件 | 判定 | 根拠 |
|---|---|---|---|---|---|---|---|---|
| COde: A benchmark multimodal oro-dental dataset | [Hugging Face](https://huggingface.co/datasets/zirak-ai/COde), [GitHub](https://github.com/zirak-ai/COde), [Data Usage Agreement](https://huggingface.co/datasets/zirak-ai/COde/blob/main/Data%20Usage%20Agreement.pdf) | 実写口腔内カラー写真、X線、臨床テキスト | 8,775 dental checkups、4,800 patients、50,000 intraoral photographs | `patient_id` と `checkup_id` を確認。ローカル検証で複数チェックアップ患者 2,339 名 | 診断、治療計画、QA。歯牙マスクは確認できない | CC BY 4.0。DUA は再識別禁止、研究・教育・開発目的、法令・倫理遵守。匿名 ID 内のペア評価に限定し、実人物の特定、再識別、実運用の本人確認には使わない | 条件付きで照合用に使える | Hugging Face README と API に規模、license、gated=false。公開 CSV に `checkup_id`、`patient_id`、`photographs` |
| FDTooth | [PhysioNet](https://physionet.org/content/fdtooth/1.0.0/) | 前歯の実写口腔内写真と CBCT | 241 patients、241 intraoral images、241 CBCT scans | 匿名患者 ID はある。PhysioNet 本文は 1 patient 1 intraoral photograph と明記 | 前歯12本の F/D/N ラベル、写真向け bbox、CBCT | PhysioNet Credentialed Health Data License 1.5.0。credentialed user、DUA、CITI training が必要 | 申請要。抽出用に条件付きで使える。照合用には不足 | PhysioNet 本文に JPEG 5760 x 3840、patient ID、bbox、241 patients。Access Policy は credentialed users with DUA |
| AlphaDent | [Zenodo](https://zenodo.org/records/16582489), [GitHub](https://github.com/ZFTurbo/AlphaDent), [Hugging Face](https://huggingface.co/datasets/ZFTurbo/AlphaDent), [論文ページ](https://computeroptics.ru/eng/KO/Annot/KO49-6/490629e.html) | DSLR 歯科写真 | 295 patients、over 1200 images | 患者数は論文ページで確認。複数セッションは確認できない | 9 class instance segmentation | Zenodo は CC BY 4.0。Hugging Face card は apache-2.0 と表示され、配布先で表記差がある | 条件付きで抽出用に使える。照合用には不足 | 論文ページは 295 patients、over 1200 images、instance segmentation、9 classes、open license と記載 |
| Annotated intraoral image dataset for dental caries detection | [Zenodo](https://zenodo.org/records/14827784), [Data tree](https://zenodo.org/api/records/14827784/files/Data%20tree.txt/content) | 実写口腔内カラー写真。正面、左右側方面、上顎咬合面、下顎咬合面 | 6,313 images | no_retractors、retractors、pilot と view 別構造。被写体 ID と複数セッションは確認できない | う蝕の物体検出。YOLO、COCO、Pascal VOC、LabelMe | CC BY 4.0 | 条件付きで疾患検出用。歯牙セグメンテーションと照合には不足 | Zenodo API は 6,313 images と object detection を記載。Data tree は Frontal、Left_Lateral、Mandibular、Maxillary_Occlusal、Right_Lateral |
| Segmentation teeth Images and Masks | [Kaggle API](https://www.kaggle.com/api/v1/datasets/view/leonardoaranguiz/segmentation-teeth-images-and-masks) | Zenodo う蝕データの派生。標準化済み 572 x 572 口腔内写真と歯牙マスク | Kaggle API は総容量 1.6 GB と記載。説明は 32 teeth plus background | patient-aware split 推奨と patient ID への言及あり。複数セッションは確認できない | 32歯 + 背景の pixel-level segmentation masks | Kaggle metadata は CC BY-NC-SA 4.0。説明文中の License は CC BY 4.0 と書かれており、表記差がある | 条件付きで抽出用に使える。ライセンス確認が必要 | Kaggle API の `licenseName` は CC BY-NC-SA 4.0。説明は Zenodo DOI 由来と歯牙マスクを記載 |
| Natural Color Tooth (NCT) dataset | [Kaggle API](https://www.kaggle.com/api/v1/datasets/view/alielhenidy/tooth-dataset) | モバイルカメラ由来の自然色歯画像 | 1,019 images。train 805、validation 214 | 被写体 ID と複数セッションは確認できない | labelImg による bbox | Apache 2.0 | 条件付きで歯検出用。前歯6本照合には不可 | Kaggle API は mobile cameras、1019 images、bounding box annotation を記載 |
| A DENTAL INTRAORAL IMAGE DATASET OF GINGIVITIS FOR IMAGE CAPTIONING | [Mendeley Data](https://data.mendeley.com/datasets/3253gj88rr/1), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39386321/) | 高解像度の実写口腔内写真。論文抄録は前歯12本と歯肉組織を対象と説明 | 1,096 images | 被写体 ID と複数セッションは確認できない | Gingivitis label と caption。歯牙マスクは確認できない | Mendeley は CC BY 4.0 | 条件付きで前歯・歯肉のドメイン適応用。照合と歯牙分割には不足 | Mendeley は 1,096 images、captions、labels、train/validation/test を記載 |
| Teeth or Dental image dataset | [Mendeley Data](https://data.mendeley.com/datasets/6zsnhrds9t/1) | 子どもの非う蝕歯の複数視点画像 | 9,562 images、8 subcategories | 被写体 ID と複数セッションは確認できない | カテゴリ別画像。歯牙マスクは確認できない | CC BY 4.0 | 条件付きでドメイン適応用。照合と教師あり歯牙分割には不足 | Mendeley は maxillary/upper front、right、left、occlusal、mandibular/lower front、right、left、occlusal を記載 |
| OMNI Dataset | [GitHub](https://github.com/RoundFaceJ/OMNI) | 実写 RGB 口腔内写真、複数視点 | 4,166 RGB images、384 participants | participants は確認。複数セッションは確認できない | Malocclusion issue labels、COCO format annotations | GitHub に LICENSE は確認できない。Google Drive 配布 | 条件付きで分類・検出の参考。ライセンス未確認のため利用前確認が必要 | GitHub README は 4,166 RGB oral cavity images、384 participants、5 views、COCO structure を記載 |
| SegmentAnyTooth | [GitHub](https://github.com/thangngoc89/SegmentAnyTooth) | データセットではなく、口腔内写真向け歯牙列挙・セグメンテーション framework | 公開データセット規模は確認できない | 該当なし | 出力 mask は FDI tooth numbers | Code は MIT。weights は署名した non-commercial license agreement が必要 | データセットではない。抽出ツールとして条件付きで使える | README は intraoral photos、front/upper/lower/left/right views、weights 申請、non-commercial license を記載 |
| DENTEX | [Grand Challenge](https://dentex.grand-challenge.org/), [Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) | パノラマ歯科 X線 | 693 quadrant、634 quadrant-enumeration、1005 fully labeled、1571 unlabeled X-rays | 患者 ID と写真セッションは対象外 | X線上の検出、FDI、診断 | CC BY-NC-SA 4.0 | 対象外 | Grand Challenge は panoramic X-rays challenge と記載。Hugging Face README は panoramic dental X-rays と記載 |
| Teeth3DS / Teeth3DS+ / 3DTeethSeg | [3DTeethSeg GitHub](https://github.com/abenhamadou/3DTeethSeg22_challenge), [Teeth3DS+](https://crns-smartvision.github.io/teeth3ds/), [arXiv](https://arxiv.org/abs/2210.06094), [Grand Challenge](https://3dteethseg.grand-challenge.org/) | 3D intraoral scans | 1,800 3D intraoral scans、900 patients。Teeth3DS+ の 3DTeethLand は 340 IOS scans | 患者 ID と上下顎スキャンはある。実写写真ではない | 3D 頂点単位の tooth labels と instances、FDI。3DTeethLand は3D landmarks | Teeth3DS データは CC BY-NC-ND 4.0。3DTeethSeg リポジトリは MIT。3DTeethLand Zenodo 登録は CC BY 4.0 | 対象外。ただし擬似画像生成元に近い | 3DTeethSeg GitHub は data license、1,800 3D intra-oral scans、900 patients、`id_patient` を記載。Teeth3DS+ は intraoral 3D scans と 340 IOS scans を記載 |
| ToothFairy2 | [公式ページ](https://ditto.ing.unimore.it/toothfairy2/), [Grand Challenge dataset](https://toothfairy2.grand-challenge.org/dataset/) | CBCT volume | Set P 417、Set F 63、合計480 | 実写写真ではない。患者単位 volume | raw images、segmentation maps、dataset.json。42 classes | Grand Challenge dataset はログインが必要。training set は CC BY-SA と表示 | 対象外 | 公式ページは MICCAI2024、CBCT volume、417 + 63、42 classes を記載 |
| STS-Tooth | [Zenodo](https://zenodo.org/records/10597292) | パノラマ X線と CBCT | STS-2D-Tooth は 4,000 images と 900 masks。STS-3D-Tooth は 148,400 unlabeled scans と 8,800 masks | 小児・成人カテゴリはある。実写写真ではない | 歯セグメンテーションマスク | CC BY 4.0 | 対象外 | Zenodo API は PXI、CBCT、各規模、license を記載 |
| PhysioNet multimodal dental dataset | [PhysioNet](https://physionet.org/content/multimodal-dental-dataset/1.1.0/) | CBCT、パノラマ X線、根尖 X線 | 169 patients、329 CBCT files、8 panoramic radiographs、16,203 periapical radiographs | 患者 ID と複数 visit はある | CBCT tooth marking、implant label など | PhysioNet Contributor Review Health Data License 1.5.0 | 対象外 | PhysioNet 本文は DICOM/TIFF の X線・CBCTであり、実写口腔内カラー写真ではない |
| IEEE DataPort: Aoralscan3 tooth segmentation dataset | [IEEE DataPort](https://ieee-dataport.org/documents/aoralscan3-tooth-segmentation-dataset), [DOI](https://doi.org/10.21227/w9mp-5w63) | Aoralscan3 の動画・画像、過去 3D model と組み合わせる歯牙データ。静止実写口腔内写真ではない | train 1,573 videos、validation 244 videos、test 244 videos。画像サイズ 640 x 480 | 照合用の患者 ID と複数セッションは確認できない | LabelMe による歯境界と分類。point cloud-based instance segmentation にも利用可能 | IEEE DataPort Subscription が必要 | 対象外 | IEEE DataPort 本文は Aoralscan3、videos、previous 3D models、point cloud-based instance segmentation、subscription access を記載 |

## COde の追加検証

COde は照合用候補であるため、GitHub で公開されている `CODeD-Dataset.zip` の `complete_dataset.csv` を標準 CSV パーサで確認した。CSV には引用符内改行が含まれるため、`cut` では列が崩れることを確認し、Python 標準ライブラリの `csv.DictReader` で構造化して読んだ。

確認コマンドの結果は次の通りである。

```text
rows 8775
patients 4800
patients_with_multiple_checkups 2339
max_checkups_per_patient 11
rows_with_photographs 8772
total_photograph_refs 49938
min_photos_per_photo_row 1
max_photos_per_photo_row 26
```

この結果から、COde は匿名 `patient_id` 内の同一患者ペアと別患者ペアを作る研究評価に使える。DUA は再識別を禁じているため、実人物の特定、再識別、実運用の本人確認を目的にした利用は行わない。セグメンテーションマスクは確認できないため、前歯6本の抽出性能評価には別データまたは自前アノテーションが必要である。

## 探索範囲

確認したデータ配布先は次の通りである。

- Grand Challenge: DENTEX、3DTeethSeg。
- Zenodo: Annotated intraoral image dataset for dental caries detection、AlphaDent。
- Kaggle: Segmentation teeth Images and Masks、Natural Color Tooth。
- Mendeley Data: Gingivitis captioning dataset、Teeth or Dental image dataset。古い `https://data.mendeley.com/datasets/9inf2ivghv/2` は Dataset Not Found であった。
- PhysioNet: FDTooth、multimodal dental dataset。
- IEEE DataPort: Aoralscan3 tooth segmentation dataset のページと DOI 解決先を直接確認した。
- 論文経由: PubMed の gingivitis dataset、arXiv の COde、Teeth3DS+、SegmentAnyTooth、AlphaDent 論文ページ。
- Teeth3DS 関連: Teeth3DS+ 公式ページ、3DTeethSeg Grand Challenge、PyTorch Geometric 参照情報。

使用した主な検索クエリは次の通りである。

- `intraoral photograph dataset tooth segmentation color images dataset`
- `dental photograph segmentation dataset RGB intraoral images`
- `oral image dataset intraoral photographs teeth segmentation dataset`
- `tooth segmentation RGB image dataset intraoral photograph`
- `Annotated intraoral image dataset for dental caries detection dataset download`
- `Varying Views of Teeth Dataset Mendeley Data`
- `site:grand-challenge.org dental intraoral photograph dataset tooth segmentation`
- `DENTEX challenge dataset dental x-ray tooth enumeration diagnosis grand challenge`
- `site:ieee-dataport.org dental intraoral photograph dataset teeth photo`
- `FDTooth intraoral photographs CBCT PhysioNet`
- `OMNI Oral Imaging for Malocclusion Issues Assessments dataset`
- `AlphaDent dataset automated tooth pathology detection`
- `SegmentAnyTooth intraoral photos dataset GitHub`
- `A dental intraoral image dataset of gingivitis for image captioning`
- `COde benchmark multimodal oro-dental dataset Hugging Face`

## 採用・不採用理由

COde を照合用の第一候補とした理由は、患者 ID、複数チェックアップ、写真ファイル参照、CC BY 4.0、非 gated 公開が一次情報とローカル検証でそろったためである。不採用にしなかった理由は、照合に必要な「同一人物の複数撮影」が確認できたためである。歯牙マスクがないため、抽出用の単独データとしては不採用である。

FDTooth を抽出用の条件付き候補とした理由は、前歯写真、匿名患者 ID、前歯12本の歯単位ラベル、bbox が確認できたためである。照合用としては、1患者1写真で本人内変動を評価できないため不採用である。申請制であるため、Plan 02 で利用する場合は PhysioNet credentialing、CITI training、DUA signing が必要である。

AlphaDent、SegmentAnyTooth、Kaggle の Segmentation teeth Images and Masks は抽出モデルの fine-tune またはベースライン比較に近い。照合用としては、被写体 ID と複数セッションが確認できないため不採用である。

DENTEX、Teeth3DS+、PhysioNet multimodal dental dataset、IEEE DataPort の代表候補は、X線、CBCT、3D scan が中心であり、計画の必須条件である実写口腔内カラー写真ではないため不採用である。

## Plan 02 / Plan 03 への反映

Plan 02 では、照合用データの短期候補を COde にする。COde の `patient_id` と `checkup_id` を使い、患者単位の train/test split と、同一患者の別チェックアップを positive pair、別患者を negative pair とする設計に進める。評価は匿名 ID 内のペア分類または類似度評価に限定し、DUA の再識別禁止に抵触する実人物の特定、再識別、実運用の本人確認は扱わない。補助認証や生体認証研究として外部に位置づける場合は、配布元の追加許諾または同意を得た自前データを先に用意する。

Plan 03 では、抽出モデルの実写適応を COde の疑似ラベルまたは別データの教師ありラベルで分ける。教師あり fine-tune には FDTooth、AlphaDent、SegmentAnyTooth、Segmentation teeth Images and Masks を候補にし、ライセンス表記差や申請要件を先に解消する。COde は歯牙セグメンテーション教師データではなく、実写ドメインでの推論対象として扱う。

## 残課題

COde は Hugging Face API の `usedStorage` が 1,929,195,758 bytes であり、`COde-Dataset.zip` の HEAD で確認した `Content-Length` / `x-linked-size` は 963,785,879 bytes である。今回の調査では画像本体のダウンロードと視覚確認は実施していない。次工程では、ファイル名、画像ビュー、前歯の写り方、同一患者内の撮影条件差をサンプルで確認する必要がある。

FDTooth は申請制であり、ファイル本体は未取得である。利用する場合は、PhysioNet の credentialing、CITI training、DUA signing を完了してから、前歯 bbox と本プロジェクトの前歯6本定義の対応を確認する。

OMNI は GitHub README でデータ内容は確認できるが、LICENSE ファイルが 404 であった。利用する場合は、著者または配布元のライセンス確認が必要である。

サブエージェント調査は2件がコンテキスト上限で失敗したため、同じ範囲を小さく切った代替サブエージェントを起動した。失敗により一次情報の確認を省略せず、主担当側で配布元ページと API を直接確認した。
