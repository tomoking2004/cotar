# Visual Representation Alignment for Multimodal Large Language Models

**著者**：Heeji Yoon，Jaewoo Jung，Junwan Kim，Hyungyu Choi，Heeseong Shin，Sangbeom Lim，Honggyu An，Chaehyun Kim，Jisang Han，Donghyun Kim，Chanho Eom，Sunghwan Hong，Seungryong Kim（KAIST AI／New York University／Chung-Ang University／Korea University／ETH Zürich）
**掲載**：arXiv:2509.07979 v2，2025-10-10（v1 は 2025-09-09）．ICLR 2026 の virtual ページに "Poster in Workshop: ICLR 2026 Workshop on Multimodal Intelligence: Next Token Prediction and Beyond" として載る．このページの要旨は arXiv v2 の要旨と文が異なり，ワークショップ版の本文は読んでいない．本会議への採否は確認できていない．以下はすべて arXiv v2 の記述である
**原文**：[arXiv](https://arxiv.org/abs/2509.07979)・[ICLR 2026 virtual](https://iclr.cc/virtual/2026/10013230)

**LLaVA-1.5 の第16層の視覚トークンを，射影 MLP を介して事前学習済みの DINOv2 の特徴へ cosine で寄せる補助損失（λ=0.5）を足すと，視覚中心のベンチマークが上がる．**

## 位置づけ

**学習時に MLLM の中間層の表現へ補助損失を掛け，その層を32層中の第16層に置いた**先行例．層を4層おきに掃引して第16層を選び，隣接する複数層へ掛けると悪化することを表で示す．補助の目標は外部の視覚基盤モデル（VFM）の特徴であり，ラベルではない．

## 主張

- テキストだけの言語モデル損失では視覚トークンへの監督が間接的で，MLLM 内部の視覚表現が視覚エンコーダの特徴から離れていく（著者の言う "visual representation misalignment"）．
- 層ごとの CKNNA で見ると，"similarity to CLIP features drops sharply after the early layers and remains low in deeper layers"（§4.1）．中間層ではこの低下が緩む．
- 中間層の視覚トークンを VFM の特徴へ整合させる損失（VIRAL）を足すだけで，追加データなしに視覚中心の課題が上がる．
- PDF v2 の要旨の言い方では "with gains reaching up to 17.3% and an average improvement of 9.4% over the baseline"（arXiv abs ページの要旨と v1 の要旨には，この数値は無い）．9.4% は §1 の貢献の箇条にも現れるが，17.3% と 9.4% の算出の仕方は本文・付録に書かれていない．表1 の最大の差は What's Up・Vicuna-1.5-13B の +17.82 ポイントである．

## 手法

### 何に，何を掛けるか

| 要素 | 内容 |
| --- | --- |
| 対象の層 | LLaVA-1.5（32層と書かれる）の**第16層**（既定） |
| 対象のトークン | **視覚トークンの位置だけ**（`e^img_ℓ ∈ R^{N×D}`）．損失の式にテキストトークンは現れない |
| 射影 | 学習可能な**3層 MLP（SiLU）** `P_π` で VFM の特徴空間へ写す |
| 教師 | 事前学習済みの VFM の特徴 `y = E(I) ∈ R^{N×d}`．**既定は DINOv2** |
| 類似度 | **cosine** |

```
L_VRA   = −(1/N) Σ_{i=1}^{N} sim( P_π(e^img_{ℓ,i}), y_i )          (6)
L_total = L_LM + λ L_VRA ,   λ = 0.5（既定，付録 A）                (5)
```

- `L_LM` は通常の自己回帰の言語モデル損失（テキストトークン K 個の平均，式 (2)）．
- **λ を変えた実験は無い**（本文・付録・表1–6 を通読して，既定値 0.5 の記述のみ）．
- 第16層という選び方は，§4.2 で "following our analysis that fine-grained visual understanding emerges most prominently in middle layers" と説明され，層の掃引（§5.3）が裏づけとして挙げられる．
- 教師の候補：DINOv2，CLIP，SAM，Depth Anything v2（DAv2），RADIOv2.5．パッチ14のモデル（DINOv2，CLIP，DAv2）は 336×336 入力で，LLaVA-1.5 の CLIP-ViT-L/14 が出す576トークンに合わせる．パッチ16のモデル（RADIO，SAM）は 384×384 に直す（付録 A）．

### 予備実験の変種（最終手法ではない）

| 変種 | 内容 |
| --- | --- |
| (b) | 射影後の視覚特徴を第16層へ残差で再注入 |
| (c) | 視覚エンコーダの生の特徴（射影前）へ cosine 整合 |
| (d) | 射影前の特徴をアダプタ経由で残差注入 |
| (e) | 射影後の特徴へ cosine 整合 |
| 関係整合 | 自己類似度行列どうしの MSE（付録 C） |

## 実験設定

- **基盤**：LLaVA-1.5．その instruction tuning の手順に従う．
- **言語モデル**：Vicuna-1.5-7B（既定），Vicuna-1.5-13B，Qwen2.5-7B．**視覚エンコーダ**：CLIP（既定），SigLIPv2．
- **学習**：LoRA．データは LLaVA-665K のみ．visual instruction tuning は計 5.2K ステップ．A100（40GB）4枚．学習時間の増分は約3%．学習率とバッチの大きさは見当たらなかった（検索した語：learning rate，batch size，epoch）．
- **評価**：CV-Bench2D（ADE20K と COCO を別々に平均せず，全体の正解率），MMVP（pair accuracy，10回の平均），What's Up（COCO_one・COCO_two の平均），POPE（COCO，random・popular の平均），MME（知覚＋認知の和），MMStar．
- **反復**：乱数種の数や分散の記述は見当たらなかった（検索した語：seed，variance，standard deviation，std，error bar）．書かれているのは MMVP の「10回の平均」だけである．

## 結果

**表1：VIRAL の効果（✓ が L_VRA あり）**

| 言語モデル | 視覚エンコーダ | L_VRA | CV-Bench2D | MMVP | What's Up | POPE | MMStar | MME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vicuna-1.5-7B | CLIP | × | 56.82 | 28.20 | 40.13 | 85.70 | 33.93 | 1650.21 |
| Vicuna-1.5-7B | CLIP | ✓ | 59.67 | 33.33 | 48.55 | 88.32 | 33.93 | 1694.52 |
| Vicuna-1.5-7B | SigLIPv2 | × | 58.90 | 28.22 | 40.90 | 90.13 | 36.53 | 1738.96 |
| Vicuna-1.5-7B | SigLIPv2 | ✓ | 62.66 | 33.11 | 44.40 | 90.77 | 37.20 | 1835.62 |
| Qwen2.5-7B | CLIP | × | 58.97 | 33.47 | 59.08 | 85.88 | 39.20 | 1743.56 |
| Qwen2.5-7B | CLIP | ✓ | 60.50 | 36.07 | 63.57 | 84.92 | 39.67 | 1765.65 |
| Vicuna-1.5-13B | CLIP | × | 57.51 | 32.30 | 44.44 | 87.12 | 34.47 | 1599.04 |
| Vicuna-1.5-13B | CLIP | ✓ | 58.97 | 37.80 | 62.26 | 87.79 | 37.00 | 1636.62 |

MME 以外は %．**MMVP と What's Up で差が大きい**（What's Up は 13B で +17.82）．**MMStar は 7B・CLIP で ±0.00，POPE は Qwen2.5-7B で −0.96**．

表の値は v2 のものである．v1 の表1 は一部の値が違う（7B・CLIP・✓ の POPE が 87.43，13B の MMVP が 39.33／45.33）．v2 の図1(b) には，7B・CLIP・VIRAL の POPE 87.43，MME 1964.62 と，表1（88.32，1694.52）と合わない値が残っている．

**表2：教師と層の比較（LLaVA-1.5-7B）**

| VFM | 層 | CV-Bench2D | MMVP | What's Up | POPE | MME |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | – | 56.82 | 28.20 | 40.13 | 85.70 | 1650.21 |
| DINOv2 | 16 | 59.67 | 33.33 | 48.55 | 88.32 | 1694.52 |
| CLIP | 16 | 59.53 | 29.33 | 44.50 | 88.10 | 1548.49 |
| SAM | 16 | 57.58 | 30.27 | 49.84 | 88.34 | 1648.77 |
| DAv2 | 16 | 58.55 | 28.67 | 47.29 | 88.70 | 1682.42 |
| RADIO | 16 | 57.59 | 31.80 | 47.35 | 88.52 | 1692.94 |
| DINOv2 | 4 | 58.55 | 30.67 | 45.05 | 87.68 | 1720.36 |
| DINOv2 | 8 | 58.28 | 27.70 | 48.32 | 88.43 | 1662.67 |
| DINOv2 | 12 | 57.77 | 28.59 | 48.19 | 88.27 | 1648.88 |
| DINOv2 | 20 | 55.22 | 27.41 | 48.04 | 88.39 | 1705.97 |
| DINOv2 | 24 | 55.77 | 27.48 | 47.99 | 88.10 | 1740.55 |
| DINOv2 | 28 | 54.87 | 27.19 | 47.82 | 88.56 | 1755.86 |
| DINOv2 | 32 | 56.12 | 26.52 | 47.60 | 87.32 | 1678.69 |

- **層は4層おき（4，8，…，32）に掃引．DINOv2 の層の中で，第16層が CV-Bench2D・MMVP・What's Up で最良**．著者は "with the 16th layer of the 32-layer model consistently yielding stronger results across multiple benchmarks" と書く（§5.3）．
- **全指標で最良ではない**：MME は第28層（1755.86）・第24層・第4層・第20層が第16層を上回り，POPE も第28層（88.56）・第8層・第20層が上回る．
- 第20層以降では CV-Bench2D と MMVP が基準線を下回る．
- 教師について，著者は DINOv2 を "the most effective and versatile" として既定に採る（§5.3）．表では DINOv2 が CV-Bench2D・MMVP・MME で最良，What's Up は SAM（49.84），POPE は DAv2（88.70）が最良である．CLIP を教師にすると MME が基準線を下回る（1548.49）．

**表5：複数層・目的関数（DINOv2，付録 C）**

| 設定 | 目的 | CV-Bench2D | MMVP | What's Up | POPE | MME |
| --- | --- | --- | --- | --- | --- | --- |
| 第16層 | Cos. Sim. | 59.67 | 33.33 | 48.55 | 88.32 | 1694.52 |
| 第15–17層 | Cos. Sim. | 59.32 | 28.00 | 47.17 | 87.61 | 1639.72 |
| 第14–18層 | Cos. Sim. | 49.62 | 22.55 | 42.58 | 87.90 | 1444.32 |
| 第16層 | Relation | 58.83 | 26.60 | 49.05 | 87.58 | 1674.30 |

**掛ける層を隣接する層へ広げるほど，CV-Bench2D・MMVP・What's Up・MME が下がり**（POPE は 88.32 → 87.61 → 87.90），**第14–18層では CV-Bench2D・MMVP・MME が基準線を下回る**．著者の要約は "applying alignment solely at the 16th layer achieves the best performance"．なお複数層は**隣接層（15–17，14–18）**であり，4層おきの層の組み合わせではない．

**表4：予備実験（付録 B）**

| 設定 | POPE | CV-Bench2D | MMVP |
| --- | --- | --- | --- |
| Baseline | 85.70 | 56.82 | 28.20 |
| (b) 残差，射影後の特徴 | 87.17 | 57.51 | 26.67 |
| (c) 整合，生のエンコーダ特徴 | 88.10 | 59.53 | 29.33 |
| (d) 残差，射影前の特徴 | 85.47 | 53.62 | 19.33 |
| (e) 整合，射影後の特徴 | 86.99 | 57.23 | 28.53 |

**表6：再構成型の目的（ROSS）との比較（Vicuna-1.5-7B，CLIP，付録 D）**：CV-Bench2D は ROSS（Default，視覚表現の最終 hidden state に掛ける元の形）54.24，ROSS（Middle，第16層）56.05，VIRAL 59.67．MMVP は 29.73，31.40，33.33．

### 分析

- **注意のエントロピー**（§5.4，付録 A）：GQA を拡張した Zhang ら（2025b）の質問応答（答えは一語・一句）のうち Relation カテゴリで，生成テキストトークンにわたる平均の空間エントロピー（Batty 1974）を測り，層・ヘッドにわたって図4に示す．結果は "our model shows consistently lower entropy—particularly at the aligned intermediate layer"．**数値は図のみで本文に無い**．
- **トークン並べ替え（表3，§5.5）**：視覚エンコーダの出力トークンを無作為に並べ替えて射影器・LM へ入れ，CV-Bench2D の空間推論788問の正答数を比べる．

| 視覚エンコーダ | L_VRA | 元 | 並べ替え | 差 |
| --- | --- | --- | --- | --- |
| CLIP | × | 400 | 374 | −26（6.5%） |
| CLIP | ✓ | 414 | 360 | −54（13.0%） |
| SigLIPv2 | × | 374 | 353 | −21（5.6%） |
| SigLIPv2 | ✓ | 436 | 353 | −83（19.0%） |

  **VIRAL の方が並べ替えで大きく落ちる**．著者の解釈は "This confirms that our loss encourages MLLMs to capture and exploit fine-grained spatial relationships."
- **CKNNA**：第16層への残差再注入で CLIP 特徴との類似度が上がり，整合損失でさらに上がる（図2，数値は図のみ）．
- その他：テキスト→画像の注意マップ，PCA の可視化（第16層，全層），学習の早さ（CV-Bench2D と MMVP の平均が，全 5.2K ステップのうち 3K ステップで基準線を上回る，§5.7・図6）．**線形プローブは無い**（"prob" を全文検索して0件）．
- **因果的な介入は見当たらなかった**（検索した語：patching，intervention，knock，zero，causal）：推論時に層やトークンの内部状態を除去・ゼロ化・差し替える実験は書かれていない．推論時の操作は入力側のトークン並べ替えだけである．

## 著者が明記した限界

付録 G．

- **アーキテクチャの一般性**："this behavior may not hold universally as more diverse architectures continue to emerge"．
- **空間格子を壊す接続器**：Resampler や Q-Former は視覚トークンの格子を "disrupt this grid, making our approach less directly applicable"．
- トークンの枝刈りに頼る手法と組み合わせると，得られる利得が小さくなりうる，とも述べる．

## 著者が述べる含意

- "Our approach helps preserve fine-grained visual semantics often discarded under text-only supervision"（§6）．

## 射程の外

- **整合の目標は外部 VFM の画像パッチ特徴**であり，ラベルによる教師あり対照損失ではない．質問・推論手順の構造に由来するラベルは扱わない．
- **掛けるのは視覚トークンの位置**であり，質問を読み終えたテキスト側のトークンには掛けない．
- **射影 MLP を介して整合する**．中間層の生の hidden state そのものに損失を掛ける設定ではない．
- **λ は 0.5 の一点**で，変えた実験は無い．層は掃引するが，第16層の優位は CV-Bench2D・MMVP・What's Up に限られ，MME・POPE では他の層が上回る．
- 層の役割を確かめる**因果的な介入（層の除去・パッチング）は見当たらず**，表現が答えの生成に使われるかは並べ替えへの感度と注意エントロピーという間接的な指標でしか測らない．線形プローブも無い．
- VIRAL を学習して評価した対象は LLaVA-1.5 系（Vicuna-1.5 の 7B・13B，Qwen2.5-7B）で，LoRA による学習．LLaVA-NeXT は CKNNA の測定だけである（付録 E）．13B と Qwen2.5-7B の層数と，それらで掛けた層は書かれていない．**GQA は評価表に無い**（全文で GQA は1か所，注意エントロピーの分析に使う質問の出どころとして現れるだけである）．
- 乱数種・分散の記述は見当たらなかった．

---

**確認**：2026-11-13．arXiv abs ページ（版の履歴は v1 2025-09-09・v2 2025-10-10 の二つ，Comments は Project Page のみ，Journal-ref 無し），PDF v2（21頁，付録 A–H を含む全文．pdftotext でテキスト化し，表1–6 と図1 は頁画像で見た），ICLR 2026 virtual ページ，著者のプロジェクトページ（"arXiv 2025" とだけ書く）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．下書き（2026-10-11 作成）から直した主な点：掲載欄に，ICLR のページの要旨が arXiv v2 と異なることを足した．「17.3%／9.4%」の出どころを HTML 全文から PDF v2 の要旨へ直し，9.4% が §1 にもあることを足した．教師の「DINOv2 が総合で最良」を，著者の言い方と表の指標ごとの最良へ直した．表5 の「広げるほど悪化」から POPE を除いた．POPE で第16層を上回る層を列挙した．乱数種・学習率・因果的介入の「無い」を，検索した語を添えた「見当たらなかった」へ直した．v1 と v2 で表1 の値が違うこと，図1(b) と表1 が合わないこと，LLaVA-NeXT は測定だけであること，13B・Qwen2.5-7B で掛けた層が書かれていないことを足した．表の太字は原表と揃わないので外した．**読めていないもの**：OpenReview の forum（自動アクセスの検証で 403），ワークショップ版の本文，PDF v1 の全文（19頁．要旨と表1 だけを v2 と比べた），図の中の値（CKNNA，注意エントロピー，図6 の曲線）．
