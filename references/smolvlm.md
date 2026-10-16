# SmolVLM: Redefining small and efficient multimodal models

**著者**：Andrés Marafioti，Orr Zohar，Miquel Farré，Merve Noyan，Elie Bakouch，Pedro Cuenca，Cyril Zakka，Loubna Ben Allal，Anton Lozhkov，Nouamane Tazi，Vaibhav Srivastav，Joshua Lochner，Hugo Larcher，Mathieu Morlon，Lewis Tunstall，Leandro von Werra，Thomas Wolf
**掲載**：arXiv:2504.05299 v1，2025-04-07
**原文**：[arXiv](https://arxiv.org/abs/2504.05299)・使用したチェックポイントは [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

**小さい VLM は大きい VLM の設計を縮めるだけでは効率が悪い．画像トークンを強く圧縮し，データの組成を小ささに合わせて選び直すことで，少ないメモリで大きいモデルに迫る．**

## 位置づけ

**この研究が整合を掛けるモデルそのものの出典である．** 本研究の数値はすべて SmolVLM-500M-Instruct の1点で得たもので，「第16層」「$H = 960$」「全32層」はこのモデルの言語側の構成から来ている．何を借りているかは次のとおりである．

- **構成**：画像は視覚エンコーダで符号化し，pixel shuffle で圧縮してから言語モデルの入力列に並べる．したがって，画像と質問文をすべて読み終えた位置が入力末尾にでき，整合を掛ける位置の選び方（研究文書 §3.3）はこの構成に乗っている．
- **規模**：論文の3つの規模（256M・500M・2.2B）の中位である．研究文書が「計算資源の都合による」とした選択は，論文が掲げる「少ないメモリで動く」設計の側にある．

## 構成

500M は，256M と同じ視覚エンコーダに，より大きい言語モデルを組み合わせる．

> "the same 93M SigLIP-B/16 paired with the larger SmolLM2-360M"

符号化した画像の特徴は pixel shuffle で並べ替えたのち，MLP で言語モデルの入力空間へ写して視覚トークンにする．

> "Pixel shuffle rearranges spatial features into additional channels"

チェックポイントの `config.json` では，言語側が32層・隠れ次元 960（注意ヘッド 15，KV ヘッド 5，語彙 49,280，位置 8,192），視覚側が12層・隠れ次元 768・入力 512×512・パッチ 16，`scale_factor` が 4 である．512×512 の画像は 32×32 = 1,024 パッチになり，4倍の pixel shuffle で縦横それぞれ 1/4 にまとめると，1枚あたり 64 トークンになる（この 64 は構成値からの計算で，論文の記述ではない）．

文脈長は小さい2つの規模で 8k トークンである——"an 8k-token limit for smaller variants"．`config.json` の位置 8,192 と一致する．

## 設計の所見

論文は設計の選択を番号付きの所見（Finding）にまとめている．この研究に関わるものを挙げる．

| 番号 | 所見（原文） |
| --- | --- |
| 3 | "Small VLMs benefit from more aggressive visual token compression." |
| 4 | "For small models, image splitting enhances performance for vision tasks, whereas video frame averaging does not." |
| 5 | "Learned positional tokens outperform raw text tokens for compact VLMs." |
| 6 | "System prompts and media intro/outro tokens significantly improve compact VLM performance, particularly for video tasks." |
| 7 | "Adding text from SFT blend proved worse than new text SFT data." |
| 8 | "Excessive CoT data harms compact model performance." |

**画像分割（所見4）**：高解像度の画像は，縮小した全体像に加えて複数の部分画像に分けて符号化する——"high-resolution images are divided into multiple sub-images along with a downsized version of the original"．256M と 500M では画像の長辺を 1920 に揃える——"resizes the longest edge of images to 1920 in the 256M and 500M models"．

## 学習

学習は画像の段階と動画の段階の2段である．画像の段階のデータは "a new mixture of the datasets used in Laurençon et al. (2024)" に手書き数式の MathWriting を足したもので，複数画像の推論も含む（"including 22% dedicated to multi-image reasoning"）．GPU メモリはバッチ 64 で "15.0GB (256M), 16.0GB (500M), and 49.9GB (2.2B)" である．

最小の 256M は，推論時に 1GB 未満のメモリで，約300倍の規模の Idefics-80B を上回ると要旨は述べる．

## この研究との違い

- **画像分割を切っている．** 本研究の `SmolVLMProcessor` は `do_image_splitting=False` で読み込み，どの画像も 512×512 の1枚として同じ長さのトークン列にする（`cotar/models/smolvlm.py`）．画像1枚が足すトークン数を一度測れば，全サンプルのプロンプト長が求まり，表現を取り出す位置（`SmolVLM._pool`）が決まるからである．所見4のとおりなら，これは視覚タスクの性能を下げる側の選択であり，研究文書の正解率（53.5%）は論文の推奨設定での値ではない．群間の比較は同じ設定の上で行うので，この選択は比較を歪めない．
- **GQA は論文に現れない．** 本文に GQA の語は無く，評価にも使われていない．画像の段階のデータの土台である The Cauldron（Laurençon ら）にも GQA のサブセットは無い．ただし学習データの全体を列挙した表までは確かめていないので，事前学習で GQA を一度も見ていないとまでは言えない．

## 射程の外

- **内部表現を扱わない．** 設計とデータの選択を正解率と GPU メモリで評価する論文であり，中間層の表現・プローブ・整合はどれもこの論文の外にある．
- **タスクの手順を扱わない．** 質問がどの手順で解かれるかは問わない．

---

**確認**：2026-10-16．arXiv の abs ページ（書誌・要旨・版）と arXiv の HTML 版の本文，Hugging Face のチェックポイントの `config.json`，The Cauldron のデータセットカード（サブセット一覧）を参照した．所見の文と構成の数値は原文と照合した．**HTML 版は数式と一部の数値（RoPE の基数，ベンチマークの表）が変換で崩れていたので，それらはここに書いていない**——引くときは PDF 本文に当たること．
