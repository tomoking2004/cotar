# SmolVLM: Redefining small and efficient multimodal models

**著者**：Andrés Marafioti，Orr Zohar，Miquel Farré，Merve Noyan，Elie Bakouch，Pedro Cuenca，Cyril Zakka，Loubna Ben Allal，Anton Lozhkov，Nouamane Tazi，Vaibhav Srivastav，Joshua Lochner，Hugo Larcher，Mathieu Morlon，Lewis Tunstall，Leandro von Werra，Thomas Wolf
**掲載**：arXiv:2504.05299 v1，2025-04-07
**原文**：[arXiv](https://arxiv.org/abs/2504.05299)・使用したチェックポイントは [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

**小さい VLM は大きい VLM の設計を縮めるだけでは効率が悪い．画像トークンを強く圧縮し，データの組成を小ささに合わせて選び直すことで，少ないメモリで大きいモデルに迫る．**

## 位置づけ

**この研究が整合を掛けるモデルそのものの出典である．** 本研究の数値はすべて SmolVLM-500M-Instruct の1点で得たもので，「第16層」「$H = 960$」「全32層」はこのモデルの言語側の構成から来ている．何を借りているかは次のとおりである．

- **構成**：画像は視覚エンコーダで符号化し，pixel shuffle で圧縮してから言語モデルの入力列に並べる．したがって，画像と質問文をすべて読み終えた位置が入力末尾にでき，整合を掛ける位置の選び方（研究文書 §3.3）はこの構成に乗っている．
- **規模**：論文の3つの規模（256M・500M・2.2B）の中位である．研究文書が「計算資源の都合による」とした選択は，少ないメモリで動かすという論文の設計の側にある．

## 構成

500M は，256M と同じ視覚エンコーダに，より大きい言語モデルを組み合わせる．

> "the same 93M SigLIP-B/16 paired with the larger SmolLM2-360M"

符号化した画像の特徴は pixel shuffle で並べ替えたのち，MLP で言語モデルの入力空間へ写して視覚トークンにする（図2の説明文．図の中の箱は Linear Layer と書く）．視覚トークンは文のトークンとつなげて，言語モデルが一緒に処理する（§2.2）．

> "Pixel shuffle rearranges spatial features into additional channels"

チェックポイントの `config.json` では，言語側が32層・隠れ次元 960（注意ヘッド 15，KV ヘッド 5，語彙 49,280，位置 8,192），視覚側が12層・隠れ次元 768・入力 512×512・パッチ 16，`scale_factor` が 4 である．**層数と次元はチェックポイントの値であり，論文の本文には書かれていない．** 言語側の `model_type` は `llama` である．全体のパラメタ数は 507,482,304 である（Hugging Face のメタ情報）．

512×512 の画像は 32×32 = 1,024 パッチになり（論文も 1024 トークンと書く），比 $r = 4$ の pixel shuffle でトークン数が $r^2$ 分の1になるので，1枚あたり 64 トークンになる．論文の本文は 64 という値そのものを書かないが，モデルカードの “64 visual tokens” と `processor_config.json` の `image_seq_len` がこの値である．

文脈長は小さい2つの規模で 8k トークンである——"an 8k-token limit for smaller variants"．`config.json` の位置 8,192 と一致する．**RoPE の基数は一致しない**——論文は "increasing the RoPE base from 10k to 273k" と書くが，このチェックポイントの `rope_theta` は 100,000 である．理由は確かめていない．

## 設計の所見

論文は設計の選択を番号付きの所見（Finding）にまとめている．この研究に関わるものを挙げる．

| 番号 | 所見（原文） |
| --- | --- |
| 3 | "Small VLMs benefit from more aggressive visual token compression." |
| 4 | "For small models, image splitting enhances performance for vision tasks, whereas video frame averaging does not." |
| 5 | "Learned positional tokens outperform raw text tokens for compact VLMs." |
| 6 | "System prompts and media intro/outro tokens significantly improve compact VLM performance, particularly for video tasks. During SFT, only train on completions." |
| 7 | "Adding text from SFT blend proved worse than new text SFT data." |
| 8 | "Excessive CoT data harms compact model performance." |

**画像分割（所見4）**：高解像度の画像は，縮小した全体像に加えて複数の部分画像に分けて符号化する——"high-resolution images are divided into multiple sub-images along with a downsized version of the original"．評価の設定（§4.2）では，256M と 500M で画像の長辺を 1920 に揃える——"resizes the longest edge of images to 1920 in the 256M and 500M models"．これは評価の節の記述で，チェックポイントの既定（`preprocessor_config.json` の `longest_edge`）は 2048 である．

## 学習

学習は画像の段階と動画の段階の2段である．画像の段階のデータは "a new mixture of the datasets used in Laurençon et al. (2024)" に手書き数式の MathWriting を足したもので，複数画像の推論も含む（"including 2% dedicated to multi-image reasoning"）．学習率・バッチ・エポック数・視覚エンコーダを凍結したかどうかは，本文に書かれていない．GPU メモリはバッチ 64 で "15.0GB (256M), 16.0GB (500M), and 49.9GB (2.2B)" である．

最小の 256M は，推論時に 1GB 未満のメモリで，約300倍の規模の Idefics-80B を上回ると要旨は述べる．

**論文の記述がすべて，使ったチェックポイントに当てはまるとは限らない．** `SmolVLM-500M-Instruct` の作成は 2025-01-20 で（Hugging Face のメタ情報），論文（2025-04-07）より前である．モデルカードは論文を引用するが，学習データに挙げるのは The Cauldron と Docmatix で，公開時のブログはこれに MathWriting を足す．動画のデータは挙げていない．一方，論文の図8右の組成（動画 33% など）と表1の 500M の動画の値（Video-MME 42.2，MLVU 47.3）は，`SmolVLM2-500M-Video-Instruct` のモデルカードの値と一致する．したがって動画の段階は SmolVLM2 の記述と読めるが，論文は両者を区別しておらず，使ったチェックポイントが動画の段階を経ていないとは確かめていない．

## この研究との違い

- **画像分割を切っている．** 本研究の `SmolVLMProcessor` は `do_image_splitting=False` で読み込み，どの画像も 512×512 の1枚として同じ長さのトークン列にする（`cotar/models/smolvlm.py`）．このとき画像は縦横比を保たず正方形に写される（transformers v4.50.0 の `image_processing_idefics3.py` で確認．実行機の版は確かめていない）．画像1枚が足すトークン数を一度測れば，全サンプルのプロンプト長が求まり，表現を取り出す位置（`SmolVLM._pool`）が決まるからである．所見4のとおりなら，これは視覚タスクの性能を下げる側の選択であり，研究文書の正解率（53.5%）は論文の推奨設定での値ではない．群間の比較は同じ設定の上で行うので条件は揃っているが，画像分割を入れたときに同じ結果になるかは測っていない．
- **GQA は論文に現れない．** 本文にも参考文献にも GQA の語は無く，評価にも使われていない．モデルカードが挙げる The Cauldron のサブセット一覧にも `gqa` は無い．ただし各サブセットの中身と画像の重なりは確かめていないので，学習で GQA の画像や似た問を一度も見ていないとまでは言えない．

## 射程の外

- **内部表現を扱わない．** 設計とデータの選択をベンチマークの得点と GPU メモリで評価する論文であり，中間層の表現・プローブ・整合はどれもこの論文の外にある．
- **タスクの手順を扱わない．** 質問がどの手順で解かれるかは問わない．

## ポスターとの対応

ポスターの文ごとに，依る原文を示す．論文に無く，チェックポイントやコードに依る部分はそう書いた．

**「モデル：SmolVLM-500M-Instruct[4]」**

- §4 の文に戻せる——"SmolVLM-500M: A mid-range model with the same 93M SigLIP-B/16 paired with the larger SmolLM2-360M."
- 論文の語は SmolVLM-500M で，`-Instruct` の付く名はチェックポイントのものである．[4] で結ぶ根拠は，モデルカードがこの論文を引用していることである．
- 判定：言い過ぎは無い．ただし論文の学習の記述をこのチェックポイントのものとして語らない（→ [学習](#学習)）．

**「𝒉：第16層（全32層の中間）の，入力の末尾の位置のベクトル（画像と質問文のすべてを参照できる位置）」**

- 「全32層」は `config.json` の `num_hidden_layers` であり，論文には層数が無い．第16層は `hidden_states[16]`，つまり第16ブロックの出力である（`[0]` は埋め込み．`cotar/models/smolvlm.py`）．
- 「入力の末尾」は，プロンプトの最後のトークンである（`SmolVLM._pool`）．チェックポイントの `chat_template` では，プロンプトは画像・質問文・`<end_of_utterance>`・改行・`Assistant:` の順に並ぶので，末尾は `Assistant:` の最後である．
- 「参照できる」のうち，論文に戻せるのは "visual tokens from the vision encoder are concatenated with textual tokens and jointly processed by a language model"（§2.2）までである．各トークンが自分より前を見ることは論文に書かれておらず，言語側が `llama` 型であることと研究文書 §3.3 の前提に依る．
- 判定：言い過ぎは無い．ただし「この位置だけ」とは言わない．質問文のあとの `<end_of_utterance>` などの位置からも，画像と質問文は同じく参照できる．

**「図：画像 → 視覚エンコーダ → 言語モデルの第1層…第32層 → 出力層 → 答え．質問文は言語モデルへ入る」**

- 図2の説明文に戻せる——"These features are first rearranged via a pixel-shuffle operation, then mapped into the LLM input space as visual tokens using an MLP projection." と "This combined sequence is passed to the LLM for text output."
- ポスターの図は pixel shuffle と射影を省いている．図2の画像分割は，本研究では切っているので描かない．
- 「出力層」は論文に無い語である（`config.json` の `tie_word_embeddings` は `false`）．
- 判定：省略であって誤りではない．聞かれたら，視覚エンコーダと言語モデルの間に pixel shuffle と射影があると補う．

## 学会で聞かれたら

答えは口で言う形で書いた．括弧の中は出所で，言わなくてよい．

**Q1．SmolVLM はどんな構成ですか．**
93M の視覚エンコーダ SigLIP-B/16 と，言語モデル SmolLM2-360M をつないだものです．画像の特徴を pixel shuffle で並べ替えて言語モデルの入力へ写し，質問文のトークンとつなげて処理します．（§4，図2）

**Q2．500M の言語部は何層，何次元ですか．**
32層で，隠れ次元は 960 です．これは原典の本文には書かれていなくて，チェックポイントの `config.json` の値です．（`num_hidden_layers`・`hidden_size`）

**Q3．画像はどう扱われますか．**
論文の設計では，大きい画像を部分画像に分け，縮小した全体像も一緒に符号化します．本研究は分割を切って，どの画像も 512×512 の1枚にしています．全サンプルのプロンプト長を求められるようにするためです．（§2.3．`cotar/models/smolvlm.py`）

**Q4．視覚トークンは何個ですか．**
512×512 の1枚で 64 個です．1,024 トークンが，比 4 の pixel shuffle で 16 分の1になります．64 という値そのものは，論文ではなくモデルカードと `processor_config.json` にあります．（§2.2．モデルカード："64 visual tokens"）

**Q5．そこまで圧縮して，情報は落ちませんか．**
論文も，比を上げると OCR のように細かい位置を要する課題が損なわれると書いています．それでも小さい VLM では強い圧縮のほうがよい，というのが所見3です．GQA への影響は原典には書かれていません．（§2.2：Finding 3）

**Q6．どんな学習をしたモデルですか．**
論文は画像の段階と動画の段階の2段と書いています．使ったチェックポイントのカードが挙げるデータは The Cauldron と Docmatix で，動画のデータは挙げていません．学習率などの値は原典には書かれていません．（§4.1．モデルカード）

**Q7．論文のモデルと，使ったチェックポイントは同じものですか．**
カードはこの論文を引用していますが，公開は 2025 年 1 月で論文より前です．論文の動画の値は SmolVLM2 のカードと一致するので，論文の記述がすべてこのチェックポイントに当てはまるかは確かめていません．（Hugging Face のメタ情報．両モデルカード．表1）

**Q8．GQA はモデルの学習に入っていませんか．**
論文には GQA の語が無く，The Cauldron のサブセット一覧にも gqa はありません．ただし各サブセットの中身や画像の重なりは確かめていないので，一度も見ていないとまでは言えません．（論文全文．The Cauldron のデータセットカード）

**Q9．SmolVLM の GQA の正解率は，論文ではいくつですか．**
原典には書かれていません．論文の評価に GQA はありません．（表1）

**Q10．なぜ小さいモデルなのですか．**
計算資源の都合です．結果はこのモデルの一点についてのもので，別の規模では測っていません．（研究文書 §4.1，§0）

**Q11．第16層とは何ですか．なぜそこなのですか．**
言語モデルの全32層の中間の層で，9回の学習すべてでここに固定しました．第8層と第24層は乱数種1つの試走だけで，読み取り精度は3点とも 95% 台でした．第16層が最も有利だとは言えません．（研究文書 §3.3，§7，§A.5）

**Q12．入力の末尾の位置には，どんな意味がありますか．**
質問を読み終えて答えを書き出す直前の位置で，画像と質問文のすべてを参照できます．答えの先頭トークンもこの位置から出たスコアで選ばれるので，整合が載せた構造が答えに届きうる位置です．（研究文書 §3.3）

**Q13．前の位置しか見ないことは，論文に書いてありますか．**
原典には書かれていません．論文が書くのは，視覚トークンと文のトークンをつなげて言語モデルで処理するところまでです．causal attention は，研究文書の前提です．（§2.2．研究文書 §3.3）

**Q14．画像分割を切ると，性能が落ちませんか．**
所見4に従えば，落ちる側の選択です．3群とも同じ設定で比べていますが，分割を入れたときに同じ結果になるかは測っていません．（§2.3：Finding 4）

---

**確認**：2026-11-06．arXiv v1 の PDF 全文（20頁．付録は無い）をテキスト化して読み，この要約の数値・引用・節番号を一つずつ照合した．§4.1 の 2% は PDF のページ画像でも見た．ほかに arXiv の abs ページ（版は v1 のみ），`SmolVLM-500M-Instruct` のモデルカード・`config.json`・`preprocessor_config.json`・`processor_config.json`・`chat_template.json`・Hugging Face のメタ情報，`SmolVLM2-500M-Video-Instruct` のモデルカード，公開時のブログ，The Cauldron のデータセットカードを読んだ．2026-10-16 の確認は HTML 版に依ったので，この確認で置き換える．**読めていないもの**：図1・3・5–7 の細かな値，SmolLM2 と Laurençon らの論文の本文，The Cauldron の各サブセットの中身．
