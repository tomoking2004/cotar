# Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?

**著者**：Abhilasha Ravichander，Yonatan Belinkov，Eduard Hovy（Ravichander・Hovy：カーネギーメロン大学 Language Technologies Institute／Belinkov：テクニオン・イスラエル工科大学）
**掲載**：EACL 2021（Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume，pp. 3363–3377，Online，2021年4月）．arXiv:2005.00719 v3，2021-03-07（初版 v1 は 2020-05-02）
**原文**：[ACL Anthology](https://aclanthology.org/2021.eacl-main.295/)・[arXiv](https://arxiv.org/abs/2005.00719)

**学習タスクに要らない性質も表現から読める——プローブの精度は，モデルがその性質に頼っていることを示さない．**

## 位置づけ

表現から性質が読み取れることと，モデルがその性質を使うことは別である，という区別（研究文書 §2，§6）について，**学習タスクに要らない性質でも読める**ことを示した論文である．研究文書はこの区別の先行研究として Belinkov の総覧と Elazar らを引いており，本論文は引いていない．本論文のプローブは線形ではなく MLP であり，性質を除く介入（Elazar ら）とも，性質を損失で足す本研究とも操作が違う．

## 主張

- 自然言語推論（NLI）を題材に，"models can learn to encode linguistic properties even if they are not needed for the task on which the model was trained"（要旨）．
- "pretrained word embeddings play a considerable role in encoding these properties rather than the training task itself"（要旨）——プローブ精度のかなりの部分は，学習タスクではなく事前学習した単語埋め込みから来る．
- 合成タスクで，**ランダムノイズとして入れた性質**さえチャンスを大きく上回って読める（要旨）．
- 考察（§6）で，プローブは "should not be used to pinpoint the information a model is relying upon"．注意深い対照とベースラインが要る．

## 手法

### 対照データセット（性質をタスクに不要にする）

"Control datasets are based on the intuition that a linguistic feature is not informative for a model to discriminate between classes if the linguistic feature remains constant across classes."（§3）

MultiNLI を性質で分割し，**前提と仮説の両方で性質の値が一つに固定された対**だけを残す．値がどの例でも同じなので，その性質だけではタスクのラベルを識別できない．固定する値は，分割したとき学習件数が最大になる値である．

| データ（表1） | 固定した値 | # Train | # Test |
| --- | --- | --- | --- |
| MultiNLI | – | 392,702 | 20,000 |
| 時制（主節動詞） | 過去 | 69,652 | 1,678 |
| 主語の数 | 単数 | 102,452 | 2,584 |
| 目的語の数 | 単数 | 43,178 | 1,060 |

### 比べる二つのモデル

| 名前 | 学習データ |
| --- | --- |
| **PT**（task-controlled） | 上の対照データセット（性質を使えないはずのモデル） |
| **DS**（downsampled） | MultiNLI の学習データを，対照データセットと同じ件数に間引いたもの |

PT と DS のプローブ精度を比べて，表現に符号化された情報が，学習タスクに有用な情報にどこまで駆動されているかを問う（§4.1）．

### 符号化器とプローブ

- **文符号化器**：CBOW（単語埋め込みの和），BiLSTM（Last／Avg／Max の3通りで固定長にする）．前提と仮説のベクトルを，要素積・差とともに連結し，tanh 層と3値の softmax 分類器へ渡す
- **埋め込み**：300 次元 GloVe で初期化（表3・表4では無作為初期化とも比べる）．埋め込みはタスクの学習中に更新する
- **プローブ**：隠れユニット 200 の1層 MLP．NLI の学習を終えたモデルの，前提と仮説の文ベクトルに当てる．学習タスクとプローブ課題の設定は Conneau et al.（2018）に従う
- Transformer 系は扱わない（脚注6）

### 合成タスク

前提・仮説は `S = {(a|b)(a|b|c)*}` の文字列（最大長30）．**仮説が前提と同じ文字（a か b）で始まれば含意**．性質は「文字 c を含むか」で，c を先頭の文字より後ろの無作為な位置に挿入する．

| 設定 | c の入れ方 | c とタスクの関係（原文の説明） |
| --- | --- | --- |
| **Noise** | 無作為に選んだ 50% の前提・仮説に挿入 | 学習データ中のノイズ |
| Uncorrelated | a で始まる前提に挿入 | 判定とは無関係だが，データ中の他の性質と相関 |
| Partial | a で始まる前提と仮説に挿入 | 判定の部分的な説明 |
| Full | 仮説が含意されるとき，前提と仮説に挿入 | 判定の完全な代替説明 |

各タスクは学習 20,000・開発 5,000・テスト 5,000（表6）．BiLSTM の隠れ層の大きさを 10，50，100，200，300，600 と変える（図2a）．プローブは1層・2層の MLP で，次元は 10，50，100，200，1000（図2b）．

## 結果

### 性質が不要でも読める（表2．Dev-ST／SS／SO は性質を固定した MultiNLI 開発集合）

| モデル | 時制 Dev-ST | 時制 Probing | 主語数 Dev-SS | 主語数 Probing | 目的語数 Dev-SO | 目的語数 Probing |
| --- | --- | --- | --- | --- | --- | --- |
| Majority | 37.90 | 50.00 | 36.88 | 50.0 | 39.52 | 50.0 |
| CBOW-DS | 57.57 | 82.36 | 58.4 | 76.55 | 55.85 | 75.49 |
| CBOW-PT | 60.31 | 82.2 | 58.2 | 75.69 | 59.15 | 74.38 |
| BiLSTM-Av-DS | 63.53 | 82.93 | 64.24 | 79.53 | 66.23 | 76.11 |
| BiLSTM-Av-PT | 65.08 | 82.79 | 66.76 | 78.81 | 67.08 | 75.48 |
| BiLSTM-Max-DS | 63.35 | 81.14 | 65.91 | 78.56 | 65.94 | 74.79 |
| BiLSTM-Max-PT | 64.6 | 81.04 | 66.87 | 79.51 | 66.98 | 72.44 |
| BiLSTM-Last-DS | 61.08 | 80.43 | 64.2 | 81.52 | 62.26 | 72.65 |
| BiLSTM-Last-PT | 63.89 | 78.44 | 66.18 | 78.9 | 66.04 | 72.82 |

**性質を使えないはずの PT でも，プローブ精度は 72.44–82.79 で，DS（72.65–82.93）と同程度である．** 著者は "models consistently display similar probing accuracies whether the property was needed for the training task or not" と述べる（表2の説明）．

### 事前学習埋め込みの寄与（表3．Dev は MNLI 開発集合）

| モデル | 時制 Dev | 時制 Probing | 主語数 Dev | 主語数 Probing | 目的語数 Dev | 目的語数 Probing |
| --- | --- | --- | --- | --- | --- | --- |
| Majority | 36.50 | 50.0 | 36.50 | 50.0 | 36.50 | 50.0 |
| CBOW-Word | 62.21 | 83.74 | 62.1 | 76.91 | 61.93 | 75.4 |
| CBOW-Rand | 56.98 | 60.14 | 56.27 | 67.01 | 56.82 | 64.71 |
| BiLSTM-Avg-Word | 70.05 | 82.48 | 70.67 | 76.53 | 69.82 | 72.29 |
| BiLSTM-Avg-Rand | 63.33 | 61.4 | 64.0 | 67.68 | 63.71 | 63.87 |
| BiLSTM-Max-Word | 68.67 | 78.34 | 69.19 | 73.96 | 69.12 | 68.53 |
| BiLSTM-Max-Rand | 62.78 | 62.89 | 63.29 | 69.51 | 63.28 | 62.84 |
| BiLSTM-Last-Word | 68.32 | 74.61 | 69.04 | 71.82 | 68.82 | 69.27 |
| BiLSTM-Last-Rand | 62.14 | 62.96 | 61.88 | 67.45 | 62.29 | 61.32 |

Word＝事前学習した単語埋め込みで初期化，Rand＝無作為初期化．**埋め込みを無作為に初期化すると，プローブ精度は時制で 74.61–83.74 から 60.14–62.96 へ下がる．**

### 埋め込みをゼロから学習しても PT と DS は同程度（表4）

| モデル | 時制 Dev-ST | 時制 Probing | 主語数 Dev-SS | 主語数 Probing | 目的語数 Dev-SO | 目的語数 Probing |
| --- | --- | --- | --- | --- | --- | --- |
| Majority | 37.90 | 50.0 | 36.88 | 50.0 | 39.52 | 50.0 |
| CBOW-Rand-DS | 49.88 | 61.33 | 51.04 | 67.32 | 49.25 | 63.63 |
| CBOW-Rand-PT | 53.28 | 61.37 | 50.97 | 67.02 | 52.45 | 63.84 |
| BiLSTM-Avg-Rand-DS | 57.21 | 63.75 | 60.76 | 68.5 | 59.53 | 63.89 |
| BiLSTM-Avg-Rand-PT | 60.91 | 63.07 | 61.18 | 69.12 | 60.57 | 63.77 |
| BiLSTM-Max-Rand-DS | 59.18 | 61.05 | 61.8 | 70.32 | 60.57 | 64.68 |
| BiLSTM-Max-Rand-PT | 60.55 | 61.53 | 63.78 | 70.6 | 63.49 | 64.26 |
| BiLSTM-Last-Rand-DS | 56.73 | 63.88 | 58.82 | 69.09 | 56.79 | 63.86 |
| BiLSTM-Last-Rand-PT | 57.39 | 62.88 | 61.88 | 68.8 | 60.75 | 61.96 |

無作為に初期化した埋め込みでも，性質は PT・DS で同程度に，多数派のベースライン（50.0）を上回って符号化されている（§4.3）．

### 合成タスク

- **4設定すべて**で c の有無をプローブで "considerably above chance-level in all four cases" 予測できた——**ランダムノイズとして入れた Noise 設定を含む**（§5.2）．
- 主タスクの精度は表現の次元によらず一貫しているが，プローブ精度は容量の小さいモデルで下がる（§5.2，図2a）．
- 合成タスクのプローブ精度は図2にのみ示され，本文に数値はない（本ノートでは載せない）．
- 付録 A は敵対的学習で性質を抑える試みである（BiLSTM，隠れ層 200，最後の隠れ状態）．表8の λ=0.0〜5.0 の7行で，主タスクの開発精度は 97.98–100.0，外部の攻撃者（attacker）の分類器の精度は 62.26–100.0 である（Noise 設定では 90.3–100.0．多数派は 50.2）．著者は Noise について "we do not find any setting of adversary weight λ that suppresses the attribute" と述べる．

## 著者が明記した限界

- **Transformer 系は扱わない**："We leave an exploration of recent transformer-based architectures to future work"（脚注6）．
- **敵対的除去は情報を消しきらない**："Adversarial learning does not remove all information from the representation"（付録 A．Elazar and Goldberg（2018）を引く）．
- **プローブ精度についての主張は，使ったプローブに相対的である**（脚注12）．

## 著者が述べる含意

- "properties may be incidentally encoded even when not required for a task"（§2）．合成タスクの結果から，プローブの精度に頼って，モデルがタスクを解くのに何の情報に頼るかを測るのは誤りである，と述べる（§5.2）．
- "We emphasize the need for probing work to establish careful controls and baselines when reporting experimental results."（§6）
- モデルの表現に介入して予測への影響を調べる因果的な方法を，今後の有望な方向として挙げる（§6）．

## 射程の外

- **対象は英語のテキスト NLI**（MultiNLI）と CBOW・BiLSTM の文符号化器である．Transformer 系は今後の課題とされ（脚注6），画像を扱うモデルやデータは本文・付録に見当たらなかった（検索した語：GQA，VQA，vision，visual，image，multimodal．visual は参考文献の題に1件のみ）．
- **プローブで読む性質は時制・主語数・目的語数**（合成タスクでは文字 c の有無）で，解き方の手順を表すラベルではない．
- **性質を損失で足す介入は行わない．** 行うのは，性質をタスクに不要にする（データの側で一定にする）操作と，付録 A の敵対的な抑制である．読めた性質が出力に使われているかを，表現への介入で測ることは今後の方向として挙げるにとどまる（§6）．
- プローブを当てるのは前提と仮説の文ベクトルである．プローブは MLP で，線形プローブは用いていない（全文で linear の語は見当たらなかった）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴 v1 2020-05-02／v2 2021-02-22／v3 2021-03-07・Comments「EACL 2021」），arXiv PDF v3（15頁，付録 A まで．pdftotext で全文をテキスト化し，表3・表4・表6・表8はページ画像でも確かめた），ACL Anthology の書誌ページ（会議名・頁・年月・Online）と同 PDF（15頁）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．v2 と v3 の差は引用1件の修正と謝辞だけで，ACL 版と v3 の差は頁の柱と未解決の引用記号だけである（数値・引用した文は同一）．v1 は表2・表4の数値が v3 と同じことだけを見た．下書き（2026-10-11 作成）から直した主な点：プローブの逐語引用を言い換えに改めた（原文は multi- と layered のあいだで改行しており，下書きの綴り "multilayered" を確かめられない）．「符号化の多くは」を「かなりの部分は」（considerable）へ，「VLM・GQA は扱わない」を検索した語を添えた「見当たらなかった」へ，「内部の感度で測ることもしない」を原文が今後の方向として書く形へ直した．表の太字（原文に無い強調）を外し，表の列名を原文に揃え，PT のプローブ精度の範囲を小数まで書いた．「位置づけ」を，研究文書が本論文を引いていないことと，プローブが MLP であることに合わせて書き直した．PT と DS の比較の読み方を，原文 §4.1 の問いの形に直した．**読めていないもの**：図2・図4の中の値（本文に数値が無く，載せていない），v1 の本文（表2・表4以外），著者のコードとデータ．
