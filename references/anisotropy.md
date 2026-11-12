# How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings

**著者**：Kawin Ethayarajh（Stanford University．脚注に "Work partly done at the University of Toronto."）
**掲載**：Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)，pp. 55–65，Hong Kong, China，2019-11．arXiv:1909.00512 v1，2019-09-02
**原文**：[arXiv](https://arxiv.org/abs/1909.00512)・[ACL Anthology](https://aclanthology.org/D19-1006/)

**ELMo・BERT・GPT-2 の文脈化された語の表現は，ELMo の入力層を除くすべての層で異方的であり（無作為に引いた語どうしの cos の平均が 0 にならない），著者はその平均を基準として差し引いたうえで，文脈化の度合いを測る．**

## 主張

ELMo・BERT・GPT-2 の各層の表現の幾何を調べる．本文書に関わる主張は次の二つである．

1. **表現は異方的である**——"Contextualized representations are anisotropic in all non-input layers."（§4.1 の段落見出し）．表現が等方的なら，一様に無作為に引いた語どうしの cos の平均は 0 になり，この平均が 1 に近いほど異方的である．著者は異方性の幾何学的な解釈を，語の表現がみな "occupy a narrow cone in the vector space" ことだと述べる．
2. **上の層ほど文脈に固有になる**——"upper layers of ELMo, BERT, and GPT-2 produce more context-specific representations than lower layers"（§6）．そして異方性で補正すると，静的な埋め込みで説明できる分は小さい："On average, less than 5% of the variance in a word’s contextualized representations can be explained by a static embedding."（§4.3 の段落見出し）

著者は，3モデルとも文脈化された隠れ層の表現のほぼすべてが入力層の表現より異方的であることから，"This suggests that high anisotropy is inherent to, or least a by-product of, the process of contextualization." と述べる（"or least" は原文どおり．§4.1）．

## 手法

### 三つの指標（§3.3）

| 指標 | 定義 | 読み方 |
| --- | --- | --- |
| **self-similarity** | ある語 $w$ が現れる $n$ 個の異なる文脈について，層 $\ell$ での $w$ の表現どうしの cos を全対で平均したもの（Definition 1） | 層がまったく文脈化しないなら 1．文脈化されているほど低くなると見込む |
| **intra-sentence similarity** | 文の中の各語の表現と，文ベクトル（その文の語の表現の平均）との cos の平均（Definition 2） | これが高く，self-similarity が低ければ，文の中の語の表現を互いに寄せるだけの文脈化を示唆する |
| **maximum explainable variance（MEV）** | 語 $w$ の各文脈での表現を並べた行列（occurrence matrix）の特異値による $\sigma_1^2 / \sum_i \sigma_i^2$．$w$ の表現の分散のうち第1主成分で説明できる割合（Definition 3） | 静的な埋め込みが文脈化表現をどこまで置き換えられるかの上限．1 なら完全に置き換えられ，0 に近いほど置き換えにならない |

### 異方性による補正（§3.4）

**著者は，文脈化の度合いを論じるときに等方性（またはその欠如）を考慮することが重要だと述べる．** 著者の例では，self-similarity 0.95 は完全に等方的な空間なら文脈化が弱いことを示唆するが，"any two words have on average a cosine similarity of 0.99." となるほど異方的な空間では，逆に，よく文脈化されていることを示唆する——$w$ の文脈ごとの表現どうしのほうが，無作為に選んだ2語よりも平均して似ていないことになるからである．

そこで層ごとに基準（baseline）を作る．

- self-similarity と intra-sentence similarity の基準：**異なる文脈から一様に無作為に引いた語の表現どうしの cos の平均**．
- MEV の基準：一様に無作為に引いた語の表現の分散のうち，第1主成分が説明する割合．

どちらの基準も，その層の表現が異方的であるほど 1 に近づく．"We then subtract from each measure its respective baseline to get the anisotropy-adjusted contexuality measure."（"contexuality" は原文どおり）．断りのない限り，以降の文脈化の指標は補正後の値を指し，生の値と基準はともに，一様に無作為に引いた 1K 個の語の表現から推定する．

## 実験設定

- **モデル**：ELMo（隠れ層2），BERT（base cased，隠れ層12），GPT-2（隠れ層12）．各モデルの入力層を第0層として含める——第0層は文脈化されていないので，後の層の文脈化を比べる基準になる，というのが著者の理由である．"The models we work with are all pre-trained on their respective language modelling tasks."（§3.1）．微調整したモデルを調べたという記述は見当たらなかった（検索した語：fine-tun，finetun．該当は関連研究で3モデルが下流タスクに向けて微調整されると述べる1文のみ）．
- **データ**：SemEval Semantic Textual Similarity タスク（2012〜2016年）の文．各語を，それが現れる文と文中の位置に対応づける．異なる文脈が5未満の語は扱わない（§3.2）．
- intra-sentence similarity は，一様に無作為に引いた 500 文の平均を図示する（§4.2）．

## 結果

### 異方性（§4.1）

- **段落見出しは「入力層以外のすべての層で異方的」であり，本文は「ほぼすべての層」と書く．** 本文が挙げる唯一の例外は ELMo の入力層で，文脈の情報も位置の情報も使わない文字レベルの静的な埋め込みを作る層である．
- **異方性は概して上の層ほど強い．** GPT-2 では，一様に無作為に引いた語どうしの cos の平均は第2〜8層でおよそ 0.6，第8〜12層で指数的に増え，最終層では "any two words have on average an almost perfect cosine similarity!"
- ただし例外がある——"the anisotropy in BERT’s penultimate layer is much higher than in its final layer."

### 文脈固有性（§4.2）

- **self-similarity は，3モデルとも上の層ほど平均して低い．** 3モデルのうち GPT-2 の表現が最も文脈に固有である．
- ELMo の層を通じた self-similarity の平均が最も低い語は ‘and’・‘of’・‘’s’・‘the’・‘to’ で，いずれもストップワードである．これらは多義的でないことから，著者は，表現のばらつきを生むのは語に内在する多義性ではなく，語が現れる文脈の多様さであることが示唆されると述べる．
- intra-sentence similarity は，ELMo では上の層ほど高い．BERT では上の層ほど語どうしが似なくなるが（例外として第12層を挙げる），全層で基準（無作為に選んだ語どうしの平均）を上回る．GPT-2 では "the anisotropy-adjusted intra-sentence similarity is close to 0 in most layers of GPT-2."

### 静的埋め込みとの比較（§4.3）

- **補正後，静的な埋め込みで説明できる分散は，ELMo・BERT・GPT-2 のどの層でも平均 5% を超えない．**
- **生の MEV が無視できない大きさになるのは GPT-2 だけである**——第2〜11層で平均およそ 30% で，著者はこれを極端に強い異方性によるものとする．ELMo と BERT では，生の MEV も全層で平均 5% 未満である．
- 文脈化表現の第1主成分を静的埋め込みとして取り出すと，3モデルとも，下の層から作ったもののほうが上の層から作ったものより有効である．Table 1 の SimLex999 では BERT 第1層が 0.315 で，GloVe（0.194）・FastText（0.239）を上回る（同じ列の最高値は BERT 第2層の 0.320）．

## 著者が明記した限界

限界を述べる節は無い（"limitation" の語は，静的な埋め込みの限界を述べる関連研究の1か所にしか現れない）．今後の課題（§5）として，"adding an anisotropy penalty to the language modelling objective" ことで，さらに良い結果が得られるかもしれないと述べる——**異方性を抑える罰則は，今後の課題として挙げられているだけで，この論文の実験には含まれない**．

## 位置づけ

**いまの研究文書（`.claude/context.md`）は，この論文を引いておらず，「異方」の語も使っていない．** 以下は，引くとすれば何が対応するかの整理である．

対応が成り立つのは，**cos の値を，その絶対値ではなく，同じ空間の別の対の水準と比べて読む**という形までである．研究文書 §4.4 は，baseline（整合前）の同署名の類似度 0.9448 と異署名の類似度 0.9183 を，差を標準偏差で割った分離度（0.816）で読む．

対応が成り立たない点は三つある．

- 原典の基準は**一様に無作為に引いた語の出現どうし**の cos の平均であり，本研究の異署名の類似度は**署名が一致しない対に限った**平均である．同じ量ではない．
- 原典は基準を**差し引く**．本研究の分離度は差を**標準偏差で割る**．研究文書 §4.4 が割る理由として書くのは「類似度そのものの大きさに左右されずに群を比べるため」であり，異方性への対処だとは書いていない．
- 本研究の表現が異方的であるかどうかは，研究文書では測っておらず，述べてもいない．異署名の類似度が高いことを異方性の現れと読むのは，研究文書に無い解釈である．

## 射程の外

- **言語モデル単体**（ELMo・BERT・GPT-2）であり，VLM は調べていない（vision・visual・multimodal を全文で検索し，該当は参考文献の題1件のみ．image は画像分類モデルの層についての類比の1文のみ）．
- 調べたのは**事前学習済みの3モデル**で，損失による表現への介入は行っていない（異方性の罰則は今後の課題として挙げるだけ）．
- 単位は**語の出現ごとの表現**と，その平均である文ベクトルである．質問全体を読み終えた位置の表現や，タスクのラベルで分けた対の類似度を扱う記述は見当たらなかった．
- **異方性や文脈固有性が，文脈化モデルそのものの下流タスクの成績にどう効くかは測っていない．** 成績を測ったのは，文脈化表現の第1主成分から作った静的埋め込みを語の埋め込みのベンチマークに掛けた Table 1 だけである．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・投稿日．版は v1 のみ，Comments は "Accepted to EMNLP 2019"），arXiv PDF v1（11頁．付録は無い），ACL Anthology の書誌ページ（会議録名・頁・年月・開催地）と ACL Anthology 版 PDF（11頁）を取得した．v1 の全文を pdftotext で読み，数式（式1〜4）と Table 1 は v1 のページ画像（3・4・7・8頁）で確かめ，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．ACL Anthology 版は，テキスト化した語の集合が v1 と一致すること（違いは arXiv の版表示と会議録のフッタ・頁番号のみ）を確かめ，通読はしていない．下書き（2026-10-11 作成）から直した主な点：冒頭の一文と §3.4 の「差し引いてから読まなければならない」を，原文の強さ（考慮することが重要，著者は差し引いて測る）に直した．「§4.1／§4.3 の見出し」を段落見出しに直した．「入力層以外のすべての層で異方的．例外は ELMo の入力層」を，見出しと本文の書き分けに直した．「急増」を原文の「指数的に増える」に，「機能語」を原文の「ストップワード」に直した．「微調整したモデルは扱わない」を「記述は見当たらなかった」に弱めた．射程の外の「下流の正解率にどう効くかは測らない」を，Table 1 が静的埋め込みのベンチマーク成績を測っている事実に合わせて限定した．位置づけの「本研究がこの論文を引く」は，いまの研究文書が引いていないので書き改め，本研究の数値を研究文書 §4.4 の値にした．開催地を Hong Kong, China にした．引用中のアポストロフィを原文の ’ に揃えた．**読めていないもの**：各層の値は図（Figure 1〜4）にしか無く，本文が文で述べる数値以外は読み取っていない．ACL Anthology 版 PDF のページ画像は見ていない．
