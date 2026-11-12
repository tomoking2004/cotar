# Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations

**著者**：Atticus Geiger，Zhengxuan Wu，Christopher Potts，Thomas Icard，Noah D. Goodman（Pr(Ai)²R Group／スタンフォード大学．Geiger と Wu は equal contribution）
**掲載**：CLeaR 2024（Proceedings of the Third Conference on Causal Learning and Reasoning，PMLR 236，pp. 160–187）．arXiv:2303.02536 v4，2024-02-21
**原文**：[PMLR](https://proceedings.mlr.press/v236/geiger24a.html)・[arXiv](https://arxiv.org/abs/2303.02536)

**高水準の因果変数は，互いに素なニューロン集合ではなく，学習した回転のもとでの部分空間に対応しうる——そこへの入れ替え介入で，標準基底の対応づけでは得られない高さの IIA が得られる．**

## 位置づけ

モデルの内部表現が高水準の変数を担っているかを，**入れ替え介入の一致率（IIA）で測る**枠組みである．低水準・高水準のモデルは凍結し，対応づけ（回転）だけを学習する．目的関数は interchange intervention training（IIT）のものを借りる——"In this paper, we use IIT objectives to learn an alignment between a high-level causal model and a deep learning model."（§2）．

本研究の研究文書（`.claude/context.md`）は「読み取れること」と「使われること」を分ける（§6）が，本論文を引いていない．本研究が「使われること」の側を測るのは第16層の相対感度（§5.3，§A.4）であり，入れ替え介入は行っていない．

## 主張

- 従来の因果抽象の手法には二つの制約があった——対応づけの**総当たり探索**と，高水準変数が**互いに素なニューロン集合**に対応するという前提（要旨）．
- **Distributed Alignment Search（DAS）** は勾配降下で対応づけを探し，**標準基底以外の基底**で表現を解析する．これにより一つのニューロンが複数の役割を担うことを許す（要旨）．
- 要旨は "Our experiments show that DAS can discover internal structure that prior approaches miss." と述べる．MoNLI の結果（§5.3）については "no localist alignment comes close to the learned distributed alignments in terms of IIA" と書く．

## 手法

### 分散的な入れ替え介入（定義3）

対象の変数の集合 `N` から空間 `Y` への可逆写像 `R` を置き，`Y` を直交する部分空間 `Y_0, …, Y_k` に分解する．`Y_0` は base の値を保ち，`Y_j`（j ≥ 1）には source `s_j` の値を入れる．

```
F*_N(v) = R^{-1}( Proj_{Y_0}(R(F_N(v))) + Σ_j Proj_{Y_j}(R(F_N(M(s_j)))) )
```

比較対象のうち，学習した分散的な対応づけに「最も近い」localist 対応づけは，回転が標準基底に揃った場合にあたる——"The rotation matrix for the localist alignment will be axis-aligned with the standard basis, possibly permuting and reflecting unit axes."（§3.7）

### 回転の学習（§3.5，定義4）

- `R` は回転（**直交行列**）とし，PyTorch の直交行列のパラメタ化（付録D：`torch.nn.utils.parametrizations`）を使って確率的勾配降下で学習する．
- 損失は任意の微分可能な損失でよく（定義4），base・source の組にわたって和をとる．実験では，入れ替え介入後の高水準モデルの出力分布と低水準モデルの出力分布のあいだの交差エントロピーを用いる．
- **低水準・高水準のモデルはどちらも凍結する**："Crucially, the low-level and high-level models are frozen during learning so we are only changing the alignment."
- 離散のハイパーパラメタとして残るのは，対象のニューロン集合 `N` と，各部分空間の次元 `|Y_0|, …, |Y_k|` である．

### IIA（定義5）

**interchange intervention accuracy（IIA）**："IIA is the proportion of aligned interchange interventions that have equivalent high-level and low-level effects." base・source の組にわたる平均である．IIA が 100% 未満のときは近似的な因果抽象として読み，**高水準モデルどうしの IIA の相対的な順位**が，どの高水準モデルがよりよい近似かを示すとする（付録D）．

## 実験設定

| 課題 | 低水準モデル | 比べる高水準モデル | 介入サイズ（表の Intervention size） |
| --- | --- | --- | --- |
| **階層的等価**（入力 `[w, x, y, z]`，ラベルは `(w=x) = (y=z)`） | 3層の ReLU 順伝播網（表1の層あたりのニューロン数 \|N\|=16／32） | 両方の等価関係（Both Equality Relations）／左の等価関係（Left Equality Relation）／第1引数の同一性（Identity of First Argument） | 1・2・8（\|N\|=16），2・4・16（\|N\|=32） |
| **単調性 NLI（MoNLI）**（前提の1語を上位語か下位語に替えて仮説を作る） | MultiNLI で微調整済みの BERT-base uncased を MoNLI でさらに微調整，**[CLS] トークンの表現のみ** | 否定と語彙的含意（Negation and Lexical Entailment）／語彙的含意（Lexical Entailment）／語彙素の同一性（Identity of Lexeme） | 64・128・256（\|N\|=768） |

- 階層的等価：無作為に生成した 1.92M 対で 10 エポック訓練して訓練正解率 100%，未見の入出力対からなる hold-out の test でも 100%（付録A.2）．
- MoNLI：元の MoNLI から無作為に取った 10K 例で 5 エポック微調整し，訓練・hold-out test とも 100%（付録A.2）．
- 表1・表2の IIA は**訓練データ上の値**（両表の caption："interchange intervention accuracies for the learned alignment on training data"）で，回転行列の学習の乱数種を変えた3回のうち最良を報告する．
- 「語彙素の同一性」の高水準モデルについて，§5.2 は "the identity of the premise word" と書き，付録A.1 は "the identity of replacing lexeme from the hypothesis sentence" と書く（原文のまま）．

## 結果

### 階層的等価（表1，第1層，IIA）

| 高水準モデル | \|N\|=16，介入サイズ 8 | \|N\|=32，介入サイズ 16 |
| --- | --- | --- |
| **両方の等価関係（DAS）** | **1.00** | 0.99 |
| 左の等価関係（DAS） | 0.90 | 0.99 |
| 第1引数の同一性（DAS） | 0.52 | 0.49 |

- **同じ「両方の等価関係」の第1層で，総当たり探索は 0.60，localist 対応づけは 0.73 にとどまる**（§4.3："the best brute-force alignment achieves only 0.60 and the best localist alignment achieves only 0.73"）．
- 「両方の等価関係」も，第2層・第3層では DAS で 0.49–0.67 である（表1）．
- 第1引数の同一性は DAS でも全層で 0.49–0.55 である（表1）．
- 著者の総括は "The best alignments found are with the ‘Both Equality Relations’ model that is widely assumed in the cognitive science literature." である．さらに，左の等価関係の部分空間を第1引数の同一性へ分解する試み（表1の最右列，0.50–0.51）が成り立たないことから，"models are truly learning to encode an abstract equality relation, rather than merely storing the identities of the inputs" と述べる（§4.3）．

### MoNLI（表2，BERT [CLS]，IIA）

| 高水準モデル | 第7層（介入サイズ 256） | **第9層（介入サイズ 256）** | 第11層（介入サイズ 256） |
| --- | --- | --- | --- |
| 否定と語彙的含意 | 0.67 | **1.00** | 0.86 |
| 語彙的含意 | 0.91 | **1.00** | 1.00 |
| 語彙素の同一性 | 0.88 | 0.96 | 0.88 |

- 第9層の「語彙的含意」の部分空間の中で，語彙素の同一性に対応する部分空間を学習すると 0.97–0.98（表2の最右列）——著者は，語彙的含意の表現が二つの語の同一性の表現へ "nearly perfectly decomposed" できると述べる（§5.3）．
- **localist 対応づけは 0.47–0.51**（表2の全列）．
- 表2の総当たり探索の行は，表1の総当たり探索の行と同じ数値の並び（0.60／0.56／0.52／0.64／0.64／0.57／0.50／0.51／0.54）である．原文に説明は無い．

## 著者が明記した限界

著者は「限界」の節を立てていない．以下は付録D（Common Questions）と §5.1 の記述である．

- **線形（直交）変換の制約**：変数が表現空間の非線形な部分多様体で表されている場合，"an orthogonal transformation results in imperfect interchange intervention accuracy"．可逆で微分可能な非線形変換のほうが適しうるが，"In practice, this transformation is computationally difficult to find" とする（付録D）．
- **規模**：BERT-base の全トークンを連結した表現に完全な回転を掛けると約 15.4B パラメタになり "becomes intractable for standard training infrastructure"（付録D）．MoNLI では "we look only at the representations of the [CLS] token"（§5.1）．
- **初期値依存**："The initialization step is important for finding the local optimal of the rotation matrix."（付録D）．対処として乱数種を変えた試行のうち最良を報告する（表1・表2は3回中の最良）．
- **前提**："We assume a partial or complete causal graph of the data generation process." また反実仮想データを抽出できること（"assume we can sample counterfactual data"）を要する（付録D）．

## 射程の外

- **学習済みモデルを凍結した解析**であって，学習時に表現へ構造を作り込む操作ではない（§3.5）．
- 実験の対象は3層の順伝播網と BERT の [CLS] トークン．**視覚言語モデル・画像入力・GQA は全文に見当たらなかった**（検索した語：vision，visual，image，multimodal，VQA，GQA）．
- 実験した二つの課題は，どちらも高水準のアルゴリズムが自明に書ける課題である（§1："two tasks that have obvious interpretable high-level algorithmic solutions with two intermediate variables"）．
- 表1・表2の IIA は**訓練データ上**の値である．未見の test 例での IIA は，付録E の図6–11に訓練エポックごとの曲線として示されるが，数値は本文に書かれていない．
- 表現が使われているかは入れ替え介入の一致率で測る．ヤコビアンや感度に基づく量は全文に見当たらなかった（検索した語：Jacobian，sensitiv）．

---

**確認**：2026-11-13．arXiv abs ページ（版の履歴 v1–v4），arXiv PDF v4（28頁，付録 A–E まで．pdftotext の全文と，表1・表2・図6の頁の画像），PMLR の公式書誌ページ（題・著者・巻・頁・BibTeX）と PMLR 版 PDF（28頁，arXiv v4 と本文を diff で突き合わせた）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．下書き（2026-10-11 作成）から直した主な点：位置づけから，原文に無い IIT との対比（「向きが逆」）と，研究文書に無い本研究との対応づけを外し，原文 §2 の文と研究文書の節番号に置き換えた．"no localist alignment comes close…" の出どころを MoNLI の §5.3 に限った．離散のハイパーパラメタから原文に無い「対象の層」を外した．表の列の名を原文の Intervention size に合わせた．「抽象的な等価関係を符号化している」の根拠を，原文どおり表1の最右列の分解の結果に付け替えた．「汎化は評価していない」を，付録E の図に test の曲線があるという事実に直した．限界の節に，著者が限界の節を立てていないことを明記した．否定の主張を検索語つきの「見当たらなかった」に直した．**読めていないもの**：arXiv v1–v3（v4 との異同は見ていない）．図5，図7–11 は caption のみで，図の中の値は読んでいない．PMLR ページの著者表記は "Noah Goodman" で，ミドルネームの頭文字は PDF にのみある．
