# Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models

**著者**：Peter Hase（Google Research／UNC Chapel Hill），Mohit Bansal（UNC Chapel Hill），Been Kim（Google Research），Asma Ghandeharioun（Google Research）
**掲載**：Advances in Neural Information Processing Systems 36（NeurIPS 2023），Main Conference Track，pp. 17643–17668．arXiv:2301.04213 v2，2023-10-16（v1 は 2023-01-10）
**原文**：[arXiv](https://arxiv.org/abs/2301.04213)・[NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)

**Causal Tracing で測った，事実が「どこに格納されているか」は，「どの層を編集すれば効くか」をほとんど予測しない——編集の成否の分散を説明するのは tracing effect ではなく，編集する層の選択である．**

## 主張

Causal Tracing（表現の denoising）で事実が特定の MLP 層に局在していると分かれば，その層を編集すれば事実を書き換えられるはずだ——ROME と MEMIT は，Causal Tracing の平均の結果にもとづいて編集層を選んだ．著者らはこの前提をデータ点ごとに検証し，局在の結果は，既存の事実を新しい事実で上書きするのにどの MLP 層を編集するのが最良かについて，何の手がかりも与えないことを見いだす．消去・増幅など編集問題の変種に広げると，そのうち1つ（Fact Forcing）では編集の成績が局在の結果と関係するが，そこでも編集層の選択のほうがはるかによく成績を予測する．結論として，著者らは "better mechanistic understanding of how pretrained language models work may not always translate to insights about how to best change their behavior" と述べる（要旨，§7）．

## 手法

### 局在——Causal Tracing

主語のトークン埋め込みにガウス雑音を加えた順伝播の途中で，ある層・位置の表現を雑音なしの順伝播のものに書き換え，正答の確率がどれだけ変わるかを測る（§3.2）：

$$
\text{Tracing Effect} = p(o_{\text{true}} \mid s_{\text{noise}}, r, v_{(t,\ell)}) - p(o_{\text{true}} \mid s_{\text{noise}}, r)
$$

分析では，介入に効果が無いときを 0，もとの確率 $p(o_{\text{true}} \mid s, r)$ を完全に回復したときを 1 とする割合の形で用いる（§4.1）．MLP 層にのみ適用し，窓幅（書き換える隣接層の数）は既定で 5，各層でトークン方向の最大値を取る．付録 B では第2の局在法として表現の zeroing も試す．

### 編集法（§3.3，§5.2）

| 手法 | 内容 |
| --- | --- |
| ROME | 1つの MLP 層の down-projection 行列への rank one の編集．GPT-J の既定層は 6 |
| MEMIT | 単一の事実を編集するときは，更新を1層でなく複数の層に分散する点だけが ROME と違う（本論文では5層に適用） |
| 制約付き微調整 FT（1層） | $\ell_\infty$ ノルム制約付きの，Adam による最適化を1層に |
| 制約付き微調整 FT（5層） | 同じものを隣接する5層に |

### 編集問題（§5.1）

| 問題 | 目的 |
| --- | --- |
| Error Injection | 偽の目的語の確率 $p(o_{\text{false}} \mid s, r)$ を最大化（§4 で扱った問題） |
| Tracing Reversal | 主語に雑音を入れた入力に対する出力 $o_{\text{noise}}$ の確率 $p(o_{\text{noise}} \mid s, r)$ を最大化 |
| Fact Erasure | 真の目的語の確率 $p(o_{\text{true}} \mid s, r)$ を最小化 |
| Fact Amplification | 真の目的語の確率 $p(o_{\text{true}} \mid s, r)$ を最大化 |
| Fact Forcing | 主語に雑音を入れた入力のもとで真の目的語の確率 $p(o_{\text{true}} \mid s_{\text{noise}}, r)$ を最大化 |

Fact Forcing は実用上の意味は乏しい（"of little practical significance"）が，"it is the most similar to Causal Tracing in its design, since it uses the same input as Causal Tracing"——入力・目標・目的の三点で Causal Tracing に合わせるよう設計された変種である（§1，§5.1）．

### 分析

編集の成否は主に rewrite score（編集が目標の確率をどれだけ上げたかを，可能な最大の改善に対する割合で表したもの，§3.4）で測る．これを，tracing effect のみ・編集層のみ（カテゴリ変数）・両者の交互作用で線形回帰し，$R^2$ を比べる（§4.3）．

## 実験設定

- **モデル**：GPT-J（60億パラメタ，28層）．頑健性の確認に GPT2-XL（48層．Error Injection と Fact Forcing のみ，付録 C）．
- **データ**：CounterFact．その 10% から，GPT-J が正しく補完できる事実に絞って "Our final sample size is n = 652."（§4.2，付録 A）．付録 B で ZSRE も用いる．

## 結果

### 局在と編集の成否は相関しない（§4）

ROME は大半の層でよく効く："Note ROME achieves an average rewrite score of 99% at layer 6 and above 96% at layers besides layer 28."（§4.2）．一方で多くの事実は編集される層の範囲の外に局在している（Figure 1："But many facts appear to be stored outside of this range, e.g. at layers 1-3 and 16-20."）．

第6層での rewrite score と tracing effect の関係は "there is not a positive relationship but a negative relationship"——線形相関は $\rho = -0.13$（$p < 10^{-3}$）である（§4.3，Figure 4）．ただし著者らは "in most layers we simply see a near-zero rather than negative correlation" と断る（§4.3）．

| ROME の編集成否を説明する $R^2$（Table 1） | Layer | Tracing Effect | Both |
| --- | --- | --- | --- |
| ROME | 0.947 | 0.016 | 0.948 |

編集層の選択を考慮に入れると，"the tracing effect is able to explain only 0.1% of the variance in edit success"（§4.3）．

### 編集問題と手法を広げても同じ（§5.3，付録の Table 3 から抜粋）

| 問題 | 手法 | Layer | Trace | Both | Diff |
| --- | --- | --- | --- | --- | --- |
| Error Injection | FT（1層） | 0.756 | 0.062 | 0.758 | 0.002 |
| Error Injection | ROME | 0.947 | 0.016 | 0.948 | 0.001 |
| Error Injection | MEMIT | 0.677 | 0.024 | 0.678 | 0.001 |
| Fact Forcing | FT（1層） | 0.697 | 0.104 | 0.724 | 0.027 |
| Fact Forcing | FT（5層） | 0.634 | 0.10 | 0.666 | 0.032 |

- 編集層のみの回帰は "which explains most of the variance in the outcome (58.5% on average across conditions in Fig. 6)"．Fig. 6 が載せるのは Error Injection を除く4問題×4手法の16条件である．
- tracing を足して増える説明力は "tracing effects explain at most an additional 3.2% of the variance in edit success"——最大は Fact Forcing を制約付き微調整（5層）で解いた場合である（Table 3 の Diff 0.032）．編集の成否と tracing effect の正の関係が最も強いのは Fact Forcing を微調整で解く場合で，他の実験では増分は 1.5% までである．
- **原文の食い違い**：Tracing Reversal の FT（1層）は，Fig. 6 では 29.4／29.6，Table 3 では 0.067／0.067 で，一致しない（Table 3 は Fig. 6 のデータに対応すると題されている）．本文の 58.5% は Fig. 6 の16条件の値の平均（58.525）に一致し，Table 3 の対応する16行の平均（約 0.571）には一致しない．どちらが正しいかは原文からは分からない．

### 著者の解釈（§6）

- "First, we note that information is gradually accumulated across layers in a Transformer forward pass"．
- 仮説として "Many layers could store a fact, and it happens that some do."——ただし，この仮説が正しければ驚くべきことだと著者らは述べ（"it should matter where information enters the residual stream"），検証は今後の課題に残す．
- "Causal Tracing answers a different question than model editing does."
- Causal Tracing の知見が無用だったとは言っていない（"this does not mean that localization insights from Causal Tracing have been useless"）．

## 著者が明記した限界（§8）

- **データ**："We work only with the CounterFact and ZSRE datasets, which we use as short English prompts"——"localization and editing analysis may yield different trends for other forms of knowledge."
- **モデル**："We work with two autoregressive Transformers chosen for their representativeness"——"the conclusions from our analysis may not generalize to models larger than GPT-J (6B parameters)"
- **手法**："We use a particular set of localization and editing methods"——"Our conclusions may not necessarily hold for the breadth of localization and editing methods"

## 射程の外

- **言語モデル単体**（GPT-J，GPT2-XL）であって，VLM ではない．GQA は使わない（全文に GQA の語は無い）．
- 扱うのは**事実知識**（主語・関係・目的語の組 $(s, r, o)$）で，質問の解き方の手順（演算子列）ではない．
- 操作は**個別の事実についての重みの編集**であり，補助損失を足した学習ではない．
- 「局在」は Causal Tracing（と zeroing）で測ったものである．線形プローブによる読み取りは用いていない（全文を probe・probing で検索して見当たらなかった）．
- 著者は手法・データ・モデルの範囲外への一般化を留保している（上記）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴・Comments），arXiv PDF v2（26頁，付録 A〜C を含む全文．pdftotext でテキスト化し，Figure 6 は頁画像で確かめた），NeurIPS 2023 proceedings の書誌ページ（巻・頁）を読み，逐語引用は空白を正規化した全文への文字列照合でも確かめて，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．PDF v1（20頁）は，本文書が引く主な数値（n = 652，ρ = −0.13，Table 1，3.2%，58.5%，Table 3 の Error Injection の行）が v2 と同じであることだけを確かめた．下書き（2026-10-11 作成）から直した主な点：99% の引用の節を §4.3 から §4.2 へ．「編集層を固定したうえで」を原文の「編集層の選択を考慮に入れると」へ．「局在が関係するのは1つだけで効果は小さい」を要旨の言い方へ．第6層の負の相関に，大半の層ではほぼ 0 という著者の断りを足した．58.5% が指す範囲（Fig. 6 の16条件）と，Fig. 6 と Table 3 の食い違いを明記した．雑音で「正答の確率を下げた」という原文に無い説明と，「部分空間による局在ではない」という確かめられない否定を外した．Table 3 が付録にあること，著者の所属の対応を補った．NeurIPS proceedings 掲載版の PDF（26頁）は通読していないが，本文書の逐語引用と上の主な数値がすべて同じ文字列で在ることを機械的に確かめた．**読めていないもの**：NeurIPS proceedings の Supplemental．proceedings 掲載版と arXiv v2 の，引用箇所以外の異同．図の中の値は Figure 6 だけを照合した．
