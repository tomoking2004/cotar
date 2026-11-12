# The Linear Representation Hypothesis and the Geometry of Large Language Models

**著者**：Kiho Park，Yo Joong Choe，Victor Veitch（シカゴ大学）
**掲載**：Proceedings of the 41st International Conference on Machine Learning（ICML 2024），PMLR 235，pp. 39643–39666．arXiv:2311.03658 v2（2024-07-17．v1 は 2023-11-07）
**原文**：[PMLR](https://proceedings.mlr.press/v235/park24c.html)・[arXiv](https://arxiv.org/abs/2311.03658)

**「概念は方向である」を出力側（unembedding）と入力側（embedding）の二通りに反実仮想で定式化し，両者は causal inner product のもとで一つに統一される．**

## 位置づけ

「線形に読める方向（プローブ）」と「足すと振る舞いが変わる方向（介入）」を，一つの幾何のもとで区別し結びつける理論．

## 主張

1. 線形表現仮説には**二つの定式化**がある——**unembedding 表現**（出力語の空間の方向）と**embedding 表現**（文脈の空間の方向）．前者は**測定（線形プローブ）**に，後者は**介入（steering）**に対応する．
2. 表現空間の内積は学習から**識別されない**．ユークリッド内積などに基づく代数的な操作が意味を持つと期待する明らかな理由はない，と著者は述べる（"there is no obvious reason to expect that algebraic manipulations based on, e.g., the Euclidean inner product, should be semantically meaningful"）．
3. **causal inner product**——因果的に分離可能な概念どうしを直交させる内積——を定義すると，定理の前提（各概念に，分離可能な d−1 個の概念があって基底をなす）のもとで，**unembedding 表現が embedding 表現に写る**．
4. この内積は，仮定（Assumption 3.3）のもとで unembedding の共分散を使った形に書け，自由度の残る部分は著者が選ぶ．LLaMA-2 の実験は，要旨の言葉で "the existence of linear representations of concepts, the connection to interpretation and control, and the fundamental role of the choice of inner product" を示す．

## 手法

### 設定

文脈 x の embedding を λ(x) ∈ Λ ≃ ℝᵈ，語 y の unembedding を γ(y) ∈ Γ ≃ ℝᵈ とし，次語分布は

```
P(y | x) ∝ exp(λ(x)ᵀ γ(y))
```

**概念** W は文脈が引き起こし出力の原因となる潜在変数で，説明を簡単にするため二値に限り，向きを持つ（例 male⇒female）．W の値だけが違う反実仮想の出力対（"man"，"woman"）で定まる．二概念 W，Z は Y(W=w, Z=z) が各 w，z について定まるとき**因果的に分離可能**（例：English⇒French と male⇒female は分離可能．English⇒French と English⇒Russian は不可）．

### 二つの線形表現

| | unembedding 表現 γ̄_W | embedding 表現 λ̄_W |
| --- | --- | --- |
| 空間 | 出力語の空間 Γ | 文脈の空間 Λ |
| 定義 | γ(Y(1)) − γ(Y(0)) がほぼ確実に Cone(γ̄_W) に入る（Definition 2.1） | P(W=1 \| λ₁)/P(W=1 \| λ₀) > 1 で，かつ W と分離可能な各 Z について P(W,Z \| λ₁)/P(W,Z \| λ₀) = P(W \| λ₁)/P(W \| λ₀) を満たす任意の λ₀，λ₁ について，λ₁ − λ₀ が Cone(λ̄_W) に入る（Definition 2.3） |
| 対応する操作 | **測定**（Theorem 2.2） | **介入**（Theorem 2.5） |
| 帰結 | 出力が Y(0) か Y(1) だと分かっているときの Y(1) の対数オッズ ＝ α·λᵀγ̄_W（α > 0 は対に依る）——ロジット線形な予測 | λ + c·λ̄_W とすると，W についての条件つき確率は c について増加し，分離可能な Z についての条件つき確率は c によらず一定 |

Cone(v) = {αv : α > 0}．符号に意味がある．

unembedding 表現は**相関しているが標的でない概念の情報を取り込まない**点で，プローブを当てはめて得る表現と異なる．

**Lemma 2.4**：分離可能な Z について λ̄_Wᵀγ̄_W > 0 かつ λ̄_Wᵀγ̄_Z = 0．逆に，ある表現がこの二つを満たし，W と分離可能な d−1 個の概念 Zᵢ があって {γ̄_W} ∪ {γ̄_Zᵢ} が ℝᵈ の基底をなすなら，その表現は W の embedding 表現である．

### 内積の非識別性

可逆な A と定数 β で γ(y) ← Aγ(y) + β，λ(x) ← A⁻ᵀλ(x) と変えても softmax は変わらない．学習の目的関数は表現に softmax の確率を通じてしか依存しないので，**γ は（せいぜい）可逆なアフィン変換を除いてしか定まらず，固定した内積の値は一般に A で変わる**．

### causal inner product

**Definition 3.1**：因果的に分離可能な任意の W，Z について ⟨γ̄_W, γ̄_Z⟩_C = 0 となる内積．

**Theorem 3.2（統一）**：任意の概念 W に，W と分離可能な d−1 個の概念 Zᵢ があって {γ̄_W} ∪ {γ̄_Zᵢ} が ℝᵈ の基底をなすとする．⟨·,·⟩_C が causal inner product なら，Riesz 同型が unembedding 表現を embedding 表現に写す——⟨γ̄_W, ·⟩_C = λ̄_Wᵀ．**内積を γ̄ᵀMγ̄′ と書き A = M^{1/2} で空間を変換すると，概念の embedding 表現と unembedding 表現は同じベクトルになる（§3.2 "Unified representations"）．**

### 推定

**Assumption 3.3**：分離可能な W，Z について，語彙から一様に引いた γ に対し λ̄_Wᵀγ と λ̄_Zᵀγ は独立．

内積を ⟨γ̄, γ̄′⟩ = γ̄ᵀMγ̄′ と書くと，直交性の制約は d(d−1)/2 個，M の自由度は d(d−1)/2 + d 個——**d 個の自由度（対角行列 D）が残る**（Theorem 3.4）．著者は D = I_d を選び，

```
⟨γ̄, γ̄′⟩_C = γ̄ᵀ Cov(γ)⁻¹ γ̄′
```

介入ベクトルは λ̄_W = Cov(γ)⁻¹γ̄_W（式 4.1）．γ̄_W は反実仮想対の差の（正規化した）平均で推定し，射影は leave-one-out で計算する．

## 実験設定

- **モデル**：LLaMA-2 7B．語彙 32,000，次元 4,096．付録 D.2 で Gemma-2B．
- **概念**：**27 個**．本文は，うち **22 個**を BATS 3.0（Big Analogy Test Set）が定める概念とする．残りは言語の4概念（English⇒French，French⇒German，French⇒Spanish，German⇒Spanish）と frequent⇒infrequent．ただし付録 C によれば，22 個のうち 13・17・19 番（small⇒big，pronoun⇒possessive，lower⇒upper）の対は ChatGPT-4 が生成し，16 番（country⇒capital）の対は別の csv に基づく．
- **測定**：本文は French と Spanish の Wikipedia から長さをランダムに取った文脈を用いる（付録 D.3 は English と French の文脈も）．
- **介入**：λ(x) + α·λ̄_C で α を **0 から 0.4** まで変える．

## 結果

**概念は方向である．** 反実仮想対の差を γ̄_W に射影すると，ランダムな語対の差に比べ，γ̄_W にかなり強く揃う（Figure 2 は "a strong right skew" と書く）．**唯一の例外は thing⇒part**——"The sole exception is thing⇒part, which does not appear to have a linear representation."

**分離可能な概念はほぼ直交する．** 推定した causal inner product のもとで，概念対の多くがほぼ直交し，ブロック対角の構造が見える（Figure 3）．

**ユークリッド内積との比較**（付録 D.2）：

| モデル | ユークリッド内積 | causal inner product |
| --- | --- | --- |
| LLaMA-2 7B | "Surprisingly, the Euclidean inner product somewhat works in the LLaMA-2 model" | 明らかに改善する，と著者は書く（例：frequent⇒infrequent は，ユークリッド内積では多くの分離可能な概念との内積が高いが，causal inner product ではずっと小さい） |
| Gemma-2B | "doesn't capture semantics" | "still works" |

**測定**：French⇒Spanish の γ̄_W への射影 γ̄_Wᵀλ(x) が，フランス語とスペイン語の文脈を予想どおりの符号（フランス語で負，スペイン語で正）で分ける．標的でない male⇒female の表現はこの課題に予測力を持たない（Figure 4）．

**介入**（文脈 "Long live the "，male⇒female 方向を加える．Table 1）：

| α | 0 | 0.1 | 0.2–0.4 |
| --- | --- | --- | --- |
| 次語の1位 | king | Queen | **queen** |

male⇒female の介入は "queen" 対 "king" のロジットを上げ，"King" 対 "king"（lower⇒upper）は変えない．**標的の概念が動き，標的でない概念は動かない**（Figure 5）．

## 著者が明記した限界

原典に限界を扱う節は無い（"limitation" の語は全文に 0 件）．以下は，本文と付録の各所で著者が断っていることである．

- **内積は一意に定まらない**："We do not have a principle for picking out a unique choice of D."
- **中間層を扱わない**："we do not address interpretability of either model parameters, nor the activations of intermediate layers."
- 概念は二値に限り，概念の値は出力語から決定的に読めると仮定する．
- 複数トークンに分かれる語を含む対は使わない．ある語が別の語のトークンの一部にもなる問題は扱えず，結果に多くの雑音を与える，と書く（付録 C）．
- Assumption 3.3 は語彙から無作為に引いた語についての仮定で，自然言語の出どころから無作為に引いた語についてのものではない．後者では，分離可能な概念のあいだに因果によらない相関がありうる，と著者は注記する（脚注2）．

## 射程の外

- **対象は，softmax で内積をとる文脈の embedding λ(x) と語の unembedding γ(y)**．中間層の活性は扱わない，と著者が書く（上記）．
- **実験は言語モデル2本**（LLaMA-2 7B と，付録の Gemma-2B）．VLM の実験は見当たらなかった（"vision-language" は関連研究の引用にだけ現れ，"image" は 0 件）．
- **実験の 27 概念は語の対で定まる二値の概念**（語形・性別・言語・大文字小文字など，Table 2）であり，推論手順や問いの構造を概念とする実験は見当たらなかった（"reasoning" は 0 件）．
- **学習によって表現を変える実験は見当たらなかった**（"training"・"trained"・"fine-tun" を検索．学習への言及は，内積が学習から識別されないという議論と，モデルの事前学習の説明だけ）．介入は，文脈の embedding に方向を加えるものだけである．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴・Comments・Journal-ref），arXiv PDF v2（24頁．本文・参考文献・付録 A–D の全文）と v1（28頁），PMLR の書誌ページ（巻・頁）を取得した．v2 を pdftotext でテキスト化して全文を読み，定義・定理・Assumption 3.3・Table 1 は PDF の 3・4・5・8 頁を画像にして確かめ，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．v1 は，本文書が引く数値と引用の有無だけを検索で見た（Gemma-2B の比較と "We do not have a principle for picking out a unique choice of D." の文の形は v2 のもの．27・22 概念，32,000・4,096，α の範囲は v1 と同じ）．arXiv に v3 は無い．下書き（2026-10-11 作成）から直した主な点：主張4の「すべてが確かめられる」を要旨の強さに，主張2の「意味上の特権はない」を原文の文に，「見かけの類似が消える」を「ずっと小さい」に直した．Theorem 3.2 の前提と embedding 表現の定義の条件を原文どおりに書いた．「22 個は BATS から」に付録 C の但し書き（4概念の対は BATS 以外）を足した．射程の外の「最終層出力」は原文に無い語なので外し，否定の主張は検索した語を添えた強さにした．**読めていないもの**：付録 B の証明は読んだが，正しさを自分で検算してはいない．Figure 2–5・7–13 の図の中の値は，本文とキャプションに書かれたものだけを照合した．Table 1 は各 α の1位の語だけを載せた．
