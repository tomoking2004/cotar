# Verbalizable Representations Form a Global Workspace in Language Models

**著者**：Wes Gurnee\*，Nicholas Sofroniew\*，Adam Pearce，Mateusz Piotrowski，Isaac Kauvar，Runjin Chen，Anna Soligo，Paul Bogdan，Euan Ong，Rowan Wang，T. Ben Thompson，David Abrahams，Subhash Kantamneni，Emmanuel Ameisen，Joshua Batson，Jack Lindsey\*†（Anthropic）
**掲載**：Transformer Circuits Thread，2026-07-06
**原文**：[Transformer Circuits](https://transformer-circuits.pub/2026/workspace/index.html)

**言語化できる表現は活性の分散のごく一部しか占めないが，因果的に効くのはそちら側である．**

## 主張

言語モデルは，はるかに大量の自動的処理の上に特権的な表現の集合を保っている——"we observe that language models maintain a privileged set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing"．

著者はこれを workspace 的な表現と呼び，人間の意識的アクセスに対応する五つの性質で定義する——**verbal report（言語的報告）・directed modulation（方向づけられた変調）・internal reasoning（内部推論）・flexible generalization（柔軟な汎化）・selectivity（選択性）**．

## 手法

### Jacobian lens（J-lens）

J_ℓ = E_{t, t′≥t, prompt} [ ∂h_{final,t′} / ∂h_{ℓ,t} ]

- 期待値は**元の位置 t，文脈内のそれ以降のすべての位置 t′，そして1,000本のプロンプト**にわたって取る．
- 結果は層 ℓ ごとに **d_model × d_model の行列1つ**．適用は `lens(h_ℓ) = softmax(W_U norm(J_ℓ h_ℓ))`．
- **J-lens ベクトルは `W_U J_ℓ` の行**である．
- **学習は含まない**．中間活性が最終層 logit に与える線形化された因果効果を平均するだけで得られる．

### J-space

**J-lens ベクトルの疎な非負結合として表せる点の集合**．分解は **gradient pursuit** で解く——**特異値分解は用いない**．

疎性パラメタ k について著者はこう断る——"For the J-space to be properly defined, we must specify an allowable sparsity level k—this parameter is somewhat arbitrary, and we vary our choice of k throughout the paper, but we typically choose it to be no more than 25."

### J-space の除去

上位 **k = 10** の J-lens 方向を指定層域で射影除去する．強度は **light / medium / heavy** の3段．

## 実験設定

主対象は **Claude Sonnet 4.5**．補強として **Claude Haiku 4.5・Claude Opus 4.5・Claude Opus 4.6**．**いずれもテキストの言語モデルである．**

層の位置は，残差ストリームの**等間隔25層を [0–100] に再指標化**して報告される．

## 結果

### 分散は小さい

- J-space 成分が活性の全分散に占める割合は "varying by layer, but never more than 10%"．
- 概念ベクトルでは "the J-space component carries a median of only **6–7%** of the concept vector's variance, with the remaining **~93%**"．

### 分散は因果を意味しない

概念ベクトルの差替（成功＝目標語が上位5件に入ること）：

| 差し替える成分 | 成功率 |
| --- | --- |
| **純粋な J-lens ベクトル** | **88%** |
| J-space 成分 | **59%** |
| 補空間（非 J-space）成分 | **5%** |

**93% の分散を担う補空間を差し替えても 5% しか効かず，6–7% を担う J-space 成分は 59% 効く．** 59% と 88% は別々の操作の成功率であって，「J-space の因果寄与の幅」ではない．

二段推論の probe 分解（n = 90 プロンプト）：J-space 成分 **61%**，非 J-space 成分 **28%**，J-space への再流入を遮断すると **6%**．

### 二段推論の中間表現の差替（n = 50 プロンプト，top-1）

Haiku 4.5 **54%**，Sonnet 4.5 **70%**，Opus 4.5 **70%**．中間の差替は答えの差替より "a median of approximately 17 percent earlier" から効き始める．

### 層方向の三分割

- **sensory**：モデルのおよそ最初の3分の1．意味のある J-lens 内容を持たない．
- **workspace**："beginning about a third of the way through (~L38) and ending shortly before the output (~L92)"．立ち上がりでは**過剰尖度・top-1 トークンの自己相関・実効次元がいずれも同じ層あたりで急に上がり**，自己相関は中間帯で高いまま推移する．
- **motor**："in the final few layers, J-lens vectors function as 'motor' representations that drive the imminent output"．

全体として "coherent content emerges only after an initial band of layers, and abstract concepts give way in the final layers to representations tied more directly to the imminent output"．

### 選択性——J-space を潰すと何が壊れるか

**強い除去でも基準線付近を保つ**："MMLU multiple choice, odd-one-out, SQuAD extractive QA, sentiment classification, CoLA... essentially unaffected even under heavy ablation"．

**大きく壊れる**："Caesar-cipher decoding, analogy completion, summarization, TriviaQA, multi-hop reasoning, translation, sonnet writing"——多段推論は統制評価でほぼゼロまで落ちる．

**連鎖思考の効果**：GSM8K を明示的な chain-of-thought で解く場合は，同じ問題を直接答える場合より "substantially more robust to ablation"．

**自動的な計算は workspace を通らない**：ある節では，linewrap 指示に従って正しい桁で折り返す流暢な継続を作りながら，数のトークンは lens にまったく現れない（"number tokens are entirely absent from the lens across the prompt"）．一方，同じ節について明示的に問えば「46」と答える．

### 同一表現の操作横断再利用（柔軟な汎化）

4カテゴリ（国名・月・動物・数詞）× カテゴリごとに4つの関数＝16関数，各関数につき12の差替対で **192試行**．

- α = 1：**76/192**
- α = 2：**101/192**

成功は "workspace loading" とよく相関し，"Country arguments have the highest loading and swap most reliably; number-word arguments have the lowest loading"．**傾向であって決定的ではない．**

## 著者が明記した限界

- **語彙の制約**："The Jacobian lens only identifies vectors associated with concepts that correspond to single tokens in the model's vocabulary, but many important concepts correspond to multiple tokens"（複数トークンへの拡張は付録で言及）．
- **道具としての不完全さ**："The Jacobian lens is an imperfect tool, which we believe only approximately and incompletely captures the model's underlying workspace structure."
- **初期層の解釈は二通り残る**：最初の3分の1に J-lens で読める内容が無いことは，"(1) the J-lens is degenerate at these depths and fails to resolve content that is in fact present" と "(2) the early-layer residual stream genuinely carries no linearly accessible and causally relevant verbalizable content" のどちらでもありうる．ゆえに "it remains possible that parts of the model's 'true workspace,' not captured by the J-lens, operate in earlier layers"．
- **疎性パラメタの恣意性**（上記の引用のとおり）．
- **脳との構造的差**："We do not claim that language models reproduce the full architecture global workspace theory ascribes to the brain—specialized, encapsulated processors competing for entry to a workspace that broadcasts back to them through recurrent connections." broadcast は "occurs within a single feedforward pass rather than through recurrent loops"．
- **ignition の不確かさ**："Although we observe some degree of competition for access to the J-space, it is unclear whether this mirrors the sharp, competitive 'ignition' that characterizes workspace entry in the brain."

## 射程の外

- **評価対象は Claude 系のテキストモデルのみ**．VLM も小規模モデルも評価に含まれない——ただし著者が「VLM では未検証」と述べているわけではなく，**評価対象一覧からそう言えるにすぎない**．
- **独立再現への言及は本文に無い**．他機関による再現を引く場合は，**本論文とは別の一次資料として扱う必要がある**．

---

**確認**：2026-08-09．Transformer Circuits の HTML 全文を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．照合で裏の取れなかった記述（言語継続実験の試行数など）は本文書に載せていない．
