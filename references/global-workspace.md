# Verbalizable Representations Form a Global Workspace in Language Models

**著者**：Wes Gurnee\*，Nicholas Sofroniew\*，Adam Pearce，Mateusz Piotrowski，Isaac Kauvar，Runjin Chen，Anna Soligo，Paul Bogdan，Euan Ong，Rowan Wang，T. Ben Thompson，David Abrahams，Subhash Kantamneni，Emmanuel Ameisen，Joshua Batson，Jack Lindsey\*†（Anthropic）
**掲載**：Transformer Circuits Thread，2026-07-06
**原文**：[Transformer Circuits](https://transformer-circuits.pub/2026/workspace/index.html)

**言語化できる表現は活性の分散のごく一部しか占めないが，言語的報告と内部推論に因果的に効くのはそちら側である．自動的な処理はそこを通らない．**

## 主張

言語モデルは，はるかに大量の自動的処理の上に特権的な表現の集合を保っている——"we observe that language models maintain a privileged set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing"．

著者はこれを workspace 的な表現と呼び，人間の意識的アクセスに対応する五つの性質で定義する——**verbal report（言語的報告）・directed modulation（方向づけられた変調）・internal reasoning（内部推論）・flexible generalization（柔軟な汎化）・selectivity（選択性）**．

扱うのは機能としての access consciousness であり，主観的経験との関係については "we take no position on this issue" と断る．

## 手法

### Jacobian lens（J-lens）

J_ℓ = E_{t, t′≥t, prompt} [ ∂h_{final,t′} / ∂h_{ℓ,t} ]

- 期待値は**元の位置 t，文脈内のそれ以降のすべての位置 t′，そして1,000本のプロンプト**にわたって取る．
- 結果は層 ℓ ごとに **d_model × d_model の行列1つ**．適用は `lens(h_ℓ) = softmax(W_U norm(J_ℓ h_ℓ))`．
- **J-lens ベクトルは `W_U J_ℓ` の行**である．
- **学習は含まない**（"requires no auxiliary training"）．中間活性が最終層 logit に与える線形化された因果効果を平均するだけで得られる．
- **既定の作り方**：128トークンの列を1,000本使い，Sonnet 4.5 では微分の先を最後から2番目の層に取る（付録）．本数を 1 から 1000 まで振ると，"J-lens beats the logit lens and tuned lens baselines with as few as 10 prompts" である．現在の位置だけに絞るなどの作り方の違いには，結果は大きく左右されない（"fairly consistent among these design choices"）．

### J-space

**J-lens ベクトルの疎な非負結合として表せる点の集合**．分解は **gradient pursuit** で解く．

疎性パラメタ k について著者はこう断る——"For the J-space to be properly defined, we must specify an allowable sparsity level k—this parameter is somewhat arbitrary, and we vary our choice of k throughout the paper, but we typically choose it to be no more than 25"．

### 線形プローブとの違い（関連研究）

線形プローブについて著者はこう書く——"They are also correlational, rather than causal, in the sense that a probe may recover information the model encodes but does not itself use"．

### J-space の除去

各位置で最も強く活性化した上位 **k = 10** の J-lens 方向を，指定層域で射影除去する．強度は **light / medium / heavy** の3段で，除去する層域の広さが違う．報告そのものを消さないよう，除去しない場合の上位10トークンは除去の対象から外す．

## 実験設定

主対象は **Claude Sonnet 4.5**．補強として **Claude Haiku 4.5・Claude Opus 4.5**，一部の節で **Claude Opus 4.6**．**定量の実験の課題はテキストである．** 画像に触れるのは序論の図3の例（"the model recognizing an image of a face"）で，図の中身は本文のテキストからは確かめていない．

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

**93% の分散を担う補空間を差し替えても 5% しか効かず，6–7% を担う J-space 成分は 59% 効く．** 摂動の大きさは揃えてある．59% と 88% は別々の操作の成功率であって，「J-space の因果寄与の幅」ではない．

二段推論の probe 分解（n = 90 プロンプト）：J-space 成分 **61%**，非 J-space 成分 **28%**，J-space への再流入を遮断すると **6%**．

### 二段推論の中間表現の差替（n = 50 プロンプト，top-1）

Haiku 4.5 **54%**，Sonnet 4.5 **70%**，Opus 4.5 **70%**．中間の差替は答えの差替より "a median of approximately 17 percent earlier" から効き始める．

### 層方向の三分割

- **sensory**：モデルのおよそ最初の3分の1．意味のある J-lens 内容を持たない．
- **workspace**："beginning about a third of the way through (~L38) and ending shortly before the output (~L92)"．立ち上がりでは**過剰尖度・top-1 トークンの自己相関・実効次元がいずれも同じ層あたりで急に上がり**，自己相関は中間帯で高いまま推移する．
- **motor**：最後の数層で，読み出しは出力に揃う．"J-lens vectors function as 'motor' representations that drive the imminent output"．ただし著者は，workspace と motor を分ける原理的な定義は示していないと断る（"somewhat post-hoc"）．

全体として "coherent content emerges only after an initial band of layers, and abstract concepts give way in the final layers to representations tied more directly to the imminent output"．

### 選択性——J-space を潰すと何が壊れるか

**強い除去でも基準線付近を保つ**：MMLU の多肢選択，odd-one-out，SQuAD の抽出型 QA，感情分類，CoLA は "essentially unaffected even under heavy ablation" である．

**大きく壊れる**："Caesar-cipher decoding, analogy completion, summarization, TriviaQA, multi-hop reasoning, translation, sonnet writing"——多段推論は統制評価でほぼゼロまで落ちる．

**連鎖思考の効果**：GSM8K を明示的な chain-of-thought で解く場合は，同じ問題を直接答える場合より "substantially more robust to ablation"．

**読めても使われない場合がある**：言語が本文から分かる8つの文章で，言語名は四つの課題すべてで J-lens に同程度に現れるが，差し替えて答えが変わるのは明示の報告と柔軟な推論だけで，続きを書く課題と異常検知はほとんど動かない．著者はこう整理する——"In some cases, the information relevant to the automatic computation is present in the J-space but unused for the task; in others, it is not present at all"．

**自動的な計算は workspace を通らない**：ある節では，linewrap 指示に従って正しい桁で折り返す流暢な継続を作りながら，数のトークンは lens にまったく現れない（"number tokens are entirely absent from the lens across the prompt"）．一方，同じ節について明示的に問えば「46」と答える．

### 同一表現の操作横断再利用（柔軟な汎化）

4カテゴリ（国名・月・動物・数詞）× カテゴリごとに4つの関数＝16関数，各関数につき12の差替対で **192試行**．

- α = 1：**76/192**
- α = 2：**101/192**

成功は元の語の "workspace loading" でよく予測され，"Country arguments have the highest loading and swap most reliably; number-word arguments have the lowest loading"．数詞の結果について著者は二通りの解釈を残す——workspace の外で計算している場合と，語の J-lens ベクトルにモデルの表現が揃っていない場合である．

## 著者が明記した限界

- **語彙の制約**：J-lens は "only identifies vectors associated with concepts that correspond to single tokens in the model's vocabulary, but many important concepts correspond to multiple tokens"（複数トークンへの拡張は付録で言及）．
- **概念の結びつきは読めない**：読み出しは概念の平らな集まりとして扱うので，概念どうしがどう結びつくかは見えない（Beyond a bag of concepts）．
- **どの課題が J-space を要るかは予測できない**："we have no way of predicting in advance, for an arbitrary computation, whether it will engage the J-space"．
- **モデルの規模**：主に大規模な製品モデルを調べたのであり，規模との関係は分からない——"Future work could investigate whether smaller models have an equally rich workspace, a proportionally smaller one, a less reliable one, or none at all."．
- **道具としての不完全さ**："The Jacobian lens is an imperfect tool, which we believe only approximately and incompletely captures the model's underlying workspace structure."
- **初期層の解釈は二通り残る**：最初の3分の1に J-lens で読める内容が無いことは，"(1) the J-lens is degenerate at these depths and fails to resolve content that is in fact present" と "(2) the early-layer residual stream genuinely carries no linearly accessible and causally relevant verbalizable content" のどちらでもありうる．ゆえに "it remains possible that parts of the model's 'true workspace,' not captured by the J-lens, operate in earlier layers"．
- **疎性パラメタの恣意性**（上記の引用のとおり）．
- **脳との構造的差**："we do not claim that language models reproduce the full architecture global workspace theory ascribes to the brain—specialized, encapsulated processors competing for entry to a workspace that broadcasts back to them through recurrent connections"．broadcast は "occurs within a single feedforward pass rather than through recurrent loops"．
- **ignition の不確かさ**："although we observe some degree of competition for access to the J-space, it is unclear whether this mirrors the sharp, competitive 'ignition' that characterizes workspace entry in the brain"．

## 射程の外

- **評価対象は Claude 系の大規模モデルのみ**．VLM を対象にした評価は無い——ただし著者が「VLM では未検証」と述べているわけではなく，**評価対象一覧からそう言えるにすぎない**．小さいモデルについては，著者が今後の問いとして明記している（上記）．
- **独立再現への言及は本文に無い**．他機関による再現を引く場合は，**本論文とは別の一次資料として扱う必要がある**．著者は再現のために実装を公開し，公開モデルの J-lens を Neuronpedia に置いている．

## 学会で聞かれたら

本研究についての答えは研究文書（`.claude/context.md`）の範囲で書いた．

- **問：どんな論文ですか．**
  答：言語モデルには，報告や内部推論に使える特権的な表現の集合があり，大量の自動的な処理の上に乗っている，という論文です．語ごとに，言語化に結びつく方向を平均のヤコビアンで求めるのが J-lens で，その方向の疎な非負結合が J-space です．（出所：原典 Introduction，Methods）
- **問：モデルに意識があると言っているのですか．**
  答：いいえ．機能としての意識的アクセスを扱っていて，主観的経験との関係には立場を取らないと断っています．脳の workspace の構造全体を再現するとも主張していません．（出所：原典 Introduction）
- **問：本研究は J-lens を使ったのですか．**
  答：考え方を借りました．答えを決める方向を，途中の層を線形化して第16層へ引き戻しています．原典は全語彙の方向を位置とプロンプトで平均しますが，本研究は答えの語の方向だけを，一つの位置で 64 問について平均するので，J-space そのものを測ったのではありません．（出所：研究文書 §A.4，原典 Methods）
- **問：64 問で足りますか．**
  答：同じ逆伝播から累積平均を見ました．相対感度は 64 問でもなお僅かに動きますが，baseline との比はどの本数でも −24% から −30% の間にあります．（出所：研究文書 §A.4）
- **問：第16層は原典の workspace の層に当たりますか．**
  答：本研究では確かめていません．原典の帯は Claude の層を 0–100 に直しておよそ 38 から 92 で，小さいモデルにも同じ構造があるかは，著者も分からないと書いています．（出所：原典 The J-space's structure supports its function，Limitations）
- **問：署名は J-space に載っていますか．**
  答：本研究では確かめていません．J-lens の方向は単一トークンの語に対応するもので，演算子の列である署名には当たる語がありません．本研究で測ったのは，答えを決める方向に署名が集中していないことです．（出所：原典 Limitations，研究文書 §5.3）
- **問：線形プローブの限界について，原典は何と言っていますか．**
  答：プローブは相関的で，モデルが符号化しているが使わない情報も回収しうると書いています．原典の課題でも，言語名が J-lens に読めるのに，続きを書く課題では差し替えても答えが変わりません．どちらも署名についての結果ではありません．（出所：原典 Related work，The J-space selectively mediates flexible but not automatic cognition）
- **問：分散は小さいのに効く，という原典の結果は本研究と関係しますか．**
  答：読める量と効く量が別だという向きは似ています．ただし原典の数字は Claude の概念ベクトルの分解で，同じ分解を本研究で行うことは確かめていません．（出所：原典 The J-space supports verbal report）

---

**確認**：2026-08-09．Transformer Circuits の HTML 全文を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．照合で裏の取れなかった記述（言語継続実験の試行数など）は本文書に載せていない．

**再照合**：2026-11-06．Transformer Circuits の HTML を取得してテキスト化し，序論・Methods・A Global Workspace・Structure の各節，Related work，Discussion の Limitations と Alignment implications を通読し，付録のうち Quantitative Methodological Comparisons と Methodological Details and Ablations の一部を読んで，本文書の数値・引用を一件ずつ照合した．直した主な点：「いずれもテキストの言語モデル」を改め，小さいモデルについて著者が書いた問いを足し，連続していない引用（motor，選択性の課題の列，文頭の大文字小文字など）を原文に揃えた．原文に無い「特異値分解は用いない」と「傾向であって決定的ではない」を除いた．言語の課題の文章数（8）は今回確かめて載せた．**読まなかった範囲**：Alignment Auditing，The Assistant's Perspective，Counterfactual Reflection Training の本文と，それ以外の付録は通読していない（image などの語の検索だけを全文に掛けた）．図の中の値と対話的な図の中身は確かめていない．
