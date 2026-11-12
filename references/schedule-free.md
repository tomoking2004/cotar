# The Road Less Scheduled

**著者**：Aaron Defazio，Xingyu (Alice) Yang（以上 Fundamental AI Research Team, Meta），Harsh Mehta（Google Research），Konstantin Mishchenko（Samsung AI Center），Ahmed Khaled（Princeton University），Ashok Cutkosky（Boston University）
**掲載**：NeurIPS 2024（Advances in Neural Information Processing Systems 37，Main Conference Track，pp. 9974–10007）．arXiv:2405.15682 v4，2024-10-29（v1 は 2024-05-24）
**原文**：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)・[arXiv](https://arxiv.org/abs/2405.15682)

**学習率スケジュールを使わず，勾配を取る点・基礎の最適化器が更新する点・評価に使う平均の3系列を補間と平均でつなぐことで，停止時刻 T を事前に定めずに，スケジュール付きの学習に並ぶか上回る．**

以下の節・図・付録の番号と逐語引用は **arXiv v4** のものである．NeurIPS の予稿集版（34頁）は組みが違い，番号も一部の文も一致しない（末尾の確認行に挙げる）．

## 位置づけ

本研究の実装（`cotar/training/trainer.py`）が使う最適化器 `RAdamScheduleFree`（`schedulefree` パッケージ）の元になった手法の論文である．実装は，学習と評価の切り替えに合わせて最適化器の `train()`／`eval()` を呼ぶ．研究文書（`.claude/context.md`）は最適化器に触れていない（§4.1 が書くのは学習率と勾配ノルムの打ち切りまで）．RAdam 版そのものは本論文に現れない（下の「RAdam 版の有無」）．

## 主張

- **停止時刻を事前に決めなくてよい**——"Our approach does not require the stopping time T to be known or set in advance."（1.1 節）
- **追加のハイパーパラメタはない**：基礎の SGD（モーメンタム付き）や Adam に対して新たなハイパーパラメタを要しない（1.1 節）．
- 凸 Lipschitz 設定では，モーメンタム係数をどう選んでも最悪ケース最適（1.1 節）．Theorem 1：ステップ幅を γ = D/(G√T) と置くと，任意の β∈[0,1] で E[F(x_T) − F(x_⋆)] ≤ DG/√T．
- **28 問題**（ロジスティック回帰から大規模深層学習まで）の評価で，入念に調整した（"heavily-tuned"）cosine スケジュールに並ぶか上回る（1.1 節）．
- Schedule-Free AdamW は，MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge の Self-Tuning track で著者らが優勝したエントリーの中核のアルゴリズムである（要旨）．

## 手法

### 3系列（Schedule-Free SGD，2 節）

`z_1 = x_1` から始め，ステップ幅 γ，モーメンタム係数 β で

```
y_t     = (1 − β) z_t + β x_t              # 勾配を取る点
z_{t+1} = z_t − γ ∇f(y_t, ζ_t)             # 基礎の SGD 系列
x_{t+1} = (1 − c_{t+1}) x_t + c_{t+1} z_{t+1},   c_{t+1} = 1/(t+1)   # 評価系列
```

| 系列 | 役割 |
| --- | --- |
| `z` | 基礎の最適化器が更新する系列（"base sequence"） |
| `x` | `z` の**等重みのオンライン平均**．**評価に使う系列**（"evaluation sequence"） |
| `y` | `z` と `x` の補間．**勾配はここで取る**（"gradient location sequence"） |

- **β = 0 で Polyak–Ruppert 平均，β = 1 で primal averaging**（勾配を `x` で取る）に一致する．
- Polyak 平均・primal averaging・Schedule-Free のいずれでも，`x` の系列は「現在の時刻 t で終わる線形減衰スケジュール」の効果を模す——著者はこれを，平均がスケジュールの代わりになる理由とする（2.3 節）．
- β は標準的なモーメンタムの値に近い β ≈ 0.9 が実用上うまく働くようだ，と著者は述べる（2 節）．

### Schedule-Free AdamW（Algorithm 1）

入力：`x_1`，学習率 γ，減衰 λ，warmup ステップ `T_warmup`，β₁，β₂，ε．`z_1 = x_1`，`v_0 = 0`．

```
y_t     = (1 − β₁) z_t + β₁ x_t
g_t     ∈ ∂f(y_t, ζ_t)
v_t     = β₂ v_{t−1} + (1 − β₂) g_t²
γ_t     = γ √(1 − β₂^t) · min(1, t / T_warmup)       # warmup と Adam のバイアス補正を学習率に含める
z_{t+1} = z_t − γ_t g_t / (√v_t + ε) − γ_t λ y_t     # weight decay は y で掛ける
c_{t+1} = γ_t² / Σ_{i=1}^{t} γ_i²
x_{t+1} = (1 − c_{t+1}) x_t + c_{t+1} z_{t+1}
```

**AdamW 版では平均の重みが `1/(t+1)` ではなく `γ_t²` に比例する**（式 23）．著者は，warmup を使うときこの重み付きの `c_t` で性能が大きく改善した（"greatly improved"）と述べる（4.4 節）．β₁ は勾配の指数移動平均としてではなく，`y` の補間として入る（Algorithm 1 の注記 "Momentum via interpolation"）．

### 実装上の注意（4.4 節）

- **warmup は依然として必要**——"Learning rate warmup is still necessary for our method."
- バイアス補正を学習率に含めたことは，初期の反復での weight decay の効き方に影響しうる．
- weight decay は `y` でも `z` でも計算できる．著者の実験は `y` で掛け，**`y` で掛けると ImageNet と NanoGPT を含む一部の問題で "significantly better performance" だった**と述べる（検定の記述はない）．
- **メモリ**：Schedule-Free SGD は，モーメンタム付き SGD に対して余分なメモリを要しない（`x` とモーメンタムの buffer の代わりに `x` と `z` を持つ）．`y` は `x` と `z` から計算できる．著者らの PyTorch 実装は `z` 用に buffer を1本持ち，**主パラメタの buffer に `x` か `y` を入れて，学習時と評価・推論時で入れ替える**．
- **BatchNorm**：running_mean・running_var は `y` で計算されるので，"For model evaluation, these buffers need to be updated to match the statistics on the x sequence."．評価の直前に `x` で少数の学習バッチを流すか，PreciseBN を使う．バッチ統計を使わない正規化層では不要．

### train/eval の切替（公式実装 README．論文本文ではない）

公式リポジトリ（facebookresearch/schedule_free）の README は，`model.train()` を呼ぶ箇所で `optimizer.train()` を，`model.eval()` を呼ぶ箇所で `optimizer.eval()` を呼び，パラメタの buffer を二つの点（勾配を取る点と，test/val の損失を計算する点）の間で切り替えるよう求める．チェックポイントを保存するときも eval モードにすべきとする．

### RAdam 版の有無

- **論文に RAdam は現れない**（arXiv v4 と予稿集版の全文で "RAdam"・"rectified" を検索し，0件）．
- 公式実装 README は `RAdamScheduleFree` を "Schedule-free variant of RAdam, which eliminates the need for both learning rate scheduling and warmup" と記載し，"(implementation community contributed)" と注記する．Releases の項に "Version 1.4 adds a RAdam implementation" とある．「warmup が不要」は README の記載であり，論文の実験・理論で示されたものではない．

## 実験設定

**深層学習 9 問題**（4.1 節．CIFAR-10・CIFAR-100・SVHN・ImageNet は Schedule-Free SGD，残りは Schedule-Free AdamW）

| 問題 | モデル | 比較したスケジュール（本文の記述） | 結果（本文の記述） |
| --- | --- | --- | --- |
| CIFAR-10 | Wide ResNet 16-8 | cosine | 上回る |
| CIFAR-100 | DenseNet | cosine | 上回る |
| SVHN | deep ResNet（3-96） | cosine | 上回る |
| ImageNet | ResNet-50 | 既存の最良のスケジュール | 並ぶ |
| IWSLT14 独英 | LSTM | cosine | 上回る |
| DLRM（Criteo Kaggle） | DLRM | 線形減衰 | 上回る |
| fastMRI | stacked U-Net | 線形減衰 | 上回る |
| MAE（ImageNet 微調整） | ViT（patch16-512d-8b） | 既存の最良のスケジュール | 並ぶ |
| NanoGPT（OpenWebText） | 124M パラメタの GPT-2 型 | cosine | 上回る |

**深層学習の数値は図の凡例（Figure 5，IWSLT14 は Figure 2）にあり，本文に精度の表はない．** 比較相手の欄は 4.1 節の本文に従った．図の凡例・付録 G と食い違う箇所がある（IWSLT14 は付録 G.6 の表が基準線を Linear Decay とし，fastMRI は Figure 5 の凡例が "Cosine Schedule" とする）．

**β**：0.9 と 0.98 の二つを掃引．**既定値の 0.9 で NanoGPT 以外のすべての問題でうまく働き**，NanoGPT では 0.9 で損失が急に増え始め，0.98 は安定した．

**AlgoPerf**（4.2 節）：Schedule-Free AdamW を NAdamW（競技の参照アルゴリズム）と比較，それぞれ**乱数種 10**．

| 結果 | 問題 |
| --- | --- |
| 上回る | WMT，ViT，fastMRI，OGBG |
| 並ぶ | Conformer，Criteo |
| わずかに下回る | DeepSpeech |

（全8問題のうち ResNet-50 は，AdamW も NAdamW も目標精度に届かないとして除外．）

**凸問題**（4.3 節）：LibSVM のロジスティック回帰 12 問題のうち **11 で，二つの平均化法（Polyak・primal）と線形減衰スケジュールより速く収束**．

## 結果

- **パラメタ感度**（5 節）：ImageNet・Schedule-Free SGD・200 エポック・学習率 1.5 固定で，**最良のモーメンタム β = 0.9 は学習の長さによらず同じ**．
- 基準の学習率の，学習の長さへの依存は，スケジュール付きの学習と同程度に穏やかである．短い学習（≤ 25 エポック）ではより大きい学習率が最良になり始める（Figure 10）．最終のテスト精度の学習率への感度は，選んだ問題で cosine スケジュール付きの基準の最適化器と同程度（付録 I）．

## 著者が明記した限界

- **学習率と weight decay の掃引が要る**——"The primary practical limitation is the need to sweep learning rate and weight decay"．最良の値がスケジュール付きのときと異なるためである（6 節）．
- **理論は予備的**——"further theory is needed to fully understand the method."（6 節．arXiv v4 の文で，予稿集版の結論にはこの句が無い）
- **BatchNorm を使うモデルには追加のコードが要る**（4.4 節）．AlgoPerf では Conformer と DeepSpeech の成績について "We attribute the performance on the Conformer and DeepSpeech tasks to their use of batch-norm" と述べ，AlgoPerf の仕組みでは `x` の系列で BN の統計を更新しにくいことを理由に挙げる（4.2 節）．
- 大きな学習率が使える条件の全体は，確率的な問題ではまだ十分に分かっていない（2.2 節）．

## 射程の外

- **対象は最適化器の収束と最終性能**である．中間層の表現や，表現に掛ける補助損失との相互作用を扱う記述は見当たらなかった（検索した語：representation，auxiliary，contrastive．該当は参考文献の会議名だけ）．
- **RAdam 版は論文の対象外**——4.1 節の深層学習の実験が使ったのは Schedule-Free SGD と Schedule-Free AdamW，4.2 節の AlgoPerf は Schedule-Free AdamW である．
- **VLM・GQA は評価問題に見当たらない**（検索した語：GQA，VLM，vision-language，visual question．0件）．"fine-tuning" と書かれた問題は MAE（ViT）の1件だけである．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴 v1〜v4），arXiv PDF v4（35頁．付録 A〜I を含む全文を pdftotext でテキスト化し，Theorem 1・Algorithm 1・式 23・Figure 5・5〜6 節・付録 G.5〜G.7 は頁の画像でも見た），NeurIPS 公式の書誌ページ（巻・頁・トラック）と予稿集版 PDF（34頁），公式実装の README（main ブランチ，最終更新 2026-06-18）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．本研究についての文は `.claude/context.md` §4.1 と `cotar/training/trainer.py` に照らした．下書き（2026-10-11 作成）から直した主な点：位置づけを，研究文書に無い「評価は平均化重み」の記述から，実装に在る事実へ直した．所属を著者ごとに対応させた．Theorem 1 にステップ幅 γ = D/(G√T) の条件を足した．AlgoPerf の「優勝した」を要旨の表現（優勝したエントリーの中核のアルゴリズム）に直した．2.3 節の記述を，三つの平均化法に共通の性質として直した．weight decay の「有意に良かった」を原文の句の引用に直した．README の `optimizer.eval()` を呼ぶ箇所を README の文どおりに直した．「理論は予備的」の引用が arXiv v4 だけに在ることを明記した．比較相手の欄が図・付録と食い違う箇所を注記した．否定の主張を，検索した語を添えた強さに直した．**読めていないもの**：付録 A〜F の証明は式の展開を検算していない．図の曲線は読まず，凡例の値も本文書には載せていない．予稿集版は通読しておらず，arXiv v4 との語単位の差分と，本文書の引用・番号の検索で比べた——予稿集版には 2.3 節（How Schedule-Free Replaces a Schedule）が無く，Implementation Concerns は 4.3 節，学習率の感度の付録は J で，`x` の説明は "a running average" である．公式実装のコード（`RAdamScheduleFree` の中身）は読んでいない．README は随時更新される文書で，上の記載は閲覧日のものである．
