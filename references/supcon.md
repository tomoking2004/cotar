# Supervised Contrastive Learning

**著者**：Prannay Khosla，Piotr Teterwak，Chen Wang，Aaron Sarna，Yonglong Tian，Phillip Isola，Aaron Maschinot，Ce Liu，Dilip Krishnan（Google Research／ボストン大学／Snap／MIT）
**掲載**：NeurIPS 2020．arXiv:2004.11362
**原文**：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)・[arXiv](https://arxiv.org/abs/2004.11362)

**自己教師ありの対照学習を教師ありへ拡張する——アンカーごとの正例を1つから，同じクラスの全サンプルへ増やす．**

## 主張

> 要旨（原文）："Contrastive learning applied to self-supervised representation learning has seen a resurgence in recent years, leading to state of the art performance in the unsupervised training of deep image models. Modern batch contrastive approaches subsume or significantly outperform traditional contrastive losses such as triplet, max-margin and the N-pairs loss. In this work, we extend the self-supervised batch contrastive approach to the *fully-supervised* setting, allowing us to effectively leverage label information. Clusters of points belonging to the same class are pulled together in embedding space, while simultaneously pushing apart clusters of samples from different classes. We analyze two possible versions of the supervised contrastive (SupCon) loss, identifying the best-performing formulation of the loss. On ResNet-200, we achieve top-1 accuracy of 81.4% on the ImageNet dataset, which is 0.8% above the best number reported for this architecture. We show consistent outperformance over cross-entropy on other datasets and two ResNet variants. The loss shows benefits for robustness to natural corruptions, and is more stable to hyperparameter settings such as optimizers and data augmentations. Our loss function is simple to implement and reference TensorFlow code is released at https://t.ly/supcon."

## 手法

### 記号

- `I ≡ {1…2N}`：multiviewed batch のアンカー添字．N 個の元サンプルをそれぞれ2通りに増強して 2N 個にする
- `A(i) ≡ I \ {i}`：アンカー以外のすべて（2N−1 個）
- `P(i) ≡ {p ∈ A(i) : ỹ_p = ỹ_i}`：**アンカーと同じクラスの全サンプル**．無作為バッチでの見込みは平均 N/C 個（C はクラス数）
- `z_ℓ = Proj(Enc(x̃_ℓ)) ∈ R^{D_P}`：射影ネットワークの出力．単位超球上に正規化される
- `τ`：温度

### 損失の定義

**自己教師あり版**——正例は `j(i)`（同一サンプルのもう一方の増強）ただ1つ．

```
L_self = Σ_{i∈I} −log [ exp(z_i·z_{j(i)}/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]
```

**教師あり版・その1「L_out」＝著者の推奨**——正例の和が **log の外**．

```
L_out = Σ_{i∈I} (−1/|P(i)|) Σ_{p∈P(i)} log [ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]
```

**教師あり版・その2「L_in」**——正例の和が **log の内側**．

```
L_in = Σ_{i∈I} −log { (1/|P(i)|) Σ_{p∈P(i)} exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) }
```

どちらも**分母は正例・負例を問わず 2N−1 個すべて**にわたる．

### どちらを使うべきか——`L_out` である

**経験的な差は大きい**．ResNet-50・バッチ 6144 で `L_out` が top-1 **78.7%**，`L_in` が **67.4%**——**11.3 ポイントの開き**．

**理由は勾配にある．** 両者の勾配は同じ形を持つ．

```
∂L_i/∂z_i = (1/τ) { Σ_{p∈P(i)} z_p (P_ip − X_ip) + Σ_{n∈N(i)} z_n P_in }
```

違いは `X_ip` だけ——

```
X_ip = exp(z_i·z_p/τ) / Σ_{p'∈P(i)} exp(z_i·z_p'/τ)   （L_in のとき）
X_ip = 1/|P(i)|                                        （L_out のとき）
```

`L_in` では `1/|P(i)|` の正規化が log の内側に入って加法定数となり，勾配に影響しない——"Without any normalization effects, the gradients of L_in^sup are more susceptible to bias in the positives"．

なお値の大小からは判定できない："Because log is a concave function, Jensen's Inequality implies that L_in^sup ≤ L_out^sup"——**決め手は上の勾配の議論と実測**である．

**暗黙の hard mining**：正規化表現と内積を使うと，勾配の寄与は容易な正例で `‖z_p − (z_i·z_p) z_i‖ ≈ 0`，難しい正例で `≈ 1` になる．triplet loss で必須の明示的な hard mining を回避できる．

### 既存損失との関係

"We show that the triplet loss is a special case... when one positive and one negative are used"，そして負例が複数になると "SupCon loss becomes equivalent to the N-pairs loss"．SupCon は**正例も負例も多数**の場合にあたる．

### 射影 head の扱い

1. **encoder** `Enc(·)`：`r ∈ R^{D_E}`，`D_E = 2048`
2. **射影ネットワーク** `Proj(·)`：`z ∈ R^{D_P}`，`D_P = 128`．隠れ層 2048 の MLP か単層線形．出力は単位超球上に正規化
3. **損失は射影ネットワークの出力の上で計算する**——"The supervised contrastive loss is computed on the outputs of the projection network"．encoder 表現 `r` の上ではない．

**推論時には `Proj(·)` を捨てる**——"we discard Proj(·) at the end... inference-time models contain exactly the same number of parameters" as a cross-entropy model．

**学習は二段**．第1段で SupCon による事前学習，第2段で**凍結した encoder 表現の上に**交差エントロピーで線形分類器を学習する．第2段は "as few as 10 epochs of additional training" で足りる．

## 実験設定

- **増強**：入力1つにつき2通りの無作為増強．AutoAugment・RandAugment・SimAugment・Stacked RandAugment を比較し，AutoAugment が ResNet-50 に，Stacked RandAugment が ResNet-200 に最良
- **バッチ**："batch sizes of 6144 for ResNet-50 and batch size 4096 for ResNet-200"．ただし "batch sizes of 2048 suffice for most purposes"．メモリバンク版（バッチ 256・バンク 8192）は **79.1%**
- **温度**："All our results used a temperature of τ = 0.1"
- **エポック**：ResNet-200 で 700，小さいモデルで 350．ResNet-50 は "even 200 epochs is likely sufficient"
- **最適化器**：事前学習は LARS，線形分類器は RMSProp．交差エントロピーの基準線は SGD+momentum が最良

## 結果

**ImageNet 分類**

| 構成 | 損失 | 増強 | top-1 | top-5 |
| --- | --- | --- | --- | --- |
| ResNet-50 | SupCon | AutoAugment | **78.7%** | 94.3% |
| ResNet-50 | 交差エントロピー | AutoAugment | 77.6% | 95.3% |
| ResNet-101 | SupCon | Stacked RandAugment | **80.2%** | 94.7% |
| ResNet-200 | SupCon | Stacked RandAugment | **81.4%** | 95.9% |
| ResNet-200 | 交差エントロピー | AutoAugment | 80.6% | 95.3% |

**他データセット（ResNet-50，SimCLR／交差エントロピー／SupCon）**：CIFAR-10 で 93.6 / 95.0 / **96.0**，CIFAR-100 で 70.7 / 75.3 / **76.5**，ImageNet で 70.2 / 78.2 / **78.7**．N-pairs は ImageNet で **57.4%** に留まる．

**自然な劣化への頑健性（ImageNet-C，rel.mCE / mCE．低いほど良い）**：ResNet-50 は交差エントロピー 96.2 / 68.6 に対し SupCon **94.6 / 67.2**，ResNet-200 は 69.1 / 52.4 に対し **66.5 / 50.6**．

**ハイパーパラメタへの安定性**：増強・最適化器・学習率を振ったとき，SupCon は出力の分散が有意に小さい．

## 著者が明記した限界

- **転移学習では優位が出ない**："SupCon is on par with cross-entropy and self-supervised contrastive loss on transfer learning performance"．
- **射影 head の最適な構造は未検討**："we leave to future work the investigation of optimal Proj(·) architectures"．

## 射程の外

- **対象は画像分類**である．言語モデル・VLM・中間層の表現整合は扱わない．
- **損失は射影後の低次元表現（`D_P = 128`）に掛けられる**．生の encoder 表現に直接掛ける設計は本論文では検討されていない．
- **クラスラベルは画像分類のカテゴリ**であり，クラス数が極端に偏る場合の挙動は論じられていない．

---

**確認**：2026-08-09．ar5iv による HTML 全文を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．
