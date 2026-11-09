# Supervised Contrastive Learning

**著者**：Prannay Khosla，Piotr Teterwak，Chen Wang，Aaron Sarna，Yonglong Tian，Phillip Isola，Aaron Maschinot，Ce Liu，Dilip Krishnan（Google Research／ボストン大学／Snap／MIT）
**掲載**：NeurIPS 2020．arXiv:2004.11362
**原文**：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)・[arXiv](https://arxiv.org/abs/2004.11362)

**自己教師ありの対照学習を教師ありへ拡張する——アンカーごとの正例を1つから，同じクラスの全サンプルへ増やす．**

以下の引用と節番号は arXiv 版による．NeurIPS 版と食い違う箇所は，その場に書く．

## 主張

> 要旨（原文）："Contrastive learning applied to self-supervised representation learning has seen a resurgence in recent years, leading to state of the art performance in the unsupervised training of deep image models. Modern batch contrastive approaches subsume or significantly outperform traditional contrastive losses such as triplet, max-margin and the N-pairs loss. In this work, we extend the self-supervised batch contrastive approach to the *fully-supervised* setting, allowing us to effectively leverage label information. Clusters of points belonging to the same class are pulled together in embedding space, while simultaneously pushing apart clusters of samples from different classes. We analyze two possible versions of the supervised contrastive (SupCon) loss, identifying the best-performing formulation of the loss. On ResNet-200, we achieve top-1 accuracy of 81.4% on the ImageNet dataset, which is 0.8% above the best number reported for this architecture. We show consistent outperformance over cross-entropy on other datasets and two ResNet variants. The loss shows benefits for robustness to natural corruptions, and is more stable to hyperparameter settings such as optimizers and data augmentations. Our loss function is simple to implement and reference TensorFlow code is released at https://t.ly/supcon."

## 手法

### 記号

- `I ≡ {1…2N}`：multiviewed batch のアンカー添字．N 個の元サンプルをそれぞれ2通りに増強して 2N 個にする
- `A(i) ≡ I \ {i}`：アンカー以外のすべて（2N−1 個）
- `P(i) ≡ {p ∈ A(i) : ỹ_p = ỹ_i}`：**アンカーと同じクラスの全サンプル**（アンカー自身のもう一方の増強を含む）．バッチがクラス数に比べて大きいとき，加わる項は平均 N/C 個（C はクラス数）と書かれる．バッチ 6144 の ImageNet では，期待値で 13 個（別サンプル6個の各2増強と，アンカーのもう一方の増強．補足 §11）
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

**経験的な差は大きい**．ResNet-50・バッチ 6144 で `L_out` が top-1 **78.7%**，`L_in` が **67.4%**——**11.3 ポイントの開き**．以後の実験はすべて `L_out` である——"Throughout the rest of the paper, we consider only"．

**著者の説明は勾配にある．** 差の原因は "We conjecture that this is due to the gradient of" `L_in` の構造，と推測として書かれる．両者の勾配は同じ形を持つ．

```
∂L_i/∂z_i = (1/τ) { Σ_{p∈P(i)} z_p (P_ip − X_ip) + Σ_{n∈N(i)} z_n P_in }
```

違いは `X_ip` だけ——

```
X_ip = exp(z_i·z_p/τ) / Σ_{p'∈P(i)} exp(z_i·z_p'/τ)   （L_in のとき）
X_ip = 1/|P(i)|                                        （L_out のとき）
```

`L_in` では `1/|P(i)|` の正規化が log の内側に入って加法定数となり，勾配に影響しない——"Without any normalization effects, the gradients of L_in^sup are more susceptible to bias in the positives"．`L_out` の `X_ip` は，正例を平均で置き換えたときの `L_in` の値に等しく（式6），著者はここから "the stabilization due to using the mean of positives benefits training" と結論する．

**値の大小についての一節は，版によって逆である．** 数学としては，log が凹なので `L_in ≤ L_out` である．

- arXiv 版は正しい向きで書く："Because log is a concave function, Jensen's Inequality [23] implies that L_in^sup ≤ L_out^sup"，"One would thus expect L_out^sup to be the superior supervised loss function (since it upper-bounds L_in^sup)"．Change Log の Version 6 に "Corrected sign mistake in Jensen's Inequality" とある．
- NeurIPS 版は不等号が逆で（`L_out ≤ L_in`），結論も逆である："One might thus be tempted to conclude that L_in^sup is the superior supervised loss function (since it bounds L_out^sup). However, this conclusion is not supported analytically."

**どちらの版でも，`L_out` を選ぶ根拠は勾配の議論と実測である．** 大小の一節に頼らない．

**暗黙の hard mining**：表現を正規化すると，勾配の寄与は容易な正例で `‖z_p − (z_i·z_p) z_i‖ ≈ 0`，難しい正例で `≈ 1` になる（補足 §7）．triplet loss で必須の明示的な hard mining を回避できる．この性質は "only if a normalization layer is added to the end of the projection network" 現れる．

### 既存損失との関係

"the triplet loss is a special case of the contrastive loss when one positive and one negative are used"，そして負例が複数になると "SupCon loss becomes equivalent to the N-pairs loss"．SupCon は**正例も負例も多数**の場合にあたる．

補足 §9 は，これらを SupCon の特殊な場合として並べる——"The SupCon loss subsumes multiple other commonly used losses as special cases of itself."

| 制限 | 得られる損失 |
| --- | --- |
| 正例を，同じ画像の増強1つに限る | 自己教師あり版 |
| 正例を，同じラベルの**別サンプル**1つに限り，温度を外す | N-pairs．"There is also no notion of temperature in the original N-Pairs loss, though it could be easily generalized to include it." |
| 正例も負例も1つに限る | triplet（補足 §8 の近似による） |

### 射影 head の扱い

1. **encoder** `Enc(·)`：`r ∈ R^{D_E}`，`D_E = 2048`．`r` も単位超球上に正規化される
2. **射影ネットワーク** `Proj(·)`：`z ∈ R^{D_P}`，`D_P = 128`．隠れ層 2048 の MLP か単層線形．出力は単位超球上に正規化
3. **損失は射影ネットワークの出力の上で計算する**——"The supervised contrastive loss is computed on the outputs of the projection network"．encoder 表現 `r` の上ではない．

**推論時には `Proj(·)` を捨てる**——"we discard Proj(·) at the end of contrastive training"．そのため "our inference-time models contain exactly the same number of parameters as a cross-entropy model using the same encoder"．

**学習は二段**．第1段で SupCon による事前学習，第2段で**凍結した encoder 表現の上に**交差エントロピーで線形分類器を学習する．第2段は "as few as 10 epochs of additional training" で足りる．この第2段は top-1 を測るための "(optional)" な段階で，表現を転移や検索に使うなら要らない．

## 実験設定

- **増強**：入力1つにつき2通りの無作為増強．AutoAugment・RandAugment・SimAugment・Stacked RandAugment を比較し，AutoAugment が ResNet-50 に，Stacked RandAugment が ResNet-200 に最良
- **バッチ**："batch size 6144 for ResNet-50 and batch size 4096 for ResNet-200"．ただし "batch sizes of 2048 suffice for most purposes"．メモリバンク版（バッチ 256・バンク 8192）は **79.1%**
- **温度**："All our results used a temperature of τ = 0.1"．固定の値で，学習はしない．小さいほど学習に有利だが，極端に小さいと数値的に不安定になる．勾配が `1/τ` に比例するので，"we rescale the loss by τ during training for stability"．ResNet-50 の top-1 で 0.1 が最適と実測し，最適な温度で "nearly 3%" 改善しうる（補足 §10）
- **エポック**：ResNet-200 で 700，小さいモデルで 350．ResNet-50 は "even 200 epochs is likely sufficient"
- **最適化器**（ImageNet）：事前学習は LARS，線形分類器は RMSProp．交差エントロピーの基準線は SGD+momentum が最良．CIFAR-10・100 では SGD+momentum が最良

## 結果

**ImageNet 分類**（表3）

| 構成 | 損失 | 増強 | top-1 | top-5 |
| --- | --- | --- | --- | --- |
| ResNet-50 | SupCon | AutoAugment | **78.7%** | 94.3% |
| ResNet-50 | 交差エントロピー（著者の実装） | AutoAugment | 77.6% | 95.3% |
| ResNet-50 | 交差エントロピー（参照論文の値） | AutoAugment | 78.2% | 92.9% |
| ResNet-101 | SupCon | Stacked RandAugment | **80.2%** | 94.7% |
| ResNet-200 | SupCon | Stacked RandAugment | **81.4%** | 95.9% |
| ResNet-200 | 交差エントロピー（著者の実装） | Stacked RandAugment | 80.9% | 95.2% |
| ResNet-200 | 交差エントロピー（参照論文の値） | AutoAugment | 80.6% | 95.3% |

要旨の "0.8% above" は 80.6 に対する差である．同じ増強の著者の実装 80.9 に対しては 0.5 ポイントである．

**他データセット（ResNet-50，SimCLR／交差エントロピー／SupCon）**：CIFAR-10 で 93.6 / 95.0 / **96.0**，CIFAR-100 で 70.7 / 75.3 / **76.5**，ImageNet で 70.2 / 78.2 / **78.7**．N-pairs は ImageNet で **57.4%** に留まる．著者はその要因を，多視点を使わないこと，温度が低くないこと，正例が少ないことに帰す（"We believe"，§4.1）．

**正例の数（補足 §11，ResNet-50，ImageNet，表5）**：上限を 1／3／5／7／9／無し（期待値 13）とすると top-1 は 69.3／76.6／78.0／78.4／78.3／78.5．ここでの「1」は同じサンプルの増強1つで，別サンプルの正例ではない．

**自然な劣化への頑健性（ImageNet-C，rel.mCE / mCE．低いほど良い）**：ResNet-50 は交差エントロピー 96.2 / 68.6 に対し SupCon **94.6 / 67.2**，ResNet-200 は 69.1 / 52.4 に対し **66.5 / 50.6**．

**ハイパーパラメタへの安定性**：増強・最適化器・学習率を一つずつ変えて比べ，交差エントロピーに対して "significantly lower variance in the output of the contrastive loss" と報告する．

## 著者が明記した限界

- **転移学習では優位が出ない**："SupCon is on par with cross-entropy and self-supervised contrastive loss on transfer learning performance"．
- **射影 head の最適な構造は未検討**："we leave to future work the investigation of optimal Proj(·) architectures"．
- **大きなバッチを要する**：hard mining が自然に働く代わりに，"This comes at the cost of requiring large batch sizes to include many positives and negatives"（補足 §8）．

## 射程の外

- **対象は画像分類**である．言語モデル・VLM は扱わない．中間層は，別の研究の紹介に現れるだけである——"[12] also uses a closely related loss formulation to ours to entangle representations at intermediate layers by maximizing the loss"．
- **損失は射影後の低次元表現（`D_P = 128`）に掛けられる**．生の encoder 表現に直接掛ける設計は本論文では検討されていない．
- **補助損失として別の損失と同時に学習する形**は，本論文の形ではない．SupCon の第1段は対照損失だけで学習する．同時に学習する形は，近い手法 CCLP との違いとして挙げられる（§2）．
- **クラスラベルは画像分類のカテゴリ**であり，クラス数が極端に偏る場合の挙動は論じられていない．

## ポスターとの対応

ポスターの [3] がこの文献である．原典の箇所は arXiv 版の節番号で書く．

| ポスターの文 | 原典の箇所 | 原文 | 読み |
| --- | --- | --- | --- |
| キーアイデア：答えに至る手順が同じ質問どうしの内部表現を近づけ，手順が違う質問どうしを遠ざける | 要旨，§1 | "Clusters of points belonging to the same class are pulled together in embedding space, while simultaneously pushing apart clusters of samples from different classes" | 損失の狙いとしては原文どおりである．**測定の結果としては言えない**——同署名の類似度も下がった（研究文書 §4.4） |
| 図の④の右の整合損失 ℒalign：同じ手順の 𝒉 が，違う手順より近くなるように学習（supervised contrastive loss[3]） | §3，§3.2.2，補足 §9 | "The supervised contrastive loss is computed on the outputs of the projection network" | 損失の名と形は [3] による．掛ける先は違う——原典は射影の出力，本研究は 𝒉 を長さ1に揃えたもの．正例が1件なので，原典の階層では N-pairs の形に温度を足したものにあたる |
| 図：同じ手順：近づける／違う手順：遠ざける（手順の一致は署名の一致．3.1） | §1，式4 | "Normalized embeddings from the same class are pulled closer together than embeddings from different classes" | 原文は比較の形である．「他より近い」であって，「前より近い」ではない |
| 工夫：画像分類で同じ種類の画像を近づける手法[3]を，答えに至る手順が同じ質問に転用 | 要旨，§1，§4 | "Clusters of points belonging to the same class are pulled together"／"by measuring classification accuracy on a number of common image classification benchmarks" | 元の使われ方（画像分類で，同じクラスを近づける）は原文どおりである．「種類」は class の言い換えである．同じクラスの代わりに署名を当てたのはこちらの選択で，原典にこの使い方は無い．損失の形は原典のとおりである．原典は対照学習の後に線形分類器を別に学習する2段の形だが，本研究は通常の損失に足して1段で学習する——この違いはポスターに載せていない |
| 工夫：元の手法は表現を小さなネットワークに通してから近づける → 通さず，𝒉（下図③）を直接近づける／通すと，𝒉 が変わらないまま，小さなネットワークの側だけで損失が下がりうるため | §3，§3.1 | "The supervised contrastive loss is computed on the outputs of the projection network"／"we discard Proj(·) at the end of contrastive training" | 「小さなネットワーク」は射影 head（隠れ層1つの MLP か線形層1つ，出力 128 次元．§3.1）の言い換えである．外した理由は原典のものではなく研究文書 §3.3 のもので，原典は生の表現に直接掛ける形を検討していない |

**「近づける」は，損失が指す向きとしては正しい．** 正例が1件のとき，式4の `z_p` の係数は `P_ip − 1 ≤ 0` なので，勾配を下る向きは常に正例へ寄る（こちらの計算）．ただし全員を同時に動かした結果の類似度は，別の量である．

## 学会で聞かれたら

### 論文そのもの

1. **SupCon を一言で言うと？**
   自己教師ありの対照学習を，ラベルのある設定へ拡張した損失です．アンカーごとの正例を1つから同じクラスの全サンプルへ増やし，負例は従来どおり多数使います．（§1）
2. **L_out と L_in の違いは？**
   正例についての和を log の外に置くのが L_out，内に置くのが L_in です．ResNet-50 の ImageNet で top-1 は 78.7% と 67.4% で，著者は以後 L_out だけを使います．（§3.2.2，表1）
3. **なぜ L_out のほうが良いのですか？**
   L_in では 1/|P(i)| の正規化が log の内側で加法定数になって勾配に効かず，正例の偏りを受けやすい，と説明されています．ただし著者はこれを conjecture，つまり推測と書いています．（§3.2.2，式4〜6）
4. **温度は何をしますか？**
   類似度を割る尺度で，勾配は 1/τ に比例します．下げると難しい正例・負例を重視する形になりますが，極端に低いと数値的に不安定です．原典は全実験で 0.1 に固定しています．（§4.5，補足 §10）
5. **射影 head とは何ですか？**
   encoder の 2048 次元の表現を 128 次元へ写す小さなネットワークで，損失はその出力に掛け，学習後は捨てます．最適な構造は今後の課題とされています．（§3.1）
6. **なぜ同じクラスを引き寄せるのですか？**
   同じクラスの全要素に揃った表現を与えると，式1より頑健なクラスタになる，というのが著者の説明で，実験で支持されるとしています．引き寄せると分類が良くなることの証明は，原典には書かれていません．（§3.2.2）
7. **自己教師ありの対照学習との違いは？**
   正例が「同じ画像の別の増強1つ」か「同じラベルの全サンプル」かの違いです．P(i) を増強だけに制限すると，SupCon の式は自己教師あり版に戻ります．（§1，補足 §9）
8. **triplet や N-pairs との関係は？**
   正例と負例を1つずつに制限すると triplet，正例を別サンプルの1つに制限して温度を外すと N-pairs の形になります．原典はこれらを SupCon の特殊な場合と位置づけています．（§3.2.3，補足 §8・§9）

### 本研究との関係

9. **なぜ射影を挟まないのですか？**
   学習可能な射影があると，表現を変えなくても射影の側だけで損失を下げられ，仮説が反証不能になるからです．それで第16層のベクトルに直接掛けました．これは原典の形からの逸脱で，生の表現に直接掛ける形は原典では検討されていません．（研究文書 §3.3，原典 §3）
10. **温度を学習可能にしたのはなぜですか？ 原典は固定です．**
    はい，原典は 0.1 の固定で，学習可能な温度は原典には書かれていません．本研究は初期値 0.07 の学習可能なパラメタとし，9回の学習すべてで同じ設定なので，温度の効果は切り分けていません．（原典 §4.5，研究文書 §3.2・§7）
    ※研究文書には，学習可能にした理由も，0.07 を選んだ理由も書かれていない．0.07 は原典の値でもない．
11. **署名をクラスとして扱ってよいのですか？**
    損失に要るのは同じか違うかのラベルだけなので，形式としては使えます．ただし原典のクラスは画像のカテゴリで，署名が解き方の正しい形式化かは検証していません．ablation が示すのはラベルに意味が要ることまでで，署名である必要があるかは未確認です．（研究文書 §3.1・§4.2・§7）
12. **同署名の類似度も下がったのに，なぜ「近づける」と言うのですか？**
    「近づける」は損失が指す向きの説明で，測定の結果ではありません．実測では同署名が 0.945 から 0.745 へ下がり，異署名が 0.918 から 0.225 へそれ以上に下がったので，分離度が 0.82 から 2.38 へ上がりました．正確には「同署名だけを相対的に近づけた」です．（研究文書 §3.2・§4.4）
13. **正例が1件なら，SupCon ではなく N-pairs ではないですか？**
    形としてはそのとおりです．16署名から2問ずつ引くので正例はちょうど1件で，増強の2視点も作らないため，原典の補足 §9 の N-pairs の形に温度を足したものにあたり，L_out と L_in の差も消えます．原典は N-pairs を SupCon の特殊な場合と位置づけています．（研究文書 §3.2，原典 補足 §9）
14. **原典のバッチは 6144 です．32 で足りますか？**
    原典は大きなバッチを要することを代償に挙げていて，32 での性能は原典には書かれていません．本研究の目的は分類精度ではなく，B = 32 で分離度と読み取り精度が上がることを3つの乱数種で確かめました．バッチの大きさは変えていません．（原典 補足 §8，研究文書 §4.1・§4.4・§5.1）

---

**確認**：2026-08-09．ar5iv による HTML 全文を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．

**照合**：2026-11-06．arXiv の PDF（ヘッダは v5，Change Log は Version 6 まで）の本文と補足の全23頁を，pdftotext の文で読み直した．§3.2.2 の不等号，表1〜3 は頁の画像でも確かめた．NeurIPS 版の本文の PDF と逐語引用を突き合わせ，食い違いは上に書いた．引用の中の数式の記号は，文字へ写したものである．確かめていないのは，図3〜7 の曲線の読み取り（値は使っていない），NeurIPS 版の補足である．研究文書についての答えは，研究文書の 2026-11-06 時点の記述による．

**再確認**：2026-11-09．ポスターの工夫の行を書き直すため，arXiv の PDF（v5）を取り直して pdftotext の文で，要旨（同じクラスを引き寄せること），§3 の冒頭（射影の出力に損失を掛けること），§3.1 の射影ネットワークの定義と学習後に捨てること，§4 の冒頭（画像分類の評価）を読み直した．「ポスターとの対応」の工夫の2行の引用は，この文から写した．
