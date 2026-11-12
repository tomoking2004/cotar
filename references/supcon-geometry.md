# Dissecting Supervised Contrastive Learning

**著者**：Florian Graf，Christoph D. Hofer，Marc Niethammer，Roland Kwitt（Graf・Hofer・Kwitt はザルツブルク大学 計算機科学科，Niethammer は UNC Chapel Hill）
**掲載**：ICML 2021（Proceedings of the 38th International Conference on Machine Learning，PMLR 139，pp. 3821–3830）．arXiv:2102.08817 v4，2023-03-02
**原文**：[PMLR](https://proceedings.mlr.press/v139/graf21a.html)・[arXiv](https://arxiv.org/abs/2102.08817)

**交差エントロピーも教師あり対照損失も，各クラスの表現が一点に縮み，その点が球に内接する正則単体の頂点をなすとき最小になる——最小化解の幾何は同じだが，最適化の振る舞いは大きく違う．**

## 主張

- 穏やかな仮定のもとで（"under mild assumptions"），交差エントロピー（CE）と教師あり対照損失（SC）の下界を示し，**どちらも等号は，各クラスの表現が一点に崩壊し，その点が正則単体の頂点をなすときに限って成り立つ**ことを証明する（CE ではこれに分類器の重みの条件が加わる）．
- 実験では，この配置が実際の学習で達成されること，最適に近い状態に達することが通例よい汎化性能を示すこと（要旨："typically indicates good generalization performance"）の経験的な証拠を示す．§5.2 では，この状態に近づくことが汎化性能と正に相関することを図が "suggests" と述べ，一般的な現象か，理論的な裏づけがあるかは今後の課題とする．
- ただし二つの損失の最適化の振る舞いは大きく異なる．ランダムに入れ替えたラベルの割合を増やすと，SC がデータに完全に合うまでの反復数は**超線形**に増える．著者はこれを，主にバッチ内の事例どうしの相互作用の結果だと論じる（"We argue that …"，§5.3）．

## 手法

### 記号

- N 個の表現 `z_n ∈ Z`，ラベル `y_n ∈ [K]`，表現の次元 `h`
- **均衡ラベル**：各クラスがちょうど N/K 個（§3 を通じた仮定）
- **`K ≤ h + 1`**：正則単体が存在するための必要十分条件（注意1．§3 を通じた仮定）．これが破れても §3 の下界は成り立つが，等号は達成されない（"still hold, but are not tight"）．
- **理想的な encoder**：表現のどんな幾何的配置も実現できる写像を仮定し，表現 Z を N 個のラベル付きの点の自由な配置として扱う（§2）．

### 正則単体（定義3）

`ζ_1, …, ζ_K ∈ R^h`（`K ≤ h+1`）が半径 ρ の球に内接する正則単体の頂点をなすとは，

- (S1) 和がゼロ
- (S2) 各ノルムが ρ
- (S3) 対ごとの内積がすべて等しい

単位球に内接する K 頂点の正則単体では，頂点どうしの内積は `−1/(K−1)`（§5.2 の本文）．

### SC 損失（定義2）——温度は半径に吸収

表現空間は**半径 ρ_Z の球面** `S^{h−1}_{ρ_Z}`．`𝓑` は `[N]` の添字からなる大きさ b の多重集合すべての集合（同じ例が一つのバッチに重なりうる）．

```
L_SC(Z; Y) = Σ_{B∈𝓑} ℓ_SC(Z; Y, B)

ℓ_SC(Z; Y, B) = − Σ_{i∈B, |B_{y_i}|>1}  (1/(|B_{y_i}|−1)) Σ_{j∈B_{y_i}\{i}}
                  log [ exp⟨z_i, z_j⟩ / Σ_{k∈B\{i}} exp⟨z_i, z_k⟩ ]
```

- `B_{y_i} = {j ∈ B : y_j = y_i}`．**バッチ内で正例のいないアンカー（|B_{y_i}| = 1）は寄与ゼロ**（脚注2・3）
- 内積に掛ける温度 `1/τ` は書かず，半径 ρ_Z に含める（脚注3）．実験では `ρ = 1/√τ`，`τ = 0.1`（§5.1）
- Khosla らの元の定義との違いとして著者が挙げるのは三つ：バッチを明示的に複製しない，多重集合に合わせた，温度を省いた（脚注3）．
- SC 損失は大きさ b の全バッチについての**和**，CE 損失は全事例の**平均**として定義する（§2.1）．

### 定理1（CE）

表現は**半径 ρ_Z の球体（ノルム上限 `‖z‖ ≤ ρ_Z`）**に置く．ラベルが均衡なら

```
L_CE(Z, W; Y) ≥ log( 1 + (K−1) exp( −ρ_Z (√K/(K−1)) ‖W‖_F ) )
```

等号は次のときに限る．

- (C1) **クラス崩壊**：`z_n = ζ_{y_n}`
- (C2) `ζ_y` が半径 ρ_Z の球に内接する正則単体をなす
- (C3) **分類器の重みが頂点の正の定数倍**：ある `ρ_W > 0` があって `w_y = (ρ_W/ρ_Z) ζ_y`

重みにノルム上限 `‖w_y‖ ≤ r_W` を課す場合（系1），`L_2` 正則化 `L_CE + λ‖W‖_F²` の場合（系2）も，等号条件は (C1)(C2) と，(C3) を置き換えた重みの条件（系1は `w_y = (r_W/ρ_Z) ζ_y`，系2は `w_y = (r_W(ρ_Z, λ)/ρ_Z) ζ_y`）である．

### 定理2（SC）

表現は半径 ρ_Z の球面上．**ラベルの配置 Y が均衡**なら（`K ≤ h+1` と `b ≥ 3` は §2・§3 と補遺 S1 の冒頭で仮定）

```
L_SC(Z; Y) ≥ Σ_{l=2}^{b} l · M_l · log( l − 1 + (b − l) exp( −K ρ_Z² / (K−1) ) )

M_l = Σ_y |{ B ∈ 𝓑 : |B_y| = l }|      （クラス y がちょうど l 回現れる (クラス, バッチ) の組の数）
```

等号は (C1) クラス崩壊 と (C2) 正則単体 のときに限る．定理1の (C3) にあたる分類器の条件は無い（定義2の SC 損失に分類器は現れない）．

### CE と SC の幾何の違い

| | CE | SC |
| --- | --- | --- |
| 表現の定義域 | 球体（ノルム上限） | 球面 |
| 最小化解の表現 | クラス崩壊＋正則単体 | クラス崩壊＋正則単体 |
| 分類器の条件 | 重みが単体の頂点の正の定数倍（C3） | なし（損失に分類器がない） |
| 損失の定義 | 事例ごとの損失の平均 | **バッチごとの損失の，全バッチについての和** |

### バッチ構成の影響

- "as the loss is defined over batches, we can not simply sum up per-instance losses to characterize the ideal N point configuration"（§3.2）．
- 各バッチの最小化配置はそのバッチのラベル構成に依存するので（図5），"there is no simultaneous minimizer for all batch-wise losses"．それでも定理2により，全クラスが単体の頂点に崩壊すれば，**全バッチの和としての下界は達成される**．
- 定理の仮定は**ラベルの配置 Y（データ全体）の均衡**である．損失は大きさ b の多重集合すべてについて和をとるので，クラスの現れ方が偏ったバッチ（図5の (7, 2, 0) など）も含まれる．バッチの構成は下界の中の `M_l` として入る．
- 著者は最適化の違いをこの性質に結びつけて論じる："While cross-entropy acts sample-wise, the supervised contrastive loss considers pair-wise sample relations, i.e., a batch is an atomic computational unit during stochastic optimization"（§6）．

## 実験設定

- **データ**：CIFAR10，CIFAR100
- **モデル**：ResNet-18 の線形分類器より前の全層（表現 512 次元で，`K ≤ h+1` を満たす）
- **比較**：CE（encoder に線形分類器をつなぎ CE 損失で学習）／SC（SC 損失で encoder を学習し，凍結して線形分類器を学習）／CE-fix（重みをあらかじめ正則単体の配置へ最適化して固定した分類器をつなぎ，CE 損失で学習）
- **最適化**：ミニバッチの SGD，momentum 0.9，`L_2` 正則化 10⁻⁴，100k 反復．"The batch-size is fixed to 256"，学習率は 0.1 から指数的に減衰
- **増強**（使う場合）：無作為切り出しと無作為の水平反転，それぞれ確率 1/2
- **SC の表現**：encoder の出力を半径 `ρ = 1/√τ`（τ = 0.1）の球面へ射影する．元の設定で通例使われる射影 head は外した——"by detaching the commonly used projective head"（§6）

## 結果

**理論の下界と実測の損失（ResNet-18，CIFAR10，増強あり，§5.2）**

| 損失 | 理論値 | 実測（100k 反復） | 実測（500k 反復） |
| --- | --- | --- | --- |
| CE | 7.64e-5 | 2.48e-4 | 2.27e-4 |
| SC | 824.487 | 824.731 | 824.523 |

SC の実測値は 1k 個の学習バッチで推定したもの．著者はこの比較から，SC で学習した表現のほうが理論上の最適配置に近く並ぶ**かもしれない**と述べ（"might arrange closer"），定理が扱うのは最小化解であって最適に近い等位集合ではないので，この仮説は "more of a first guess and not predicted by the theory" と断る．

**数値模擬（定理2の確認，§3.2・図6）**：単位円 S¹ 上の 12 点（K = 3，各クラス4点），b = 9，全 167,960 バッチ，SGD 100k 反復で，バッチごとの損失の平均の実測 **12.12016** に対し下界 **12.12015**．

**単体への近さ（§5.2・図7）**：指標は著者が "cosine similarity" と呼ぶ `γ(x, y) = 1 − cos⁻¹(⟨x/‖x‖, y/‖y‖⟩)/π`（値域 [0, 1]．通常の cosine そのものではない）で，クラス平均どうし，分類器の重みどうし，各表現とそのクラス平均の三つを見る．増強ありの学習で，SC は "an arrangement of the class means much closer to the ideal simplex configuration" と，学習表現のクラス平均まわりへのより強い集中をもたらす．著者は図7（右）のテスト誤差から，この状態に近づくことが汎化性能と正に相関すると "suggests" の語で述べる．ただし標準の ResNet の encoder 出力は ReLU により座標が非負で，クラス平均どうしの γ は 0.5 より小さくなれない．このずれは CIFAR100（K = 100）では目立たず，CIFAR10（K = 10）で顕わになる．

**テスト誤差［%］（図7 右）**

| | CIFAR10 増強あり | CIFAR10 増強なし | CIFAR100 増強あり | CIFAR100 増強なし |
| --- | --- | --- | --- | --- |
| CE | 6.3 | 16.4 | 27.0 | 41.8 |
| CE-fix | 6.2 | 14.7 | 26.3 | 41.3 |
| SC | 5.7 | 13.8 | 24.9 | 41.5 |

SC が最小でないのは CIFAR100 の増強なし（CE-fix が 41.3）だけである．

**500k 反復（補遺 S3・図9・10）**：500k 反復まで延ばすと最適状態への収束は進むが非常に遅く，SC は（ほぼ）どの評価時点でも CE より単体に近い表現を与える，と著者は述べる．

**ランダムラベル（§5.3・図8）**：ラベルを無作為に入れ替える割合を増やすと，訓練誤差ゼロに達するまでの反復数（上限 100k）は，どちらのデータセットでも CE と CE-fix ではおよそ線形に，SC では "remarkably superlinear" に増える．CIFAR10 では "SC cannot achieve zero error beyond 80% corruption"．

## 著者が明記した限界

- **元の SupCon の設定を簡略化した**："While we simplified the original setup of supervised contrastive learning, in particular, by detaching the commonly used projective head, we hope that our results provide a viable starting point for further analyses."（§6）
- **確率的最適化と，対の制約をもつ損失との相互作用の理論的理解は今後の課題**："a better theoretical understanding of the profound interaction between stochastic optimization and loss functions that capture pairwise constraints (rather than instance losses), could be a promising avenue to be explored"（§6）
- **`K > h+1` の領域は解いていない**："Studying the loss minimizing configurations in this regime is much harder."（注意1）
- **定理は最小化解だけを扱う**："our results only cover the loss minimizers and not (close to optima) level sets"（§5.2）

## 射程の外

- **定理の対象は最小化解の幾何**であり，理想的な encoder を仮定する．有限の学習でどこまで近づくかは実験（§5.2，補遺 S3）で見ているだけである．SC 損失を他の損失と足し合わせて学習する場合の記述は見当たらなかった（検索した語：auxiliar，weighted）．
- **ラベルはデータ全体で均衡**と仮定する．クラス頻度が偏るラベルについての定理・実験は見当たらなかった（検索した語：imbalanc，unbalanc，balanced）．
- **`K ≤ h+1` を仮定する．** これが破れると下界は成り立つが等号は達成されず，最小化配置は調べられていない（注意1）．
- **実験は画像分類**（CIFAR10・CIFAR100，ResNet-18）．言語モデル・VLM についての記述，encoder の出力以外の中間層の表現についての記述は見当たらなかった（検索した語：language，multimodal，vision-language，intermediate，hidden）．
- 表現は encoder の出力を球面へ射影したもので，射影 head は外してある（著者が簡略化と明記）．

---

**確認**：2026-11-13．arXiv abs ページ（版の履歴，Comments，Journal-ref），PDF v4（39頁．本文9頁＋参考文献，補遺 S1〜S3 は 11〜39頁），PMLR の書誌ページ（v139，pp. 3821–3830）を取得し，pdftotext の抽出テキストと，3・4・5・8・9頁のページ画像（定義1〜3，定理1・2，系1・2，§5.1〜5.2，図7の表）で，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．PDF v1（34頁）・v3（37頁）も取得し，数値模擬の値（12.12016／12.12015，167,960），バッチサイズ 256，テスト誤差の表，80% の記述が同じであること，理論値と実測の損失の比較（7.64e-5 ほか）は v3 と v4 で同じで v1 の抽出テキストには見当たらないことを確かめた．下書き（2026-10-11 作成）から直した主な点：「SC は下界にほぼ達する」の断定を著者の留保（"might"，"first guess"）に替えた．最適化の違いの原因を著者の論（"We argue"）として書いた．定理2の仮定の出どころ（`b ≥ 3`・`K ≤ h+1` は定理の文でなく §2・§3・補遺の冒頭）を分けた．SC 損失の式の符号の位置を原文に合わせた．「均衡なバッチは要求しない」を定義から言える形に直した．"cosine similarity" が著者の定義した γ であることを足した．原文に無い「CIFAR-10 で −1/9」を外した．`K > h+1` では下界が成り立つが等号が達成されないことを足した．下書きが確定できずに外していたテスト誤差の表を，ページ画像で行と列を確かめて載せた．射程の外の否定の記述を，全文検索に基づく「見当たらなかった」の強さに直した．データセットの表記を原文の CIFAR10／CIFAR100 に揃えた．**読めていないもの**：補遺 S1・S2 の証明（補題1〜14）は見出しと定理・系の再掲を確かめただけで，導出を追っていない．系2の `r_W(ρ_Z, λ)` を定める方程式は本文書に載せていない．図（2・4・7・8・9・10）の中の値は，本文と表に書かれたもの以外は照合していない．arXiv v2 と，PMLR に掲載された PDF そのものは取得していない（PMLR は書誌ページのみ）．
