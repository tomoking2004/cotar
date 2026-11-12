# Understanding Dimensional Collapse in Contrastive Self-supervised Learning

**著者**：Li Jing，Pascal Vincent，Yann LeCun，Yuandong Tian（Facebook AI Research）
**掲載**：International Conference on Learning Representations（ICLR 2022）．arXiv:2110.09348（v1 2021-10-18，v2 2022-02-02，v3 2022-04-23）．本文書は v3 に拠る
**原文**：[arXiv](https://arxiv.org/abs/2110.09348)・[OpenReview](https://openreview.net/forum?id=YevsQ05DEN7)

**負例で反発させる対照学習でも，埋め込みは低次元の部分空間に潰れる——強い augmentation と，多層の暗黙の正則化がその原因で，射影 head は表現空間をこの崩壊から守っている．**

## 位置づけ

本研究 §3.3 は射影 head を外して第16層の生の表現に SupCon を掛けるが，その理由（学習可能な射影があれば，表現を変えずに射影の側を変えるだけで損失を下げられる）は本論文の論点とは別である．本論文が示すのは，射影 head なしで学習した SimCLR では**表現空間そのもの**に次元崩壊が起きること，そして射影 head が表現空間をそれから守っていることである．ただし本論文の対象は，ImageNet で学習した ResNet-50 の自己教師の対照学習である（→ 射程の外）．本研究の設定で同じことが起きるかは，本論文からは言えない．

## 主張

自己教師の表現学習には，全表現が一点に縮む完全な崩壊（"whereby all representation vectors shrink into a single point"）のほかに，埋め込みが空間全体ではなく低次元の部分空間しか占めない**次元崩壊**（dimensional collapse）がある．これは非対照の手法で観察されてきたが，"while they prevent complete collapse, contrastive methods still experience a dimensional collapse"（§3）．貢献は三つである（§1）．

1. 対照型の自己教師学習でも次元崩壊が起きることを実験で示す．
2. その機構を二つ示す——（1）特徴の次元に沿った強い augmentation，（2）モデルを低ランクの解へ向かわせる暗黙の正則化．
3. 訓練可能な射影 head を使わずに表現空間を直接最適化する **DirectCLR** を提案する．

**表現**（representation）は ResNet-50 の出力で，"Each representation vector has a dimension of 2048."（図7の注）．§3 で測る SimCLR の埋め込み（embedding）は $d = 128$ 次元である．

## 手法

### 次元崩壊の測り方（§3）

埋め込み $\mathbf{z}_i$（$N$ 件）の共分散

$$
C = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{z}_i - \bar{\mathbf{z}})(\mathbf{z}_i - \bar{\mathbf{z}})^{\mathsf{T}},
\qquad \bar{\mathbf{z}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{z}_i
$$

を特異値分解し，特異値の対数を大きい順に並べる（スペクトル）．0 に落ちた特異値が潰れた次元にあたる．

### 機構1——強い augmentation（§4）

1層の線形モデル $\mathbf{z} = W\mathbf{x}$ に InfoNCE を掛け，augmentation を加法的な雑音とする（"the augmentation is an additive noise"）．最適化は運動量も重み減衰も無い SGD で，勾配流（学習率無限小）として扱う．

- **補題1**：$\dot W = -G$，$G = \sum_i(\mathbf{g}_{\mathbf{z}_i}\mathbf{x}_i^{\mathsf{T}} + \mathbf{g}_{\mathbf{z}'_i}\mathbf{x}'^{\mathsf{T}}_i)$．$\mathbf{z}_i = W\mathbf{x}_i$ からこれは $G = -WX$ と書ける（式5．$X$ の定義は式6）．
- **補題2**：$X = \hat\Sigma_0 - \hat\Sigma_1$．$\hat\Sigma_0 = \sum_{i,j}\alpha_{ij}(\mathbf{x}_i - \mathbf{x}_j)(\mathbf{x}_i - \mathbf{x}_j)^{\mathsf{T}}$ は重み付きのデータ分布の共分散，$\hat\Sigma_1 = \sum_i(1-\alpha_{ii})(\mathbf{x}'_i - \mathbf{x}_i)(\mathbf{x}'_i - \mathbf{x}_i)^{\mathsf{T}}$ は重み付きの augmentation 分布の共分散で，$\alpha_{ij}$ は埋め込みどうしの類似度の softmax である．
- augmentation の振幅が，$X$ が正定値かどうかを決める．augmentation が小さければ $X = \hat\Sigma_0 - \hat\Sigma_1 \succ 0$（§5.2）．
- **定理1**："With fixed matrix X (defined in Eqn 6) and strong augmentation such that X has negative eigenvalues, the weight matrix W has vanishing singular values."（証明は付録 B.3．$W(t) = W(0)\exp(Xt)$ による）
- **系1**："With strong augmentation, the embedding space covariance matrix becomes low-rank." $C = \sum_i W(\mathbf{x}_i-\bar{\mathbf{x}})(\mathbf{x}_i-\bar{\mathbf{x}})^{\mathsf{T}}W^{\mathsf{T}}/N$ なので，$W$ の特異値が消えれば $C$ も低ランクになる．

数値実験：等方ガウスの入力，$16\times16$ の $W$，augmentation の共分散を $\mathrm{block\_diagonal}(\mathbf{0}, k\ast I)$（塊の大きさは $8\times8$）として振幅 $k$ を変え，$W$ の特異値のスペクトルを描く（図3）．強い augmentation で $W$ の特異値が消える．

### 機構2——暗黙の正則化（§5）

強い augmentation が無く $X$ が半正定値に留まるなら，1層の線形モデルは次元崩壊しない．それでも深いネットワークでは実際に次元崩壊が起きる，という問いに，バイアス無しの2層線形 $\mathbf{z} = W_2W_1\mathbf{x}$（$W_1, W_2 \in \mathbb{R}^{d\times d}$，埋め込みは正規化しない）で答える．

- **補題3**：$\dot W_1 = -W_2^{\mathsf{T}}G$，$\dot W_2 = -GW_1^{\mathsf{T}}$（$G$ は補題1の定義）．この模型では $G = -W_2W_1X$ となる（式9）．
- $W_1 = U_1S_1V_1^{\mathsf{T}}$，$W_2 = U_2S_2V_2^{\mathsf{T}}$ とし，層間の**整列行列** $A = V_2^{\mathsf{T}}U_1$ を置く．
- **定理2（重みが整列する）**：すべての $t$ で $W_2(t)W_1(t) \neq 0$，$X(t)$ が正定値，$W_1(+\infty)$・$W_2(+\infty)$ が相異なる特異値を持つなら，$A \to I$．
- **定理3（特異値の動き）**：整列のもとで $\dot\sigma_1^k = \sigma_1^k(\sigma_2^k)^2(\mathbf{v}_1^{k\mathsf{T}}X\mathbf{v}_1^k)$，$\dot\sigma_2^k = \sigma_2^k(\sigma_1^k)^2(\mathbf{v}_1^{k\mathsf{T}}X\mathbf{v}_1^k)$．特異値の勾配が特異値自身に比例するので，最も小さい特異値の群は伸びが著しく遅い，と著者は説明する．
- **系2**："With small augmentation and over-parametrized linear networks, the embedding space covariance matrix becomes low-rank."

付録Cは，層を増やした線形の模型（$L = 1$〜$4$）と，層の間に ReLU を挟んだ模型でも同様の崩壊が起きることを，$16\times16$ の重みの数値実験で示す（図11）．こちらは定理ではなく実験である．

### 射影 head の役割と DirectCLR（§6）

**動機**：表現空間のスペクトルにも射影 head の有無で差が出る——"The dimensional collapse in representation space happens when the model is trained without a projector." 著者の結論は "The projector in contrastive learning is essential to prevent dimensional collapse in the representation space." である．機構2の理論から，線形の射影 head について二つの命題を立てる．

- **命題1**："A linear projector weight matrix only needs to be diagonal."（重みの直交成分は効かず，射影 head の振る舞いは特異値だけで決まる）
- **命題2**："A linear projector weight matrix only needs to be low-rank."

**DirectCLR**：訓練可能な射影 head を置かず，表現 $\mathbf{r}$ の先頭 $d_0$ 次元 $\mathbf{z} = \mathbf{r}[0{:}d_0]$ を正規化（$\hat{\mathbf{z}} = \mathbf{z}/|\mathbf{z}|$）して InfoNCE を掛ける（$d_0$ はハイパーパラメタ）．これは実質的に低ランク対角の射影を使うことにあたる．$\mathbf{r}[d_0{:}]$ には損失の勾配が掛からないが，最後の非線形の畳み込みブロックを通ると勾配はフルランクになり，隠れ層 $\mathbf{h}$ は全チャネルに勾配を受ける．順方向では "This hidden layer h is directly fed to the representation vectors via the residual connection."（§6.2）．図10の注は "Therefore, the entire representation vector r is meaningful." と結ぶ．

## 実験設定

- §3 のスペクトル：SimCLR（2層 MLP の射影 head），ImageNet で 100 エポック，ImageNet 検証集合の埋め込み（$d = 128$）．
- §6：ResNet-50（表現 2048 次元），ImageNet で 100 エポック（"standard training recipe"），評価は ImageNet での線形プローブの精度．

## 結果

### 次元崩壊の観察

SimCLR の埋め込みのスペクトルについて "We observe that a number of singular values collapse to zero, thus representing collapsed dimensions."（§3，図2）

### ImageNet の線形プローブ精度（表1，100 エポック）

| 損失 | 射影 head | 精度 |
| --- | --- | --- |
| SimCLR | 2層の非線形 | 66.5 |
| SimCLR | 1層の線形 | 61.1 |
| SimCLR | なし | 51.5 |
| **DirectCLR** | **なし** | **62.7** |

DirectCLR は1層の線形 head の SimCLR を上回る（著者の主張）．表の値では，2層の非線形 head の SimCLR には及ばない．図9の注によれば "Similar to SimCLR with projectors, DirectCLR is able to prevent dimensional collapse in the representation space."

### 射影 head の ablation（表2，SimCLR）

| 射影 head | 対角 | 低ランク | top-1 |
| --- | --- | --- | --- |
| なし | | | 51.5 |
| 直交 | | | 52.2 |
| 訓練可能 | | | 61.1 |
| 訓練可能な対角 | ✓ | | 60.2 |
| 固定の低ランク | | ✓ | 62.3 |
| 固定の低ランク対角 | ✓ | ✓ | 62.7 |

## 著者が明記した限界

- §4 の理論は線形のネットワークに限られる——"Our theory in this section is limited to linear network settings."
- 非線形の head までは説明しない——"But our theory is not able to fully explain why a nonlinear projector is able to prevent dimensional collapse."（§6.2 の Disclaimer）
- DirectCLR 自身も非線形の部分に頼っている——"DirectCLR also still relies on the mechanism of a nonlinear projector to prevent dimensional collapse"（同．文の前半．後半は，それを実質的に担うのがバックボーンの最後のブロックだと続く）．
- 定理2の仮定が成り立たず特異値が縮退しているとき，特異値分解は一意でなく，完全な整列は観察されない．重みを乱数で初期化した現実の場合には，整列行列は縮退した特異値の群ごとのブロックからなるブロック対角の行列に収束するのが観察されるだけだ，と著者は注記する（§5.3）．

## 射程の外

- **自己教師の対照学習**（正例は同じ画像の二つの augmentation）であり，ラベルで正例を定める SupCon への言及は見当たらなかった（検索した語：supervised contrastive，SupCon，Khosla，label）．
- 定理は**線形モデル**（1層・2層）についてである．多層と ReLU は付録Cの小さな模型の数値実験に留まる．Transformer を対象にした実験は見当たらなかった（transformer の語は参考文献の題に1度現れるだけである）．
- 実験の対象は画像の ResNet-50／ImageNet で，**VLM・言語モデルの中間層は扱わない．** GQA も使わない（検索した語：GQA，language model，vision-language）．
- 測ったのは **ImageNet で 100 エポック学習したモデル**であり，学習済みモデルの微調整に補助項として足す設定は見当たらなかった（検索した語：fine-tun，finetun）．
- 表現の評価は，共分散の特異値スペクトルと ImageNet の線形プローブの精度による．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴），arXiv PDF v1（15頁）・v2（18頁）・v3（17頁．付録A〜Fを含む全文）を pdftotext でテキスト化して読み，v3 の 4〜6頁と9頁はページ画像でも見て（式3〜11，定理2，表2のチェック印），ICLR 2022 の公式ページ（iclr.cc/virtual/2022/poster/6792．題・著者4名と，OpenReview の forum id YevsQ05DEN7 への参照を確認）と合わせて，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．表1の値は v1〜v3 で同じである．命題1・2と表2は v2 以降，§6.2 の Disclaimer（限界の引用2件）は v3 にだけ在る．下書き（2026-10-11 作成）から直した主な点：次元崩壊が「非対照の手法の問題とされてきた」を原文どおり「観察されてきた」へ；「augmentation が強いと $X$ は負の固有値を持つ」を定理1の条件としての書き方へ；系1を原文の文で引用；「非零の塊は 8×8」を原文どおり「塊の大きさ」へ；射程の外の「非線形は扱わない」を，付録Cが多層と ReLU の模型を数値実験で扱う事実に合わせて直し，付録Cの記述を手法に足した；「事前学習を最初から行う」（原文に from scratch の語は無い）を「ImageNet で 100 エポック学習したモデル」へ；「表現がモデル自身の出力に使われるかは問わない」を，原文が用いた評価の列挙へ；"Therefore, the entire representation vector r is meaningful." の出どころを図10の注と明記；否定の主張に検索した語を添えた；位置づけに，対象の違いから本研究へはそのまま移せない旨を足した．**読めていないもの**：OpenReview の forum ページ（ブラウザ認証の画面で取得できず．査読・最終版 PDF は見ていない）．付録A・Bの証明は式の導出を一行ずつ検算してはいない．図（2・3・5・6・7・9・11・12）の中の値は，本文と注に書かれたことだけを照合した．DirectCLR で用いた $d_0$ の値は本文・付録に書かれておらず（付録Eの図12が掃引を示すのみ），載せていない．
