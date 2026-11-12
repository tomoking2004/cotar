# Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere

**著者**：Tongzhou Wang，Phillip Isola（MIT Computer Science & Artificial Intelligence Lab (CSAIL)）
**掲載**：ICML 2020（Proceedings of the 37th International Conference on Machine Learning，PMLR 119:9929–9939）．arXiv:2005.10242 v10，2022-08-15．**以下の数値と引用は arXiv v10 のもの**である（PMLR 掲載版との違いは「版による違い」に記す）
**原文**：[PMLR](https://proceedings.mlr.press/v119/wang20k.html)・[arXiv](https://arxiv.org/abs/2005.10242)

**対照損失は，正例対の特徴を近づける alignment と，超球上の特徴の分布を一様にする uniformity の二つを，負例数を無限にした極限で最適化する．**

## 主張

- 単位超球上に正規化した特徴について，対照損失に関わる二つの性質を特定し，それぞれを最適化可能な指標として定義する（要旨，§1）．
  - **alignment**：正例対をなす二つのサンプルが近い特徴へ写ること（§4）．
  - **uniformity**：特徴ベクトルが単位超球上でおおよそ一様に分布すること（§4）．
- "We prove that the contrastive loss optimizes for alignment and uniformity asymptotically."（§1 の貢献の列挙）
- 二つの指標は下流タスクの性能と強く一致する（"strong agreement between both metrics and downstream task performance"，§1）．
- 二つの指標を**他の損失なしで直接最適化**すると，下流タスクで対照学習と同等以上の性能になる（"comparable or better performance at downstream tasks than contrastive learning"，§1）．

## 手法

### 対照損失（§3）

encoder `f` はデータを単位超球 `S^{m−1}` 上の特徴へ写す．`p_pos` は正例対の分布，`p_data` はデータ分布，`M` は負例数（固定の正の整数），`τ > 0` は温度．

```
L_contrastive(f; τ, M) ≜ E_{(x,y)~p_pos, {x_i^-}_{i=1}^{M} ~iid p_data}
    [ −log  e^{f(x)ᵀf(y)/τ} / ( e^{f(x)ᵀf(y)/τ} + Σ_i e^{f(x_i^-)ᵀf(y)/τ} ) ]
```

分析の大半を占める STL-10 と NYU-Depth-V2 の実験では，負例はミニバッチ内の他のサンプルである（§5）．ImageNet-100・ImageNet は MoCo 系，BookCorpus は Quick-Thought Vectors 系の手法を，L_align と L_uniform も使えるように改めて学習する．MoCo は memory queue と momentum encoder を導入する（§5）．

### 二つの指標（§4.1）

```
L_align(f; α)   ≜ E_{(x,y)~p_pos} [ ‖f(x) − f(y)‖₂^α ],                    α > 0

L_uniform(f; t) ≜ log E_{x,y ~iid p_data} [ e^{−t‖f(x) − f(y)‖₂²} ],        t > 0
```

- **完全な alignment**：`(x, y) ~ p_pos` についてほとんど確実に `f(x) = f(y)`（§4.2）．
- **完全な uniformity**：`x ~ p_data` についての `f(x)` の分布が，`S^{m−1}` 上の一様分布 `σ_{m−1}`（§4.2）．
- uniformity は **Gaussian potential kernel（RBF カーネルとも呼ばれる）** `G_t(u, v) = e^{−t‖u−v‖₂²}` の対ごとの平均の対数である．一様分布はこの対ポテンシャルの期待値を最小にする**唯一の分布**であり（命題1），N 点の最小化配置に対応する正規化した数え上げ測度は一様分布へ弱*収束する（命題2）．
- 対ごとの内積の平均やユークリッド距離の平均は，平均ゼロの分布なら何でも最適にしてしまう，と著者は述べる（§4.1.2）．
- L_uniform の値域は `[−2t + log ₀F₁(; m/2; t²), 0]` で，下界は完全に uniform な encoder だけが，上界 0 はほとんど確実に固定の特徴ベクトルを出す退化した encoder だけが達成する（付録 A.1.1 系1）．

### 定理1——対照損失の漸近形（§4.2）

τ を固定して M → ∞ のとき，

```
lim_{M→∞} [ L_contrastive(f; τ, M) − log M ]
  = −(1/τ) E_{(x,y)~p_pos} [ f(x)ᵀf(y) ]                                    ← 第1項
    + E_{x~p_data} [ log E_{x^-~p_data} [ e^{f(x^-)ᵀf(x)/τ} ] ]             ← 第2項
```

1. 第1項は f が**完全に aligned のとき，かつそのときに限り**最小．
2. 完全に uniform な encoder が存在すれば，それらが第2項の**厳密な最小化解**をなす．
3. 極限からの絶対誤差は `O(M^{−1/2})` で減衰する（v10 の値．PMLR 掲載版では `O(M^{−2/3})`）．

### 退化した解と，二つの性質の両立（§4.2，§5）

- STL-10 で L_align と L_uniform の重みを振った実験（図7）では，L_align の重みを L_uniform よりずっと大きくすると "degenerate solution occurs and all inputs are mapped to the same feature vector (exp L_uniform = 1)"．
- `p_data` と `p_pos` が有限のデータセットの増強から作られる場合，完全に aligned かつ完全に uniform な encoder は存在しえない．完全な alignment は，一つの要素のすべての増強が同じ特徴ベクトルを持つことを意味するからである（§4.2）．
- 完全な uniformity も常に達成できるとは限らない（例：データ多様体の次元が特徴空間 `S^{m−1}` より低いとき）（§4.2）．

## 実験設定

v10 の §5："We optimize a total of 304 STL-10 encoders, 64 NYU-DEPTH-V2 encoders, 45 IMAGENET-100 encoders, and 108 BOOKCORPUS encoders without supervision."

| データ | encoder 数 | 構成 | 下流評価 |
| --- | --- | --- | --- |
| STL-10 | 304（v9 までと PMLR 掲載版は 306） | AlexNet 系 | encoder 出力または中間活性（fc7）の上の線形分類・5-NN |
| NYU-Depth-V2 | 64 | CNN | 畳み込み層の後の中間活性（conv5・conv4）の上の深度推定（MSE） |
| ImageNet-100 | 45 | MoCo 系，ResNet50 | 最終手前の層の活性の上の線形分類 |
| BookCorpus | 108 | Quick-Thought Vectors 系，RNN（双方向 GRU）の文 encoder | encoder 出力の上の MR・CR の二値分類（ロジスティック分類器） |

- 正例対は，画像では同じ画像の独立な二つの増強，BookCorpus では隣り合う文である（§5）．
- 三つの損失の重み（ゼロを含む），温度 τ，α ∈ {1, 2}，t ∈ {1, 2, …, 8}，バッチサイズ，埋め込み次元，エポック数と学習率，初期化（一からか，事前学習済み encoder からか）を振る（§5）．
- 評価では encoder を凍結し，その上で分類器・深度推定器を学習する（付録 B）．

## 結果

**指標と性能の一致**（§5，図5）："We observe that the two metrics strongly agrees the representation quality overall."——最良の encoder は，L_align と L_uniform がともに低いものである．相関係数の数値は見当たらなかった（検索した語：correlation，Pearson，Spearman．§5 の本文は図9 について "still correlated with the downstream task performances" と書くが，数値は無い）．

**直接最適化と対照損失の比較**（表1〜5．値は検証セット）

| データ | 最良の L_contrastive のみ | 最良の L_align と L_uniform のみ |
| --- | --- | --- |
| STL-10（出力＋線形，精度） | 80.46%（τ = 0.19） | **81.15%**（0.98·L_align(α=2) + 0.96·L_uniform(t=2)） |
| STL-10（fc7＋線形，精度） | 83.89% | **84.43%** |
| NYU-Depth-V2（conv5，MSE↓） | 0.7024（0.5·L_contrastive，τ = 0.1） | **0.7014**（0.75·L_align(α=2) + 0.5·L_uniform(t=2)） |
| ImageNet-100（top-1） | 72.80%（τ = 0.07） | **74.60%**（3·L_align(α=2) + L_uniform(t=3)） |
| BookCorpus MR | **77.51%**（τ = 0.075） | 73.76%（0.9·L_align(α=2) + 0.1·L_uniform(t=5)） |
| BookCorpus CR | **83.86%**（τ = 0.05） | 80.95%（同上） |
| ImageNet（MoCo v2，top-1） | 67.5% ± 0.1%（τ = 0.2．MoCo v2 公式実装の値，5回の平均と標準偏差） | **67.69%**（3·L_align(α=2) + L_uniform(t=3)） |

- 表1〜4 の「最良」は，学習セットの交差検証で選んだものである（表5 は選抜でなく，MoCo v2 と，その L_align・L_uniform 版の2行の比較）．表1（STL-10）は，出力 128 次元・バッチ 768 で一から学習した 150 個の encoder の中から選ぶ．表2 は出力 128 次元・バッチ 128 の 64 個，表3 は出力 128 次元・バッチ 128 の 45 個，表4 は出力 1200 次元・バッチ 400 の 108 個から選ぶ（各表の注）．
- 表1・2 について著者は，L_align と L_uniform だけで学習した encoder が対照損失のものを "consistently outperform" すると述べる（表2 の conv4 の列では，対照損失 0.7575 に対し 0.7592 で，対照損失の方が MSE が低い）．
- 表3・4 について著者は "directly optimizing them also leads to comparable or better representation quality" と述べる．表4（BookCorpus）の数値では，MR・CR とも対照損失のみの方が高い．
- 表1〜5 に載る L_align と L_uniform のみの設定は，すべて α = 2 で，t は 2・3・5 である．

**二つの性質はどちらも要る**（§5，図7・図8）：重みの組み合わせを振ると精度は逆 U 字（"inverted-U-shaped accuracy curve"）になり，"as long as the ratio between two weights is not too large (e.g., < 4)" 表現の質は比較的よいままで，重みの細かな選び方に敏感でない．準最適な温度（τ = 2.5）の対照損失で学習した encoder を微調整すると，一方の性質だけを最適化したときは，その指標は改善するがもう一方の指標と性能は悪化し，両方を最適化したときは表現の質が上がり続ける．

## 版による違い

- **encoder 数**：STL-10 は v9 までと PMLR 掲載版で 306，v10 で 304．v10 の変更履歴は，他の正則化項を使っていた 2 個を図5 と表8 から外したとし，L_align・L_uniform・下流性能の関係には影響しないと述べる．
- **定理1 の収束の速さ**：PMLR 掲載版と v1 は `O(M^{−2/3})`．変更履歴（2020-11-06）で `O(M^{−1/2})` に訂正された．
- **ImageNet 全体の MoCo v2 の結果（表5）**：変更履歴（2020-08-24）で加えられた．PMLR 掲載版には無い．

## 著者が明記した限界

原文に限界を掲げる節は無い．§6 は今後の課題として次を挙げる．

- **なぜ超球がよい特徴空間なのかは厳密には答えられていない**："the question of why the unit hypersphere is a nice feature space is not yet rigorously answered"．

## 著者が述べる含意

- 本論文の分析は対照学習と，超球上の alignment・uniformity との関係に絞ったものだが，表現学習で ℓ2 正規化が広く使われていることから，この結びつきはより一般的でありうる，と著者は述べる．より広い種類の表現を超球上の uniformity や alignment に関係づけることを，今後の方向として挙げる（§6）．
- 表1・2 の差について，有限個の負例しか使えないときには二つの性質を直接最適化する方が実用上優れうることを示唆する，と述べる（§5）．

## 射程の外

- **対象は教師なしの対照学習**であり，正例対は同じサンプルの二つの増強（画像）や隣り合う文（BookCorpus）で作る．クラスラベルで正例を決める対照損失は見当たらなかった（"supervised" の語は，図3 で比較に使う交差エントロピーの教師あり学習と，"without supervision" などに限られる）．
- **別の主損失に補助として足す設定は見当たらなかった．** encoder は L_contrastive・L_align・L_uniform の重み付き和だけで学習し，凍結した encoder の上で下流を評価する．
- **損失を掛けるのは encoder の出力で，ℓ2 正規化したもの**である（例外は BookCorpus の 108 個のうち，改変しない Quick-Thought Vectors で学習した正規化なしの 1 個．表11 の注）．下流評価には中間の活性（fc7，conv5 など）も使う．大規模言語モデルの中間層の表現は扱わない（"language model"，"hidden state" の語は本文に無い）．
- 言語の実験は BookCorpus の文 encoder（Quick-Thought Vectors 系）のみである．視覚言語モデルや VQA は見当たらなかった（検索した語：VQA，question answering，vision-language，multimodal，GQA）．
- 下流評価は線形分類・5-NN・深度推定・ロジスティック分類であり，生成のタスクは無い．

---

**確認**：2026-11-13．arXiv abs ページ（版の履歴 v1〜v10，Comments），arXiv の PDF v10（41頁．付録 A の証明と付録 B の実験の細部，表8〜11 を含む）を pdftotext でテキスト化して読み，数式は 3・5 頁を画像でも見た．版の違いを見るために arXiv の PDF v1（36頁）・v9（41頁）と，PMLR の書誌ページ，PMLR 掲載版の本文 PDF（11頁）・補遺 PDF（26頁）も取得し，encoder 数・収束の速さ・表の有無を検索で比べた．本文書の書誌・数値・逐語引用・節番号を一件ずつ v10 に照合した．下書き（2026-10-11 作成）から直した主な点：STL-10 の encoder 数の引用を 306 から v10 の 304 へ直し（306 は v9 までと PMLR 掲載版の値），版による違いの節を足した．「alignment だけなら定数 encoder で自明に達成される」は原文に無いので外し，付録の L_uniform の値域に置き換えた．「有限の M では誤差が残る」は著者が限界として書いていないので限界の節から外した．言語で対照損失が上回るという読みに，著者自身の記述（"comparable or better"）を並べた．評価時の encoder の凍結は付録 B に明記があるので戻した．実験設定の表から，全 encoder に共通でない次元・バッチの値を，表の選抜母集団の説明へ移した．**読めていないもの**：付録 A の証明は主張と系1 を確かめたのみで，導出の各行は追っていない．表8〜11 の個々の行（各 encoder の設定と値）と，図の中の値は照合していない．v2〜v8 は取得していない．
