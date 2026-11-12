# AtP*: An efficient and scalable method for localizing LLM behaviour to components

**著者**：János Kramár，Tom Lieberum，Rohin Shah，Neel Nanda（Google DeepMind）
**掲載**：arXiv のみ確認．arXiv:2403.00745 v1，2024-03-01
**原文**：[arXiv](https://arxiv.org/abs/2403.00745)

**活性パッチの一次近似（AtP）は速いが，attention の飽和と効果の打ち消し合いで偽陰性を生む——その二つに手当てをした AtP* を提案し，残る偽陰性の効果の大きさに統計的な上界を与える診断法を添える．**

## 主張

- 活性パッチ（activation patching）は振る舞いを部品へ直接帰属させるが，網羅するには部品数に比例する費用が要り，大規模言語モデルでは高くつく（要旨）．
- 勾配に基づく近似 **Attribution Patching（AtP）** は速い．著者は AtP を，活性パッチを速める他の手法と比べ，"AtP significantly outperforms all other investigated methods, with AtP* providing further significant improvement" と述べる（要旨）．
- AtP には**偽陰性を生む二種類の失敗モード**がある．AtP* はその二つに手当てをしたもので，さらに**残る偽陰性の大きさを統計的に上から押さえる診断法**を添える（1節の貢献）．

## 手法

### 活性パッチ（基準）

`x^clean` は振る舞いが起きるプロンプト，`x^noise` は介入に使う参照プロンプト，`L: ℝ^V → ℝ` は振る舞いを測る指標（脚注2の例：次トークンの予測損失，正誤トークンの対数確率の差，正解トークンの確率など）．

```
I(n; x^clean, x^noise) = L(M(x^clean | do(n ← n(x^noise)))) − L(M(x^clean))        （式2）
c(n) = |𝓘(n)|,   𝓘(n) = E_{(x^clean, x^noise)~𝒟} [ I(n; x^clean, x^noise) ]         （式1）
```

方向は **noising**（clean プロンプト上のノードを，参照プロンプトでの値に置き換える）："We follow Chan et al. (2022); Wang et al. (2022) in using noising, however denoising is also widely used in the literature"（2.1節）．

### Attribution Patching（AtP）＝一次近似

`n(x^noise) ≈ n(x^clean)` のまわりで `I` を一次の Taylor 展開する（原文は Figurnov ら 2016，Molchanov ら 2017，Nanda 2022 を挙げ，この手続きを Nanda 2022 に従って Attribution Patching と呼ぶ）．

```
Î_AtP(n) = (n(x^noise) − n(x^clean))ᵀ · ∂L(M(x^clean))/∂n |_{n = n(x^clean)}       （式4）
ĉ_AtP(n) = E[ |Î_AtP(n; x^clean, x^noise)| ]      （式5．絶対値は期待値の内側）
```

**一つのプロンプト対につき，前向き2回と後ろ向き1回で全ノードの推定値が出る**（2.2節）．ただし著者の手順では，AtP の推定値の大きい順にノードを活性パッチで検証する（verification）．検証は1ノードずつ行い，費用は前向き計算の回数で数える．

### 失敗モード1：attention の飽和 → QK fix

- AtP は各活性での勾配が介入下の真の振る舞いを反映することに依る．**clean 側の前活性が活性化関数の平坦な領域にあり（例：attention 重みが飽和している），noise 側はそうでない**とき，局所の勾配は効果をよく近似しない（3.1節）．key と query は attention の softmax に入るので，これが起きる（Figure 3：端点の一方または両方が softmax の飽和域にあると，attention 確率の線形近似は "a particularly poor approximation"）．
- **修正**：パッチした query（または key）で **attention の softmax を厳密に計算し直し**，そこから先だけを線形近似する．query については clean の key と noise の query から `attn(n)_patch` を作り，

```
Î^Q_AtPfix(n) = (attn(n)_patch − attn(n)(x^clean))ᵀ · ∂L(M(x^clean))/∂attn(n)        （式7）
```

- 費用：key 側は素朴に計算すると attention head・プロンプト対ごとに O(T³) flops だが，attention logit を併せてキャッシュする効率版（付録 A.2.1）は O(T²)．**QK fix 全体で transformer の前向き2回未満**（前向き計算は MLP の計算も含むため）．

### 失敗モード2：打ち消し合い → GradDrop

- 総効果が**直接効果と間接効果の和**で，両者がほぼ打ち消すとき，間接項の推定の小さな乗法的誤差によって，推定の絶対値が真の効果より桁違いに小さくなりうる（3.1節）．
- **修正（GradDrop）**：各層 ℓ ∈ {1,…,L} について，その層の残差への寄与にかかる勾配をゼロにして（それより前の層への伝播も含む）AtP の推定を計算し，絶対値を層について平均する．直接効果の経路の寄与が変わらないよう L/(L−1) を掛ける（その経路は，ノード自身の層を落としたときにゼロになる）．

```
ĉ_AtP+GD(n) = E[ (1/(L−1)) Σ_ℓ |Î_AtP+GD_ℓ(n)| ]        （式11）
```

- 費用：AtP の費用に加えて，同じ clean の前向き計算の中間活性からの後ろ向き計算 L 回（中間結果の再利用で (L+1)/2 に減らせる，脚注4）．**AtP* ＝ AtP + QK fix + GradDrop**（Figure 2 と Figure 4 の説明）．

### 偽陰性の扱い

- **偽陽性は検証で分かる，偽陰性が問題**："false positives are typically not a big issue; they are simply revealed during verification"．一方 "false negatives are not so easy to remedy without verifying all nodes"（4.2節）．
- **診断法（3.2節）**：AtP* の推定値の上位 K 個（Figure 6 では K = 1024）に入らないノードについて，無作為な部分集合を丸ごとパッチする subset sampling（Algorithm 1）で，ノードが部分集合に入る／入らない条件での効果の平均・標本標準偏差・件数を集める．各ノードで帰無仮説 |𝓘(n)| ≥ θ に対し片側の Welch の t 検定を行い，全ノードの最大の p 値を取り，これを反転して所与の信頼水準（Figure 6 では 90%・99%・99.9%）を満たす最小の θ を求める．**＝ AtP* が見落としたかもしれないノードの効果の大きさの上側信頼限界．**
- 検定は母集団がほぼ不偏・正規で歪んでいないことに依る（脚注6）．追加の計算を要するが，より信頼できるかもしれない代替としてノンパラメトリックなブートストラップ検定を挙げる（同）．

## 実験設定

- **モデル**：Pythia の 410M〜12B パラメタ．検証済み再現率の費用の図（Figure 1・7・8）は Pythia-12B．
- **ノード**：NeuronNodes（MLP の各ニューロン）と AttentionNodes（各 head の query・key・value と，線形層の前の head ごとの attention 出力）．どちらもトークン位置ごとに別のノードとする．規模の例として，Chinchilla 70B で長さ 1024 のプロンプトならニューロンノードは 2.7·10⁹（1節）．
- **プロンプト**：単一の対として IOI-PP・CITY-PP と，The Pile から無作為に選んだ RAND-PP．分布として，6名に絞った IOI（6 × 5 × 4 = 120 対，AttentionNodes で評価）と，不定冠詞 ' a'／' an' を出力させる A-AN（NeuronNodes で評価）．
- **指標**：clean プロンプトの目標トークンの負の対数確率．費用は前向き計算の回数．要約の指標として，上位ノードを真の寄与の大きい順に検証するオラクルに対する相対費用を使う（4.2節）．
- **比較手法**：Iterative（活性パッチを順に行う），Subsampling，Blocks，Hierarchical（3.3節）．

## 結果

- **著者によれば，AtP は調べた他の手法すべてを有意に上回り，AtP* はさらに改善する**（要旨）．ただし GradDrop には前払いの費用があるため，集約した相対費用では AtP* が不利に見えることがある（Figure 2 の説明，4.2節）．相対費用の値は図（Figure 2・9）で示され，本文に数値はない．
- **GradDrop が直した偽陰性の直接効果比**：打ち消し合いが原因である証拠として，著者は直接効果と総効果の比（direct effect ratio）を計算する．Figure 5（Pythia-12B，CITY-PP，NeuronNodes）で GradDrop が直した最も大きな偽陰性の比は **5.35，12.2，0（直接効果なし）**．比の中央値は，全ノードで数えると 0，直接効果をもつノードだけで数えると 0.77（3.1.2節）．
- Figure 1 に現れる AtP* の偽陰性は "a small minority of nodes" である（Figure 7 の説明．目標再現率 90%）．具体的な件数は本文に示されない．
- **推奨（5.4節）**：AtP が適切な場合，最良の変種はおそらく，単一のプロンプト対には AtP*，分布に対しては AttentionNodes で AtP+QKFix，NeuronNodes で AtP．

## 著者が明記した限界

- **プロンプトの範囲**（5.1節）："We only considered a small set of prompt pair distributions, which often were limited to a single prompt pair, since evaluating the ground truth can be quite costly."
- **ノードの粒度**（5.1節）："we only considered relatively fine-grained nodes"．NeuronNodes では MLP ニューロンを分析の基本単位とした．
- **他のモデル**（5.1節）：Pythia 以外の LLM にも結果が移る見込みは高いとしつつ，SotA 規模のモデルや標準的な decoder-only transformer から大きく外れる構造では特に，"cannot rule out qualitatively different behavior without further evidence"．
- **denoising は予備的**（限界の節ではなく 5.2節 Extensions/Variants）：denoising は "moderately different results" を生みうるとし，付録 B.4 に予備的な証拠を示して，詳しい検討は今後に残す．付録 B.4 は，IOI で効果を denoising で評価すると AtP* の性能が "notably worse" に見え，満足のいく説明はまだ無いと書く．

## 射程の外

- **VLM は扱わない．** 実験はすべて Pythia（410M〜12B）の decoder-only の言語モデルで行う（全文で vision・visual・image・multimodal を検索し，該当なし）．
- **学習時の介入は扱わない．** 手法はすべて，プロンプト対の上での前向き・後ろ向き計算による帰属（局所化）である（全文で fine-tun を検索し該当なし．training は参考文献の題名にだけ現れる）．
- **帰属の単位はニューロンと attention の query・key・value・head 出力**である．ニューロン活性の空間の中の方向を見つけることは "out of scope for this paper" とする（5.1節）．残差ストリームのノードについては，層正規化を固定する変種の誤差を付録 C.1 で解析するにとどめ，実験はしていない．
- **評価はプロンプト対（clean／noise）上の指標 L の変化**である．表現から何が線形に読めるか（probing）は測っていない（probe は全文に現れず，probing は参考文献の題名に1回だけ現れる）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴．版は v1 のみ，Comments と Journal-ref の記載なし）と arXiv PDF v1（46頁，付録 A〜D を含む全文）を取得し，pdftotext でテキスト化して全文を読み，数式と記号は 4・5・7・8・9頁の画像で確かめて，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．掲載先は Semantic Scholar と DBLP の照会が応答せず（429・ボット判定）確かめられなかったので，arXiv のみ確認とした．下書き（2026-10-11 作成）から直した主な点：(1) 結果の見出し「GradDrop が拾った偽陰性は直接効果が大きい」を，原文の量（直接効果比．三つのうち一つは直接効果なしの 0）に合わせた．(2) denoising の記述は 5.1節 Limitations ではなく 5.2節 Extensions/Variants に在ると明記し，付録 B.4 の所見を足した．(3) RAND-PP だけが The Pile 由来であると分かるように直した．(4) 冒頭の「統計的に抑える」を「上界を与える」に，「桁違いに小さくなる」を原文の強さ「なりうる」に，「部品数に依らない」「AtP の命名は Nanda 2022 のブログ記事」を原文に在る表現に直した．(5) 逐語引用の末尾の句点（原文では引用の後に文献の括弧が続く）を引用の外へ出した．(6) QK fix の式を式7の形にし，各式と各主張に式番号・節番号を足した．数値の誤りは無かった．**読めていないもの**：図の中の値（Figure 1・2・4〜17 の曲線と散布図は，説明文と本文に書かれたことだけを照合した）．付録 A.1.1 の式15，付録 C.2 の Table 2 は pdftotext で崩れており，本文書はその内容に触れていない．Nanda 2022 など引用先の文献そのものは読んでいない．
