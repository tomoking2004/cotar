# Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching

**著者**：Aleksandar Makelov，Georg Lange，Atticus Geiger，Neel Nanda（会議版の表記：Makelov と Lange は SERI MATS，Geiger はスタンフォード大学言語学科，Nanda の所属は書かれていない．Makelov と Lange は equal contribution）
**掲載**：ICLR 2024（poster），pp. 26486–26515．arXiv:2311.17030 v2，2023-12-06（v1 は 2023-11-28．arXiv 版の著者は Makelov・Lange・Nanda の3名で，Geiger は会議版にのみ載る．arXiv の Comments 欄は "NeurIPS 2023 Workshop on Attributing Model Behavior at Scale"）
**原文**：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/70b8505ac79e3e131756f793cd80eb8d-Paper-Conference.pdf)・[arXiv](https://arxiv.org/abs/2311.17030)

**部分空間を差し替えて出力が動いても，その部分空間が特徴を担っているとは限らない——出力から因果的に切り離された方向と，休眠している方向の和が，見かけ上の「特徴の在りか」を作る．**

本文書は会議版（30頁）に拠る．節・図・表・付録の番号は，断らないかぎり会議版のものである．会議版は脚注で arXiv 版を "full version" と呼ぶ．arXiv v2（58頁）は節立て・図番号・一部の記述が会議版と違い，違いが本文書の記述に関わる箇所ではその旨を書いた．

## 主張

- 部分空間への活性パッチ（DAS などで最適化した1次元方向の差し替え）は，**モデルの振る舞いを操作すること**と，**その背後の特徴を部分空間に帰属させること**を同時に狙う．著者はこの二つの狙いが乖離しうると示す（arXiv v2 の要旨："we demonstrate that these two aims diverge"．会議版の要旨はこの言い方をせず，素朴な部分空間介入が "interpretability illusions" を生みうると述べる）．
- 錯覚は**休眠（dormant）方向**と**因果的に切り離された（causally disconnected）方向**の組み合わせで生じる．前者は原理的には出力を意図した向きに動かせるが，二つのプロンプトで活性化のされ方が違わない．後者は二つのプロンプトを区別するが，下流のすべての部品の零空間にある（§3）．
- 実例は GPT-2 Small の IOI と GPT-2 XL の事実想起．さらに **1次元の活性パッチは，等価な rank-1 重み編集の存在を含意する**（§6.2）ので，rank-1 の事実編集は事実がその層に格納されているかに関わらず成功しうる．
- 一方で，**残差ストリーム**では DAS が忠実な方向を見つける成功例も示す（§5）．

## 手法

### 部分空間パッチ

単位ベクトル `v` に沿った1次元パッチは，プロンプト B の活性の `v` 成分だけを A のものに置き換える（§3，式1）．

```
act_B^patched = act_B + (p_A − p_B) v,   p_A = vᵀ act_A,  p_B = vᵀ act_B
```

一般の部分空間 U では，U への射影を他方の入力のものに置き換え，直交補空間 U^⊥ への射影はそのまま残す（付録 A.1）．

### 錯覚の機序

パッチが出力を意図どおりに動かすには，方向が **(a) 二つのプロンプトで異なって活性化され**，かつ **(b) モデルの出力と因果的につながっている**必要がある（§3）．錯覚の要は，`v` が (a) と (b) を**無関係な二つの方向から別々に得る**ことである．

| 成分 | 二つのプロンプトで値が変わるか | 下流が読むか | MLP 層の場合 |
| --- | --- | --- | --- |
| `v_disconnected` | 変わる | 読まない（下流のすべての部品の零空間にある） | `ker W_out`（出力射影の零空間）にある成分（§4.3） |
| `v_dormant` | 変わらない | 読む（原理的には出力を意図した向きに動かせる） | W_out の行空間にある成分（§4.3）．§7 の議論では `W_out⁺ v` |
| 和 | 変わる（disconnected 成分による） | 読む（dormant 成分による） | — |

和の方向に沿ってパッチすると，"the variation in the disconnected part activates the dormant part, which then achieves the causal effect"（§3）．**どちらかの成分だけに沿ったパッチはこの効果を生まない**——disconnected 成分だけでは定義により出力が変わらず，dormant 成分だけでは二つの例の活性化が似ているので効果が弱い（§3 脚注3）．

会議版の §3 は和を `v = v_disconnected + v_dormant` と書く．付録 A.2（Lemma A.1）は，全例の `v_dormant` への射影が一定値に等しいという仮定の下で，`v = v_disconnected cos α + v_dormant sin α` に沿ったパッチによる `v_dormant` 方向の射影の変化の期待値の大きさが α = π/4（cos α = sin α = 1/√2）で最大になることを証明する（理論的結果）．arXiv v2 は本文の式2でこの形を `v_illusory = (v_disconnected + v_dormant)/√2` と書く．

**零空間の正体**：MLP の隠れ活性が出力に効く経路は出力射影 W_out との積だけなので，`ker W_out` にある成分は MLP の出力を変えない（arXiv v2 §3.2）．GPT-2 Small では W_out ∈ R^{768×3072} なので，その核は少なくとも 2304 次元あり，MLP 活性の空間の 3/4 を占める（arXiv v2 §4.4．会議版は §7 で，MLP 層の内部次元が残差ストリームより大きいと述べる）．

### 分解による検証

DAS で得た方向 `v_MLP` を，**`ker W_out` への直交射影（nullspace 成分）**と**W_out の行空間への直交射影（rowspace 成分）**に直交分解し，それぞれだけでパッチして効果を比べる（§4.3）．著者は nullspace 成分を "causally disconnected component"，rowspace 成分を "causally relevant component" と呼ぶ．残差ストリームの方向では，rowspace 成分だけにしても効果が大きくは減らない（§5）．

## 実験設定

- **IOI**：GPT-2 Small．**第8層 MLP の隠れ活性・最終トークン位置**．Wang ら（2023）の IOI 回路では S-Inhibition head が第7・8層，Name Mover head が第9・10・11層にある（§4.1）．**著者は MLP8 が IOI に効かないという IOI 論文の分析を "convincing" とする**（付録 B.5："How certain are we that MLP8 doesn't actually matter for the IOI task? While we find the IOI paper analysis convincing, …"）．DAS で「名前の位置情報」を媒介する1次元方向を，第8層 MLP の活性（`v_MLP`）と第8層の残差ストリーム出力（`v_resid`）について最終トークンで求める——"We find that relatively small sample sizes (e.g., 500) are sufficient"（§4.2）．指標はすべて，学習に使わないテスト集合で報告する．
- **指標**（§4.3）：fractional logit difference decrease（FLDD）＝ `1 − logitdiff^P(x)/logitdiff(x)` の平均．0 はパッチが平均して logit 差に効果を持たないこと，100% 超はパッチによってモデルが IO より S の名前を好む例のほうが多いことを示す．interchange accuracy は，IO の名前の logit が S より高く，介入がそれを逆転させたプロンプトの割合．
- **事実想起**（§6.1，付録 D.1）：GPT-2 XL（1.5B）．COUNTERFACT の先頭 1000 例から，モデルが正しく想起する事実を残し，関係ごとに対をつくって **8 関係・40 対**を得る．プロンプト B の主語 s' の最終トークンから，A の主語 s の最終トークンへパッチする．計算上の困難のため，回転行列ではなく単位ベクトルを射影勾配法で直接最適化する．

## 結果

### IOI（Table 1）

| パッチ対象 | FLDD | interchange accuracy |
| --- | --- | --- |
| full MLP（MLP8 の隠れ活性の全体） | −8% | 0.0% |
| **`v_MLP`** | **46.7%** | 4.2% |
| `v_MLP` の rowspace 成分 | 13.5% | 0.2% |
| `v_MLP` の nullspace 成分 | 0% | 0.0% |
| full residual stream（残差ストリームの全体） | 123.6% | 54.8% |
| **`v_resid`** | **140.7%** | **74.8%** |
| `v_resid` の rowspace 成分 | 127.5% | 63.1% |
| `v_resid` の nullspace 成分 | 13.9% | 0.4% |
| `v_grad` | 111.5% | 45.1% |
| `v_grad` の rowspace 成分 | 106.47（会議版の表に % なし．arXiv v2 は 106.47%） | 40.6% |
| `v_grad` の nullspace 成分 | 2.2% | 0.0% |

- 表の数値は会議版と arXiv v2 で同じである．
- **`v_MLP` に沿ったパッチは FLDD 46.7% を生むが，`ker W_out` の成分を除くと 13.5% に，MLP の活性の全体をパッチすると −8% になる**（著者は −8% を "actually increasing confidence" と書く）．§4.3 の見出しは "The causally disconnected component of v_MLP drives the effect" である．
- 付録 B.5：MLP8 の出力のノルムに揃えて無作為に取った重みと偏りの MLP を用意する．層そのものは置き換えず，無作為な MLP の活性の上で DAS により部分空間を学習し，パッチ前後の無作為な MLP の出力の差を，実際の MLP8 の出力に足す．この設定で，`v_MLP` 方向よりさらに logit 差を下げる部分空間が見つかる（数値は示されない）．著者はこれを，`v_MLP` の存在が MLP8 の含む情報よりも**MLP8 がネットワークのどこにあるか**に依ることを示唆すると読む．

### 成功例：残差ストリーム（第8層の後・最終トークン，§5）

- `v_resid` は DAS で求める．`v_grad` は，Name Mover head 3本の，S と IO の名前の位置への attention スコアの差の勾配を，テスト分布の 2000 例について平均し，ℓ2 ノルム 1 に正規化したもの（§4.2）．ここでの零空間は，Name Mover head の query 行列を積んだ行列の核について取る——著者はこれを，残差ストリームの因果的に切り離された部分空間を決めるための "proxy" として使う（会議版は行列を R^{768×192}，arXiv v2 は R^{192×768} と書く）．
- **nullspace 成分を除いても効果は大きくは減らない**（`v_resid` 140.7% → rowspace 127.5%）．`v_MLP` と対照的である．
- 終端間の効果以外の検証もある：ABB と BAB のプロンプトで `v_resid` との内積が異なる head は **S-Inhibition head だけ**（付録 C，Figure 7）．著者によれば，パッチする位置の残差ストリームは計算の "full bottleneck" なので，`v_MLP` と違い，手前の部品が skip 接続でパッチを迂回してこの方向を活性化することがありえない．

### 事実想起（§6.1）

- 著者は IOI の錯覚の方向と同じ定性的現象の "a stronger version" を報告する：(i) DAS の方向は強い因果効果をもつ（o を o' に変える），(ii) **W_out の零空間の成分を除くと効果は消える**，(iii) MLP の活性の全体のパッチは，正しい目的語と誤った目的語の logit の差に無視できる効果しかもたない．
- 主語の最終トークンの活性の差は，見つかった部分空間の因果的に切り離された成分と強く揃う（**cosine similarity ∼ 0.9**，付録の Figure 12）．著者はここから，rowspace 成分とは弱くしか揃わないこと（**cosine similarity ∼ 0.4**）が導かれると述べる——"hence, the rowspace component is relatively dormant compared to the causally disconnected component"．この二つの値は会議版の本文にあり，arXiv v2 の本文には見当たらなかった．
- 層ごとの成功率と FLDD の分布は図（Figure 4・11）でのみ示され，本文に数値はない．

### rank-1 編集との等価性（§6.2，付録 D.3–D.5）

- ROME の編集は `W'_out = W_out + abᵀ`．MLP の活性空間の任意の方向 `v` について，`a = ((u_B − u_A)ᵀ v) W_out v`，`bᵀ u_A = 1` とする rank-1 編集が，プロンプト A のトークン t_A で，`u_B` から `u_A` への `v` に沿ったパッチと等しい MLP 出力を与える（付録 D.4，Lemma D.2）．`b = Σ⁻¹ u_A / (u_Aᵀ Σ⁻¹ u_A)` が，Meng らの意味でのモデルへの変化を最小にする．
- 編集は恒久的ですべてのトークンに掛かるので，なお o' を予測させられるかは自明でない，と著者は断る．実測ではパッチと編集の logit 差・成功率は "quite closely track each other"（付録 D.5，Figure 14・15）．
- ⇒ "rank-1 model edits can work by creating a dormant pathway of an MLP layer, regardless of whether the fact is stored there"（§2）．著者はこれを，事実の局所化と編集の不一致（Hase ら 2023）の機構的説明として示す．

### 遍在性の議論（§7，付録 E）

著者は，あるタスクの計算に使われない MLP が，残差ストリームのある方向を介して合成する二つの部品（どちらも使われる）のあいだにあるときは常に，錯覚が生じると予想する．"We do not prove this, but hope to illustrate it with an informal argument."——W_out は通例フルランクであり（付録 E.1），特徴は GELU 後の MLP 活性からも線形に回収できる（付録 E.2）．付録 E.2 の実測は GPT-2 Small と OpenWebText-10k による：残差ストリームと無作為な方向との内積を GELU 後の活性から線形回帰で回収すると，テスト集合で R² = 0.71 ± 0.17．無作為な方向を足すか引くかを当てる分類の平均精度は，倍率 z = 0.0001／0.001／0.01／0.1 で 0.69／0.83／0.87／0.996（Table 2）．

## 著者が明記した限界

会議版に限界の節は無い（arXiv v2 の §8 の題は "Discussion, Limitations, and Recommendations"）．以下は会議版の本文から拾った．

- **遍在性は証明されていない**（§7）："We do not prove this, but hope to illustrate it with an informal argument."
- **実証は二つの設定に限られる**（§7）："We only exhibit our illusion empirically in two settings, IOI and factual recall, but we believe it is likely prevalent in practice."
- **MLP8 が IOI に効かないという判断は IOI 論文（Wang ら）の分析に依る．** 著者は，効く可能性に対して結果をより頑健にするために，付録 B.5 の無作為な重みの実験を加えたと述べる．
- 会議版の §1 は "Our results suggest this effect is strongest when overfitting to a small dataset." と述べるが，これに当たる実験は会議版の PDF には見当たらなかった．arXiv v2 の付録 B.7 は，2つの名前だけの小さな IOI データで DAS を学習すると，学習時の FLDD がテスト時より有意に高くなると報告し，同時に，他の層での FLDD は著者の予想に反して高くなかったとも書く．

## 著者が述べる含意

- 最適化による部分空間パッチは "can find both faithful and illusory features with respect to the model's computation"（§7）．
- 推奨（§7）："We recommend running such methods in activation bottlenecks such as the residual stream, as well as using validations beyond end-to-end evaluation to ascertain the precise role of such features."

## 原文の表記の食い違い

- 会議版の Table 1 の caption は指標を "fractional logit diff (higher means less successful patch; 100% means no change)" と説明し，本文 §4.3 の FLDD の定義（0 が効果なし，大きいほどパッチが効く）と食い違う．arXiv v2 の caption は "fractional logit difference decrease (FLDD, higher means more successful patch; 0% means no change)" で，本文の定義と一致する．
- 会議版の付録 D.1 は "40 facts spanning 8 different relations" と書き，本文 §6.1 の "40 pairs of facts across 8 relations" と食い違う．arXiv v2 の付録 D.1 は "40 fact pairs spanning 8 different relations" である．
- interchange accuracy の定義は版で違う．arXiv v2 §4.3 は "the fraction of patches for which the model predicts the S (i.e., wrong) name for the patched run" と書く．
- 付録 C の本文は "contributes to the value of projections on v_MLP" と書くが，続く計算と Figure 7 は `v_resid` との内積である．

## 射程の外

- **VLM は扱わない．** 実験の対象は GPT-2 Small と GPT-2 XL のテキスト言語モデルである（会議版・arXiv v2 の全文で GQA，VQA，multimodal，image を検索し，視覚を扱う実験は見当たらなかった．"computer vision" は関連研究の紹介に一度出る）．
- **学習時の介入は扱わない．** 操作はすべて，学習済みモデルへの推論時のパッチと，それと対応づけた rank-1 重み編集である（DAS が学習するのは方向であり，モデルの重みではない）．
- **方向は DAS または勾配で事後に求めたもの**である．学習で表現に性質を足した場合の，読み取りやすさと使用の関係を論じる記述は見当たらなかった（検索した語：fine-tun，finetun，during training，training time，probe，probing）．
- **錯覚の実例は MLP の隠れ活性（`ker W_out` をもつ位置）のもの**で，残差ストリームは著者が "safer and less prone to illusions"（§1）とする側として扱われる．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴・Comments），会議版 PDF（proceedings.iclr.cc，30頁，付録 A–G まで），arXiv PDF v2（58頁，付録 A–E まで）と v1（58頁），proceedings.iclr.cc の BibTeX（著者・頁），ICLR 2024 virtual の poster ページ（掲載区分）を取得し，pdftotext でテキスト化して読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．会議版は本文と付録 A–E の全文を読み，Table 1・Table 2 は PDF のページ画像でも確かめた．arXiv v2 は本文 §1–§8 と付録 B.4–B.7・C・D.1–D.2 を読み，それ以外の付録は検索で当たった．v1 と v2 は，テキストを語の単位で比べると，版番号と日付の行のほかは記号3か所（⊤ が ⊥ に直っている）だけが違う．下書き（2026-10-11 作成）から直した主な点：錯覚の方向の記号 `v_illusory` と係数 1/√2 が arXiv v2 と付録 A.2 のものであると明記した．「どちらの成分も単独では効果を生まない」を脚注3の強さ（dormant 成分だけでは弱い効果）に直した．付録 B.5 を「重みを替える」から原文の手順（層は置き換えず，無作為な MLP の出力の差を足す）に直した．interchange accuracy の定義を原文の言い方に直した．「MLP8 全体を差し替えても効果はない」を「FLDD −8%（確信を上げる）」に，「効果の多くは組み合わせから生じる」を原文の見出しと数値に直した．Table 1 の caption の食い違いは会議版だけのもので arXiv v2 では解消していること，"40 facts" も同様であることを足した．cosine ∼0.4 が著者の導いた値として書かれていることを明記した．「過学習で強まる」の一文に当たる実験が会議版に無いことと，arXiv v2 付録 B.7 の内容を足した．零空間の説明の出どころを arXiv v2 に付け，著者の所属を会議版の表記どおりにした．**読めていないもの**：OpenReview のページ（閲覧の確認画面に阻まれ，取得できなかった）．図は caption と本文の言及だけを照合し，図の中の値は読んでいない．arXiv v2 の付録 A・D.3–D.7・E は通読していない．
