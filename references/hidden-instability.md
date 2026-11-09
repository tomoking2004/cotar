# Same Answer, Different Representations: Hidden instability in VLMs

**著者**：Farooq Ahmad Wani，Alessandro Suglia，Rohit Saxena，Aryo Pradipta Gema，Wai-Chung Kwan，Fazl Barez，Maria Sofia Bucarelli，Fabrizio Silvestri，Pasquale Minervini（ローマ・サピエンツァ大学／エディンバラ大学／オックスフォード大学／Martian／コートダジュール大学／CNRS／Inria／I3S／Miniml.AI）
**掲載**：arXiv:2602.06652 v2，2026-09-15（v1 は 2026-02-06）．以下は v2 による
**原文**：[arXiv](https://arxiv.org/abs/2602.06652)

**出力が変わらないことは，内部処理が安定していることを意味しない．**

## 主張

VLM の頑健性は通例**出力レベルの不変性**で測られ，予測が安定していれば多モーダル処理も安定していると暗黙に仮定されている．本論文はこの仮定が不十分だと論じ，内部埋め込みの drift・スペクトル感度・構造的滑らかさ（視覚トークンの空間的一貫性）を標準の正誤指標と並べて測る枠組みを導入する．そこから三つの failure mode が現れる．

1. **答えを保ったまま表現が大きく漂う**——"for text overlays this drift can approach inter-image variability in our smallest model, indicating that representations move toward regions typically occupied by unrelated inputs despite unchanged outputs, with the perturbation ordering consistent across families"．画像間変動に迫るのは最も小さいモデルでのことで，族をまたいで揃うのは摂動の順序である．本文はこのモデルを名指ししない．control に迫る値が載っているのは Qwen3-VL-2B の表4だけである．
2. **頑健性は規模で改善しない**——"larger models achieve higher accuracy but exhibit equal or greater sensitivity, consistent with sharper yet more fragile decision boundaries"．
3. **影響はタスク依存**——粗い視覚手がかりと細かい手がかりの統合を乱すと推論は悪化するが，幻覚ベンチマークでは答えが保守化して偽陽性が減ることがある（"they can reduce false positives"）．

## 手法

### 摂動

**摂動は画像にのみ加える．質問文は摂動しない．** 付録 A.6 の族はすべて画像の操作で，POPE の付録 D は "Perturbations are applied exclusively to the image, while the semantic content of the query remains unchanged" と書く．MMMU の複数画像の問では，全画像に同じ摂動を加える（付録 A.5）．

| 摂動 | パラメタ | 1枚あたりの件数 |
| --- | --- | --- |
| 平行移動（巡回） | Δx ∈ {−16, −12, …, 16} \ {0} px の水平巻き込みシフト | 8 |
| pad / crop | n ∈ {−16, −12, …, 16} \ {0} px の対称な余白付加・切り出し | 8 |
| 拡縮 | α = 0.9 で縮小し元解像度へ戻す | 1 |
| 拡縮＋pad | 縮小後に一様な背景余白を付加 | 2 |
| 回転 | 面内 ±30°，"with interpolation" | 2 |
| テキスト重畳3種 | "three overlay families with identical geometry but different content"——意味のある指示文（"short directive phrases such as 'Answer is A/B/C/D'"）／"same-sized text region filled with random strings of comparable length and ink density"／"the same box region rendered without text" | 指示文は最大3．残り2種は表2に無い |

件数は表2の GT inst. の列（3,493 枚に 27,944／27,944／3,493／6,986／6,986／10,430 件）を割った値である．"Each perturbation type is applied multiple times per image, producing a set of perturbed variants"——族ごとにパラメタの値の数だけ摂動画像を作る．

**族の数え方**：§3.2 は "six families of natural perturbations" と書く．付録 A.6 の箇条がこの六つ（平行移動・pad/crop・拡縮・拡縮＋pad・回転・テキスト重畳）で，重畳の中を "three overlay families" と呼ぶ．表1の行は，重畳を開いた八つと union である．

**表の TextOverlay は指示文の重畳である**（"Unless explicitly stated otherwise, TextOverlay refers to the semantic overlay variant"）．§3.2 はこれを "adversarial text" とも呼ぶ．Qwen3-VL-2B の表3・表4にランダム文字列と空の箱の行は無く，二つの drift の表は Gemma-3 にだけある．

**指示文の選び方は v2 で定められた**（付録 A.6 "Overlay letter assignment"）："For each image, TextOverlay renders 'Answer is L' for up to three option letters L, chosen as the options other than the model's own base prediction."——指示文はモデル自身の摂動前の予測以外の選択肢を名指しし，1枚に最大3つ作る．正解を名指しする重畳は，摂動前に誤答した画像にしか現れない．

**本文に書かれていないこと**：余白・回転・pad の背景色，補間法，画素値の基準となる入力解像度，拡縮＋pad の縮小率と1枚2件の中身，重畳の箱の位置・大きさ・フォント，ランダム文字列の生成法．再現するときは，これらを自分で決めて明記するほかない．画素数で決めた摂動がモデルの入力のどれだけにあたるかは族ごとの前処理で違う，と著者は限界に書き，族をまたぐ結論を順序の比較にとどめている．

**空間関係の質問は部分的な例外**（付録 A.4）："Spatial-relation questions are a partial exception, since rotation or translation can alter the reference frame"——回転や平行移動は，空間関係を問う質問では正解を変えうる．SEEDBench の評価次元ごとの flip rate は "as a check in future work" とされ，載っていない．代わりに，物の位置を動かさない写真的な劣化でも答えが変わることを確かめたとする（付録 A.4 は参照先を C.1 と書くが，その内容は付録 A.7 にある）．

### 指標

**答えは生成でなく，選択肢の対数尤度で決める**（§3.1，"a log-likelihood-based evaluation protocol that ranks answer options by their conditional probabilities"）．Instance Flip Rate（IFR）は全ての摂動の対のうち予測が変わった割合，Image Vulnerability（IV）は一つ以上の摂動で予測が変わる画像の割合である（§3.2）．

**drift は元入力と摂動入力の埋め込み間の cosine distance（1 − cos）**（表3・表4．ℓ2 距離は表3と付録の表7）．これを単体では読まず **control 分布**と比べる——"For each base image, we compute distances between its embedding across regimes and the embeddings of randomly sampled other images, yielding a control-drift distribution"（§3.3）．付録 A.9 の言い方は "derived from 1,000 random pairs of unrelated images" で，Gemma-3 の節の数え方は別である（SEEDBench の 12B は200枚に1枚あたり8対，MMMU は600件に1枚あたり32対．付録 C.4.1・C.4.2）．control の埋め込みは別のサンプルのものなので問も異なる対と読めるが，**問の扱いは原典に明記されていない**．著者はこの無作為な対の control を "deliberately conservative" とし，より近い対を control にすれば重なりはさらに大きくなるので，解析は深刻さの下限だと述べる．

**報告は control の平均に対する比と Cohen's d の組である．** 比は "Drift is expressed as a percentage of mean control drift"（表4の注），平均 ± 標準偏差で書く．d の読み方は "Large negative values indicate that perturbation-induced drift is substantially smaller than inter-image variability, while values closer to zero indicate non-local displacement in representation space"（§3.3）である．

**d の分母の標準偏差をどう合わせるか（標本数で重みを付けるか）は書かれていない．** 付録 A.9 が定めるのは二つの距離の集合（D_pert と D_ctrl）だけで，表7の列も μ_pert・σ_pert・μ_ctrl・Cohen's d で，control の標準偏差は載っていない．ただし表の値から逆算すると，表7の6行すべてが d = (μ_pert − μ_ctrl) / √((σ_pert² + σ_ctrl²) / 2) と σ_ctrl ≈ 84.76 の一つの値で揃い，表4も四捨五入の範囲で σ_ctrl ≈ 0.072 で揃う．件数が族で8倍違うのに一つの σ_ctrl で揃うので，二つの分散を標本数で重みを付けずに等しく合わせたと読める．これは表の値からの推定で，原典の記述ではない．

**drift の集計を，予測が変わらなかった事例に限ってはいない**（§4.3・§4.4，表3・表4にその限定は無い．"Even with unchanged predictions, representations may no longer reside in neighbourhoods of the original image" は読みであって集計の条件ではない）．答えが変わらなかった事例だけの drift の値は，どの表にも無い．表28は Dirichlet energy の全事例と flip の事例の2列である．

補助指標として視覚トークン格子上の **Dirichlet energy** を測り，パッチ埋め込みの空間的滑らかさを見る．これは LLM の hook 点ではなく，"within the vision encoder" の局所構造を見る量である（付録 E）．

### hook 点

5点すべて **LLM バックボーンの最終層**（"the final transformer layer (before the output projection)"．付録 A.8 も "from the final layer of the LLM backbone"）から取る．途中の層は測っていない．位置は5点で違い，ctx の2点は生成の直前の1トークン，ans の3点は生成した答えのトークンの平均である．

| 名前 | 取り方 |
| --- | --- |
| `ctx_open` | 自由記述プロンプト下での文脈最終トークン（"the last context token before generation begins"） |
| `ctx_mcq` | 多肢選択プロンプト下での文脈最終トークン |
| `ans_open` | 生成した answer トークンの平均プーリング |
| `ans_mcq` | 多肢選択条件下での同上 |
| `ans_mcq_free` | 自由に生成させたうえで多肢選択の選択肢に照らして評価する場合の同上 |

**著者は，自由に生成する ans_open と ans_mcq_free を結論の土台にしていない．** ctx の2点は生成の前に読むので答えの文の違いに影響されない．そこで "we therefore base our representation-drift conclusions on these context probes and on the option-constrained ans_mcq, and treat the freely generated ans_open/ans_mcq_free as additionally carrying a generation-variability component"（§4.3）と書く．一方で control との比較は，§4.4 が "the ans_mcq_free regime, which exhibits the strongest effects" で報告する．

## 実験設定

**モデル**（すべて zero-shot）——Qwen3-VL Instruct **2B / 4B / 8B / 32B**，LLaVA-OneVision **0.5B / 7B**，Gemma-3 Instruct **4B / 12B**．"All models are evaluated without fine-tuning"（付録 A.1）．乱数種は 0 に固定し，FP16 で評価する（付録 A）．

**データセット**

| データ | 分割 | 件数 |
| --- | --- | --- |
| SEEDBench | Full | 約14,000（3,500枚の互いに素な部分集合を4回） |
| MMMU | validation | 847 |
| POPE | Adversarial | 3,000 |

**断りのない数値は Qwen3-VL-2B と SEEDBench のものである**（"Unless stated otherwise, results in this section are reported for a single representative model, Qwen3-VL-2B Instruct and SEEDBench dataset"，§4）．他の条件の標本数は節で違う．Gemma-3-4B は SEEDBench 3,500 件（drift は約600），Gemma-3-12B は 1,000 件（drift は200），周波数の実験は Qwen3-VL-2B の400件，POPE は Qwen3-VL-2B と 8B である（付録 C.4・C.4.1・D・F）．

## 結果

### ラベルの不安定（SEEDBench，Qwen3-VL-2B，摂動なしの平均正解率 61.7%）

| 摂動 | Instance Flip Rate | Image Vulnerability |
| --- | --- | --- |
| 平行移動 | 6.2% | 16.8% |
| pad/crop | 6.5% | 16.9% |
| 拡縮 | 7.9% | 7.9% |
| 拡縮＋pad | 8.0% | 10.0% |
| 回転 | 12.2% | 16.6% |
| **テキスト重畳（指示文）** | **19.2%** | **23.9%** |
| テキスト重畳（ランダム文字列） | 6.4% | 8.6% |
| テキスト重畳（空の箱） | 4.3% | 4.4% |
| いずれか | 7.9% | **37.6%** |

**37.6% の画像が，何らかの摂動で1回以上 flip する．** 値は4回の実行の平均で，実行間の標準偏差は IFR で 0.005 以下，IV で 0.008 以下である（表1の注）．

**重畳の順序は，文字の中身が効くことを示す．** 指示文 19.2%，ランダム文字列 6.4%，空の箱 4.3% の差から，著者は "We therefore interpret TextOverlay as pixel-space prompt injection, not visual misalignment" と読む（§4.1）．

### 埋め込みの不変性

各欄は cosine 類似度／L2 距離である（表3）．

| 摂動 | `ctx_open` | `ctx_mcq` | `ans_open` | `ans_mcq` | `ans_mcq_free` |
| --- | --- | --- | --- | --- | --- |
| 平行移動 | 0.989 / 226 | 0.995 / 104 | 0.980 / 141 | 0.994 / 82 | 0.991 / 65 |
| 回転 | 0.963 / 458 | 0.976 / 262 | 0.959 / 276 | 0.983 / 182 | 0.980 / 142 |
| **テキスト重畳** | **0.866 / 1027** | 0.918 / 551 | 0.922 / 465 | 0.963 / 310 | 0.820 / 484 |

多肢選択で条件づけた `ctx_mcq` は `ctx_open` より安定で，この安定は `ans_mcq` と `ans_mcq_free` には及ばない（§4.3）．

### control に対する drift

**比（% of control）が載るのは表4だけで，一つのモデルの中で摂動の種類どうしを比べる使い方である．** 条件は Qwen3-VL-2B，SEEDBench，`ans_mcq_free` である．**μ_ctrl = 0.228** はこの条件の control の平均で（表4の注は "the typical distance between unrelated images" と言う），6行に共通の分母になる．

| 摂動 | 比（% of control） | Cohen's d |
| --- | --- | --- |
| 平行移動 | 4.1 ± 14.8 | −3.88 |
| pad/crop | 4.6 ± 16.7 | −3.77 |
| 拡縮 | 4.8 ± 21.9 | −3.50 |
| 拡縮＋pad | 5.7 ± 18.7 | −3.63 |
| 回転 | 8.8 ± 18.5 | −3.52 |
| **テキスト重畳** | **77.7 ± 88.6** | **−0.34** |

幾何摂動は局所的な変形にとどまり（"indicating local deformations"），テキスト重畳の drift の平均（約 0.177）だけが control の平均に迫る（§4.4）．

> 表4の注は "near-complete displacement to unrelated regions" と書く．しかし**標準偏差が平均を超える**（88.6 > 77.7）ので，「無関係画像並みに漂う**場合がある**」までが正確な読み方になる．これは要約者の読みである．

**77.7% は `ans_mcq_free` の値である．** 著者が結論の土台にする `ctx_open`・`ctx_mcq`・`ans_mcq` の control は報告されていない（→ hook 点）．表3の `ans_mcq` では，テキスト重畳の 1 − cos は 0.037 である．

**control はモデル・データ・埋め込みごとに別の値で，モデルどうしは距離そのもので比べる．** Gemma-3 の表は比を載せず，drift・control・d の3列である．

| 表 | モデル | データ | 埋め込み | 平行移動の drift | テキスト重畳の drift | control | テキスト重畳の d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Gemma-3-4B | SEEDBench | `ans_mcq_free` | 0.024 | 0.042 | 0.229 | −6.88 |
| 15 | Gemma-3-12B | SEEDBench | `ans_mcq_free` | 0.033 | 0.066 | 0.314 | −5.74 |
| 12 | Gemma-3-4B | SEEDBench | `ans_open` | 0.045 | 0.099 | 0.394 | −2.98 |
| 16 | Gemma-3-12B | SEEDBench | `ans_open` | 0.033 | 0.112 | 0.350 | −2.35 |
| 24 | Gemma-3-4B | MMMU | `ans_mcq_free` | 0.026 | 0.038 | 0.422 | −7.94 |
| 25 | Gemma-3-12B | MMMU | `ans_mcq_free` | 0.031 | 0.061 | 0.479 | −7.54 |
| 26 | Gemma-3-4B | MMMU | `ans_open` | 0.067 | 0.118 | 0.610 | −5.04 |
| 27 | Gemma-3-12B | MMMU | `ans_open` | 0.062 | 0.132 | 0.601 | −4.51 |

- **4B と 12B の比較は距離そのものである**——表15の注は "drift is larger across nearly every perturbation, including TextOverlay" と書き，分母が 0.229 から 0.314 へ変わることには触れない．こちらで割ると，SEEDBench の `ans_mcq_free` のテキスト重畳は 18.3% から 21.0% へ，平行移動は 10.5% のままである（原典の値ではなく要約者の計算で，3桁の表からなので目安である）．
- **Gemma-3 のテキスト重畳は control に迫らない**——`ans_mcq_free` で control の約1〜2割，d は −5.74 から −7.94 である．
- **Qwen3-VL の 4B 以上と LLaVA-OneVision に，control との表は無い．** パラメタ数で最も小さい LLaVA-OneVision 0.5B も含まれない．
- **族の順序は厳密には揃わない**——Gemma-3-4B の `ans_mcq_free` では回転 0.045 がテキスト重畳 0.042 を上回り（表11），MMMU では同じ 0.038 である（表24）．付録 C.4 の "TextOverlay remains the most disruptive perturbation" は，答えの変化（表8）では成り立つ．

### 視覚トークンの滑らかさ（ΔDirichlet energy）

平行移動 +10.34 ± 67.49，テキスト重畳 −33.87 ± 60.14，回転 −72.73 ± 99.95．

### 正誤の遷移

すべての摂動で正→誤と誤→正の両方が起き，"induce movement across decision boundaries in both directions rather than acting as uniform noise"．指示を含まない幾何摂動の双方向性（平行移動 694／618，回転 390／262）が決定境界の不安定の直接の証拠とされる．テキスト重畳の誤→正 881 件 ＞ 正→誤 685 件の非対称は，重畳がモデル自身の予測を名指ししないことによる設計上のもの（"by design"）である．指示に一様に従うわけではなく，摂動前に正答した画像が誤った文字の重畳で変わるのは 10.7%，誤答した画像が正されるのは 22% である（§4.2，付録 A.6）．

### 規模との関係

**主張の根拠は，答えの側と内部表現の側で別である．**

答えの側は表5（SEEDBench）である．

| モデル | 正解率（%） | テキスト重畳の IFR | union の IV |
| --- | --- | --- | --- |
| Qwen3-VL-2B | 61.7 | 0.192 | 0.376 |
| Qwen3-VL-4B | 65.4 | 0.270 | 0.476 |
| Qwen3-VL-8B | 65.7 | 0.196 | 0.397 |
| Qwen3-VL-32B | 44.1 | 0.343 | 0.547 |
| LLaVA-OV-0.5B | 33.0 | 0.076 | 0.199 |
| LLaVA-OV-7B | 55.9 | 0.087 | 0.255 |
| Gemma-3-4B | 47.0 | 0.203 | 0.471 |
| Gemma-3-12B | 53.0 | 0.316 | 0.671 |

- **読みは対に限られる**——表5の注の "base accuracy rises with scale while flip rate and vulnerability do not decrease" が挙げるのは，Qwen 2B→4B，LLaVA 0.5B→7B，Gemma 4B→12B である．Qwen3-VL の 4B→8B では IFR は 0.270 から 0.196 へ下がる．
- **32B は根拠にしていない**——正解率が 44.1% に下がり，著者は "we do not rely on 32B" と断る（§5.1）．
- **Gemma-3 の2行は件数が違う**——12B は 1,000 件（表13），4B は 3,500 件（付録 C.4）である．
- **MMMU でも同じ向きである**——"every larger Qwen3-VL model is less robust than the 2B"（付録 C.2）．

"larger models develop sharper but more fragile decision boundaries, rather than uniformly improved robustness" は，規模とともに正→誤と誤→正の率がともに上がること（図5）からの読みである．3系統・8規模（0.5B–32B）で頑健性は精度の向上に伴わない，が §5.1 のまとめである．

内部表現の側の表があるのは **Gemma-3 の 4B と 12B だけ**で，距離そのものか類似度で比べている．SEEDBench の `ctx_open` のテキスト重畳は類似度 0.957 → 0.918（§5.1，表10・14），`ans_mcq_free` の drift は 0.042 → 0.066（表11・15），MMMU では 0.038 → 0.061（表24・25）である．付録 C.4.1 の "larger representation drift across all regimes" は，表とは合わないところがある．`ans_open` では 12B の drift のほうが小さい族が多い（SEEDBench で8族中7族，MMMU で8族中6族．表12・16，表26・27）．

### 幻覚ベンチマーク（POPE Adversarial，Qwen3-VL-2B・8B）

無地の画像を与えると，モデルはほぼ全件で "No" と答える（§7）．摂動前に物体があると誤答した事例では，摂動は予測マージンを負側へ寄せる（平均 flip margin ≈ −0.96．マージンは Yes と No の対数尤度の差で，付録 D の定義による）．著者はこの drift-to-prior を，この設定の幻覚が言語事前分布ではなく**脆い視覚特徴**から生じている証拠と読む．偽陽性は減るが再現率を犠牲にする．

摂動の種類で向きは違う．回転と拡縮は正しい Yes を No へ変える向きが多く，テキスト重畳と切り出し系は真陰性を偽陽性へ変える向きが多い（付録 D.2）．POPE で最も答えを変えるのはテキスト重畳でなく回転である（"Rotation consistently emerges as the most disruptive transformation"，付録 D.1）．

### 周波数解析

"high-frequency perturbations are just as effective at inducing flips as low-frequency ones (Figure 18 in Appendix), contradicting the low-frequency dominance hypothesis"．また "margins degrade smoothly rather than abruptly, indicating that VLMs do not rely on a single 'truth' band but require spectral coherence across components"．周波数を制約した PGD 攻撃（ε = 8/255，400件）の成功率は，画素空間 81.25%，低周波 79.00%，高周波 82.25% である（表29）．

## 著者が明記した限界

- **対象の限定**："Our analysis focuses on a specific set of vision-language models and benchmarks, which may not fully capture the diversity of architectures and task distributions in deployment."
- **摂動の範囲**："the parameter ranges and specific transformations were chosen to reflect natural variations rather than exhaustive adversarial exploration—more extreme perturbations or targeted attacks may reveal different failure modes"
- **指標の限定**："The frequency-domain analysis provides interpretable insights but relies on specific metrics (Dirichlet energy, spectral norms) that may not capture all aspects of representational drift."
- **hook 点の限定**："our hook points target the final LLM layer, which is typically the target layer used for probing and model editing"．しかし "this choice may ignore the nuances that characterise the early fusion dynamics within vision encoders or cross-attention mechanisms"．
- **頑健性は一様に測れない**："robustness cannot be evaluated uniformly across all applications"
- **族をまたぐ比較は順序まで**："we therefore report cross-family conclusions as ordinal (the ranking of perturbations and probes) rather than absolute-magnitude comparisons"
- **対処法を提案しないが，方向を挙げる**："while we document the scale-robustness decoupling, we do not propose architectural modifications or training objectives to address it; our findings nonetheless suggest concrete directions, such as representation-aware training objectives that penalise embedding drift under augmentation and Dirichlet-energy regularisation to enforce spatial coherence in vision tokens, both of which we leave for future work"
- **診断であって機構の説明ではない**："Our analysis is intentionally diagnostic rather than mechanistic."——drift が視覚エンコーダ・接続部・LLM のどの層で生じるかは特定していない．

## 射程の外

- **軸は「同一入力への摂動不変性」**であって，別々の入力どうしの表現が揃っているかではない．
- **GQA は主評価に含まれない**（主評価は SEEDBench・MMMU・POPE）．v2 の付録 A.7 が補足として，Qwen3-VL-2B で GQA の 100 枚に写真的な劣化を加えたときの答えの変化だけを報告している（ガウスぼかし 8.0%，JPEG-70 7.4% など）．著者はこの値を "not directly comparable to the main SEEDBench tables" とする．
- **SmolVLM は評価モデルに含まれない**（Qwen3-VL・LLaVA-OneVision・Gemma-3 のみ）．
- **質問文の摂動は扱わない．**
- **対処法は実装も評価もされない**——方向として挙げる学習目的は，同じ入力を拡張したときの埋め込みの drift を罰するものであり（上記），別々の入力どうしの表現を近づけるものではない．

## ポスターとの対応

ポスターの文ごとに，原典の依りどころを示す．本研究の側の設計は [.claude/context.md](../.claude/context.md) の §A.6 にある．

**Wani ら[1]の研究**

- **目的「VLM の頑健性を，答えだけでなく内部表現でも評価」**——要旨 "measures internal embedding drift, spectral sensitivity, and structural smoothness (spatial consistency of vision tokens), alongside standard label-based metrics"．§1 の問いは "Does output-level robustness in VLMs similarly mask representation drift?" である．
- **方法「正解が変わらない範囲で画像を加工し」**——§4 "meaning-preserving transformations"，付録 A.4 "content-preserving for a human observer under mild transforms"．同じ段落に "Spatial-relation questions are a partial exception" とある．
- **方法「加工の前後で答えと内部表現を比較」**——§3 "decouple decision instability (output changes) from representation instability (representation drift)"，§4 "how often do predictions change under meaning-preserving transformations"，"what happens internally when predictions do not change"．
- **結果「答えが同じでも，内部表現は大きく変化」**——要旨 "models frequently preserve predicted answers while undergoing substantial internal representation drift"，§1 "in some cases, the model preserves its predicted answer under perturbation despite a substantial representation drift"．数値は表4で，幾何摂動は "below 10% of inter-image variability"，テキスト重畳は "approaches 78%" である．

**3.3 動きの比（%）**

- **「Wani ら[1]の尺度」**——表4の注 "Drift is expressed as a percentage of mean control drift"．比は原典が使う尺度である．ただし原典は一つのモデルの中で加工の種類どうしを比べるのに使い，本研究は別々に学習した群を比べる（→ 問15）．倣ったのは，距離を control の平均で割って % で書く形である．
- **「加工の前後の 𝒉 の距離」**——表4の題 "measured using cosine distance"，§3.3 の Embedding Stability．𝒉 の層と位置は本研究のもので，原典は LLM の最終層である．
- **「÷ 別の画像に替えたときの距離」**——§4.4 "a control baseline of distances to randomly sampled other images"．分母は control の分布の平均である．本研究は同じ質問に別の画像を付ける（→ 問19）．
- **「加工による動きが別の画像の何%か」**——表4の注の定義による．比が小さいほど，加工による動きは別の画像との距離に比べて小さい．§3.3 の d の読み方 "Large negative values indicate that perturbation-induced drift is substantially smaller than inter-image variability" とも符合する．群どうしでは分母が違うので，比の低下は距離の低下を意味しない（→ 問16）．

**加工の例**

- **「Wani ら[1]に倣う23通り」**——付録 A.6 "We define the perturbation families with the following parameter sweeps"．原典の族の数は §3.2 の "six families of natural perturbations" である．ポスターは族の数を載せず，23は本研究の数え方である（→ 問9）．
- **「全問に適用」**——本研究の設計である．原典は "Each perturbation type is applied multiple times per image"（付録 A.6）と書く．

## 学会で聞かれたら

答えの末尾に出所を添える．本研究の数値はすべて研究文書の §A.6 による．比は対照の平均に対する %，距離の単位は 10⁻³ である．

### ポスターの図と背景の文について

- **baseline でも署名は 82.9% 読めるのに，「手順の特徴を獲得できていない可能性」と言えるのですか．** 線形に読めることと，内部表現が手順ごとにまとまっていることは別です．baseline では同署名と異署名の類似度の差が小さく（0.945 と 0.918），分離度は 0.82 にとどまります．背景の文は研究の動機としての可能性で，結果としては，整合の前から署名の情報は線形に取り出せました．（出所：研究文書 §4.4，§5.1）
- **背景の図の例は，どのくらい動いたのですか．** 答えは5乱数種とも racket のままで，内部表現の距離は別の画像に替えたときの平均42%です（5乱数種の距離の平均どうしの比）．乱数種ごとの比は7–114%と大きく揺れるので，一つの問の値は例として見てください．全問の平均は比で読みます．（出所：`analyses/probe-drift.pt`，第16層，文字の重ね書き．この値はポスターに載せていない——背景の図は Wani らの方法の模式で，本研究の測定値を置くと Wani らの結果と読まれるため．2026-11-08）

### 論文そのものについて

1. **この論文は何をしたのですか．** 画像を意味を保って加工し，答えの変化と LLM 最終層の埋め込みの動きを一緒に測りました．答えが保たれていても内部表現は動き，文字の重ね書きでは無関係な画像どうしの距離に迫ることがある，と報告しています．（要旨，§4.3，§4.4）
2. **どのモデルとデータですか．** Qwen3-VL の 2B・4B・8B・32B，LLaVA-OneVision の 0.5B・7B，Gemma-3 の 4B・12B を，学習せず zero-shot で評価しています．データは SEEDBench 約14,000件，MMMU の validation 847件，POPE の Adversarial 3,000件です．本文の細かい数値は Qwen3-VL-2B と SEEDBench のものです．（§4，付録 A.1・A.2）
3. **答えと内部表現はどう測ったのですか．** 答えは生成でなく，選択肢の対数尤度が最大のものです．内部表現は LLM の最終層の5点で，答えの直前の1トークンが2点，生成した答えのトークンの平均が3点です．途中の層は測っていません．（§3.1，§3.3，付録 A.8）
4. **どのくらい動くのですか．** Qwen3-VL-2B の ans_mcq_free という埋め込みで，幾何的な加工は対照の 4.1〜8.8%，Cohen's d は −3.88〜−3.50 です．指示文の重ね書きは 77.7%，d は −0.34 です．（表4）
5. **なぜ対照で割るのですか．** 距離の値だけでは大きいか小さいか判断できないからです．動きが局所的か，別の画像との違いに匹敵するかを見るために，無関係な画像どうしの距離と比べています．（§4.4）
6. **Cohen's d はどういう意味ですか．** 加工による距離の分布と対照の分布がどれだけ離れているかを，標準偏差を単位に表した値です．大きな負の値なら加工の動きは画像間の違いよりずっと小さく，0 に近ければ二つの分布が重なります．分母の合わせ方は原典には書かれていませんが，表7の値からは二つの分散を等しく合わせたと逆算できます．（§3.3，表4・表7，付録 A.9）
7. **78% はどのモデルでも出る値ですか．** いいえ．Qwen3-VL-2B の ans_mcq_free の値で，要旨も最も小さいモデルに限っています．Gemma-3 の同じ埋め込みでは，表の値を割ると対照の1〜2割で，著者も ans_mcq_free には生成のばらつきが混ざると書いています．（要旨，§4.3，表4・11・15・24・25）
8. **答えが変わらなかった事例だけで集計したのですか．** 原典には書かれていません．表3と表4にその限定は無く，答えが変わらなかった事例だけの値はどの表にもありません．本研究はその集計も出していて，比は 14.31／9.62／20.60% でほとんど変わりません．（§4.3，§4.4，研究文書 §A.6）
9. **加工は何族ですか．23通りは原典と同じですか．** 原典は6族と数え，重ね書きの3種を開いた表1の行は8つです．ポスターの23通りは本研究の数え方です．定められた値は付録 A.6 のとおりで，1枚あたりの件数は違います．原典は縮小と余白が2件，指示文が最大3件で，本研究はどちらも1通りです．（§3.2，付録 A.6，表2，研究文書 §A.6）
10. **文字の重ね書きが効くのは，画像が隠れるからではないですか．** 答えが変わる割合は，空の箱 4.3%，無作為な文字列 6.4%，指示文 19.2% で，隠れることだけでは説明できません．原典は指示文を "pixel-space prompt injection" と解釈しています．ただし指示に一様に従うわけではなく，正答していた画像が誤った文字で変わるのは 10.7% です．（§4.1，§4.2，表1）
11. **加工で正解が変わることはないのですか．** 原典は人が見て内容が分かる範囲の加工だとしますが，空間関係の問は "partial exception" だと認めています．そこで物の位置を動かさない劣化でも答えが変わることを，GQA の100枚で確かめています．本研究の GQA には空間関係の問が多いので，この前提はその分だけ弱いです．（付録 A.4・A.7，研究文書 §A.6）
12. **大きいモデルほど頑健なのではないですか．** 原典はそうならないと言います．表5で Qwen 2B→4B，LLaVA 0.5B→7B，Gemma 4B→12B は，正解率が上がっても答えの変わる割合が下がりません．内部表現の側の表があるのは Gemma-3 の 4B と 12B だけです．（§5.1，表5，付録 C.4）
13. **この論文で何が言えて，何が言えないのですか．** 言えるのは，答えが変わらないことは内部表現が安定していることを保証しない，までです．drift がどの部品で生じるかは特定しておらず（"diagnostic rather than mechanistic"），対処法も示していません．検定や信頼区間は原典には書かれていません（表1は実行間の標準偏差だけです）．（要旨，Limitations，表1の注）

### 本研究との関係について

14. **なぜ比を使うのですか．** 表現の広がり方が群ごとに違うからです．proposal は異署名の類似度を 0.92 から 0.22 まで下げたので，距離そのものは群のあいだで同じ物差しになりません．そこで同じ問に別の画像を与えたときの距離を物差しにしました．（研究文書 §A.6）
15. **Wani らも比で群やモデルを比べたのですか．** いいえ．Wani らの比は表4だけで，一つのモデルの中で加工の種類どうしを比べるので分母は共通で，モデルどうしは Gemma-3 の 4B と 12B を距離そのもので比べています．本研究は別々に学習した群を比べるので分母が群ごとに違い，そのため比と距離の両方を示しています．（§4，表4・11・15，研究文書 §A.6）
16. **なぜ「安定した」と言わないのですか．** 比と距離で向きが逆だからです．proposal の第16層の比は baseline より 4.82pt 小さく（区間 [−7.87, −1.77]），距離そのものは +1.97 と大きいです（[+1.58, +2.37]．距離が約3倍に，対照が約5倍になったので比が下がりました）．言えるのは，画像を差し替えたときの動きに比べて加工による動きが小さい，までです．（研究文書 §A.6）
17. **Cohen's d で群を比べないのはなぜですか．** d は対照の分布の幅が変わるだけで動くからです．対照の変動係数は群で 0.34〜0.78 と違い，proposal で d が 0 に近づいたのは対照の分布が広がったためで，加工の動きが対照に近づいたからではありません．（研究文書 §A.6）
18. **Wani らとモデル・データ・層はどう違いますか．** 本研究は SmolVLM-500M-Instruct を GQA で学習した群で，第16層と第32層の，答えを書き出す直前の位置を測ります．Wani らは最終層で，位置が近いのは ctx の2点ですが，比が載る表4は生成した答えの平均です．モデルもデータも違うので，借りたのは指標の形で，数値は比べられません．（§3.3，§4.4，研究文書 §3.3・§A.6）
19. **対照の作り方はどう違いますか．** 本研究は同じ問に別の画像を付けたときの距離で，Wani らは別の画像との距離です．Wani らの問の扱いは原典には書かれていませんが，問も別と読めます．本研究の位置は質問の末尾で，整合が質問どうしの配置を変えているので，問まで替えると画像の寄与でなくその作り変えを測ってしまいます．（§3.3，付録 A.9，研究文書 §A.6）
20. **指示文の決め方はどう違いますか．** Wani らはモデル自身の加工前の予測以外の選択肢を名指しする文を1枚に最大3つ作ります．本研究は「Answer is yes」か「Answer is no」の一方を問ごとに固定します．予測に応じて文を変えると同じ問が群ごとに違う加工を受けるうえ，GQA には選択肢が無いからです．（付録 A.6，研究文書 §A.6）

---

**確認**：2026-08-09．arXiv abs ページ（書誌・要旨・著者所属）と arXiv HTML 全文 v1 を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．「射程の外」に挙げた不在は，論文が明示する評価対象一覧と著者の限界表明から言えるものに限る．

**再確認**：2026-10-23．本研究で摂動と drift の指標を再現するにあたり，arXiv HTML v1 の §3.2・§3.3・§4.4・付録 A.6〜A.8 を再度読み，摂動の定義・hook 点・control の作り方・集計の条件を逐語で照合し，本文に書かれていない事項を列挙した．

**再確認**：2026-10-24．本研究の d が原典と同じ作り方かを確かめるため，arXiv HTML v1 の §3.3・§4.4・表4の注・付録 A.8・表6の列名を再度読み，d の計算式とプールの仕方が本文に無いこと，control が「1,000 random pairs of unrelated images」であることを照合した．

**再確認**：2026-10-29．v2（2026-09-15）が出ていたので，arXiv HTML v2 の全文（本文と付録 A）を v1 と突き合わせ，本要約の数値・逐語引用を v2 に照らして一件ずつ照合し直した．v2 で変わったのは，要旨の限定（最も小さいモデルで，摂動の順序が族をまたいで揃う），Gemma-3 の追加，MMMU の 3,000 件の削除，指示文の選び方（付録 A.6）と，それによる正誤の遷移の非対称の解釈，control が保守的だという記述，空間関係の質問の例外（付録 A.4），限界の段落（族をまたぐ比較，対処の方向，診断であること），付録と表の番号（control は A.8 から A.9，ℓ2 の表は表6から表7）である．摂動の族と値，hook 点，control の作り方，d のプールの仕方が書かれていないことは v1 と変わらない．表1の拡縮と回転の Image Vulnerability は v1 にも載っていたので，空欄だったのを埋めた．

**再確認**：2026-11-05．ポスターの文のために，arXiv HTML v2 の題，§3.2 の族の数，付録 A.4 の内容保持と空間関係の問の例外，表2の件数，表4の比を読み直した．

**再確認**：2026-11-06．比の使い方を確かめるため，arXiv HTML v2 の §4 冒頭・表4・§5.1・付録 C.4 の該当箇所を読んだ．control に対する比（表4）は Qwen3-VL-2B の1モデルの中で加工の種類どうしを比べるのに使われ，モデルどうしの比較は付録で距離そのもの（Gemma-3 の 4B と 12B．表11・表15など）で行われている．規模についての主張は，表5では正解率と答えが変わる割合により，内部表現については Gemma-3 の 4B と 12B の表による．

**照合**：2026-11-06．学会の前に，arXiv HTML v2 を取得してテキスト化し，要旨，§1〜§8，Limitations，付録 A〜F の本文，表1〜29の全行，図の注を読んで，本要約の数値・引用・節番号・表番号を一件ずつ原文に照らした．abs ページで，版が v2 までであることと v1・v2 の日付を確かめた．直したのは，POPE の節の文を全体の定めとして引いていたこと，`ctx_mcq` の「約0.99」，POPE のマージンの条件，PGD の「79% 超」，SEEDBench の分割，途中を略した引用，問の扱いを確定として書いていたことである．足したのは，§4.3 の ans_mcq_free の但し書き，control の表，規模の根拠の分解，族の数え方，d の分母の逆算，ポスターとの対応，想定問答である．逆算と，Gemma-3 の比は，表の値からの計算で原典の記述ではない．図の中の値は読んでいない（画像なので注だけを読んだ）．PDF 版は照合していない．

**再確認**：2026-11-09．ポスターの 3.3 に「Wani ら[1]の尺度」と書くため，arXiv HTML v2 で表4の注 "Drift is expressed as a percentage of mean control drift"，§4.4 "we compare against a control baseline of distances to randomly sampled other images"，§3.3 "we compute distances between its embedding across regimes and the embeddings of randomly sampled other images" を読み直した．要約の引用と一致した．
