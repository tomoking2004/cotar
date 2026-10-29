# Same Answer, Different Representations: Hidden instability in VLMs

**著者**：Farooq Ahmad Wani，Alessandro Suglia，Rohit Saxena，Aryo Pradipta Gema，Wai-Chung Kwan，Fazl Barez，Maria Sofia Bucarelli，Fabrizio Silvestri，Pasquale Minervini（ローマ・サピエンツァ大学／エディンバラ大学／オックスフォード大学／Martian／コートダジュール大学／CNRS／Inria／I3S／Miniml.AI）
**掲載**：arXiv:2602.06652 v2，2026-09-15（v1 は 2026-02-06）．以下は v2 による
**原文**：[arXiv](https://arxiv.org/abs/2602.06652)

**出力が変わらないことは，内部処理が安定していることを意味しない．**

## 主張

VLM の頑健性は通例**出力レベルの不変性**で測られ，予測が安定していれば多モーダル処理も安定していると暗黙に仮定されている．本論文はこの仮定が不十分だと論じ，内部埋め込みの drift・スペクトル感度・構造的滑らかさ（視覚トークンの空間的一貫性）を標準の正誤指標と並べて測る枠組みを導入する．そこから三つの failure mode が現れる．

1. **答えを保ったまま表現が大きく漂う**——"for text overlays this drift can approach inter-image variability in our smallest model, indicating that representations move toward regions typically occupied by unrelated inputs despite unchanged outputs, with the perturbation ordering consistent across families"．画像間変動に迫るのは最も小さいモデルでのことで，族をまたいで揃うのは摂動の順序である．
2. **頑健性は規模で改善しない**——"larger models achieve higher accuracy but exhibit equal or greater sensitivity, consistent with sharper yet more fragile decision boundaries"．
3. **影響はタスク依存**——粗い視覚手がかりと細かい手がかりの統合を乱すと推論は悪化するが，幻覚ベンチマークでは答えが保守化して偽陽性が減る．

## 手法

### 摂動

**摂動は画像にのみ加える．質問文は摂動しない**（"Perturbations are applied exclusively to the image"）．

| 摂動 | パラメタ |
| --- | --- |
| 平行移動（巡回） | Δx ∈ {−16, −12, …, 16} \ {0} px の水平巻き込みシフト |
| pad / crop | n ∈ {−16, −12, …, 16} \ {0} px の対称な余白付加・切り出し |
| 拡縮 | α = 0.9 で縮小し元解像度へ戻す |
| 拡縮＋pad | 縮小後に一様な背景余白を付加 |
| 回転 | 面内 ±30°，"with interpolation" |
| テキスト重畳3種 | "three overlay families with identical geometry but different content"——意味のある指示文（"short directive phrases such as 'Answer is A/B/C/D'"）／"same-sized text region filled with random strings of comparable length and ink density"／"the same box region rendered without text" |

"Each perturbation type is applied multiple times per image, producing a set of perturbed variants"——族ごとにパラメタの値の数だけ摂動画像を作る．

**指示文の選び方は v2 で定められた**（付録 A.6 "Overlay letter assignment"）："For each image, TextOverlay renders 'Answer is L' for up to three option letters L, chosen as the options other than the model's own base prediction."——指示文はモデル自身の摂動前の予測以外の選択肢を名指しし，1枚に最大3つ作る．正解を名指しする重畳は，摂動前に誤答した画像にしか現れない．

**本文に書かれていないこと**：余白・回転・pad の背景色，補間法，画素値の基準となる入力解像度，重畳の箱の位置・大きさ・フォント，ランダム文字列の生成法．再現するときは，これらを自分で決めて明記するほかない．画素数で決めた摂動がモデルの入力のどれだけにあたるかは族ごとの前処理で違う，と著者は限界に書き，族をまたぐ結論を順序の比較にとどめている．

**空間関係の質問は部分的な例外**（付録 A.4）："Spatial-relation questions are a partial exception, since rotation or translation can alter the reference frame"——回転や平行移動は，空間関係を問う質問では正解を変えうる．

### 指標

**drift は元入力と摂動入力の埋め込み間の cosine distance（1 − cos）**（表4．ℓ2 距離は付録の表7）．これを単体では読まず **control 分布**と比べる——"For each base image, we compute distances between its embedding across regimes and the embeddings of randomly sampled other images, yielding a control-drift distribution"（§3.3），付録 A.9 では "a control distribution … derived from 1,000 random pairs of unrelated images"．各画像はそれぞれの問を伴うので，無関係な画像の対は問も異なる対である．報告は control の平均に対する比（%，"Drift is expressed as a percentage of mean control drift … values are reported as mean ± std"）と，摂動 drift の分布と control 分布のあいだの Cohen's d の組．付録 A.9 が定めるのは二つの距離の集合（D_pert と D_ctrl）だけで，**d の分母の標準偏差をどう合わせるか（標本数で重みを付けるか）は書かれていない**．表7の列も μ_pert・σ_pert・μ_ctrl・Cohen's d で，control の標準偏差は載っていない．control の平均は **μ_ctrl = 0.228**（ans_mcq_free）．著者はこの無作為な対の control を "deliberately conservative" とし，より近い対を control にすれば重なりはさらに大きくなるので，解析は深刻さの下限だと述べる．**drift の集計を，予測が変わらなかった事例に限ってはいない**（§4.3・§4.4，表3・表4にその限定は無い．"Even with unchanged predictions, representations may no longer reside in neighbourhoods of the original image" は読みであって集計の条件ではない）．

補助指標として視覚トークン格子上の **Dirichlet energy** を測り，パッチ埋め込みの空間的滑らかさを見る．

### hook 点

5点すべて **LLM バックボーンの最終層**（"the final transformer layer (before the output projection)"）から取る．

| 名前 | 取り方 |
| --- | --- |
| `ctx_open` | 自由記述プロンプト下での文脈最終トークン（"the last context token before generation begins"） |
| `ctx_mcq` | 多肢選択プロンプト下での文脈最終トークン |
| `ans_open` | 生成した answer トークンの平均プーリング |
| `ans_mcq` | 多肢選択条件下での同上 |
| `ans_mcq_free` | 自由に生成させたうえで多肢選択の選択肢に照らして評価する場合の同上 |

## 実験設定

**モデル**（すべて zero-shot）——Qwen3-VL Instruct **2B / 4B / 8B / 32B**，LLaVA-OneVision **0.5B / 7B**，Gemma-3 Instruct **4B / 12B**．

**データセット**

| データ | 分割 | 件数 |
| --- | --- | --- |
| SEEDBench | — | 約14,000（3,500枚の互いに素な部分集合を4回） |
| MMMU | validation | 847 |
| POPE | Adversarial | 3,000 |

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

**37.6% の画像が，何らかの摂動で1回以上 flip する．**

### 埋め込みの不変性

- 平行移動・`ctx_open`：cosine 類似度 0.989，L2 距離 226
- **テキスト重畳・`ctx_open`：cosine 類似度 0.866，L2 距離 1027**
- 多肢選択で条件づけた `ctx_mcq`（約0.99）は `ctx_open` より安定

### control に対する drift（`ans_mcq_free`）

- 平行移動：control の **4.1 ± 14.8%**，Cohen's d = −3.88
- **テキスト重畳：control の 77.7 ± 88.6%**，Cohen's d = **−0.34**
- 幾何摂動は総じて control の **4.1–8.8%**，Cohen's d は **−3.88 から −3.50**＝局所的な変形にとどまる．テキスト重畳だけが非局所的な移動になる．

> **標準偏差が平均を超える**（88.6 > 77.7）ので，「無関係画像並みに漂う**場合がある**」までが正確な読み方になる．

### 視覚トークンの滑らかさ（ΔDirichlet energy）

平行移動 +10.34 ± 67.49，テキスト重畳 −33.87 ± 60.14，回転 −72.73 ± 99.95．

### 正誤の遷移

すべての摂動で正→誤と誤→正の両方が起き，"induce movement across decision boundaries in both directions rather than acting as uniform noise"．指示を含まない幾何摂動の双方向性（平行移動 694／618，回転 390／262）が決定境界の不安定の直接の証拠とされる．テキスト重畳の誤→正 881 件 ＞ 正→誤 685 件の非対称は，重畳がモデル自身の予測を名指ししないことによる設計上のもの（"by design"）である．

### 規模との関係

精度は上がるが flip rate は同等以上で，"larger models develop sharper but more fragile decision boundaries, rather than uniformly improved robustness"．LLaVA-OneVision と Gemma-3 でも同じ傾向で，3系統・8規模（0.5B–32B）で頑健性は精度の向上に伴わない．

### 幻覚ベンチマーク（POPE Adversarial）

摂動は予測マージンを一貫して負側へ寄せる（平均 flip margin ≈ −0.96）．著者はこの drift-to-prior を，この設定の幻覚が言語事前分布ではなく**脆い視覚特徴**から生じている証拠と読む．偽陽性は減るが再現率を犠牲にする．

### 周波数解析

"high-frequency perturbations are just as effective at inducing flips as low-frequency ones (Figure 18 in Appendix), contradicting the low-frequency dominance hypothesis"．また "margins degrade smoothly rather than abruptly, indicating that VLMs do not rely on a single 'truth' band but require spectral coherence across components"．周波数を制約した PGD 攻撃は低周波・高周波いずれの帯域でも **79% 超**の成功率．

## 著者が明記した限界

- **対象の限定**："Our analysis focuses on a specific set of vision-language models and benchmarks, which may not fully capture the diversity of architectures and task distributions in deployment."
- **摂動の範囲**："the parameter ranges and specific transformations were chosen to reflect natural variations rather than exhaustive adversarial exploration—more extreme perturbations or targeted attacks may reveal different failure modes"
- **指標の限定**："The frequency-domain analysis provides interpretable insights but relies on specific metrics (Dirichlet energy, spectral norms) that may not capture all aspects of representational drift."
- **hook 点の限定**："Our hook points target the final LLM layer, which is typically the target layer used for probing and model editing … However, this choice may ignore the nuances that characterise the early fusion dynamics within vision encoders or cross-attention mechanisms."
- **頑健性は一様に測れない**："robustness cannot be evaluated uniformly across all applications"
- **族をまたぐ比較は順序まで**："we therefore report cross-family conclusions as ordinal (the ranking of perturbations and probes) rather than absolute-magnitude comparisons"
- **対処法を提案しないが，方向を挙げる**："while we document the scale-robustness decoupling, we do not propose architectural modifications or training objectives to address it; our findings nonetheless suggest concrete directions, such as representation-aware training objectives that penalise embedding drift under augmentation and Dirichlet-energy regularisation to enforce spatial coherence in vision tokens, both of which we leave for future work"
- **診断であって機構の説明ではない**："Our analysis is intentionally diagnostic rather than mechanistic."——drift が視覚エンコーダ・接続部・LLM のどの層で生じるかは特定していない．

## 射程の外

- **軸は「同一入力への摂動不変性」**であって，別々の入力どうしの表現が揃っているかではない．
- **GQA は主評価に含まれない**（主評価は SEEDBench・MMMU・POPE）．v2 の付録 A.7 が補足として，Qwen3-VL-2B で GQA の 100 枚に写真的な劣化を加えたときの答えの変化だけを報告している．
- **SmolVLM は評価モデルに含まれない**（Qwen3-VL・LLaVA-OneVision・Gemma-3 のみ）．
- **質問文の摂動は扱わない．**
- **対処法は実装も評価もされない**——方向として挙げる学習目的は，同じ入力を拡張したときの埋め込みの drift を罰するものであり（上記），別々の入力どうしの表現を近づけるものではない．

---

**確認**：2026-08-09．arXiv abs ページ（書誌・要旨・著者所属）と arXiv HTML 全文 v1 を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．「射程の外」に挙げた不在は，論文が明示する評価対象一覧と著者の限界表明から言えるものに限る．

**再確認**：2026-10-23．本研究で摂動と drift の指標を再現するにあたり，arXiv HTML v1 の §3.2・§3.3・§4.4・付録 A.6〜A.8 を再度読み，摂動の定義・hook 点・control の作り方・集計の条件を逐語で照合し，本文に書かれていない事項を列挙した．

**再確認**：2026-10-24．本研究の d が原典と同じ作り方かを確かめるため，arXiv HTML v1 の §3.3・§4.4・表4の注・付録 A.8・表6の列名を再度読み，d の計算式とプールの仕方が本文に無いこと，control が「1,000 random pairs of unrelated images」であることを照合した．

**再確認**：2026-10-29．v2（2026-09-15）が出ていたので，arXiv HTML v2 の全文（本文と付録 A）を v1 と突き合わせ，本要約の数値・逐語引用を v2 に照らして一件ずつ照合し直した．v2 で変わったのは，要旨の限定（最も小さいモデルで，摂動の順序が族をまたいで揃う），Gemma-3 の追加，MMMU の 3,000 件の削除，指示文の選び方（付録 A.6）と，それによる正誤の遷移の非対称の解釈，control が保守的だという記述，空間関係の質問の例外（付録 A.4），限界の段落（族をまたぐ比較，対処の方向，診断であること），付録と表の番号（control は A.8 から A.9，ℓ2 の表は表6から表7）である．摂動の族と値，hook 点，control の作り方，d のプールの仕方が書かれていないことは v1 と変わらない．表1の拡縮と回転の Image Vulnerability は v1 にも載っていたので，空欄だったのを埋めた．
