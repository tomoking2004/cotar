# Same Answer, Different Representations: Hidden instability in VLMs

**著者**：Farooq Ahmad Wani，Alessandro Suglia，Rohit Saxena，Aryo Pradipta Gema，Wai-Chung Kwan，Fazl Barez，Maria Sofia Bucarelli，Fabrizio Silvestri，Pasquale Minervini（ローマ・サピエンツァ大学／エディンバラ大学／オックスフォード大学／CNRS／i3S／Martian／Miniml.AI）
**掲載**：arXiv:2602.06652 v1，2026-02-06
**原文**：[arXiv](https://arxiv.org/abs/2602.06652)

**出力が変わらないことは，内部処理が安定していることを意味しない．**

## 主張

VLM の頑健性は通例**出力レベルの不変性**で測られ，予測が安定していれば多モーダル処理も安定していると暗黙に仮定されている．本論文はこの仮定が不十分だと論じ，内部埋め込みの drift・スペクトル感度・構造的滑らかさ（視覚トークンの空間的一貫性）を標準の正誤指標と並べて測る枠組みを導入する．そこから三つの failure mode が現れる．

1. **答えを保ったまま表現が大きく漂う**——テキスト重畳では drift が画像間変動の大きさに迫り，"representations move to regions typically occupied by unrelated inputs despite unchanged outputs"．
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
| 回転 | 面内 ±30° |
| テキスト重畳3種 | "three variants with identical geometry but different content"——意味のある指示文／同程度の長さとインク密度のランダム文字列／空の箱 |

### 指標

**drift は元入力と摂動入力の埋め込み間の cosine distance**．これを単体では読まず，**無関係な画像1,000対から作った control 分布**と比べ，control に対する比（%）と Cohen's d の組で報告する．control の平均は **μ_ctrl = 0.228**．

補助指標として視覚トークン格子上の **Dirichlet energy** を測り，パッチ埋め込みの空間的滑らかさを見る．

### hook 点

5点すべて **LLM バックボーンの最終層**から取る．

| 名前 | 取り方 |
| --- | --- |
| `ctx_open` | 自由記述プロンプト下での文脈最終トークン |
| `ctx_mcq` | 多肢選択プロンプト下での文脈最終トークン |
| `ans_open` | 生成した answer トークンの平均プーリング |
| `ans_mcq` | 多肢選択条件下での同上 |
| `ans_mcq_free` | 自由に生成させたうえで多肢選択の選択肢に照らして評価する場合の同上 |

## 実験設定

**モデル**（すべて zero-shot）——Qwen3-VL Instruct **2B / 4B / 8B / 32B**，LLaVA-OneVision **0.5B / 7B**．

**データセット**

| データ | 分割 | 件数 |
| --- | --- | --- |
| SEEDBench | — | 約14,000（3,500枚の互いに素な部分集合を4回） |
| MMMU | validation | 847 および 3,000 |
| POPE | Adversarial | 3,000 |

## 結果

### ラベルの不安定（SEEDBench，Qwen3-VL-2B，摂動なしの平均正解率 61.7%）

| 摂動 | Instance Flip Rate | Image Vulnerability |
| --- | --- | --- |
| 平行移動 | 6.2% | 16.8% |
| pad/crop | 6.5% | 16.9% |
| 拡縮 | 7.9% | — |
| 回転 | 12.2% | — |
| **テキスト重畳** | **19.2%** | **23.9%** |
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

### 正誤の遷移（テキスト重畳）

正→誤 685 件に対し**誤→正 881 件**．**双方向であって，一様な性能劣化ではない．**

### 規模との関係

精度は上がるが flip rate は同等以上で，"larger models develop sharper but more fragile decision boundaries, rather than uniformly improved robustness"．LLaVA-OneVision でも挙動は同様．

### 幻覚ベンチマーク（POPE Adversarial）

摂動は予測マージンを一貫して負側へ寄せる（平均 flip margin ≈ −0.96）．著者はこの drift-to-prior を，この設定の幻覚が言語事前分布ではなく**脆い視覚特徴**から生じている証拠と読む．偽陽性は減るが再現率を犠牲にする．

### 周波数解析

"high-frequency perturbations are just as effective at inducing flips as low-frequency ones (Figure 18 in Appendix), contradicting the low-frequency dominance hypothesis"．また "margins degrade smoothly rather than abruptly, indicating that VLMs do not rely on a single 'truth' band but require spectral coherence across components"．周波数を制約した PGD 攻撃は低周波・高周波いずれの帯域でも **79% 超**の成功率．

## 著者が明記した限界

- **対象の限定**："Our analysis focuses on a specific set of vision-language models and benchmarks, which may not fully capture the diversity of architectures and task distributions in deployment."
- **摂動の範囲**："the parameter ranges and specific transformations were chosen to reflect natural variations rather than exhaustive adversarial exploration—more extreme perturbations or targeted attacks may reveal different failure modes"
- **指標の限定**："the frequency-domain analysis provides interpretable insights but relies on specific metrics (Dirichlet energy, spectral norms) that may not capture all aspects of representational drift."
- **hook 点の限定**："Our hook points target the final LLM layer, which is typically the target layer used for probing and model editing. However, this choice may ignore the nuances that characterise the early fusion dynamics within vision encoders or cross-attention mechanisms."
- **頑健性は一様に測れない**："robustness cannot be evaluated uniformly across all applications"
- **対処法を提案しない**："While we document the scale-robustness decoupling, we do not propose architectural modifications or training objectives to address it—identifying effective mitigation strategies remains an open challenge."

## 射程の外

- **軸は「同一入力への摂動不変性」**であって，別々の入力どうしの表現が揃っているかではない．
- **GQA は評価データに含まれない**（SEEDBench・MMMU・POPE のみ）．
- **SmolVLM は評価モデルに含まれない**（Qwen3-VL と LLaVA-OneVision のみ）．
- **質問文の摂動は扱わない．**
- **対処法（mitigation strategies）は提案されない**——著者自身が未解決と明記する（上記）．

---

**確認**：2026-08-09．arXiv abs ページ（書誌・要旨・著者所属）と arXiv HTML 全文 v1 を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．「射程の外」に挙げた不在は，論文が明示する評価対象一覧と著者の限界表明から言えるものに限る．
