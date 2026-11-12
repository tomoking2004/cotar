# Inducing Causal Structure for Interpretable Neural Networks

**著者**：Atticus Geiger\*，Zhengxuan Wu\*，Hanson Lu\*，Josh Rozner，Elisa Kreiss，Thomas Icard，Noah D. Goodman，Christopher Potts（全員 Stanford University．\* は Equal contribution）
**掲載**：ICML 2022（Proceedings of the 39th International Conference on Machine Learning，PMLR 162:7324–7338，2022）．arXiv:2112.00826（v1 2021-12-01，v2 2022-07-20）．本要約の節番号・数値・引用は v2 による（PMLR 掲載版の PDF と本文が一致する）
**原文**：[PMLR](https://proceedings.mlr.press/v162/geiger22a.html)・[arXiv](https://arxiv.org/abs/2112.00826)

**高水準の因果モデルの変数をネットワークの表現に対応づけ，その表現を別の入力のものと入れ替えたときの出力が，因果モデルの反実仮想の出力と一致するよう学習させる．損失が 0 なら，因果モデルはネットワークの因果抽象である．**

## 位置づけ

学習時の目的関数で，ネットワークの内部に特定の構造を持たせる手法である——著者は従来の構造的な分析との違いを "we go beyond passive study of static models, by pushing them to learn specific causal structures as part of optimization" と述べる（§1）．本研究も補助損失を通常の学習に足して中間層の表現に手を入れる（研究文書 §3.2）が，損失の中身は違う．IIT の損失は「表現を入れ替えたときの出力」に掛かり，本研究の整合損失は署名が一致する質問どうしの表現の近さに掛かる（研究文書 §3.2）．研究文書は本論文を引いていない．

## 主張

- **interchange intervention training（IIT）** を提案する．因果モデルの変数をネットワークの表現に対応づけ（align），対応づけた表現を source 入力での値に置いたときの，base 入力に対する因果モデルの反実仮想的な振る舞いに，ネットワークが一致するよう学習する（要旨）．
- IIT は "fully differentiable, flexibly combines with other objectives, and guarantees that the target causal model is a causal abstraction of the neural model when its loss is zero"（要旨）．
- 3課題（MNIST-PVR，ReaSCAN，MQNLI）で multi-task の目的関数・データ拡張と比べ，"In all our experiments, IIT achieves the best results and produces neural models that are more interpretable in the sense that they more successfully realize the target causal model."（要旨）．
- probing への立場：プローブは "fundamentally limited in a way that is central to our present goals: there is no guarantee that probed information plays a causal role in the network's behavior"（§2 Related Work の Probes の段）．

## 手法

節番号は §3．

### 記号

- `GetVals(M, input, V)`：入力 `input` のもとでモデル `M` の変数集合 `V` がとる値
- `M_{V←v}`：`V` の構造方程式を定数 `v` に置き換えたほかは `M` と同じモデル
- `Π`：高水準の中間変数を，低水準の変数の互いに重ならない部分集合へ写す対応づけ（alignment）
- `κ`：出力の値を低水準から高水準へ写す，あらかじめ定めた写像（例：確率分布を出すモデルなら arg max）

### interchange intervention（式1）

base 入力に対するモデルの出力を，変数 `V` だけを source 入力のときの値に置いて求める．

```
IntInv(M, base, source, V) = GetVals( M_{V ← GetVals(M, source, V)}, base, V_Out )
```

### 因果抽象と IntInvAcc（式2・式3）

高水準モデル `M_H` の中間変数 `V_H` について，`V_In`・`V_Out`・`V_H` 以外の変数を周辺化したモデルを `M_H*` と書く．入力の対 `b, s` のすべてで次が成り立つことを，低水準モデル `M_L` と `M_H*` が因果抽象の関係にあることの定義とする（式2）．

```
IntInv(M_H*, b, s, V_H) = κ( IntInv(M_L, b, s, Π(V_H)) )
```

**interchange intervention accuracy（IntInvAcc，式3）** は，対応づけた入れ替え介入のうち，同じ出力になるものの割合である．すべての入力対を数えて 1 なら，二つのモデルは因果抽象の関係にある．入力対の空間が膨大なので，無作為に抜いた対で近似することが多いと著者は述べる．また "In practice, we rarely observe perfect INTINVACC in complex networks" とし，値が高いほど "the more we have license to reason about the high-level causal model instead of reasoning directly about the low-level network" と読む．

### IIT の損失（式4）

中間変数が一つの高水準モデルについて，

```
Σ_{b,s ∈ V_In} Loss( IntInv(C, b, s, V), IntInv(N^θ, b, s, Π(V)) )
```

- 損失関数はロジットを直接受け取るので，`κ` は適用しない．
- **`Loss` が非負で，`Loss(x, y) = 0 ⇒ x = κ(y)` なら，この損失が 0 のとき `C` は `N^θ` の因果抽象である**（付録A）．逆は成り立たない——損失が 0 でなくても因果抽象でありうる（図1がその例）．著者はこれを "a desirable property of the method, since we do not expect our loss functions to be zero in general" と述べる．
- **typed IIT**（式6・式7，MNIST-PVR）：中間変数 `Y_TL, Y_TR, Y_BL, Y_BR` は値の空間を共有し，対応する表現 `Q_TL, Q_TR, Q_BL, Q_BR` も同じなので，別の変数のあいだで入れ替える介入にも目的関数を広げる．

### 比較対象

| 名前 | 内容 |
| --- | --- |
| STANDARD | 原文は standard training と呼ぶだけで，定義の文は置かない．"without any structural training objectives" の学習として言及される（§4） |
| MULTI | 対応づけた表現から中間変数の値を予測する分類器を学習し，勾配を対象モデルの重みへ流す（MNIST-PVR では線形分類器4つ，ReaSCAN では単層の線形分類器，MQNLI では線形分類器）．§2 は multi-task training を，対象モデルのパラメタが変わることを許した supervised probing として捉えられると述べる |
| AUGMENT | 無作為に選んだ2例から，因果モデルでラベルを付けた新しい入力を作るデータ拡張（MNIST-PVR では象限の入れ替え，MQNLI では quantified verb phrase の置き換え）．"by definition restricted to input manipulations"．ReaSCAN の結果（図3b）に AUGMENT の行は無い |
| IIT ＋ MULTI | 両者の併用 |

## 実験設定

| 課題 | モデル | 対応づけ |
| --- | --- | --- |
| **MNIST-PVR**（§4．MNIST の数字4枚の格子．左上の画像が，他の3枚のどれを選ぶかを指す） | ResNet18 | 最初の max-pooling を掛けた層の4象限 `Q_TL, Q_TR, Q_BL, Q_BR` ↔ `Y_TL, Y_TR, Y_BL, Y_BR`．"aligning layers closer to the classifier head was ineffective" |
| **ReaSCAN**（§5．命令文と格子世界から行動列を予測する．関係節を含まない最も単純な命令構造のみ） | CNN-LSTM（ReaSCAN の元の baseline） | 名詞トークンの上の LSTM encoder 出力 `e_Shape`（75 次元）を 25 次元×3 ↔ 対象の大きさ・色・形．decoder LSTM の初期隠れ状態 `h_0`（100 次元）を 50 次元×2 ↔ 行方向・列方向の位置差 |
| **MQNLI**（§6．2文の意味関係を entailment・contradiction・neutral に分ける合成の NLI） | 事前学習済みの BERT（付録C：uncased-base，12層，隠れ次元 768） | 中間変数 `QP_Obj` ↔ 第1文の動詞の上の表現．層は {0, 2, 4, 6, 8, 10}（図4は層ごとの値を示す） |

- **MNIST-PVR の分布シフト**：Zhang ら（2021）の分割で，右上の数字が {1,2,3}，左下が {4,5,6}，右下が {0,7,8,9} の**いずれか**に当たる訓練例を除く．
- **ReaSCAN**：評価は行動列の完全一致（exact match）．分割は Novel color／Novel size／Novel direction／Novel length．各実験を乱数種3つで繰り返す（付録B）．
- **MQNLI**：訓練 500K・dev 60k・test 10K（"the most difficult generalization scheme of Geiger et al. (2019)"）．base 50,000 例それぞれに source を 20 例組み，同じ 100 万対を毎エポック使う（付録C）．

## 結果

### MNIST-PVR（表1，%）

| 学習 | 振る舞い train | 振る舞い test | 入れ替え介入 train | 入れ替え介入 test |
| --- | --- | --- | --- | --- |
| STANDARD | 99.10 | 0.00 | 88.80 | 20.60 |
| IIT | 99.60 | 93.93 | 99.00 | 94.85 |
| MULTI | 99.64 | 0.00 | 89.35 | 20.50 |
| **IIT ＋ MULTI** | 99.60 | **96.01** | 99.10 | **96.64** |
| AUGMENT | 99.40 | 90.90 | 98.90 | 92.00 |
| NO TYPING | 99.41 | 0.09 | 99.47 | 16.88 |

- **STANDARD と MULTI は，振る舞いの test がどちらも 0.00**："Neither the standard nor multi-task models learned the behavioral objective in a way that generalizes, with total failure on the testing data (0%)."
- typed IIT の目的関数を外すと（NO TYPING），test で振る舞いも入れ替え介入も落ちる："Typing our variables is crucial for generalization."

### ReaSCAN（図3b，exact match %）

括弧内の数が何であるか，括弧の外の数が乱数種3つの平均であるかは，図にも本文にも書かれていない（検索した語：standard deviation，std，mean，average，variance，error bar）．

| 学習 | Novel color | Novel size | Novel direction | Novel length |
| --- | --- | --- | --- | --- |
| 振る舞い：STANDARD | 55.98 (6.31) | 41.67 (6.24) | 0.00 (0.00) | 5.72 (3.44) |
| 振る舞い：MULTI | 76.91 (5.02) | 39.46 (7.68) | 0.00 (0.00) | 9.05 (5.28) |
| 振る舞い：IIT | 74.12 (6.00) | 65.65 (4.26) | 0.26 (0.14) | 10.20 (6.08) |
| 振る舞い：**IIT ＋ MULTI** | **80.37** (0.88) | **74.84** (0.04) | **14.72** (3.54) | **25.82** (0.37) |
| 入れ替え介入：STANDARD | 44.26 (2.76) | 35.57 (2.64) | 0.00 (0.00) | 0.30 (0.21) |
| 入れ替え介入：MULTI | 68.42 (0.20) | 45.83 (2.45) | 0.00 (0.00) | 0.19 (0.05) |
| 入れ替え介入：IIT | 70.63 (9.33) | 65.18 (2.84) | 5.24 (3.07) | 4.75 (2.06) |
| 入れ替え介入：**IIT ＋ MULTI** | **70.73** (6.86) | **75.34** (0.91) | **11.79** (2.57) | **8.49** (1.53) |

### MQNLI（図4．数表は無く，本文の記述のみ載せる）

- IIT は "solves MQNLI with near perfect accuracy and IIT accuracy (≈100%)"．`QP_Obj` を BERT の最初の数層に対応づけると MQNLI の正解率は下がり，第6層以降ではほぼ完全な正解率と入れ替え介入の正解率になる（§6）．
- **データ拡張は振る舞いでは完全だが，入れ替え介入の正解率は "very low"（図4の説明文）**——"Data augmentation results in models with perfect behavioral performance." 著者はこれを，データ拡張が領域外への汎化の問題を取り除くので意外ではないとする．図4の説明文は "IIT is necessary for an interpretable model with high-performance." と結ぶ．

## 著者が明記した限界

限界を題とする節は無い．本文と付録で次を述べる．

- **対応づけの選び方が結果を左右する**（付録B，ReaSCAN）："we find that the specific mapping chosen substantially affects model performance and generalizability."
- **因果構造は一意でない**（付録B）："Our released symbolic causal structures for solving ReaSCAN is not unique, and may not be the optimal one for improving generalizability."
- **base–source 対の作り方が効く**（付録B）："We found that the formulation of the pairs affects performance." 対の作り方の影響は今後の研究に残している．
- **因果抽象の保証は損失が 0 のときのものである**．著者は損失が一般に 0 になるとは見込まず（§3），複雑なネットワークで完全な IntInvAcc はまれだとする（上記）．著者はこれを限界としては書いていない．

## 著者が述べる含意

"These initial findings suggest that IIT is a flexible and powerful way to bring high-level insights about causal structure into a data-driven learning process."（§7 Conclusion）

## 射程の外

- **対象は MNIST-PVR（ResNet18）・ReaSCAN（CNN-LSTM）・MQNLI（BERT）の3課題**である．視覚質問応答・GQA は全文に現れない（検索した語：GQA，VQA，visual question，vision-language．いずれも 0 件）．ReaSCAN の CNN-LSTM は命令文と格子世界を入力に取り，原文は "multimodal sequence-to-sequence model" と呼ぶ．
- **高水準の因果モデルを課題ごとに著者が定め，それを oracle として反実仮想の出力のラベルを得る**ことが前提である．本研究のように，ラベルの一致する表現どうしを対照損失で近づける設計（研究文書 §3.2）は現れない（contrastive は 0 件）．
- 対応づける先は，課題ごとに著者が選んだ部位（象限，ベクトルの分割，第1文の動詞の上の表現）である．本研究が整合を掛けるプロンプト末尾の位置（研究文書 §3.3）への対応づけは現れない（last token，final token は 0 件）．
- 評価は，振る舞いの正解率と，入れ替え介入の正解率である．本研究が答えの位置への依存を測るのに使う，ヤコビ行列による相対感度（研究文書 §A.4 第3段）にあたる測定は現れない（Jacobian は 0 件）．勾配に基づく帰属法は関連研究（§2）で触れるだけである．逆に，本研究は整合を掛けた表現を壊して出力の変化を見る方法を採らず，性質を除く介入もしていない（研究文書 §A.4）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴），arXiv PDF v2（15頁，付録 A〜C を含む全文）を pdftotext でテキスト化して読み，3・5・6・7・9・13頁はページ画像でも見て（式の記号，表1，図3b，付録A），PMLR の公式書誌ページ（巻・頁）と PMLR 掲載版 PDF（15頁．テキストは arXiv v2 と arXiv の刻印を除き一致），arXiv PDF v1（14頁）を取得し，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．v1 は MNIST-PVR と ReaSCAN の2課題だけで，MQNLI は v2 で加わった．表1と ReaSCAN の数値は v1 と v2 で同じである（ReaSCAN の表は v1 では表2，v2 では図3b）．下書き（2026-10-11 作成）から直した主な点：冒頭の一文と「位置づけ」から，原文に無い「読めるだけでなく使われる」「ラベルの線形な読み取りやすさ」を外した．式2を原文の記号（`M_H*`，`M_L`，`V_H`）に戻した．IntInvAcc の読み（「良い近似」）を原文の句に替えた．STANDARD に定義の文が無いこと，ReaSCAN に AUGMENT の行が無いこと，ReaSCAN の数が平均であるとは書かれていないことを明記した．MULTI の test 0.00 に付けた解釈の文を原文の引用に替えた．結論の引用を一文の逐語に直した．「損失 0 でなければ保証されない」を，著者が限界として書いていないと分かる形に直した．「射程の外」の否定の主張を検索した語とともに書き直し，「hidden state 全体への割り当ては検討されていない」（MQNLI は動詞の位置の表現をそのまま対応づける）を外した．本研究についての文に研究文書の節番号を付けた．**読めていないもの**：図4の曲線の値（本文と説明文の記述だけを載せた）．図1・図2・図5の図中の値は，説明文と本文に書かれたものだけを見た．付録Bの表2（ReaSCAN の分割の件数）は読んだが，本文書には載せていない．Zhang ら（2021），Wu ら（2021），Geiger ら（2019・2020）など引用先の論文は読んでいない．
