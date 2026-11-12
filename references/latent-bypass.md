# What's Holding Back Latent Visual Reasoning?

**著者**：André G. Viveiros（Instituto Superior Técnico, Universidade de Lisboa／Instituto de Telecomunicações／Gandara），Nuno Gonçalves（Instituto Superior Técnico／Instituto de Telecomunicações／Carnegie Mellon University），André F. T. Martins（Instituto Superior Técnico／Instituto de Telecomunicações／Gandara），Matthias Lindemann（Instituto de Telecomunicações）．所属は v3 の表記．v2 では3番目の所属が TransPerfect で，Viveiros には付いていなかった．
**掲載**：NeurIPS 2026（ポスター．neurips.cc の発表ページで確認，proceedings は未確認）．arXiv:2605.18445 v3，2026-11-07（v1 は 2026-05-18，v2 は 2026-05-19）
**原文**：[arXiv](https://arxiv.org/abs/2605.18445)

**latent 視覚推論のモデルは，連続 latent トークンを答えにほとんど使っていない．妨げているのは，「画像以上の情報をほとんど持たない oracle latent」と，「推論時に oracle から離れて狭い領域に潰れる予測 latent」の二つである．**

本文書の節・表・付録の番号は v3 のものである．v2 では付録が A–I の9つで，v3 の付録 F–M が v2 の B–I に，Table 10・11 が v2 の Table 6・7 にあたる．下に載せた数値のうち v2 にもあるもの（Table 1–5，v2 の Table 7，データの件数・比率）は，v2 と v3 で同じである．

## 主張

latent 視覚推論は，中間的な視覚ステップを連続 latent トークンとして生成させてから答えさせる．本論文はその latent が予測に因果的に効いているかを調べ，こう結論する——"Surprisingly, we find that models largely do not take latent tokens into account for their answers."

著者は二つの failure mode を挙げる．

1. **latent bypass problem**——"The latent bypass problem refers to the tendency of the model to ignore latent representations at inference time"．著者はこれが学習データから生じるとする．既存データの oracle latent は中間画像から作られるが，中間画像は入力画像の部分領域であることが多く，"making them easily recoverable from the input image itself"．学習時に latent に頼る誘因が小さい．
2. **latent representation collapse**——推論時に生成される latent は，対応する oracle latent から離れ，互いによく似た狭い領域へ潰れる．モデルが latent に頼る場合でも，これが利益を妨げる．

latent が最終予測を十分に支えるよう作った診断データで微調整すれば，モデルは latent に因果的に頼りうる．

## 手法

### 対象の学習目的（Eq. 1）

次トークンの交差エントロピーに latent 整合項を足す．

```
L = −(1/T) Σ_t log P_θ(y_t | y_<t, z, x)  +  γ (1/K) Σ_k ℓ(ẑ_k, z*_k)
```

- K は latent トークン数，ẑ は生成 latent，z* は oracle latent．z は teacher forcing で学習するなら z*，そうでなければ ẑ．
- **ℓ は MSE または cosine distance**（"either the mean squared error (MSE) or cosine distance"）．
- γ は Tetris 型データでの LanteRn の再学習で 0.05（表の記載は "0.05 (default: 0.1)"），ℓ は MSE（付録 J，Table 11）．既製の4手法それぞれの γ と ℓ の選択は，見当たらなかった（検索した語：gamma，γ，MSE，cosine）．
- 2段階：まず生成 latent を oracle latent に整合させ，続いて整合項を外して下流課題の性能を（"typically through RL"）最適化する．

### 対象の4手法

| 手法 | 規模・段階 | oracle latent の作り方 |
| --- | --- | --- |
| LVR | 7B（付録 F の Table 10 の段階表記は "Base"．Figure 3 と Table 5 の列見出しは "RL"） | 凍結した視覚エンコーダで**画像全体**の埋め込みを計算し，該当する部分領域の埋め込みを選ぶ．平均プーリングで K 個にする |
| LanteRn | 3B，SFT／RL | 凍結した視覚エンコーダで**中間画像を独立に**符号化し，平均プーリングで K 個にする |
| ILVR | 7B（Table 10 は "SFT-Stage2"） | latent ブロックを複数生成し，そのすべてに整合損失を課す．目標は momentum 教師（オンラインのパラメタの指数移動平均）が生成する |
| Monet | 7B，SFT／RL | 教師モデルの hidden state から蒸留する．oracle latent を条件にせず，生成した latent を次の入力に与える free-running で学習する |

ベース VLM として名前が出るのは，Tetris 型での LanteRn の再学習に使う **Qwen2.5-VL-3B** だけである（Table 11）．既製の checkpoint については，規模と段階と配布先のリンクが載る（Table 10）．

### 介入（Section 3）

既製の checkpoint をそのまま走らせ（standard inference），生成された latent を次で置き換える．

| 介入 | 中身 |
| --- | --- |
| Random Subregion | 無関係な画像の切り出しから，学習時と同じ作り方で作った oracle latent |
| Zeros | 全零 |
| Noise | N(0, 1) のガウス雑音 |
| Skip Latents | latent_start の後に latent_end を置き，latent の経路を無効にする |

これとは別に，VisCoT では正解の中間画像から作った **Oracle Latents** を与えた場合も測る．

### Pause Tokens 基準（Section 4）

学習時に oracle latent を placeholder トークンへ置き換えて学習した基準．"the same number of time steps and compute but cannot condition on the intermediate image"．推論時は，LanteRn には oracle latent を与え，Pause Tokens 基準は通常の自己回帰で復号する．

### 予測 latent と oracle latent を比べる指標（Section 5）

- **Retrieval@k**：予測 latent ごとに，全サンプルの oracle latent を cosine 類似度で順位づけ，対応する oracle が上位 k（1・5・10）に入る割合．
- **USP**（unrelated self-generated preference）：予測 latent が，対応する oracle latent よりも**無関係な別の予測 latent**に近い頻度．Table 4 の Random Baseline の行は 50.0．
- **Sim. Within Pred／Oracle**：予測 latent どうし，oracle latent どうしの平均の類似度．

## 実験設定

- **学習データ**：LanteRn・LVR・Monet は VisCoT（中間の bounding box 付き）を主なデータ源とし，ILVR は COMT と VSP を主なデータ源とする．
- **LanteRn データの濾過**（付録 I）：元の 143,024 件から，Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 が画像なしで正答した例を除き，**約30%** 減らした．評価用に 5k 件を取り置く．
- **評価**：VisCoT，BLINK（物体の定位と空間推論に関わる課題），V\* Bench（細かな視覚的詳細の探索と相対的な空間推論）．v3 は付録 C で GQA と BabyVision を加えた．
- **Tetris 型の診断データ**：A と，A を回転した B を見て，同じ回転を C に施した結果を4つの候補から選ぶ．中間画像からは色を除く．**8K 件**を作り，学習 4,000 件・評価 400 件を使う．ペントミノ 約41%，ヘキソミノ 約35%，テトロミノ 約24%（付録 K）．
- **Tetris 型での再学習**（付録 J，Table 11）：Qwen2.5-VL-3B，latent_size 8，15 エポック，学習率 1e-5，cosine scheduler，warmup 比 0.05，γ = 0.05，MSE．
- **Monet-RL の答えの取り出し**（付録 L）：介入下では答えの書式が出ないことがあるので，Monet-RL に限り，定型の接尾辞を足して続きを復号させる手順を使う．

## 結果

### latent を置き換えても正解率はほとんど変わらない（Table 1，VisCoT，%）

| 条件 | ILVR | LVR | LanteRn RL | LanteRn SFT | Monet RL | Monet SFT |
| --- | --- | --- | --- | --- | --- | --- |
| Random Subregion | 64 | 73 | 82 | 78 | 81 | 75 |
| Noise | 61 | 74 | 82 | 80 | 82 | 71 |
| Zeros | 59 | 76 | 82 | 79 | 85 | 41 |
| Skip Latents | 60 | 78 | 82 | 79 | 83 | 78 |
| Standard Inference | 62 | 73 | 83 | 78 | 83 | 76 |
| Oracle Latents | 63 | 77 | 83 | 79 | 86 | 77 |

- LVR と Monet SFT では，latent を生成させない Skip Latents が標準推論を上回る（73→78，76→78）．
- 標準推論から10ポイント以上落ちるのは，この表では Monet SFT の Zeros（76→41）だけである．
- **oracle latent を与えても上積みは一貫しない**（ILVR 62→63，LVR 73→77，Monet RL 83→86，LanteRn RL 83→83）．本文は "providing oracle latent tokens does not lead to consistent performance improvements" と書く．

### BLINK と V\*（Figure 3，付録 A の Table 5，正解率）

Figure 3 の説明文："For every model and dataset, there is always at least one intervention that maintains accuracy within 2 percentage points"．

| データ | 介入 | ILVR 7B SFT | LVR 7B RL | LanteRn 3B RL | LanteRn 3B SFT | Monet 7B RL | Monet 7B SFT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BLINK | Noise | 0.62 | 0.75 | 0.67 | 0.65 | 0.61 | 0.57 |
| BLINK | Random Subregion | 0.59 | 0.70 | 0.67 | 0.65 | 0.61 | 0.59 |
| BLINK | Skip Latents | 0.60 | 0.73 | 0.65 | 0.65 | 0.60 | 0.57 |
| BLINK | Standard Inference | 0.63 | 0.68 | 0.65 | 0.65 | 0.63 | 0.60 |
| BLINK | Zeros | 0.61 | 0.76 | 0.66 | 0.62 | 0.64 | 0.50 |
| V\* | Noise | 0.60 | 0.81 | 0.70 | 0.64 | 0.80 | 0.76 |
| V\* | Random Subregion | 0.61 | 0.81 | 0.71 | 0.63 | 0.79 | 0.77 |
| V\* | Skip Latents | 0.46 | 0.81 | 0.71 | 0.62 | 0.78 | 0.78 |
| V\* | Standard Inference | 0.63 | 0.80 | 0.72 | 0.63 | 0.82 | 0.75 |
| V\* | Zeros | 0.65 | 0.80 | 0.70 | 0.60 | 0.80 | 0.77 |

### GQA（v3 で加わった付録 C，Table 7，正解率 %）

v2 に GQA は現れない．v3 は GQA を "close to the training distribution of the evaluated models" と述べて評価に加えた．用いた分割と件数は書かれていない．Lat. % は，標準推論でモデルが latent トークンを生成したサンプルの割合である．

| モデル | Standard | Rand. Subregion | Noise | Zeros | Skip Latents | Lat. % |
| --- | --- | --- | --- | --- | --- | --- |
| LanteRn SFT | 79.2 | 78.8 | 79.0 | 79.4 | 78.6 | 100.0 |
| LanteRn RL | 81.2 | 81.9 | 81.6 | 81.8 | 81.8 | 100.0 |
| LVR-7B | 78.1 | 76.7 | 81.1 | 80.5 | 83.7 | 100.0 |
| Monet SFT | 70.2 | 69.8 | 41.7 | 33.8 | 33.1 | 97.0 |
| Monet RL | 74.7 | 74.5 | 75.0 | 75.5 | 75.1 | 97.0 |
| ILVR | 83.9 | 83.9 | 83.9 | 83.9 | 83.9 | 1.1 |

著者の読み："On GQA, every model has at least one intervention that performs about as well as or better than standard inference."．Monet SFT は Noise・Zeros・Skip Latents で大きく落ちるが，Random Subregion では 0.4pp 以内にとどまる．ILVR は latent トークンを 1.1% のサンプルでしか生成しない．

### oracle latent は課題をほとんど易しくしない（Section 4）

**Pause Tokens 基準も 79%**．中間画像を条件にできない Pause Tokens 基準が，oracle latent を与えた LanteRn と同じ正解率になる（Table 2 の Std. 列）．

### latent が有益なら使われるようになる

**部分領域をマスクして学習**（Table 2，VisCoT，LanteRn，%）——入力画像の該当領域を隠し，中間画像は元のままにする．Std. は濾過した元データでの学習・評価，Masked は該当領域を隠した画像での学習である．

| latent の出所 | Std. | Masked |
| --- | --- | --- |
| Oracle latents | 79 | 75 |
| Random Subregion | 78 | 61 |
| Noise | 79 | 59 |
| Zeros | 80 | 59 |
| Pause Tokens | 79 | 59 |

Masked について著者は "This is the first setting where we observe a clear gap between oracle latents and the interventions" と書く（75 対 59–61）．

**Tetris 型の回転**（Table 3，LanteRn の設定，正解率）．部分領域の介入の代わりに，無作為な中間画像を使う．

| latent の出所 | 正解率 |
| --- | --- |
| Oracle latents | 0.86 |
| Standard Inference | 0.33 |
| Random Inter. Image | 0.30 |
| Noise | 0.26 |
| Zeros | 0.25 |
| Random baseline（v3 で加わった行） | 0.25 |
| Pause Tokens | 0.34 |

- oracle latent を課題と無関係な情報に置き換える介入は，どれも正解率を偶然水準へ落とす（v3 は偶然水準を 25% と明記する）．著者はこれを，latent がモデルの予測で重要な役割を果たしている証拠とする．
- **モデルが自分で生成した latent では 0.33**で，偶然水準をわずかに上回るだけである．著者は "revealing a large gap between oracle and generated latents" と書き，これを Section 5 の出発点にする．Pause Tokens 基準は 0.34 である．

### 予測 latent は oracle から離れ，潰れる（Table 4，VisCoT の 300 件）

| モデル | Retrieval@1 (%) ↑ | @5 (%) ↑ | @10 (%) ↑ | USP (%) ↓ | Sim. Within Pred | Sim. Within Oracle |
| --- | --- | --- | --- | --- | --- | --- |
| ILVR | 0.3 | 3.8 | 6.9 | 99.0 | 0.82 | 0.33 |
| LVR | 3.3 | 6.3 | 10.0 | 99.6 | 0.85 | 0.22 |
| LanteRn SFT | 0.3 | 4.3 | 6.3 | 100.0 | 0.98 | 0.60 |
| LanteRn RL | 0.3 | 3.0 | 5.7 | 100.0 | 0.97 | 0.60 |
| Monet SFT | 1.0 | 2.7 | 7.0 | 100.0 | 0.91 | 0.96 |
| Monet RL | 2.0 | 2.7 | 7.0 | 100.0 | 0.91 | 0.96 |
| Random Baseline | 0.3 | 1.6 | 3.3 | 50.0 | – | – |

v3 はこの表に Tetris 型データの2行を加えた（LanteRn SFT：0.0，1.2，2.2，100.0，0.98，0.90．Random Baseline：0.2，1.2，2.4，50.0）．

- **予測 latent は対応する oracle latent をほとんど引き当てない**（Retrieval@10 でも 10.0% 以下）．
- **USP は 99.0–100.0%**——予測 latent は，対応する oracle よりも別の予測 latent に近い．
- 予測どうしの類似度は 0.82–0.98，oracle どうしは 0.22–0.60 である．Monet だけは oracle どうしも 0.96 と高い．
- **時間とともに潰れる**：連続する時刻の予測 latent は "collapse to increasingly similar representations over time, a trend not reflected in the oracle latents"（Figure 6）．Monet は oracle latent も時間とともに似てくる例外で，著者は，Monet だけが oracle latent を教師モデルから自己回帰的に生成するためかもしれないと述べる．

### v3 で加わり，本文書が要約していないもの

介入の有意性の検定（付録 B），BabyVision の結果（付録 C），latent の位置への注意の割合（付録 D），部分的な teacher forcing による誤差の蓄積の実験（付録 E）．

## 著者が明記した限界

Section 6（Discussion and Limitations）による．

- **対象の範囲**：4手法・2規模（3B と 7B）・学習段階（SFT と RL）で結果は一貫するとしたうえで，監督信号やベースモデルの設計が大きく違う場合にどうなるかは "it remains an open question to what extent" と未解決にする．
- **全モデルは見ていない**："a comprehensive analysis of all existing models is infeasible"．代わりに，今後の研究が評価の一部として latent の bypass を検査することを勧める．
- **実データは未解決**：Tetris 型の実験は幾何変換を有望な領域として示唆するが，latent 視覚推論のための包括的な実世界データがどのようなものかは開かれた問題とする．

## 著者が述べる含意

- **データ設計**：中間ステップはモデルの助けになる足場を与える必要があり，データ作成時に，中間画像を条件にしない pause tokens のような単純な基準で検査できる．
- **collapse の診断**：予測 latent どうしの類似度と，予測と oracle の類似度を監視することを勧める．
- 今後の進歩は，有益な中間ステップを持つ質の高いデータと，より正確な latent トークンの予測の二つにかかる（要旨）．

## 射程の外

- **問うのは「モデルが生成した latent トークンを答えが使うか」である．** latent は，hidden state を言語モデルのヘッドに通さずに自己回帰で生成する連続トークンの列で，本研究が整合を掛ける第16層の末尾位置の表現（研究文書 §3.3）とは別のものである．
- **操作は推論時の latent の置き換えと，診断データでの再学習である．** 線形プローブによる表現の読み取りは見当たらなかった（v2・v3 の全文で検索した語：prob，linear）．
- **整合項の目標は，中間画像から作る oracle latent（Monet では教師の hidden state）である．** 課題の種類をラベルにした対照学習は見当たらなかった（検索した語：contrastive）．本研究の整合損失は署名をラベルとする supervised contrastive loss である（研究文書 §3.2）．
- **GQA は v2 には現れず，v3 の付録 C で評価データに加わった．** GQA で測ったのは介入下の正解率と Lat. % で，GQA の program を使った分析は見当たらなかった．
- **SmolVLM は現れない**（検索した語：SmolVLM）．対象の規模は 3B と 7B，ベースとして名前が出るのは Qwen2.5-VL-3B だけである．本研究のモデルは SmolVLM-500M-Instruct である（研究文書 §4.1）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・v1–v3 の履歴．Comments と Journal-ref の記載は無い），PDF v3（18頁，付録 A–M まで）と PDF v2（16頁，付録 A–I まで）の全文を pdftotext でテキスト化したもの，v3 の 1・3・5・7・8・13–16 頁の画像（Eq. 1，Table 1–11，所属），neurips.cc の NeurIPS 2026 の発表ページ（題と著者）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．v1 は，テキスト化したものが v2 と arXiv の刻印の行でしか違わないことだけを確かめた．下書き（2026-10-11 作成）から直した主な点：掲載を「arXiv のみ」から NeurIPS 2026 へ，版を v2 から v3 へ改め，所属を著者ごとに書いた（v3 で TransPerfect が Gandara に変わった）．「GQA は評価データに含まれない」は v2 でしか成り立たないので，v3 の付録 C の内容に直した．節・表・付録の番号を v3 に合わせた．原因の二つを，bypass の原因（データ）と，別の問題としての collapse に分けた．原文に無い語（drift，上限の参照，「予測の不正確さに帰される」）を原文の表現に直した．「oracle を与えても上積みは小さい」を Section 4 から Table 1 の節（原文では Section 3）へ移した．Figure 3 の説明文の引用を，VisCoT の節から BLINK と V\* の節へ移した．Monet-RL の強制補完は著者が限界として書いていないので実験設定へ移した．表の太字は原文の太字と一致しなかったので外した．否定の主張（γ・ℓ の選択，probing，対照学習，SmolVLM）を検索した語つきの「見当たらなかった」に直した．**読めていないもの**：OpenReview のページ（検索で見つからなかった），NeurIPS の proceedings，Figure 3・6 の図から読み取る値（本文と説明文に書かれたものだけを照合した），v2 の頁の画像（v2 の表はテキスト化したものだけで照合した），論文がリンクするコード・モデル・データ．
