# Cosine Misleads: Auxiliary Losses Reshape Vision Language Models, Not Their Latents

**著者**：XiuYu Zhang，Junfeng Fang，Zhenkai Liang（National University of Singapore）
**掲載**：arXiv のみ確認．arXiv:2606.05753 v1，2026-06-04
**原文**：[arXiv](https://arxiv.org/abs/2606.05753)

**latent トークンを視覚標的へ寄せる補助損失は，名目上それが最適化する latent を通じてではなく，共有パラメタを通じて言語モデルを作り替える——cosine の整合度は5変種を通じて正解率と負に相関し，単独では誤解を招く信号である．**

## 位置づけ

補助損失で中間表現に性質を足したとき，その表現から答えが「読める」ことと，その表現が「使われる」ことを分けて測った論文である．研究文書（`.claude/context.md`）は，2026-11-13 の時点で本論文を引いていない．

## 主張

- latent visual reasoning（LVR）は，知覚と答え生成のあいだに教師つきの latent トークンを挟む．著者によれば，分野では latent と視覚標的の整合（cosine 類似度または MSE）が**学習損失と質の指標を兼ねて**いる．
- 5変種を通じて **cosine 整合度は V\*Bench 正解率と負に相関する（Pearson r = −0.94）**．
- 推論時の診断 PRISM を二つ導入する——どこで答えが線形に読めるかを測る**線形プローブ**と，latent が答えを支えているかを測る **corruption test**．
- 答えは latent より下流（答えを生成する状態）では読めるが，latent の位置では読めない．latent を壊しても正解率は "at most four points" しか動かない（要旨）．
- 著者の解釈（§6）："because the auxiliary loss reshapes the VLM through gradient flow into shared parameters"．

## 手法

### 学習目的

```
L_LVR = L_CE + λ · MSE(h_t, v_t)
```

`h_t` は LVR 位置の hidden state，`v_t` は teacher-forced の視覚標的（式1）．**λ = 0.1 を全変種の共通の損失で固定**（§5.1："with λ = 0.1 throughout"．ただし D-LVR は途中から 0 にする）．**5変種の共通の損失は MSE である．** 関連研究には MSE の項を cosine や patch 単位の整合損失に替える，または他の損失と混ぜるものがあると著者は述べる（§3.1）．本論文が cosine を報告するのは測る量としてである——LVR 位置の予測 hidden state と teacher-forced 標的の cosine を位置で平均し，Visual-CoT の held-out 検証片（全変種で同じ片）で，teacher-forced の生成のもとで測る（付録 D.4）．

### 5変種

基盤は Qwen2.5-VL-3B-Instruct．視覚塔と visual merger は凍結し，言語モデルを学習する（§5.1）．

| 変種 | 構成 |
| --- | --- |
| LVR | K 位置の latent ブロック1つ．各位置を teacher の視覚埋め込みで教師する．追加の制約なし |
| N-LVR | 学習時に teacher-forced 入力へ平均0のガウス雑音（σ = 0.3）．標的は雑音なし．推論時は雑音なし |
| D-LVR | LVR の step 1500 の checkpoint から再開し，再構成項を0（λ_rec = 0）として 1000 step 追加学習（この間の教師は答えテキストへの CE のみ） |
| P-LVR-2 | 2段の latent：文脈段（bounding box を α = 1.5 で拡大）→ 標的段（元の box） |
| P-LVR-3 | 文脈段 → 自由段 → 標的段．自由段に多様性正則化 λ_div = 0.01 |

K の値は明記されない（Table 4 の caption に，corruption test の実行条件として steps=8／steps=16 per stage の記載はある）．

### PRISM 1：線形プローブ

| 位置 | 読む状態 |
| --- | --- |
| (a) | 最初の答えテキストトークンを生成する反復での hidden state．LM head が読む，答えを生成する状態 |
| (b) | 自己回帰ループが次の位置の入力埋め込みとして再注入する，LVR 位置の生の hidden state `h`（間に学習された射影なし） |

- 多クラスのロジスティック回帰で {A, B, C, D} を当てる．次元 D = 2048．scikit-learn，L2 正則化 C = 1.0，LBFGS，各次元を訓練 fold で z-score．層化 5-fold 交差検証の平均精度（付録 A.1–A.2）．
- **対照**：V\*Bench の各問に {A, B, C, D} から一様無作為なラベルを固定して同じ手順を回す．**selectivity ＝ 実ラベルでの精度 − 対照ラベルでの精度**（付録 A.3）．
- **decodability gap G ＝ acc_probe(a) − acc_probe(b)．**
- 読む層の番号は書かれていない（全文で "layer" を検索した）．

### PRISM 2：corruption test

LVR モードの各反復で，入力埋め込みを K/V 射影の前に壊す（付録 B.6）．greedy 復号，最大 512 新トークン（付録 D.2）．

| 種類 | 操作 |
| --- | --- |
| truncation | `h ← 0`（Qwen2.5-VL の K/V 射影に bias がないので K・V は0になるが，softmax の分母を通じて注意の質量は占める） |
| noise | `h ← h + σ·η`，`η ~ N(0, I)`，σ ∈ {0.1, 0.3, 1.0} |
| swap | 同じ LVR step の donor の `h` に置換．donor は clean パスで記録した 64 サンプルの pool から復元抽出 |

Δ が負なら壊すと正解率が下がる＝自分の latent が役立っている．

## 実験設定

- **学習**：Visual-CoT-438k．2500 step，学習率 10⁻⁵（cosine schedule），実効バッチ 64．
- **評価**：V\*Bench 191 問（direct_attributes 115，relative_position 76），MMVP 300 対，BLINK の5つの検証部分集合（Counting，IQ_Test，Jigsaw，Relative_Reflectance，Spatial_Relation）計 697 問．
- corruption test（Table 4 の caption）はバッチ 1，単段の変種は steps=8，P-LVR-2／3 は段ごとに steps=16．

## 結果

### 主表（Table 1）

| 変種 | cos | V\*B | MMVP | BLINK | probe (a) | probe (b) | G | trunc | σ=.1 | σ=.3 | σ=1.0 | swap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LVR | 0.555 | 70.2 | 49.7 | 53.4 | 69.1 | 32.5 | 36.6 | −2.6 | −1.6 | +1.0 | −1.0 | +1.6 |
| N-LVR | 0.556 | 71.7 | 50.0 | 52.9 | 66.0 | 41.9 | 24.1 | −2.6 | −0.5 | −1.0 | −2.1 | −0.5 |
| D-LVR | 0.464 | 69.6 | 49.3 | 51.4 | 64.9 | 33.0 | 31.9 | −1.1 | 0.0 | +0.5 | 0.0 | +2.1 |
| P-LVR-2 | 0.777 | 57.1 | 48.7 | 47.3 | 50.2 | 35.6 | 14.6 | 0.0 | +2.1 | 0.0 | −0.5 | +2.1 |
| P-LVR-3 | 0.769 | 57.1 | 48.0 | 48.5 | 48.7 | 34.6 | 14.1 | +2.1 | +2.6 | +4.2 | +3.1 | +2.6 |

正解率・プローブは %，G はポイント，corruption は V\*Bench 正解率の差（ポイント）．

- **cosine が最も高い P-LVR-2／3 が，V\*Bench の正解率が最も低い．** 著者は序論で，段階化の変種が再構成誤差を 3.79 から 1.55 に下げるが，V\*Bench を 13 ポイント失うと述べる．
- **D-LVR は cosine が5変種で最も低い 0.464 だが，V\*Bench は 69.6 で LVR（70.2）に近い**（Table 1 の値）．
- **probe (b) は全変種で 32.5–41.9% にとどまる**（chance 25%）．probe (a) は Table 1 で 48.7–69.1%．
- selectivity は位置 (a) で全変種 25% 超，**位置 (b) では一様に 10% 未満**（付録 A.3．変種ごとの値は載っていない）．
- corruption による変化は −2.6 から +4.2 ポイント．P-LVR の2変種では，P-LVR-2 の σ = 1.0（−0.5）を除いて 0 か正である．

### 変種間の相関（Table 2，対 V\*Bench 正解率と corruption 応答）

| 診断量 | 対 V\*B | 対 trunc Δ | 対 σ=.1 Δ |
| --- | --- | --- | --- |
| cosine | −0.94 | +0.75 | +0.84 |
| probe (a) | +0.98 | −0.92 | −0.98 |
| probe (b) | +0.20 | −0.26 | −0.02 |
| G | +0.86 | −0.77 | −0.93 |

- **相関は5点（5変種）から計算した Pearson 相関である．** cosine 対 V\*B の bootstrap 95% CI は [−0.987, −0.611]（点推定 −0.939，付録 A.5）．
- probe (a) 対 V\*B：leave-one-out で [0.969, 0.996]，bootstrap 95% CI [+0.694, +0.993]．BLINK とは +0.96，MMVP とは +0.92（付録 A.5）．

### 付録の表（Table 3：プローブ，Table 4：corruption）

| 変種 | probe (a) | CE@(a) | probe (b) | Table 4 clean | Table 4 trunc |
| --- | --- | --- | --- | --- | --- |
| LVR | 69.1±2.2 | 1.151 | 32.5±3.8 | 70.2 | −2.6 |
| N-LVR | 66.0±3.3 | 1.307 | 41.9±3.9 | 71.7 | −2.6 |
| D-LVR | 64.9±2.4 | 1.122 | 33.0±2.5 | 69.1 | −1.0 |
| P-LVR-2 | 50.2±4.8 | 1.954 | 35.6±2.8 | 55.5 | 0.0 |
| P-LVR-3 | 48.7±1.7 | 1.935 | 34.6±1.8 | 56.0 | +2.1 |

Table 3 の精度は 5-fold の平均±標準誤差．H(Y) = 1.267 nats（191 問のラベル内訳 A 71，B 70，C 27，D 23）．

### 原文の数値の食い違い

- **D-LVR の trunc Δ が Table 1 で −1.1，Table 4 で −1.0．** clean 正解率も D-LVR 69.6／69.1，P-LVR-2 57.1／55.5，P-LVR-3 57.1／56.0 と食い違う（前が Table 1，後が Table 4）．Table 4 の caption は，clean の列が Table 1 の値とおよそ 2 ポイント以内で合うとしたうえで，"the residual difference is sample-order variance from the faith script’s single-pass greedy generation" と説明する．
- **"at most four points" に対し，P-LVR-3 の σ = 0.3 は +4.2．** 付録 B.5 は "within roughly 4 V\*Bench points" と書き，最大がこの +4.2 であると明記する．
- **probe (a) の下限が本文と表で食い違う．** §5.4 の本文は "ranges from 50.2 (P-LVR-3) to 69.1 (LVR)" と書くが，Table 1・Table 3 では 50.2 は P-LVR-2 の値で，P-LVR-3 は 48.7 である．

## 著者が明記した限界

- **単一モデル・単一コーパス**："Our evidence comes from one base model (Qwen2.5-VL-3B-Instruct) and one fine-tuning corpus (Visual-CoT-438k)."
- **プローブは使用を測らない**："Linear probes measure linear decodability rather than what downstream layers actually use"
- **摂動の範囲**："The corruption test covers truncation, three noise levels, and donor swap but does not exhaust the perturbation space."
- **ベンチマークの偏り**：細粒度の知覚（V\*Bench・MMVP・BLINK）に絞っており，"holistic reasoning benchmarks with less spatially localized questions may exhibit different cosine–accuracy patterns."
- Limitations の節の外では，付録 A.5 が，相関は5点からなるので記述的なものであり，bootstrap の区間から符号は頑健だが大きさは不確かである，と述べる．

## 射程の外

「本研究」についての記述は研究文書（`.claude/context.md`）の §3.2・§3.3・§4.1・§4.2 に照らした．

- **プローブと介入の対象は，LVR 位置の状態（自己回帰ループが再注入するもの）と答えを生成する状態の2か所**である．層ごとの中間表現を測ったという記述は見当たらず，プローブの層番号も書かれていない．本研究が整合損失を掛け，プローブで読むのは，第16層の，質問を読み終えた位置の表現である（研究文書 §3.3）．
- **補助損失は視覚標的への回帰（MSE）**であって，ラベルによる対照学習（本研究の整合損失は SupCon，研究文書 §3.2）ではない．標的は bounding box の領域の視覚埋め込みで，本研究のタスク署名のようなラベルではない．全文に "contrastive" の語は見当たらなかった．
- **補助損失を最初から掛けない群は見当たらなかった．** 5変種の共通の損失は λ = 0.1 で，D-LVR も LVR の step 1500 から再開したものである．最初から λ = 0 で学習した群や，未調整の基盤モデルの評価は，Table 1–4 にも本文にも見当たらなかった（検索した語：baseline，pretrained，zero-shot）．本研究は λ = 0 の baseline を置く（研究文書 §4.2）．
- **無作為ラベルはプローブの対照（selectivity）に使われる．** 学習時の標的やラベルを無作為にした群は見当たらなかった（検索した語：random）．本研究の ablation は学習時のラベルをバッチの中で並べ替える（研究文書 §4.2）．
- **GQA と SmolVLM は全文に現れない**（検索した語：GQA，SmolVLM）．基盤は Qwen2.5-VL-3B-Instruct，学習は Visual-CoT-438k，評価は V\*Bench・MMVP・BLINK である．本研究は SmolVLM-500M-Instruct と GQA を使う（研究文書 §4.1）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴．v1 のみで v2 は存在しない．Comments・Journal-ref の記載は無い）と PDF v1（15頁．本文8頁，Limitations と参考文献，付録 A–E を含む全文）を pdftotext でテキスト化して読み，Table 1・2（6頁），Table 3（10頁），Table 4（12頁）はページ画像でも見て，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．掲載先は，題での検索で arXiv とその二次的な紹介ページしか見つからなかったので「arXiv のみ確認」とした．下書き（2026-10-11 作成）から直した主な点：(1) 「主表には固有の caption がない」という食い違いの項を削った（PDF v1 の Table 1 には固有の caption がある）．(2) §5.4 の本文と表で probe (a) の下限が食い違うことを足した．(3) 「相関は5点で，符号は頑健・大きさは不確か」は Limitations の節でなく付録 A.5 の記述なので，そう書き分けた．(4) 冒頭の一文を原文の強さに直した（「モデル全体を作り替える」→言語モデル，「質の指標にならず」→単独では誤解を招く信号）．(5) 位置づけの「先行例として引く」を，研究文書が本論文を引いていないという事実に直した．(6) 射程の外の否定の主張を，全文検索で確かめた強さ（見当たらなかった，と検索語）に直し，本研究の側の記述に研究文書の節番号を付けた．(7) 限界の引用の "cosine-accuracy" を原文の "cosine–accuracy"（en dash）に，所属を原文の表記に直し，表の中の太字（原文の表には無い強調）を外した．**読めていないもの**：arXiv の HTML 版と TeX ソース，著者の公開コードは読んでいない．図1・図4・図5 の点の値は，本文と表に書かれた値だけを照合した．Table 3 の MI_LB・A1′ の列と偏相関（付録 A.5）は読んだが，下書きに無いので載せていない．
