# Roses are Red, Violets are Blue... But Should VQA expect Them To?

**著者**：Corentin Kervadec（Orange, Cesson-Sévigné／LIRIS, INSA-Lyon, UMR CNRS 5205），Grigory Antipov（Orange），Moez Baccouche（Orange），Christian Wolf（LIRIS, INSA-Lyon, UMR CNRS 5205）
**掲載**：CVPR 2021（Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition），pp. 2776–2785．arXiv:2006.05121（v1 2020-06-09，v2 2020-12-08，v3 2021-04-07）．以下の数値・引用・節番号は arXiv v3 のものである
**原文**：[arXiv](https://arxiv.org/abs/2006.05121)，[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Kervadec_Roses_Are_Red_Violets_Are_Blue..._but_Should_VQA_Expect_CVPR_2021_paper.html)

**全体の正解率は頻出する問と答えの組に押し上げられている——文脈ごとに稀な答え（tail）で測ると，評価した VQA モデルはどれも，頻出の答え（head）で測るより低い．**

## 主張

VQA モデルは推論の代わりにデータセットの偏りに頼りがちである．要旨は，標準の評価指標である in-domain の全体正解率は誤解を招くと主張する——質問と概念の分布が偏っているので，訓練集合の統計を利用するモデルに有利になる．一方，訓練と評価のあいだに人工的な分布のずれを素朴に作る方法も，ずれが現実の傾向を反映せず，その設定に合わせ込んだモデルを生むので十分でない，とする（本文は，このベンチマークを VQA-CP の代替と位置づける）．本論文は，**同じ文脈（質問の型と概念）の中で稀な答えと頻出の答えを分けて測る**ベンチマーク GQA-OOD を提案し，7つの VQA モデルと3つの偏り低減手法を調べる．GQA-OOD の全標本での正解率 acc-all については，§4.2 で "it is mechanically increased by the high scores obtained on the most frequent question-answers" と述べる．

## 手法

### GQA-OOD の作り方（§3）

GQA（balanced，version 1.2）を組み直す．**訓練集合は変えない**——"the two share the same (existing) training set"．ずれは検証と評価の側（val と testdev）に作る．

1. **文脈で問をまとめる．** GQA の注釈にある "local groups" を使う．これは質問の型（"What color ...?" など）を定め，質問に関係する概念（"zebra" など）にも依るまとまりである．
2. **偏りの強いまとまりを残す．** まとまりごとの答えの分布について，Shannon エントロピーを答えの種類数 $d$ の対数 $\log(d)$ で割った正規化エントロピーを計算し，"we keep groups with a normalised entropy smaller than a threshold empirically set to T=0.9."
3. **まとまりの中で答えを head と tail に分ける．** 答えのクラス $i$ の標本数を $|a_i|$，まとまり内の平均の標本数を $\mu(a)$ として，"We define the tail classes as classes i with |a_i| ≤ α μ(a)"．まとまり全体から tail を除いた残りが head．"We empirically set the parameter α=1.2"．$\alpha$ を小さくすると，最も稀な問と答えの組だけが評価される．

**ずれは文脈に依存する**——著者の例では，バラの色を問う文脈では red が頻出で blue が稀な（OOD の）答えだが，花がスミレに変われば OOD の答えは blue ではなく red になる．ずれは全体で一つではない（"this shift is not global but depends on the context"）．

なお tail の定義は版で異なる．arXiv v1 は $|a_i| \le (1+\alpha)\mu(a)$，$\alpha = 0.2$ と書く（v2・v3・CVPR 版は上のとおり）．

### 規模（Table 1）

| 分割 | 全体 | head | tail | まとまり数 |
| --- | --- | --- | --- | --- |
| GQA-OOD val | 51,045 | 33,882 | 17,163 | 3,849 |
| GQA-OOD testdev | 2,796 | 1,733 | 1,063 | 471 |
| （参照）GQA testdev | 12,578 | – | – | 7,803 |

（GQA testdev に head／tail の区分は無い．）

### 指標

- **acc-all**：GQA-OOD の全標本（head＋tail）での正解率．
- **acc-tail**：OOD 標本，すなわち文脈のもとで最も稀な答え（答えの分布の tail）の標本での正解率．
- **acc-head**：まとまりごとの分布の head での正解率．
- **Δ(tail,head)** $= (\text{acc-head} - \text{acc-tail}) / \text{acc-tail}$——"how much is the error prediction imbalanced between frequent and rare answers"．

## 実験設定

- **モデル**：Question Prior（画像を使わない），LSTM（画像を使わない），BUTD，BAN4，MCAN，MMN，LXMERT，および GQA の注釈から取った正解の物体を入力にする VIS-ORACLE（正解の注釈（場面グラフ）が train と validation にしか無いので，testdev では評価しない——脚注3）．
- **偏り低減手法**（BUTD に適用）：RUBi，RUBi＋QB（質問だけの枝を評価時にも残す変種），BP，LM．
- GQA-OOD で評価するモデルは，断りが無いかぎり GQA（balanced）の訓練集合で学習し，GQA-OOD の val で検証する．標準偏差は，得られる場合に，4つ以上の乱数種で計算したものである．

## 結果

### 主要モデル（Table 3-a，GQA-OOD testdev，%）

| モデル | acc-all | acc-tail | acc-head | Δ |
| --- | --- | --- | --- | --- |
| Question Prior | 21.6 | 17.8 | 24.1 | 35.4 |
| LSTM | 30.7 | 24.0 | 34.8 | 45.0 |
| BUTD | 46.4±1.1 | 42.1±0.9 | 49.1±1.1 | 16.6 |
| MCAN | 50.8±0.4 | 46.5±0.5 | 53.4±0.6 | 14.8 |
| BAN4 | 50.2±0.7 | 47.2±0.5 | 51.9±1.0 | **9.9** |
| MMN | 52.7 | 48.0 | 55.5 | 15.6 |
| LXMERT | **54.6** | **49.8** | **57.7** | 15.9 |

（太字は原表のとおり．）

**表のどのモデルも，acc-tail が acc-head より低い．** acc-tail が最も高い LXMERT で 49.8 である．著者は，画像を使わない二つのモデルで acc-tail と acc-head の開きが最も大きいと述べ，これを質問の偏りだけに頼るためと説明する．

### 偏り低減手法（Table 3-b，BUTD に適用，GQA-OOD testdev，%）

| 手法 | acc-all | acc-tail | acc-head | Δ |
| --- | --- | --- | --- | --- |
| BUTD（そのまま） | 46.4±1.1 | 42.1±0.9 | 49.1±1.1 | 16.6 |
| ＋RUBi＋QB | 46.7±1.3 | 42.1±1.0 | 49.4±1.5 | 17.3 |
| ＋RUBi | 38.8±2.4 | 35.7±2.3 | 40.8±2.7 | 14.3 |
| ＋LM | 34.5±0.7 | 32.2±1.2 | 35.9±1.2 | 11.5 |
| ＋BP | 33.1±0.4 | 30.8±1.0 | 34.5±0.5 | 12.0 |

"Surprisingly, none of the three bias-reduction methods succeed to improve acc-tail (cf. Table 3-b), they even deteriorate acc-head."（§4.3）——表の値では，RUBi・LM・BP の Δ は BUTD より小さいが，acc-tail も acc-head も BUTD より低い．

### 訓練データの偏りと tail の深さ（Table 2，GQA-OOD val，%）

BUTD を GQA の balanced 訓練集合（BUTD+bal）と，偏りを残した unbalanced の訓練集合全体（BUTD+all）で学習して比べる（§4.1）．

| | Tot. Acc.（Baseline benchm.） | acc-tail（α=1.2） | acc-tail（α=0.5） | acc-tail（α=0.3） |
| --- | --- | --- | --- | --- |
| BUTD＋bal | 60.7±0.4 | 45.4±0.3 | 33.8±0.5 | 24.6±0.5 |
| BUTD＋all | 59.8±0.1 | 41.9±0.1 | 29.5±0.3 | 18.3±0.6 |
| Δ（relative） | −1.4% | −7.7% | −12.9% | −25.7% |

**全体正解率での相対差は 1.4% にすぎないが，acc-tail では差が開き，$\alpha$ を小さくするほど大きくなる．**

### $\alpha$ を掃引したとき（Figure 3，§4.2）

著者の記述では，どのモデルも，問と答えの組のおよそ半分にあたる tail の大きさから先で "tail accuracy starts to linearly decrease until reaching a dramatically low score (about 30 pts lower than the overall accuracy)"．例外は "An exception is VIS-ORACLE: its dynamics is nearly flat, prediction error is almost decorrelated from answer rareness."．正解が tail の問に head の答えを返す混同（Figure 3-b）については，"For α=1.2, LXMERT confuses answers for 25% of questions, which increases up to 42% for α=0.3."

### 質問の型別（Figure 5，§4.2）

LXMERT について，GQA-OOD の val で推定した推論ラベルを GQA の質問の型ごとに見た分析では，"compare and query questions are the most prone to bias exploitation" とされる（原文では compare の直後に脚注6 "only 1% of the tail questions are typed as compare." が付く）．

## 著者が明記した限界

- **合成された質問**（§3 の Limitations の段落．v3 と CVPR 版に在り，v2 には無い）："The proposed benchmark is built on the GQA dataset, whose questions have been automatically generated, resulting in a limited vocabulary (GQA covers 70% of VQA2 answers [15]) and a synthetic syntax."．著者は，データを制御しやすいことと，モデル化されていない外部知識を排せることを挙げ，質問が合成であることの利点が限界を大きく上回ると論じる．

以下の二つは，Limitations の段落ではなく，本文と脚注での断りである．

- **推論ラベルは推定**：入力ごとの bias／reason／other・unknown のラベルについて，"In absence of GT information, we propose to estimate these labels from proxy rules"（§4.2）．
- VIS-ORACLE は testdev で評価していない（脚注3．正解の注釈が train と validation にしか無いため）．

## 射程の外

- **稀さの単位は「文脈（local group）の中での答え」**である．本研究の署名（GQA の program の演算子列，研究文書 §3.1）は原典に現れず，本研究 §A.3 のような，署名が学習データに現れた回数での層別は行っていない（全文で "program" は MMN の説明の1箇所だけ，"functional" は 0 件）．
- **評価のためのベンチマークと指標の提案であり，新しい学習手法は提案しない．** 偏り低減手法は既存のもの（RUBi・BP・LM）を評価し直すだけである．
- **内部表現を測る実験は見当たらなかった**（検索した語：representation，hidden，probe／probing，intervention）．報告されるのは正解率，Δ，head／tail の混同，推定した推論ラベルの分布である．
- 評価したモデルは上の8つと偏り低減手法の組み合わせで，本研究が使う SmolVLM（研究文書 §4.1）は含まれない．
- 入力を加工したときの表現の動き（研究文書 §A.6）にあたる測定は見当たらなかった（検索した語：perturb，noise．"robust" は OOD に対する頑健性の意味で本文に2箇所）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴．Comments・Journal-ref の欄は無い），arXiv PDF v3（14頁，付録 A–D を含む全文），v2（14頁），v1（11頁），CVF Open Access の CVPR 2021 の書誌ページ・本文 PDF（10頁）・supplemental PDF（5頁）を取得した．v3 を全文読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．表（Table 1–3）と tail の定義式・Δ の式は v3 のページ画像でも確かめた．v2 と CVPR 版は，本文書の数値と逐語引用がすべて在ること（v2 は Limitations の段落だけ無い），CVPR 版の本文が v3 と文献番号を除いて一致すること，supplemental の文章が v3 の付録と一致することを，テキストの機械的な突き合わせで確かめた．v1 は v2 以降と内容が異なり（Table 2 の相対差の値や Limitations の段落が無い），tail の定義と Table 1 の値が在ることだけを見た．下書き（2026-10-11 作成）から直した主な点：掲載先を「arXiv のみ確認」から CVPR 2021, pp. 2776–2785 へ（CVF Open Access で確認）；v2 の日付と，v1 では tail の定義式と α の値が異なることを足した；"mechanically increased" の引用が標準の全体正解率ではなく acc-all についての文であることを明記；文脈依存の例を原文のバラとスミレの例に直した；acc-tail の「主たる OOD の指標」を原文の定義に直した；「大きく落ちる」「にとどまる」を表の値の記述に直した；偏り低減手法の Δ についての読みを表の値の記述に直した；VIS-ORACLE の testdev の件を「限界」の段落のものではなく脚注3として分けた；型別の分析が LXMERT についてのものであることを足した；Table 1 の GQA testdev のまとまり数 7,803 を埋めた；射程の外の否定を，検索した語を添えた強さに直し，本研究の節番号をいまの研究文書に合わせた．原文の本文と表の食い違い（本文は LM の acc-tail を 33% と書くが Table 3-b は 32.2，など）は表の値を採った．**読めていないもの**：Figure 3・4・10 の曲線の値と付録の図（Figure 6–14）の値は，本文に書かれたもの以外を照合していない．Table 4（VQA2・VQA-CP2 との比較）は本文書に載せていない．v1 と v2 は通読していない．
