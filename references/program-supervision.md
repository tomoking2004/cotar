# Supervising the Transfer of Reasoning Patterns in VQA

**著者**：Corentin Kervadec，Christian Wolf，Grigory Antipov，Moez Baccouche，Madiha Nadri（Orange Innovation／LIRIS，INSA-Lyon／LAGEPP，Université de Lyon）
**掲載**：NeurIPS 2021（Advances in Neural Information Processing Systems 34），pp. 18256–18267．arXiv:2106.05597 v1（2021-06-10）．本文書の数値・引用・文献番号・図表番号は NeurIPS 版（本文と Supplemental）による．arXiv v1 と違う箇所はその場で断る
**原文**：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/9766527f2b5d3e95d4a733fcfb77bd7e-Abstract.html)・[arXiv](https://arxiv.org/abs/2106.05597)

**VQA モデルの分類 head 直前の表現から GQA の reasoning program（演算・引数・依存）を予測させる損失を正則化項として足すと，oracle transfer の質が上がり，GQA と GQA-OOD の正解率が上がる．**

## 位置づけ

GQA に付く reasoning program を，答えとは別の教師信号として**学習時に内部表現へ掛ける**研究である．掛ける先は分類 head 直前（cross-modal 層の後）の表現で，予測させるのは演算の列だけでなく，引数と依存まで含む program 全体である．

## 主張

- program の予測を損失の正則化項として足すと，GQA と GQA-OOD の正解率が上がる．
- その効果は oracle transfer（[21]．CVPR 2021．著者の一部が本論文と重なる）や BERT 式（LXMERT）の事前学習と**相補的**である．
- PAC 学習の枠組みで，program の予測は穏当な仮定のもとで標本複雑度を下げうる，と論じる（第4節）．
- 著者は手法を，program を実行するものではなく表現を豊かにするものと位置づける（第2節）："our approach uses program supervision to enrich its internal representations instead of inferring program’s execution."

## 手法

### 土台：VL-Transformer と oracle transfer

- **モデル**：LXMERT [30] の VL-Transformer のコンパクト版．言語のみ9層，視覚のみ5層，cross-modal 5層，隠れ次元128，26M パラメータ（補遺 C）．
- **oracle transfer**：まず正解の視覚入力（ground-truth の one-hot 視覚入力）で学習した oracle を作り，その後で物体検出器の特徴に切り替えて学習を続ける．各段に program の監督を正則化として掛ける．

### program デコーダ（図2・第3節・補遺 B）

**入力は VL-Transformer の最後から2番目の埋め込み**——図2の説明："The decoder is fed with the VL-Transformer’s penultimate embedding (just before the VQA classification head)"．y_CLS トークンの埋め込みが GRU の隠れ状態を初期化する．

| 段 | 何を予測するか | 構成 | 損失 |
| --- | --- | --- | --- |
| 粗（演算） | 演算の系列．STOP 演算で止まる | 1層 GRU（隠れ128）→ 2層 MLP（128→64→N_op） | 交差エントロピー L_op |
| 細（テキスト引数） | 各演算と各質問トークンの親和度 | 2層 FFN（256→64→1） | 二値交差エントロピー L_qarg |
| 細（視覚引数） | 各演算と各検出物体の親和度．正解は正解物体と検出物体の IoU に基づく軟らかい目標 | 同上 | 二値交差エントロピー L_varg |
| 依存 | 各演算が前のどの演算の結果を使うか（program を木にする） | 演算ごとの隠れ状態で初期化した1層 GRU（隠れ128）→ 射影（128→N_maxop） | 二値交差エントロピー L_dep |

構成の欄は補遺 B による．本文と補遺で表記が違う箇所が二つある．演算の分類は，本文では "classified with a linear projection"，補遺では2層 MLP である．依存の射影は，本文では "a linear projection of the hidden state"，補遺では "a nonlinear projection (128 → Nmaxop)" である．

演算ラベルは **N_op = 212 種**，program の長さは**最大9演算**（N_maxop = 9）．

### 損失の重み

L = L_vqa + α·L_op + β·L_dep + γ·L_qarg + δ·L_varg．

本文は重みを記号でしか示さない．数値は補遺 B にある："Hyper-parameters are set to α = 1, β = 1, γ = 1 and δ = 100."——**視覚引数の項だけが100倍**である．

### PAC 学習による論証（第4節・補遺 A）

| 要素 | 内容 |
| --- | --- |
| 設定 | 過剰パラメータ化した2層 MLP を勾配降下で学習（実験で使う注意機構のモデルではない） |
| 定理4.1 | 過剰パラメータ化した MLP の標本複雑度の上界（[34] の Theorem 3.5 の再掲）．著者は後で，これを単一の推論モードを学ぶ場合の上界として使う |
| 仮定1 | 推論関数は推論モードの混合 y = Σ_r π_r g_r(v)．各 g_r は視覚入力だけを見て，重み π は質問で決まる |
| 仮定2 | 質問の埋め込みはモードごとに群れ，モードの推定は群れの中心 γ_r との内積で書ける（上界では softmax を落とす） |
| 定理4.2 | 多モードの推論関数を通常の学習で学ぶときの上界．項ごとに ‖γ_r‖ の因子が乗る |
| 仮定3 | program の監督があると，推論モードの選択が隠れ活性に符号化され，モード推定器の学習（program 出力につながる枝）と各モードの学習（互いに独立）とに分かれる．著者はこのときの複雑度が定理4.2 のものより低いとする |

埋め込みがバッチ正規化されている（γ の各成分が N(0,1) に従う）と置くと，差を決める因子 C は埋め込み次元 m とともに単調に増える．"For the order of m=512 to m=768 used for state-of-the-art LXMERT models [30], complexity grows by a factor of around ∼20."

**この節で比べているのは二つの上界である．** 著者は，上界の比較はモデルの順序に確定的な答えを与えないとしたうえで，両者は同じ代数操作で導かれており比較できると主張する．また "this analysis very probably under-estimates the difference in complexity" と添える．NeurIPS 版は，これとは別に，学習データの量を 5% から 100% に変えた実験（図3．oracle transfer なし，LXMERT 事前学習なし）を載せ，program の監督を足すと少ないデータで baseline と同程度の正解率に届くと述べる．本文に数値は無い．arXiv v1 にこの実験は無い．

## 実験設定

- **学習段**：oracle は GQA balanced train で正解の視覚入力により最大40エポック → 検出器特徴へ移して10エポック．LXMERT 式事前学習は GQA unbalanced で20エポック．微調整は GQA balanced で4エポック（補遺 C）．
- **視覚特徴**：既定は Faster R-CNN の36物体．100物体と VinVL の変種もある．
- **評価**：GQA の test-dev・test-std，GQA-OOD（acc-tail／acc-head）．ablation は GQA val と GQA-OOD val．± は乱数種3つの平均と標準偏差で，± の無い値は1回の実行である．

## 結果

### 表1：oracle transfer と LXMERT 事前学習との組み合わせ

| | 事前学習 | Oracle transf. | Prog. sup. | GQA-OOD acc-tail | acc-head | GQA test-dev | test-std | AUC（視覚引数） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) Baseline | scratch | | | 42.9 | 49.5 | 52.4 | − | / |
| (b) Oracle transfer | scratch | ✓ | | 48.2±0.3 | 54.6±1.1 | 57.0±0.3 | 57.3 | / |
| (c) Ours | scratch | ✓ | ✓ | 48.8±0.1 | 56.1±0.3 | 57.8±0.2 | 58.2 | 97.1 |
| (d) Baseline | + lxmert | | | 47.5 | 55.2 | 58.5 | − | / |
| (e) Oracle transfer | + lxmert | ✓ | | 47.1 | 54.8 | 58.4 | 58.8 | / |
| (f) Ours | + lxmert | ✓ | ✓ | 48.0±0.6 | 56.6±0.6 | 59.3±0.3 | 59.7 | 96.4 |

（原表にある binary・open の列は省いた．AUC は GQA val 上の AUC@0.66．）

- **LXMERT 事前学習ありでは，表の上で oracle transfer 単独（e）は baseline（d）を test-dev で上回らない**（58.4 対 58.5）．program の監督を足した（f）で 59.3．
- **oracle transfer なし・program 監督あり・LXMERT あり**の値は表になく，本文にだけ現れる："we achieve an accuracy of 58.8 on the testdev set of GQA." 続けて "This is lower than oracle transfer’s accuracy, demonstrating the complementary of the two methods." とある．表1の test-dev は（e）が 58.4，（f）が 59.3 であり，**"oracle transfer’s accuracy" がどの行を指すかは本文に書かれていない**．
- **AUC の帰属に食い違いがある．** 本文は "Models (c) and (e) achieve, respectively, 97.1 and 96.4 AUC scores" と書くが，表では 96.4 は（f）の欄にあり，（e）は program 監督を持たない．
- 表1の binary／open は test-std 上の値（キャプションの注）．著者は伸びの大半が open 問題にあると述べる．

### 表4：視覚特徴を変えたとき（GQA test-dev）

| | 特徴 | Prog. sup. | test-dev |
| --- | --- | --- | --- |
| (g) | R-CNN 100物体 | | 57.0±0.4 |
| (h) | R-CNN 100物体 | ✓ | 58.2±0.1 |
| (i) | VinVL | | 59.6±0.1 |
| (j) | VinVL | ✓ | 60.9±0.2 |
| (k) | VinVL + LXMERT | | 61.4 |
| (l) | VinVL + LXMERT | ✓ | 61.8 |

（いずれも oracle transfer あり．（k）（l）の test-std は 62.5／63.0．）

### 表2：監督の種類の ablation（コンパクトモデル，LXMERT/BERT 事前学習なし，oracle なし，val）

| | GQA-OOD acc-tail | GQA val |
| --- | --- | --- |
| (1) VQA only | 46.9 | 62.2 |
| (2) Coarse only（演算のみ） | 46.5 | 62.5 |
| (3) Coarse + dep. | 46.8 | 62.8 |
| (4) Full w/o v.arg | 47.3 | 63.7 |
| (5) Full (ours) | 49.9 | 66.2 |

**演算の列だけを監督したときの GQA val は 62.2→62.5 である．** 表の上で最も大きく伸びるのは視覚引数の監督を足したところ（63.7→66.2）で，著者も視覚引数の監督が伸びに最も寄与すると述べる．

### 表3：監督を掛ける位置の ablation（oracle transfer あり，LXMERT なし，val）

| | GQA-OOD acc-tail | GQA val |
| --- | --- | --- |
| (6) program 監督なし | 50.0 | 66.4 |
| (7) uni-modal 層の直後 | 49.9 | 66.5 |
| (8) cross-modal 層の後（標準） | 50.4 | 67.4 |

**比べたのは2か所だけで，層を一つずつ掃引してはいない．** キャプション："The supervision is more efficient when used after cross-modal interactions." 著者は，uni-modal 層の直後に掛けたもの（7）は監督なし（6）を改善しないと述べる．

### 無作為 program の対照（Supplemental の表2，補遺 F．arXiv v1 では表7）

学習時に正解 program を別の質問の program に差し替えると，GQA-OOD acc-tail 45.7，GQA val 61.4——**VQA のみ（46.9／62.2）より低い**．設定はコンパクトモデル，事前学習なし，oracle なし．

## 著者が明記した限界

- **program の注釈に依存する**（第6節）："reasoning program annotations, which are costly to annotate, especially when dealing with human generated questions."
- **理論は簡略化したモデルの上界比較である**——注意機構のモデルでなく過剰パラメータ化した MLP であり，先行する定理の仮定に乗り，二つの別々の上界を比べている（第4節）．
- **今後**（第6節）：program の注釈が稀か不完全な設定へ広げること．NeurIPS 版はさらに，答えを program の予測で明示的に条件づけること——"explicitly conditioning the VQA system’s answer by the program prediction module"——を挙げる（arXiv v1 にこの文は無い）．

## 射程の外

- **表現が答えに使われているかを確かめる介入は，全文に見当たらなかった**（検索した語：interven，causal，probe，ablat）．証拠は program 予測の質（AUC），損失成分と監督位置の ablation，無作為 program の対照である．答えは分類 head が出し，program デコーダはその直前の埋め込みから分岐する（図2）．答えを program の予測で明示的に条件づけることは今後の道として挙げられている．
- **標準の構成で監督を掛けるのは分類 head 直前の表現である．** 位置の比較は uni-modal 直後と cross-modal 後の2点に限る．
- **予測させるのは program 全体である．** 演算の列に加えて引数（質問トークン・検出物体）と依存を予測し，損失は交差エントロピーと二値交差エントロピーである（contrastive の語は全文に無い）．演算の列だけの監督（表2 (2)）では GQA val が 62.2→62.5 である．
- **モデルは 26M パラメータのコンパクトな VL-Transformer（と，GQA unbalanced 上での LXMERT 式の事前学習）である．** 答えは答えの辞書の上の分類として出す．
- **単一データ**：学習も評価も GQA と，その分割である GQA-OOD に限られる．

---

**確認**：2026-11-13．arXiv abs ページ（v1 のみ，2021-06-10），arXiv PDF v1（18頁，補遺 A–F を含む），NeurIPS Proceedings の書誌ページと BibTeX（巻34，pp. 18256–18267），NeurIPS 版の本文 PDF（12頁，チェックリストまで）と Supplemental PDF（7頁，補遺 A–F）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．表1・表4 と図2 は，NeurIPS 版 PDF の該当頁（4・7・9頁）を画像にして，行・列・チェック印の対応を目で確かめた．下書き（2026-10-11 作成）から直した主な点：冒頭の一文を原典の主張だけにした．どの版の値かを明記し，版で違う箇所（今後の課題の文，図3 の実験，補遺の表番号）を断った．[21] を「同著者ら」から「著者の一部が重なる」へ，「約26M」を「26M」へ，「functional program」を原文の語 reasoning program へ直した．定理4.1 の説明を原文の題に合わせた．上界の比較を「目安にとどまる」とした言い換えを原文の言い方に直した．演算の分類についての本文と補遺の表記の差，± の意味，図3 の実験を足した．58.8 の文と表3 (7) の読みを原文の強さに揃え，否定の主張に検索語を添えた．**読めていないもの**：補遺 A の証明は式を一行ずつ追ってはいない（定理と仮定の文言は読んだ）．図3 は本文の記述だけを照合し，グラフから値は読んでいない．OpenReview の査読ページは読んでいない．
