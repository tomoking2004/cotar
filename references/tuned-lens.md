# Eliciting Latent Predictions from Transformers with the Tuned Lens

**著者**：Nora Belrose，Igor Ostrovsky，Lev McKinney，Zach Furman，Logan Smith，Danny Halawi，Stella Biderman，Jacob Steinhardt（Eleuther AI／FAR AI／University of Toronto／Boston University／UC Berkeley）
**掲載**：arXiv のみ確認．arXiv:2303.08112 v6，2025-11-11（v1 は 2023-03-14）
**原文**：[arXiv](https://arxiv.org/abs/2303.08112)

**凍結した事前学習モデルの各ブロックにアフィンのプローブを学習すれば，どの隠れ状態も語彙上の分布として読み出せる．この tuned lens は，最終層の unembedding をそのまま当てる logit lens より予測が良く，信頼でき，偏りが小さい．**

## 主張

- transformer を**反復推論**（iterative inference）の観点で見る——"we view each layer in a transformer language model as performing an incremental update to a latent prediction of the next token"（§1）．
- その潜在予測を，凍結した事前学習モデルの各ブロックに**アフィンのプローブ**を学習して，語彙上の分布として読み出す．これが tuned lens で，要旨は先行する logit lens の改良（"a refinement"）と呼ぶ．20B パラメタまでの自己回帰言語モデルで試し，logit lens より "more predictive, reliable and unbiased" であることを示したとする（要旨）．
- 因果実験で，tuned lens がモデル自身と似た特徴を使っていることを示したとする．潜在予測の軌跡は，悪意ある入力の検出にも使えるとする（要旨）．

## 手法

### logit lens とその問題

```
LogitLens(h_ℓ) = LayerNorm[h_ℓ] W_U            （式3．W_U：unembedding 行列）
```

最終層の unembedding を中間層の隠れ状態 `h_ℓ` へそのまま当てる（ℓ より後の層の残差を 0 と置くことにあたる）．論文が挙げる問題は次のとおり（前の二つは §2，三つ目は §1 と §3）．

| 問題 | 内容 |
| --- | --- |
| **信頼できない（Unreliability）** | BLOOM と OPT 125M では，logit lens の top-1 予測が "often the input token, rather than any plausible continuation token, in more than half the layers"（Figure 18）．GPT-Neo 向けに nostalgebraist (2021) が導入した，最終層を残す拡張 `LayerNorm[h_ℓ + F_L(h_ℓ)] W_U`（式4）も "only partially successful" |
| **偏り（Bias）** | 最終層に比べ，特定の語彙に系統的に多くの確率を置く．"We find the bias of the logit lens can be quite large: around 4 to 5 bits for most layers."（GPT-Neo-2.7B，Figure 3）．比較として，Pythia 160M の最終層分布の Pythia 12B に対する偏りは 0.0068 bits |
| **表現のずれ（representational drift）** | "features may be represented differently at different layers of the network"（§1）．隠れ状態の共分散行列は，隔てる層の数が増えるほど離れていく強い傾向がある（Figure 6 下）．最終層の共分散は前の層に対してしばしば急に変わり，logit lens が前段の表現を "misinterpret" しうることを示唆すると著者は書く．分散のごく大きい少数の次元（"rogue dimensions"）も層のあいだで不均一に分布する傾向がある（Figure 6 上） |

### tuned lens ＝層ごとのアフィン translator

共分散のずれを直す "simple, general" な方法として，層ごとに**学習可能な基底変換の行列** `A_ℓ` を入れ，残差の和の代わりに学習可能な定数 `b_ℓ` を入れる（§3）．`(A_ℓ, b_ℓ)` を第 ℓ 層の translator と呼ぶ．

```
TunedLens_ℓ(h_ℓ) = LogitLens(A_ℓ h_ℓ + b_ℓ)                      （式8）
argmin E_x[ D_KL( f_{>ℓ}(h_ℓ) ‖ TunedLens(h_ℓ) ) ]               （式9）
```

（式9の TunedLens の添字は，原文では `k` と書かれている．）

- **目標は正解ラベルではなく，モデル自身の最終層の出力**（`f_{>ℓ}` は第 ℓ 層より後の残りの transformer）．著者はこれを，最終層の分布をソフトラベルとする蒸留損失と見なせるとし，プローブが，モデルの学んだ以上の情報を学ぶ誘因を持たないようにすると述べる．
- **初期値は恒等変換**．
- 学習（著者は "our initial experiments" の設定として書く）：SGD（Nesterov momentum），学習率は線形減衰で 250 ステップ，基礎学習率 1.0（最終層を残す場合 0.25），勾配ノルムを 1 で切る，1ステップあたり 2^18 トークン，重み減衰 10⁻³．データは，入手できる場合は事前学習の検証集合の一部（評価には別の部分），BLOOM と GPT-2 は Pile の検証集合．文書を連結して長さ 2048 に等分する．
- 費用：8×A40 の1ノードで "a full set of probes in under an hour"（§6）．
- **v6 で加わった "Muon optimizer" の段落**（§3．取得した v4・v5 の PDF には無い）：SGD の代わりに Muon で学習すると，初期の実験で可能だったよりずっと低い KL 損失に達するとし，"all of the tuned lenses trained and evaluated in the earlier versions of this paper were severely undertrained" と書く．続けて "We opted not to re-do all our experiments with tuned lenses trained using Muon" と書く．

### 因果基底抽出（causal basis extraction，CBE）

ある単位ベクトルの方向の情報を消す（平均値で置き換える mean ablation）前後の出力の期待 KL を，その方向の影響度と定義し，既に得た方向すべてに直交する中で影響度が最大の方向を，一つずつ順に探す（最適化は L-BFGS）．tuned lens について得た各方向を，モデルの隠れ状態でも消し，**tuned lens への影響**と**モデルの出力への影響**を比べる（§4.1）．

## 実験設定

- **モデル**：GPT-2，GPT-Neo，GPT-NeoX-20B，OPT，Pythia（12B まで），BLOOM 560M．微調整モデルへの転移は LLaMA 13B と Vicuna 13B．
- **評価**：各モデルの事前学習の検証集合から無作為に取った 16.4M トークン．
- **応用**（§5）：Pythia 12B でのプロンプト注入の検出（9 つの多肢選択タスク，§5.3），例題の難しさとの相関（§5.4）．このほか §5.1（言語化されない知識の引き出し．v6 で加わった節で，結果は "mixed" と書く）と §5.2（Halawi らの実験の再現）があるが，本要約では扱わない．

## 結果

### 予測の質

- "We find that tuned lens predictions have substantially lower perplexity than logit lens predictions, and are more representative of the final layer distribution."（§1．層別の値は図のみ）
- GPT-Neo-2.7B の例（Figure 1）："The logit lens fails to elicit interpretable predictions before layer 21, but our method succeeds."
- tuned lens の偏りは "significantly lower"（§3，Figure 3）．

### 層間の転移と表現のずれ

- ある層で学習した translator を別の層で評価したときの交差エントロピー損失の増分（transfer penalty）は，最大の Pythia モデルで，**共分散の類似度と強く負に相関する（Spearman ρ = −0.78）**．transfer penalty は全体に低く，近い層どうしではとくに低い．
- 転移は対称でない．外れ値次元のある層（第5層以降）で学習し，無い層で試すほうが，逆より transfer penalty が高い．
- **微調整モデルへの転移**：LLaMA 13B で学習した lens を Vicuna 13B に転用すると（unembedding は Vicuna のものを使う），RedPajama では，モデルの最終出力への KL の増加が最悪でも 0.3 bits per byte．Anthropic の Helpful Harmless の対話データでは，転用した lens と Vicuna で学習した lens に有意な差は無いとする．

### 因果的な忠実さ（Pythia 410M，第18層，Figure 8）

- 特徴の tuned lens への影響とモデルへの影響に**強い相関（Spearman ρ = 0.89）**．「tuned lens には影響するがモデルには影響しない」特徴（図の右下）は観察されなかったとする．

### プロンプト注入の検出（Table 1．Pythia 12B，テスト集合の AUROC．括弧は 95% ブートストラップ信頼区間）

| タスク | Tuned iForest | Tuned LOF | Logit iForest | Logit LOF | SRM（基準） | 正解率 通常→注入 |
| --- | --- | --- | --- | --- | --- | --- |
| ARC-Easy | 0.59 (0.54, 0.62) | 0.73 (0.71, 0.76) | 0.53 (0.50, 0.57) | 0.59 (0.56, 0.62) | 0.73 (0.70, 0.75) | 72.8% → 31.7% |
| ARC-Challenge | 0.71 (0.65, 0.77) | 0.81 (0.77, 0.84) | 0.73 (0.67, 0.79) | 0.80 (0.77, 0.83) | 0.57 (0.53, 0.61) | 43.5% → 24.7% |
| BoolQ | 0.99 (0.98, 0.99) | 1.00 (1.00, 1.00) | 0.89 (0.87, 0.91) | 0.61 (0.57, 0.66) | 1.00 (1.00, 1.00) | 67.1% → 0.0% |
| MC TACO | 0.74 (0.71, 0.77) | 0.68 (0.66, 0.70) | 0.68 (0.66, 0.69) | 0.55 (0.53, 0.59) | 1.00 (1.00, 1.00) | 0.40 → 0.06 F1 |
| MNLI | 0.98 (0.98, 0.99) | 1.00 (1.00, 1.00) | 0.95 (0.94, 0.96) | 1.00 (1.00, 1.00) | 1.00 (1.00, 1.00) | 54.3% → 0.0% |
| QNLI | 0.99 (0.99, 1.00) | 1.00 (1.00, 1.00) | 0.93 (0.92, 0.95) | 0.68 (0.63, 0.71) | 1.00 (1.00, 1.00) | 54.3% → 0.0% |
| QQP | 1.00 (0.99, 1.00) | 1.00 (1.00, 1.00) | 0.90 (0.89, 0.90) | 0.79 (0.76, 0.81) | 1.00 (1.00, 1.00) | 60.7% → 6.5% |
| SciQ | 0.62 (0.57, 0.69) | 0.64 (0.59, 0.70) | 0.75 (0.71, 0.79) | 0.70 (0.65, 0.74) | 0.75 (0.72, 0.78) | 95.5% → 62.6% |
| SST-2 | 1.00 (0.98, 1.00) | 1.00 (1.00, 1.00) | 0.78 (0.72, 0.83) | 0.61 (0.56, 0.65) | 1.00 (1.00, 1.00) | 82.9% → 49.1% |

著者は，tuned lens の検出器が5タスク（BoolQ，MNLI，QNLI，QQP，SST-2）で完全かほぼ完全な AUROC に達するとしたうえで，基準の SRM について "the tuned lens only outperforms it on one task (ARC-Challenge)" と書き，MC TACO と SciQ では SRM が上回ると書く（§5.3）．

### 例題の難しさ（Table 2．Pythia 12B (deduped)．iteration learned〔ある例に対するモデルの top-1 予測が固定される最初の学習ステップ〕と prediction depth〔top-1 予測が変わらなくなるまでの層の数〕の Spearman 順位相関）

| タスク | tuned lens | logit lens | Final acc |
| --- | --- | --- | --- |
| ARC-Easy | 0.577 | 0.500 | 69.7% |
| ARC-Challenge | 0.547 | 0.485 | 32.4% |
| LogiQA | 0.498 | 0.277 | 21.4% |
| MNLI | 0.395 | 0.435 | 40.4% |
| PiQA | 0.660 | 0.620 | 76.1% |
| QNLI | 0.409 | −0.099 | 53.0% |
| QQP | 0.585 | −0.340 | 0.381（F1） |
| RTE | 0.156 | 0.347 | 60.0% |
| SciQ | 0.530 | 0.505 | 91.9% |
| SST-2 | 0.555 | 0.292 | 64.7% |
| WinoGrande | 0.517 | 0.537 | 63.9% |

著者は，tuned lens の prediction depth が "in 8 out of 11 tasks" で logit lens のものより iteration learned とよく相関し，調べた全タスクで有意な正の相関があったと書く（§5.4）．なお本文は評価タスクの数を "twelve" と書き，Table 2 が載せるのは11タスクである（原文の表記のとおり）．

## 著者が明記した限界

"Limitations and future work"（§6）に書かれたもの．

- **層ごとに学習が要る**："One limitation of our method is that it involves training a translator layer for each layer of the network"．logit lens は学習なしで使える，と対比する．
- **CBE は重い**："Causal basis extraction, as presented in this work, is computationally intensive"．
- **言語モデルに絞った**："Due to space and time limitations, we focused on language models in this work, but we think it's likely that our approach is also applicable to other modalities."

限界の段落ではなく，脚注に置かれた前提．

- **方向としての特徴を仮定**："For simplicity we assume the "features as directions" hypothesis"（脚注4）．
- **pre-LN 構造を主な対象とする**："Both the logit lens and the tuned lens are designed primarily for the pre-LN architecture"（脚注2）．

## 射程の外

- **VLM は扱わない．** 実験の対象は言語モデルである（上の限界の引用のとおり．全文を VQA・GQA・multimodal で検索し，該当は無かった）．
- **学習時の介入は扱わない．** モデルは凍結し，translator だけを学習する．
- **読み出す対象は語彙上の分布（次トークンの潜在予測）である．** 隠れ状態から特定の概念を取り出す分類器の学習は，§1 が "One common approach" として挙げ，本論文は "instead" 反復推論の観点を取るとする．
- **translator は層ごとに一つのアフィン変換**で，最終層の出力への KL で学習される．答えトークンに限って，隠れ状態に対する出力の感度を測る量は，見当たらなかった（検索した語：sensitiv，Jacobian，derivative．"causally sensitive" の一語が §4.1 にあるのみ）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・v1〜v6 の日付．Comments／Journal-ref の記載は無い），PDF v6（25頁，付録 A〜E を含む全文）を pdftotext でテキスト化して読み，Table 1 と式9・Muon の段落は頁の画像でも見て，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．PDF v4・v5（各25頁）も取得し，本文書に載せた数値（Table 1・Table 2 の全行，ρ = −0.78／0.89，0.0068 bits，0.3 bits per byte，16.4M，250 ステップ，"8 out of 11"，"twelve"）が v6 と同じであることを検索と行の差分で確かめた（v4・v5 は通読していない）．下書き（2026-10-11 作成）から直した主な点：Muon の段落は「PDF v6 に無い」のではなく PDF v6 の §3 に在る（v4・v5 に無い）と直した．冒頭の一文を，原文に無い因果（基底のずれゆえに偏る）から要旨の主張へ直した．「最終層で急変する」を原文の "often" の強さに，「Muon でやり直していない」を原文の引用に，ρ = −0.78 を「最大の Pythia モデル」の値に，Table 1 の括弧を 95% ブートストラップ信頼区間に直し，原文に無い太字（0.81）を外した．式9の添字の原文表記（k）を注記した．限界の節を，§6 の限界と脚注の前提に分け，言語モデルに絞った旨の原文を足した．射程の外の否定の主張を，検索で確かめられる強さに直した．**読めていないもの**：v1〜v3 の PDF．図の中の値（図から読み取る数値は載せていない）．掲載先は公式の書誌ページで確かめられていない（DBLP は自動取得を拒まれた．arXiv abs に Journal-ref は無い）．
