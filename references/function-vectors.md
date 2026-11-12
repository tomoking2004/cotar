# Function Vectors in Large Language Models

**著者**：Eric Todd，Millicent L. Li，Arnab Sen Sharma，Aaron Mueller，Byron C. Wallace，David Bau（Khoury College of Computer Sciences, Northeastern University）
**掲載**：ICLR 2024．arXiv:2310.15213 v2，2024-02-25（v1 は 2023-10-23）
**原文**：[ICLR 2024 会議録](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html)・[arXiv](https://arxiv.org/abs/2310.15213)

**文脈内学習では少数の attention head が，例示されたタスクのコンパクトな表現を運ぶ．その head の平均出力を足し合わせた一本のベクトル（function vector）を，例示のない入力の最後のトークンの隠れ状態に足すと，そのタスクの実行が引き起こされる．**

## 位置づけ

タスクのベクトル表現を，相関としてではなく因果的な効果を持つものとして，モデルの内部から取り出して別の文脈に足せることを示した文献である——"Our work differs because FVs are causal, not just correlative, so they can be explicitly extracted and inserted"（§4）．

## 主張

- 文脈内学習（ICL）のタスクに因果媒介分析を当てると，**少数の attention head** が，例示されたタスクのコンパクトな表現を運んでいる．これを **function vector（FV）** と呼ぶ．
- FV は，取り出した ICL の文脈に似ていない入力（zero-shot や自然文）でもタスクの実行を引き起こす．因果的な効果は**中間層で強い**．
- FV は多くの場合タスクの出力空間を符号化する情報を含むが，その情報だけでは FV を再構成するのに足りない．
- FV は，ある程度まで，足し合わせて新しい複合タスクを引き起こすベクトルを作れる．

（以上は要旨による．）

## 手法

### 平均 head 出力

各層の計算は最後のトークン位置で見る（§2.2）．タスク t の，ラベルを乱していない 10-shot ICL プロンプト（|P_t| = 100 本，付録 C）にわたって，層 ℓ の head j の出力を平均したものを `ā^t_ℓj` とする（式 2）．P_t には，モデルが正答を1位に予測したプロンプトだけを入れる（§2.2）．

### 因果媒介分析で head を選ぶ

- **劣化プロンプト**：ICL 例のラベルを並べ替え（shuffled labels），入力と出力の間に系統的な関係が無いようにした 10-shot プロンプト．
- **CIE**：劣化プロンプト上で head `a_ℓj` の出力を `ā^t_ℓj` に差し替えたとき，正答に割り当てられる確率がどれだけ上がるか（式 3）．
- **AIE**：CIE をタスクと劣化プロンプトにわたって平均したもの（式 4）．タスクあたり劣化プロンプト 25 本．GPT-J では |T| = 18 タスク（10-shot の ICL の成績が多数派ラベルの基準を上回るタスクだけを含める．付録 C）．
- **集合 A**：全層を通して AIE が最も高い head．

| モデル | \|A\| |
| --- | --- |
| GPT-J | 10 |
| Llama 2（7B） | 20 |
| Llama 2（13B），GPT-NeoX | 50 |
| Llama 2（70B） | 100 |

（§2.3 の脚注1．）GPT-J の上位10 head は "clustered primarily in early-middle layers of the network"（§2.3）．

### function vector

```
v_t = Σ_{a_ℓj ∈ A} ā^t_ℓj
```

（式 5．）**層をまたいで選んだ head の平均出力の和**であり，単一層の隠れ状態の平均（下の layer average）とは別のものである．

### 足す場所

**プロンプトの最後のトークン位置**で，**層 ℓ ≈ |L|/3** の隠れ状態 `h_ℓ` に FV を足す——"which we found works well in practice"（付録 C）．

| モデル | 層数 \|L\| | head/層 | FV を足す層 |
| --- | --- | --- | --- |
| GPT-J（6B） | 28 | 16 | 9 |
| GPT-NeoX（20B） | 44 | 64 | 15 |
| Llama 2（7B） | 32 | 32 | 11 |
| Llama 2（13B） | 40 | 40 | 14 |
| Llama 2（70B） | 80 | 64 | 26 |

（層数と head 数は Table 1，足す層は付録 C．）

### 比較対象（layer average）

ある層 ℓ での最後のトークンの隠れ状態を，タスクの ICL プロンプトにわたって平均し（`h̄^t_ℓ`），その平均を層 ℓ の表現に足す（§2.1）．

## 実験設定

- **モデル**：デコーダのみの自己回帰言語モデル5本（Table 1）．
- **タスク**：40を超える "relatively simple tasks"．本文の主対象は6つ（Antonym，Capitalize，Country-Capital，English-French，Present-Past，Singular-Plural）．
- **ICL**：10-shot．
- **評価**：shuffled-label（ラベルを乱した ICL プロンプト）と zero-shot（質問だけで例示なし）の2条件で，FV を足したときの top-1 正解率を測る．**§3.1 の正解率は，モデルが 10-shot の ICL プロンプトで正答できたテスト質問だけに絞った部分集合の上の値である**（乱数種5つ）．絞らない場合の結果も似ていると著者は述べる（付録 D）．

## 結果

### 主結果（6タスク平均，Table 2）

| 構成 | shuffled-label | zero-shot |
| --- | --- | --- |
| GPT-J | 39.1 ± 1.2% | 5.5 ± 0.8% |
| GPT-J + layer average | 79.5 ± 3.1% | 9.5 ± 1.8% |
| **GPT-J + FV** | **90.8 ± 0.9%** | **57.5 ± 1.7%** |
| GPT-NeoX | 32.5 ± 1.3% | 6.7 ± 0.1% |
| GPT-NeoX + FV | 90.7 ± 0.6% | 57.1 ± 1.5% |
| Llama 2（70B） | 52.3 ± 2.2% | 8.2 ± 0.7% |
| **Llama 2（70B）+ FV** | **96.5 ± 0.5%** | **83.8 ± 0.7%** |
| GPT-J + FV（追加34タスク） | 80.4 ± 0.6% | 46.1 ± 3.7% |
| Llama 2（70B）+ FV（追加34タスク） | 93.0 ± 0.5% | 74.2 ± 3.1% |

値は v2 のもの．v1 は平均が同じで，± の値が異なる（例：GPT-J の基準が 39.1 ± 22.3%）．

**GPT-J では，layer average は zero-shot で 9.5%，FV は 57.5% である．** 著者は，FV が両方の条件で layer average を上回ると述べる（§3.1）．

**ICL のテンプレートを20通りに変えても**（GPT-J，6タスク）：shuffled-label 32.3 ± 12.8% → 76.2 ± 13.8%，zero-shot 6.2 ± 4.3% → 40.0 ± 16.7%（§3.1）．

### 自然文の文脈（Antonym の FV，GPT-J，Table 3）

生成した5トークンの中に正しい対義語が含まれるかで測る．

| プロンプト | GPT-J | + Antonym FV |
| --- | --- | --- |
| The word "x", means | 1.5 ± 1.1% | 55.2 ± 3.8% |
| When I think of the word "x", it usually means | 0.3 ± 0.2% | 67.7 ± 3.0% |
| When I think of x, I usually | 0.0 ± 0.0% | 61.1 ± 2.4% |
| While reading a book, I came across the word "x". I looked it up in a dictionary and it turns out that it means | 2.7 ± 1.9% | 46.0 ± 4.6% |
| The word x can be understood as a synonym for | 2.4 ± 1.7% | 52.7 ± 11.0% |

### 効く層

- 因果的な効果が最も高いのは早期〜中間層に足したときで，"with a sharp drop in performance to near-zero at the later layers of the network"（付録 A．Figure 4・14・16・26 を指す）．
- Figure 4 の説明文："Adding the FV to early-middle layers pushes models to perform the target task without any exemplars"．
- 著者は，後半の層で効果が急に落ちることから，FV は線形に働くだけではなく，後半の層の非線形の計算を引き起こすと示唆する（§3.1）．

### FV の語彙への射影

"for most tasks, the decoded tokens lie within the task's output space"（§3.2．Table 5 の例：Country-Capital → Moscow，Bangkok，Paris…）．例外として，English-French は "nonsense tokens" に復号される．

**ただし語彙の情報だけでは，働く FV は通例作れない**——FV を復号した分布を再現するように最適化で作ったベクトルを FV の代わりに足すと，上位100語を合わせたものは成績が低く，全語彙を合わせたものでも FV より低い（Table 6，付録 A．Antonym：FV 48.2 ± 2.0%，上位100語の再構成 4.8 ± 2.0%，全語彙の再構成 39.6 ± 2.6%．上位100語の再構成が最も高いのは Country-Capital の 58.1 ± 18.5% で，FV は 83.2 ± 2.7%）．

### FV の合成（GPT-J，Table 7）

`v*_BD = v_AD + v_BC − v_AC`（式 7）で複合タスクのベクトルを作る（例：Last-Capital = Last-Copy + First-Capital − First-Copy）．この表だけは，モデルが正答できた質問に絞らず，全サンプルで測っている．

| タスク | ICL（10-shot） | FV（zero-shot） | 合成（zero-shot） |
| --- | --- | --- | --- |
| Last-Antonym | 0.25 ± 0.02 | 0.02 ± 0.01 | 0.07 ± 0.02 |
| Last-Capitalize | 0.91 ± 0.02 | 0.64 ± 0.03 | 0.76 ± 0.04 |
| Last-Country-Capital | 0.32 ± 0.02 | 0.15 ± 0.03 | 0.60 ± 0.02 |
| Last-English-French | 0.45 ± 0.04 | 0.16 ± 0.02 | 0.06 ± 0.02 |
| Last-Present-Past | 0.89 ± 0.02 | 0.18 ± 0.02 | 0.29 ± 0.03 |
| Last-Singular-Plural | 0.90 ± 0.01 | 0.28 ± 0.01 | 0.29 ± 0.02 |
| Last-Capitalize-First-Letter | 0.75 ± 0.01 | 0.76 ± 0.02 | 0.95 ± 0.00 |
| Last-Product-Company | 0.35 ± 0.03 | 0.30 ± 0.02 | 0.41 ± 0.03 |

著者のまとめ（§3.3）：合成できる FV があり，合成が FV を，いくつかのタスクでは ICL をも上回る．一方，ICL や FV の成績がよいものを含めて，合成が効かないタスクもある．

## 著者が明記した限界

独立した Limitations 節は見当たらなかった（検索した語：limitation）．考察（§5）で，FV が ICL の全容ではないことを明記する——"Although FVs are not yet a complete accounting of how ICL works"．

## 射程の外

- **対象はテキストの自己回帰言語モデル**（GPT-J，GPT-NeoX，Llama 2 の 7B・13B・70B）．VLM や画像を入力とするタスクは評価に見当たらなかった（検索した語：vision，visual，multimodal，image，VQA，GQA．該当は注意重みの可視化を指す visualizations と参考文献の題だけだった）．本研究が用いる GQA（研究文書 §4.1）は含まれない．
- **FV は学習済みモデルに推論時に足す介入**である．本研究のように学習時に補助損失を足して表現を整える設計（研究文書 §3.2）は見当たらなかった（検索した語：fine-tun，finetun，gradient，during training．該当は参考文献の題と，関連研究の節で先行研究を述べる一文だけだった）．
- **タスクは著者の言う "relatively simple tasks" の ICL** である．本文の6タスクは語から語への写像（対義語・首都・翻訳など）で，付録 E には文を入力とする分類（SST-2，AG News），選択式の質問応答（CommonsenseQA），固有表現の抽出（CoNLL-2003）なども含まれる．
- **FV は選んだ head の平均出力の和**であり，ある層の隠れ状態そのもの（やその平均）とは別物である．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴・Comments），PDF v2（52頁，付録 A〜N を含む）と PDF v1（43頁），ICLR 2024 会議録の一覧ページ（proceedings.iclr.cc．題・著者・Conference トラック）を取得し，v2 を pdftotext でテキスト化して読み，Table 1・2・6 と §2.1 はページ画像で確かめて，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．下書き（2026-10-11 作成）から直した主な点：(1) 射程の外の「タスクは単純な語→語の写像」を，付録 E に文の分類・質問応答・固有表現抽出があることに合わせて直した．(2) "with a sharp drop in performance to near-zero…" の出どころを Figure 4 から付録 A へ直した．(3) §3.1 の正解率が，10-shot で正答できた質問に絞った部分集合の値であること（Table 7 だけは絞らない）を足した．(4) Table 2 の ± が v2 の値で，v1 では異なることを明記した．(5) Table 3 と Table 7 を，抜粋から全行へ改め，Table 6 の記述を原文の強さ（上位100語では低く，全語彙でも FV より低い）に直した．(6) 否定の主張を「見当たらなかった（検索した語）」の強さに直した．(7) 原文のリンクを，取得できなかった OpenReview から ICLR の会議録へ替えた．(8) 冒頭の一文・位置づけ・主張を要旨と §4 の表現に寄せた．**読めていないもの**：OpenReview のページ（自動取得が認証の確認で止められ，取得できなかった）．付録の図の中の値と，付録の表のうち本文書が引かないものは照合していない．付録 K の Table 20・21（Llama 2）は読んだが本文書には載せていない．ICLR の会議録の PDF そのものは取得していない（arXiv v2 の頭書き "Published as a conference paper at ICLR 2024" と会議録の一覧で掲載を確かめた）．
