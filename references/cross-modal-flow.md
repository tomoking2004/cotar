# Cross-modal Information Flow in Multimodal Large Language Models

**著者**：Zhi Zhang（アムステルダム大学 ILLC），Srishti Yadav（アムステルダム大学 ILLC／コペンハーゲン大学），Fengze Han（ミュンヘン工科大学），Ekaterina Shutova（アムステルダム大学 ILLC）
**掲載**：CVPR 2025，pp. 19781–19791．arXiv:2411.18620（v1 2024-11-27，v2 2025-03-25）
**原文**：[arXiv](https://arxiv.org/abs/2411.18620)・[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.html)

**画像の情報は下層から中間層にかけて二度，質問の位置へ移り，そのあと質問の位置から最終位置へ移る——画像の位置から最終位置への直接の流れは，ほとんどの問の型で予測に大きくは効かない．**

## 主張

自己回帰型の多モーダル LLM（MLLM）で，視覚と言語の情報がどこでどう統合されて答えになるかを，GQA の視覚質問応答で追う．attention knockout で位置間の注意を層ごとに遮断した結果を，要旨は次の三段にまとめる．

1. 下層で，画像全体のより一般的な視覚特徴が質問のトークンの表現へ移る．
2. 中間層で，質問に関わる特定の物体の視覚情報が，対応する質問のトークンの位置へ移る．
3. 上層（"in the higher layers"）で，できあがった多モーダル表現が入力列の最終位置へ伝わり，最終の予測に使われる．

1 と 2 について序論は "visual information undergoes a two-stage integration into the language representation within the lower-to-middle layers" と書く．3 の層について，本文（§5）と結論は要旨と違って「中間層」と書く——結論は "In the middle layers, this integrated multimodal information is propagated to the hidden representation of the last position for the final prediction."．

## 手法

### attention knockout

注意は，入力の異なる位置の隠れ表現どうしが情報をやり取りする唯一のモジュールなので，注意の辺を遮断して位置間の情報の流れを止める（"we therefore adopt an attention knockout approach, proposed by Geva et al."）．元の位置集合 $S$ から先の位置集合 $T$ への注意を，第 $\ell$ 層のまわりの $k = 9$ 層の窓で，注意マスクの対応する要素を $-\infty$ にして止める（付録Bで，LLaVA-1.5-13b の ChooseAttr について $k = 1, 3, 5, 7, 9, 11, 15$ を試し，情報の流れについての観察は窓の大きさによらず一貫していると述べる．窓を広げると二つの流れは一つにまとまっていき，確率の低下は大きくなる）．

遮断の組み合わせは次のとおり．「最終位置」は "The last position means N-th position in the input sequence and it is also the first generated sub-word for the predicted answer" である．

| 実験 | 遮断する流れ |
| --- | --- |
| 実験1（§5） | 画像 → 最終位置，質問 → 最終位置，最終位置 → 最終位置 |
| 実験2（§5） | 画像 → 質問 |
| §6 | 質問に関わる画像パッチ → 質問／それ以外の画像パッチ → 質問（関わるかどうかは，パッチが質問中の物体のバウンディングボックスを含むかで決める） |

### 指標

遮断前にモデルが最も高い確率で生成する答え $a$ の確率を $p_1$，遮断後の同じ答え $a$ の確率を $p_2$ として，相対変化 $p_c\% = ((p_2 - p_1)/p_1) \times 100$．

### 答えの生成の追跡（§7）

"motivated by the approach of logit lens [1], we monitor the probability of the correct answer from the hidden representations at the last position of the input sequence across all layers"——各層の最終位置の隠れ表現に unembedding 行列を掛けて，答えの語の確率を読む．多くの MLLM のトークナイザが語の大文字・小文字（とくに語頭）を区別するので，語頭が大文字の形と小文字の形の両方を追う．

## 実験設定

- **データ**：GQA の検証データ（"from the validation set of GQA dataset"）から集める．GQA の問は構造5種 × 意味5種の2軸で15群に分かれ，そこから6群を選ぶ．付録Aの手順は，まず verify 型を除き，LLaVA-1.5-13b での平均正解率が 80% を超える型として ChooseAttr・ChooseCat・ChooseRel・LogicalObj を残し，ChooseGlo は検証データに 556 件しか無いので除き，問の型を増やすために CompareAttr と QueryAttr からは正解率 80% 超の下位の型（CompareAttr は twoCommon，QueryAttr は positionQuery）だけを取る，というものである．6つの型それぞれで，LLaVA-1.5-13b が正答した問を最大 1,000 件取る（付録A．本文 §4 は，群ごとに平均 920 件を「この論文で使うモデルの多くが正答した」問から取ると書く）．分析には，モデルごとに正答した問だけを使う（"we only use correctly predicted samples for analysis"．集めたデータでの各モデルの正解率は 95% を超える）．答えは "only one word or phrase"．

| 問の型 | 構造の型 | 意味の型 | 件数 |
| --- | --- | --- | --- |
| ChooseAttr | Choose | Attribute | 1000 |
| ChooseCat | Choose | Category | 1000 |
| ChooseRel | Choose | Relation | 964 |
| CompareAttr | Compare | Attribute | 570 |
| LogicalObj | Logical | Object | 991 |
| QueryAttr | Query | Attribute | 1000 |

（Table 1．LogicalObj だけが yes/no の Binary で，ほかは Open である．）

- **モデル**：本文は主に LLaVA-1.5-13b（言語側 Vicuna-v1.5-13b，40層）の結果を示す．付録Iで LLaVA-1.5-7b，LLaVA-v1.6-Vicuna-7b（いずれも Vicuna-v1.5-7b，32層），Llama3-LLaVA-NEXT-8b（Llama3-8b，32層）．画像エンコーダは4つとも CLIP-ViT-L-336px である．

付録の記号は arXiv v2（CVF の補足資料と同じ）のものである．v1 には付録E〜H（v2）に当たる節が無く，他モデルの実験は付録Eにある．

## 結果

以下，層番号はすべて LLaVA-1.5-13b（40層）のものである．

### 最終位置の予測に直接効くのは質問の位置で，画像の位置ではない（実験1）

- **質問 → 最終位置**を止めると，6つの型すべてで予測の確率が大きく落ちる．"For example, in the *ChooseAttr* task, this decreases the prediction probability by up to ∼ 30%."，そして "It is worth noting that this information flow pattern is observed primarily in the middle layers,"．
- **画像 → 最終位置**を止めても，"does not directly and significantly impact the final prediction in most tasks"．例外は QueryAttr で，画像から最終位置へのわずかな流れが見られるが，"this direct influence is negligible compared to its indirect effects, discussed below."
- **最終位置 → 最終位置**を止めても，予測の確率の変化は無視できる．

### 画像は二段階で質問の位置へ移る（実験2・§6）

- **画像 → 質問**を止めると，"We first observe a sharp drop in layers ∼ 0 − 4 and then a second smaller drop around 10th layer."
- 第1の落ち込みは "reduces the prediction probability by an average of ∼ 60% across all six tasks."．第2の落ち込みは "In the second drop, tasks such as ChooseAttr, ChooseCat, ChooseRel, and QueryAttr show another average reduction of ∼ 21%"（CompareAttr と LogicalObj の低下はより小さい）．
- §6 でパッチを分けると，第1段では「それ以外の画像パッチ → 質問」の遮断が，第2段では「質問に関わる画像パッチ → 質問」の遮断が，確率の低下の主な部分を占める．著者らはこれを，下層で画像全体の情報を質問の位置へ統合してより一般的な表現を作り，その後の層で質問に関わる領域に注意を向ける，と読む．

### 答えはまず意味として，次に表記として決まる（§7）

- 語頭が小文字の答えの確率は，モデルの中間層あたりで "rises sharply from near 0 to a range of ∼20% to ∼70% across the six VQA tasks"——著者らによれば，多モーダルの統合がちょうど済み，質問の位置から最終位置への移動がまだ続いている層である．
- その後この確率は 0 近くへ徐々に下がり，語頭が大文字の形は "remains low in the initial layers following 20th but starts to increase in subsequent layers."．著者らは "the model has already semantically inferred the answer by about halfway through layers" と読み，上層では答えの統語的な正しさを整え始めるとする．
- 付録I：LLaVA-1.5-7b と LLaVA-v1.6-Vicuna-7b は，Figure 6 と整合する結果を示す．Llama3-LLaVA-NEXT-8b では，語頭が大文字の答えの確率の変化は本文の結果とよく合うが "no such pattern is observed for the Noncapitalized Answer"．著者らはこれを，意味の生成と統語の修正という中間の段を経ずに，統語的に正しい答えを直接生成することを示唆する，と読む．

### 効率への応用（付録F．arXiv v2 と CVF の補足資料にあり，v1 には無い）

"removing image tokens from final 20 layers keeps accuracy intact while reducing inference time by ∼40%"（付録Fはモデルを "LLaVA-v1.6-13B" と書いており，§4 のモデル一覧には無い名前である．）

## 著者が明記した限界

独立した限界の節は無い（"limitation" は "space limitations" の形でしか現れない）．将来の方向として，要旨は "thereby facilitating future research into multimodal information localization and editing"，結論は "offering new research directions for better understanding the interaction of the two modalities in MLLMs" と述べるにとどまる．

## 射程の外

- **測ったのは LLaVA 系列の4モデル（7b・8b・13b）だけ**であって，SmolVLM は含まれない（全文に "Smol" は現れない）．これ以外のモデルで同じ層配置になるかは示されていない．
- **本文の分析は GQA の6つの問の型・正答した問だけ**である．verify 型と ChooseGlo は除かれ，CompareAttr と QueryAttr は下位の型1つずつに限られ，正答しなかった問は分析に入らない．付録H（v2）は OKVQA と AOKVQA でも結果が本文の所見と合うと述べる．GQA の問を構造と意味の型で分けており，本研究の署名（演算子列）に当たる単位での分析は見当たらなかった（検索した語：program，signature，operation）．
- **学習による介入をしない．** 操作は，学習済みのモデルに対する推論時の注意の遮断と，各層の確率の読み取りである．
- 本研究 §3.3 が整合を第16層（全32層の中間）・プロンプトの末尾の位置に掛けたことと関わる．本論文の LLaVA-1.5-13b（40層）では，画像の位置から最終位置への直接の影響は，質問の位置を経る間接の効果に比べて無視でき，質問の位置から最終位置への流れは主に中間層で見られ，答えは層の半ばあたりまでに意味として推論されている．32層の SmolVLM（本研究 §4.1）の第16層がこれに当たる段階にあるかは，本論文からは言えない——SmolVLM は測られておらず，Llama3-LLaVA-NEXT-8b のようにモデルによって答えの生成の形が変わる例もある．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者4名・v1 2024-11-27／v2 2025-03-25・Journal reference「CVPR2025」），arXiv PDF v2（23頁，付録A〜Kを含む全文）と v1（21頁，付録A〜G），CVF Open Access の論文ページ（著者・CVPR 2025・pp. 19781-19791），CVF の本文 PDF（11頁）と補足資料 PDF（12頁）を取得して pdftotext で読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．Table 1・Table 2 と著者の所属記号は v2 の頁画像で確かめた．逐語引用は v2 の文字列で，付録F・Hに依らないものは v1 にも同じ文字列がある．下書き（2026-10-11 作成）から直した主な点：(1) データの選び方——「正解率 80% 超の問の型を選び，verify 型と ChooseGlo は除く」を付録Aの手順に直した（CompareAttr・QueryAttr は型全体では 80% に届かず下位の型だけを取ること，ChooseGlo を除く理由は件数であること）．(2) 最終位置への流れの層——要旨は「上層」，本文と結論は「中間層」と書く食い違いを明記した．(3) 逐語引用の境界——原文に無い句点で閉じていた3件（最終位置の定義，∼ 21%，∼40%）を原文どおりにし，"~30%"・"~0-4" の記号と空きを原文の "∼ 30%"・"∼ 0 − 4" に揃えた．(4) 付録の記号が版で違うこと（付録F・Hは v2 だけ，他モデルの実験は v1 で付録E・v2 で付録I）を足した．(5) 付録Bの窓の比較が ChooseAttr だけであること，§6 と付録Iの記述が著者の読みであること，付録Hが OKVQA・AOKVQA を扱うことを足し，冒頭の一文の「ほとんど使われない」を原文の強さに直した．(6) 「署名単位の分析もしない」を「見当たらなかった（検索した語）」に，「自然な分布での全体像は扱わない」を原文にある除外の事実に直した．**読めていないもの**：図（Figure 3〜37）の曲線の値は，本文が文で述べたものだけを照合し，図からは読み取っていない．IEEE Xplore の最終版と，著者の公開コードは見ていない．
