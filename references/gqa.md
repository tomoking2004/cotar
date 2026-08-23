# GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering

**著者**：Drew A. Hudson，Christopher D. Manning（スタンフォード大学）
**掲載**：CVPR 2019（oral），pp. 6700–6709．arXiv:1902.09506 v3
**原文**：[arXiv](https://arxiv.org/abs/1902.09506)・[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)

**各質問に，答えへ至る手順を書き下した functional program が付く——質問の意味が，人手を介さず機械可読な形で与えられる．**

## 位置づけ

**この文献を引くのは，データの出所としてだけではない．** タスク署名は GQA の functional program から演算子だけを取り出して作られており，署名という形式化そのものがこの論文の設計に乗っている．署名と食い違う量として引き合いに出す一貫性（consistency）指標と含意関係も，この論文が定義したものである．

## データセット

Visual Genome の scene graph から，22,669,678 問を 113,018 枚の画像について生成する．質問が scene graph を辿る手続きから作られるので，**答えだけでなく答えに至る道筋も既知**である——これが，解き方が同じかどうかを人手を介さず判定できる根拠になっている．

## functional program

各質問は原子的な演算の系列として表される．論文の例では “What color is the apple on the white table?” が

> “select: table → filter: white → relate(subject,on): apple → query: color”

となる．演算には対象の選択・関係の辿り・属性の照合・論理演算（AND・OR・NOT）があり，各ステップが引数と依存先を持って連なる．

**なお，論文が「構造型」と呼ぶのは verify・query・choose・logical・compare の5種**であって，上の `select`・`relate` のような演算子とは別の層である．演算子の系列を取り出す操作は，この5分類とは無関係に定義される．

## balanced 版

生の 22M 問は答えの分布が大きく偏るので，棄却サンプリングで 1.7M 問へ落とす．頻度の高い答えから順に重みを付け直すが，

> “we also maintain minimum and maximum ratios between each pair of subsequent answers (sorted by frequency). This ensures that the relative frequency-based answer ranking stays the same.”

——**偏りを消すのではなく，順位を保ったまま緩める．** 現実の出現傾向は残す，という設計である．

## 分割

> “We split the dataset into 70% train, 10% validation, 10% test and 10% challenge, making sure that all the questions about a given image appear in the same split.”

**切るのは画像単位である**——だから同じ画像についての質問が分割をまたがない．質問 ID だけでなく画像 ID でも排他になるのは，この設計の帰結であって偶然ではない．

## 一貫性（consistency）と含意関係

> “For each question-answer pair (q,a), we define a set E_q = {q_1, q_2, …, q_n} of entailed questions, the answers to which can be unambiguously inferred given (q,a).”

一貫性は，正解した質問それぞれについてその含意先での正解率を測り，全体で平均したものである．人間は 98.4%，論文の MAC は 81.59% にとどまる．

**含意関係が結ぶのは，同じ場面についての問である**——論文の例では，「白い皿の左に赤いりんごがあるか」への “yes” が「皿はりんごの右にあるか」の答えを決める．場面を共有することと，手順を共有することは別である．

## 評価指標

accuracy のほか，consistency，validity（その質問に理論上ありうる答えを返しているか），plausibility（現実にありうる答えか），grounding（注意が関連領域に向いているか．注意機構を持つモデルのみ），distribution（予測の答えの分布が真の分布とどれだけ一致するか．カイ二乗）を定義する．

## 結果

質問文だけを見る LSTM が 42.1%，当時の強い VQA モデルが 54.1%，人間が 89.3%．

## 射程の外

- **内部表現を一切扱わない．** データセットと評価指標の論文であり，モデルの中で何が起きているかは問わない．表現の比較・プローブ・整合はどれもこの論文の外にある．
- **タスク署名はこの論文のものではない．** program から演算子の系列だけを取り出し，引数を捨てて一致を見るのは，こちらの構成である．論文は program を「その質問の意味」として持つが，**二つの質問が同じ手順かどうか**を問う道具としては使っていない．
- **`testdev` は論文に現れない．** 論文が述べる分割は train／validation／test／challenge の4つで，`testdev` は配布物の側が用意したものである．その位置づけの根拠は論文ではなく公式サイトにある．
- **一貫性は答えの整合を求める指標であって，表現の整合とは無関係である．** 含意関係にある対に，同じ内部表現を要求してはいない．

---

**確認**：2026-08-23．arXiv abs ページ（書誌・要旨・版）と ar5iv 版の本文 HTML，GQA 公式の evaluation ページ，DBLP の書誌（頁）を参照した．**CVF Open Access の PDF 本文は精読していない**——取得した PDF が保護されており，本文は ar5iv 経由でしか読めていない．したがって上の逐語引用は**その経路を通ったもの**であり，publication で引くときは PDF 本文に当たり直すこと．本文の細部（生成手続きの各段・付録の統計）も上記の範囲にとどまる．
