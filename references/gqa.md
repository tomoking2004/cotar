# GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering

**著者**：Drew A. Hudson，Christopher D. Manning（スタンフォード大学）
**掲載**：CVPR 2019（oral），pp. 6700–6709．arXiv:1902.09506 v3
**原文**：[arXiv](https://arxiv.org/abs/1902.09506)・[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)

**各質問に，答えへ至る手順を書き下した functional program が付く——質問の意味が，質問ごとの人手の注釈を要さず機械可読な形で与えられる．**

## 位置づけ

**この文献を引くのは，データの出所としてだけではない．** タスク署名は GQA の functional program から演算子だけを取り出して作られており，署名という形式化そのものがこの論文の設計に乗っている．署名と食い違う量として引き合いに出す一貫性（consistency）指標と含意関係も，この論文が定義したものである．

## データセット

Visual Genome の scene graph を正規化し，質問パターンを当てはめて，22,669,678 問を 113,018 枚の画像について生成する．scene graph は人の注釈で（test 用の 5k 件はクラウドソーシングで足した），パターンは 524 個（117 の質問群）である．うち 250 個は手作りで，274 個は VQA1.0 の質問をテンプレート化して得た（§3.2）．program は質問ごとに後から付けるのではなく，パターンの側に結びついている．

> “Each question pattern is associated with a structured representation in the form of a functional program.”

質問が scene graph を辿る手続きから作られるので，**答えだけでなく答えに至る道筋も既知**である——これが，解き方が同じかどうかを人手を介さず判定できる根拠になっている．ただしパターンは質問の深さを縛らない．欄には入れ子の参照が入るので（“the patterns do not strictly limit the structure or depth of each question”），一つのパターンから長さの違う program が出る．

## functional program

各質問は原子的な演算の系列として表される．論文の例では “What color is the apple on the white table?” が

> “select: table → filter: white → relate(subject,on): apple → query: color”

となる．演算には対象の選択・関係の辿り・属性の照合（§3.3）と，論理演算 or・and・not（付録10）がある．図6では，手順の数は 1 が 6%，2 が 42%，3 が 44%，4以上が 8% である．

配布データでは各ステップが `operation`・`argument`・`dependencies` を持つ（公式 Download ページ）．**引数と依存先という形式は配布物の側の記述であり，論文の本文には無い．**

**なお，論文が structural type と呼ぶのは verify・query・choose・logical・compare の5種**で，program の最後の演算から決まる（§4.1）．上の `select`・`relate` のような演算子とは別の層である．表2はさらに細かい 23 の型と，その program の雛形を挙げる（表2の列見出しは本文の定義と逆で，Semantic の列に query・verify が，Structural の列に global・attribute が並ぶ）．演算子の系列を取り出す操作は，これらの分類とは無関係に定義される．

## balanced 版

生の 22M 問から，答えの分布の均し・意味の近い問の除去・型ごとの間引きを経て 1.7M 問へ落とす（§3.4，付録8）．均すのは全体の分布ではなく，**質問群ごとの答えの分布**である．program から global（答えの型．例は color）と local（主語まで含む．例は apple-color）の2つのラベルを作り，それぞれの群を棄却サンプリングで滑らかな分布へ近づける．頻度の高い答えから順に重みを付け直すが，

> “we also maintain minimum and maximum ratios between each pair of subsequent answers (sorted by frequency). This ensures that the relative frequency-based answer ranking stays the same.”

——**偏りを消すのではなく，順位を保ったまま緩める．** 現実の出現傾向は残す，という設計である．答えの分布のエントロピーは 72% 増える．

## 分割

> “We split the dataset into 70% train, 10% validation, 10% test and 10% challenge, making sure that all the questions about a given image appear in the same split.”

**切るのは画像単位である**——だから同じ画像についての質問が分割をまたがない．質問 ID だけでなく画像 ID でも排他になるのは，この設計の帰結であって偶然ではない．

**論文の比率と配布の問数は合わない．** 1.7M の 70% は約 119 万問だが，配布の `train_balanced` は 943,000 問である（`analyses/audit-dataset.json`）．理由は，論文にも，読んだ範囲の公式サイトにも書かれていない．

## 一貫性（consistency）と含意関係

> “For each question-answer pair (q, a), we define a set E_q = q_1, q_2, . . . , q_n of entailed questions”

（式の添字は下線で写した．原文の式に波括弧は無い．）続けて，その答えが (q, a) から一意に導ける問の集合だと定める．一貫性は，正解した質問それぞれについてその含意先での正解率を測り，全体で平均したものである．人間は 98.4%，論文の MAC は 81.59% にとどまる．

**含意関係が結ぶのは，同じ場面についての問である**——論文の例では，“Is there a red apple to the left of the white plate?” への “yes” から “Is the plate to the right of the apple?” の答えが導ける．場面を共有することと，手順を共有することは別である．

## 評価指標

accuracy のほか，consistency，validity（答えが質問の範囲に入っているか．色の問に色で答えるなど），plausibility（答えが質問の主語とデータ全体で少なくとも一度は共起するか），grounding（注意が関連領域に向いているか．注意機構を持つモデルのみ），distribution（予測の答えの分布が真の分布とどれだけ一致するか．カイ二乗で，低いほどよい）を定義する．

## 結果

質問文だけを見る LSTM が 42.1%，当時の強い VQA モデルが 54.1%，人間が 89.3%（要旨）．**LSTM の値は要旨と本文で異なる**——§4.2 と表1は 41.07% で，arXiv 版と CVF 版の両方でそうなっている．差の説明は無い．表1の強いモデルは MAC（54.06%）で，値はすべて test 分割のものである．人間の値は無作為な 4,000 問に5人の答えの多数決を取ったものである（脚注5）．

## 射程の外

- **内部表現を扱わない．** データセットと評価指標の論文である．モデルの中身に触れるのは，注意の向きを見る grounding までで，表現の比較・プローブ・整合はどれもこの論文の外にある．
- **タスク署名はこの論文のものではない．** program から演算子の系列だけを取り出し，引数を捨てて一致を見るのは，こちらの構成である．論文は program を質問の意味の表現として持ち，含意関係の計算・型の分類・答えの分布の均しに使うが，**演算子の系列の一致で質問を組にすること**はしていない．
- **`testdev` は論文に現れない．** 論文が述べる分割は train／validation／test／challenge の4つで，`testdev` は配布物の側が用意したものである．その位置づけの根拠は論文ではなく公式サイトにある——“We recommend using the new Test-Dev split to evaluate your performance during development”（Evaluation ページ）．
- **一貫性は答えの整合を求める指標であって，表現の整合とは無関係である．** 含意関係にある対に，同じ内部表現を要求してはいない．

## ポスターとの対応

ポスターの文ごとに，依る原文を示す．「こちらの定義」「こちらの設定」と書いた部分は，原典の主張ではない．

**「タスク署名：データセット GQA[2] が各質問に付ける手順の記述から，操作の名前だけを並べた列」**

- 前半は §3 の文に戻せる——“Each question is associated with a functional program which lists the series of reasoning steps needed to be performed to arrive at the answer.”
- 「操作の名前」は配布データの `operation` である（Download ページ：“The reasoning operation.”）．論文の語では “atomic operations”（§3.3）にあたる．名前には `filter color`・`verify rel` のように種類まで含むものがある（`analyses/audit-dataset.json`）．
- 「並べた列」はこちらの操作で，論文には無い．
- 判定：言い過ぎは無い．

**「手順の判定：タスク署名（以下，署名）が一致すれば，同じ手順とみなす」**

- 対応する原文は無い．こちらの定義である．論文が program から求めるのは “entailment and equivalence relations between different questions”（§3.3）である．
- 捨てているのは引数だけではない．`task_signature` は `operation` しか読まないので，依存先も捨てている．列が同じで依存の形が違う program がどれだけあるかは測っていない．
- 判定：「みなす」と書いているので言い過ぎではない．口頭でも「同じ解き方である」と言い切らない（研究文書 §3.1）．

**「例：select: girl／relate: holding／query: name（縦に一段ずつ）→ select > relate > query（操作の名前だけを並べた列）」**

- 問 201593903（“What's the girl holding?”，答え racket）の `semantic` は `select: girl (5)` → `relate: _,holding,o (11)` → `query: name` である．公式の `questions1.2.zip` の `testdev_balanced_questions.json` から読んだ（2026-11-08）．
- ポスターは「操作: 対象」の形を原データのとおりに残し，対象の物体 ID（`(5)`・`(11)`）と関係の向きの記号（`_,…,o`）だけを略している．論文 §3 の例も同じ形（“select: table → filter: white → …”）で書かれている．同じ署名の例に並べた問 201889340（“Who is wearing a helmet?”）は `select: helmet` → `relate: person,wearing,s` → `query: name`，問 20794339（“What is the color of the mustard bottle?”）は `select: mustard bottle` → `query: color` である．
- 判定：略しただけで，操作の名前と順は原データのとおりである．

**「学習：GQA[2] 942,116問から抽出（1バッチは16署名 × 2問）」**

- 942,116 は論文の数ではない．配布の `train_balanced` 943,000 問から，同署名の相手がいない 884 問を除いた値である（研究文書 §4.1）．
- 論文が述べるのは “a 1.7M balanced dataset”（§1）と “70% train”（§3.4）までである．「16署名 × 2問」はこちらの設定である．
- 判定：不正確ではない．ただし「抽出」のとおり全問を使うわけではなく，一度でも出る問は多くて約 38 万問である（研究文書 §4.1．上限の計算）．「942,116問で学習した」とは言わない．

**「評価：12,578問（学習に不使用）」**

- 12,578 は `testdev_balanced` の問数である．論文には `testdev` の語も，この数も無い．
- 「学習に不使用」を支えるのは手元の監査である——`train` と `testdev` の間で，質問 ID も画像 ID も重なりは 0 だった（`analyses/audit-dataset.json`）．論文の “all the questions about a given image appear in the same split”（§3.4）は，論文の4分割についての文である．
- 判定：言い過ぎは無い．

**「正解率：GQA 公式の評価（12,578問）」**

- 指標の定義は Evaluation ページの “we give 1 point if the predicted answer p matches a and 0 otherwise” である．論文の語では “the standard accuracy metric”（§4.4）にあたる．
- 実際に行ったのは，公式の評価スクリプトを手元で `testdev_balanced` に掛けることである（`cotar/data/_gqa_eval.py`．冒頭の注記によれば，手元の修正は採点に触れない）．モデルの答えは，小文字化と句読点の除去をしてから渡す（`cotar/training/metrics.py`）．
- 判定：**やや不正確である．** 「公式の評価」は，公式の採点サーバや test 分割の値とも読める．代案は「正解率：GQA 公式の評価スクリプト（testdev_balanced，12,578問）」である．論文の表1（test）の値とは直接比べられない．

## 学会で聞かれたら

答えは口で言う形で書いた．括弧の中は出所で，言わなくてよい．

**Q1．functional program は何から作られるのですか．**
質問パターンからです．524 個のパターンに program と言い回しが結びついていて，そこへ scene graph の内容を入れると質問文ができます．（§3.2–3.3：“Each question pattern is associated with a structured representation in the form of a functional program.”）

**Q2．質問や program は人が書いたのですか．**
質問は自動生成で，一問ずつ人が書いたものではありません．ただし元の scene graph は人の注釈で，パターンも 524 個のうち 250 個は手作りです．（§3.2：“We begin from a seed set of 250 manually constructed patterns”．付録8：“22M auto-generated questions”）

**Q3．balanced とはどういう意味ですか．**
質問群ごとの答えの分布を均した版で，22M 問から 1.7M 問に減らしています．答えの順位は保つので，偏りをなくしたのではなく緩めたものです．（§3.4，付録8）

**Q4．testdev と test は何が違うのですか．**
論文の分割は train・validation・test・challenge の4つで，testdev は原典には書かれていません．公式サイトが開発中の評価に勧めている分割です．test は手元で採点できないので，本研究は testdev で報告しています．（§3.4．Evaluation ページ．研究文書 §4.1）

**Q5．1.7M の 70% と，学習データの 943,000 問は合わないのでは．**
合いません．70% なら約 119 万問ですが，配布の `train_balanced` は 943,000 問です．この差の理由は原典には書かれていません．（§3.4．`analyses/audit-dataset.json`）

**Q6．評価の問は，学習に入っていませんか．**
入っていません．手元で質問 ID も画像 ID も排他であることを確かめていて，報告に使う 398 枚の画像は学習中に一度も現れません．（研究文書 §4.1）

**Q7．言語の偏りや，データセットの限界は．**
質問文だけを見る LSTM でも 41.07% 当たるので，偏りは残っています（要旨の値は 42.1% です）．論文自身も，質問は自然文でなく，why の問や OCR，外部知識の問は含まないと書いています．（表1．§2：“Even though our questions are not natural as in other VQA datasets”．付録11：“Some VQA question types are not covered by GQA, such as intention (why) questions or ones involving OCR or external knowledge.”）

**Q8．人間の 89.3% に比べて，53.5% は低いのでは．**
89.3% は論文の test の値で，53.5% は testdev の値なので，直接は比べられません．本研究が見ているのは，同じ設定の群どうしの差です．（表1．研究文書 §4.1，§5.2）

**Q9．なぜ GQA なのですか．**
各質問に functional program が付いていて，解き方が同じかどうかを人手を介さずに判定できるからです．（研究文書 §3.1）

**Q10．署名は人が付けたラベルですか．**
いいえ．データに付いている program から演算子の名前を順に取り出すだけで，機械的に得られます．引数は意図して捨てています．引数まで一致を求めると，正例対がほとんど同じ質問になるからです．（研究文書 §3.1）

**Q11．質問文を読めば署名が分かるのではありませんか．**
作りからして相関しています．質問文と program は同じ質問パターンから生成されるからです．そこで質問文だけを入力にしたプローブを対照に置いていて，その精度は 69.7% で，整合後の表現の 95.9% を大きく下回ります．（§3.2．研究文書 §3.1，§5.1）

**Q12．GQA の question type とは何が違うのですか．**
論文の型は表2の 23 種の分類で，署名は演算子の列そのものです．学習データの署名は，正例対を組めるものだけで 1,575 種あります．論文の型をラベルにして整合する実験はしていません．（表2．研究文書 §4.1）

**Q13．GQA の一貫性指標と，署名の関係は．**
別のものを捉えています．含意関係は同じ場面についての問を結び，署名は解き方を区別するので，`val_balanced` の含意対 20,867 対のうち 52.1% は署名が異なります．（§4.4．研究文書 §7）

**Q14．署名が同じなら，本当に同じ解き方と言えますか．モデルは program どおりに解いているのですか．**
言えるのは署名が一致するところまでで，署名が解き方の正しい形式化かは本研究では検証していません．モデルが program どおりに解くかどうかは，原典には書かれていません．（研究文書 §3.1，§7）

---

**確認**：2026-11-06．arXiv v3 の PDF 全文（本文と付録7–12，18頁）と CVF Open Access の PDF（本文10頁）をテキスト化して読み，この要約の数値・引用・節番号を一つずつ照合した．引用は両版で一致することを機械的に確かめた．テキスト化で崩れた箇所（program の例の矢印，$E_q$ の式，図6，表2）は PDF のページ画像で見た．公式サイトの About・Evaluation・Download・Challenge の各ページも読んだ．2026-08-23 の確認は ar5iv 経由だったので，この確認で置き換える．**読めていないもの**：図5・7・10・11 の細かな値，配布物の質問データそのもの（問数と重なりは `analyses/audit-dataset.json` に依る）．

**追記**：2026-11-08．ポスターの例の3問（201593903・201889340・20794339）の `semantic` を，公式の `questions1.2.zip` から `testdev_balanced_questions.json` を範囲指定で読んで確かめた．読んだのはこの3問の質問・答え・`semantic` だけである．
