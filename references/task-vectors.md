# In-Context Learning Creates Task Vectors

**著者**：Roee Hendel（Tel Aviv University），Mor Geva（Google DeepMind），Amir Globerson（Tel Aviv University，Google）
**掲載**：Findings of the Association for Computational Linguistics: EMNLP 2023，pp. 9318–9333（Singapore，2023年12月）．arXiv:2310.15916 v1，2023-10-24
**原文**：[ACL Anthology](https://aclanthology.org/2023.findings-emnlp.624/)・[arXiv](https://arxiv.org/abs/2310.15916)

**ICL は例示の集合 S を一本のタスクベクトル θ へ圧縮し，残りの層は θ と問い x から出力を作る，と見なせる．**

## 位置づけ

ICL を「例示の集合 S を単一のタスクベクトルへ圧縮し，そのベクトルで transformer を変調して出力を作る」過程と見る見方を提示し，その θ を中間層 L の "→"（プロンプト末尾のトークン）の表現として取り出した文献である．

## 主張

ICL の計算 `T([S, x])` は二段に分解できる（§2.1）．

```
T([S, x]) = f(x; A(S))
```

- **A（"Learning Algorithm"）**：例示 S を**タスクベクトル θ** へ写す．問い x には依存しない．
- **f（"Rule Application"）**：θ に基づいて x を出力へ写す．S には直接依存しない．
- 仮説クラスは `H = {f(·; θ) | θ}`．

要旨は，ICL が学習する関数が "often" この単純な構造を持つ，という強さで述べる．

## 手法

### θ の取り出し（§2.2，§3.1）

- 入力は例示 S の後に問い x と "→" を置く．**"→" がプロンプトの最後のトークン**である．
- **最初の L 層を通した後の "→" の表現を θ** とする．
- θ が実際の問い x に依存しないよう，**ダミーの問い x′**（Fig. 2 の例：Plum）を使って θ を取る．x への注意を遮る方法は "led to poor performance"．

### θ の適用（§3.1）

`[x, →]` だけを順伝播し，**層 L の "→" の位置に θ を差し込む**（patching）．f が S に直接依存するのを防ぐためである．

### L の選び方（§3.3）

L は自由パラメタであり，**開発セットでの正解率**で選ぶ．L ごとの正解率（Fig. 3．LLaMA 以外は付録 Fig. 7）について "all models exhibit a performance peak at a similar intermediate layer"——続けて "irrespective of their parameters and layer count differences"．**具体的な層番号は本文に記されていない**（図で示される）．

## 実験設定

**モデル**（§3.2，付録 Table 4）

| モデル | パラメタ | 層数 |
| --- | --- | --- |
| LLaMA | 7B／13B／30B | 32／40／60 |
| GPT-J | 6B | 28 |
| Pythia | 2.8B／6.9B／12B | 32／32／36 |

**タスク**：4分類18タスク（§3.2，付録 Table 5）．出力は単一トークンに限る．

| 分類 | タスク |
| --- | --- |
| Algorithmic | List first，List last，Next letter，Previous letter，To lowercase，To uppercase |
| Translation | French to English，Spanish to English，English to Spanish，English to French |
| Linguistic | Present to gerund，Present to past，Singular to plural，Antonyms |
| Knowledge | Country to Capital，Person to Language，Location to Continent，Religion |

上の「English to French」は Table 5 の印字どおりではない．Table 5 では，この行のタスク名が "English to Spanish" と重複して印字されている（説明文は "Given a word in English, translate to French"，Table 6 の略記は "En fr"）．

**比較**（§3.4）：Regular＝通常の ICL `T([S, x])`／Hypothesis＝θ の patching による手順／Baseline＝例示なしの `T([x, →])`．

## 結果

### 仮説に基づく予測の精度（§3.4）

"Across all models, our procedure maintains around 80-90% of the accuracy of regular ICL, while the baseline reaches only 10-20%."

**モデルごとの平均値は図（Fig. 4）でのみ示され，本文に数値は無い．** タスク別の例（付録 Table 6，LLaMA 13B，List last）：Baseline 0.07，Hypothesis 0.70，Regular 0.92．

### θ の頑健性（§4）

LLaMA 7B でタスクあたり50本の θ（S と x′ を変える）を t-SNE で見ると（Fig. 5），**タスクごとに別のクラスタ**をなす．距離のヒストグラム（付録 Fig. 8）では，同じタスクの θ どうしの方が，異なるタスクの θ との間より近い．

### θ の支配性（§5，Table 2）

S はタスク A の例示，差し込む θ はタスク B のもの．

| タスク A（S） | タスク B（θ） | Regular（A の正解率） | Conflicting（B の正解率） |
| --- | --- | --- | --- |
| Next Letter | To Upper | 0.92 | 0.77 |
| List Last | List First | 0.95 | 0.78 |
| Present to Past | to Gerund | 0.96 | 0.95 |

"This implies that the model mainly relies on θ, largely disregarding the demonstrations S for task A."

### θ の語彙への射影（§6，Table 3）

θ を語彙へ射影した分布の上位トークンに，複数の場合で，タスクを直接表す語が見られる（LLaMA 13B，Country Capital の上位10語に Paris，capital，cities，Madrid，London など）．"Importantly, these terms never explicitly appeared in the context."

## 著者が明記した限界

Limitations の節による．

- **単純なタスクのみ**："We study relatively simple tasks, whereas ICL can learn to perform more complex tasks"．
- **単一トークンの出力のみ**："We also focus on tasks where the output is a single token, while some other tasks require multi-token outputs."
- **機構の説明は無い**："we do not provide a mechanistic explanation for how the task vector is formed or how it is used"．
- 単一のタスクベクトルで足りる場合に焦点を当てており，より複雑な ICL ではより込み入ったパラメタ化が要るかもしれないことも明記する．

## 射程の外

- **対象はテキストの言語モデル**（LLaMA，GPT-J，Pythia）．VLM は評価に含まれない．
- **θ は推論時の順伝播で計算され，patching で使われる**．学習時に表現を整える設計は扱わない．
- **タスクは単一トークン出力の単純な写像**であり，より複雑なタスクに同じ機構が当てはまるかは "It remains to be seen" と著者自身が述べる．
- θ は ICL の例示から計算される．例示を伴わない順伝播は Baseline の正解率としてだけ現れ，その表現にタスクが表れるかは分析されていない．

---

**確認**：2026-11-13．arXiv abs ページ（版は v1 のみ，Comments は "Accepted at Findings of EMNLP 2023"），arXiv PDF v1（16頁，付録 A.1・A.2 を含む全文），ACL Anthology の書誌ページと Anthology 版 PDF（16頁）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．表（Table 2・4・5・6）と Fig. 4・7・8 はページ画像でも確かめた．Anthology 版 PDF の本文は，抽出したテキストの語の集合が arXiv v1 と一致した（違いは頁番号とヘッダ・フッタだけ）．下書き（2026-10-11 作成）から直した主な点：著者の所属を著者ごとに書き分けた（Globerson は Tel Aviv University と Google）．タスク名を原文の "A to B" の形に戻し，"English to French" が Table 5 の印字どおりでないことを表の直下に書いた．語彙への射影を原文の強さ（"In multiple cases"）に直した．「位置づけ」と「射程の外」の末項を原文に戻せる表現に直した．節・表・図の番号を足した．数値と逐語引用に修正は無い．**読めていないもの**：図の中の値（Fig. 3・4・7 の高さ）は読み取っておらず，載せていない．例示の数，開発・テストの件数は全文に記載が見当たらなかった（検索した語：demonstrations per，examples per，test set，shot）．公開コード（GitHub）は読んでいない．
