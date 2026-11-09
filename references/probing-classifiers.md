# Probing Classifiers: Promises, Shortcomings, and Advances

**著者**：Yonatan Belinkov（テクニオン）
**掲載**：Computational Linguistics 48(1)，pp. 207–219，MIT Press，2022-03（squib）．arXiv:2102.12452（2021）
**原文**：[ACL Anthology](https://aclanthology.org/2022.cl-1.7/)・[arXiv](https://arxiv.org/abs/2102.12452)

**probing を批判的に総覧した論考——この枠組みの約束と，限界と，その後の前進．**

## 位置づけ

**本研究がこの文献を引くのは，個別の主張のためではなく，probing という枠組みそのものへの批判が一本の総説にまとまっている場所として**である．新しい実験を提案する論文ではなく，Computational Linguistics 誌の squib（短い論考）である．研究文書は §2 で引く．**ポスターと予稿の文献一覧には無い．**

## 主張

probing classifier の考え方は単純である——"a classifier is trained to predict some linguistic property from a model's representations"——，そして広く使われてきた．しかし著者は "recent studies have demonstrated various methodological limitations of this approach" として，この枠組みを批判的に総覧し，約束（promises）・欠点（shortcomings）・前進（advances）を整理する．

## 本文の論点

**枠組み（§2）．** 元のモデル `f` の中間表現 `f_l(x)` から，性質 `z` を当てる分類器 `g` を学習し，その性能を報告する．性能は `g` と用いたデータだけでなく，`f` とその学習データにも依る．

**約束（§3）．** 性能が高いことは，表現の質・読みやすさ・取り出しやすさを示すと解釈されてきた．ただし "different terms (quality, readability, usability, etc.) appear abstractedly without precise definitions"．

**欠点と前進（§4）．**

| 節 | 欠点 | 前進 |
| --- | --- | --- |
| §4.1 比較と対照 | 性能の数値だけでは高いか低いか分からない．プローブが記憶しただけかもしれない | 多数派の基準線，単純な表現での分類器，無作為な重みの基準線．上限との比較．対照タスクと selectivity（Hewitt and Liang），対照関数（Pimentel ら），対照データセット（Ravichander ら） |
| §4.2 分類器の選び方 | 分類器の表現力が結論を左右する | 線形など単純なプローブの推奨．最小記述長（Voita and Titov）．精度と複雑さのトレードオフの報告．パラメタを学習しないプローブ |
| §4.3 相関と因果 | `g` と `f` が切り離されている．"it does not tell us whether this property is involved in predictions of f" | 表現への介入．Giulianelli ら，Tucker ら，Elazar ら，Feder ら |
| §4.4 データセットとタスク | データセットはタスクの不完全な代理である | 同じデータで異なるモデルを，同じモデルを異なるデータで比べること．実験は足りていない |
| §4.5 性質は前もって決める | 調べる性質を先に定めるので，既存の注釈に縛られ，偏った結論を導きうる | 潜在クラスタを学習する試み（Michael ら） |

**対照タスク（§4.1）．** Hewitt and Liang は，プローブ用データのラベルを無作為にして，記憶でしか解けない課題を作る．本来の課題との性能の差が selectivity である．線形プローブは selectivity が高く，非線形プローブは低い傾向がある．語レベルの性質に向き，"it is less clear how to apply this idea more broadly, such as in sentence-level properties"．

**読めることと使われることの関係は，総覧の中で一つの答えに定まっていない（§4.3）．**

| 研究 | Belinkov の記述 |
| --- | --- |
| Giulianelli ら | プローブの勾配で表現を変え，数の一致を試す例で元の性能が上がった．"They conclude that probing classifiers can identify features that are actually used by the model." |
| Elazar ら | 性質を除いた．"they conclude that probing classifiers do not always identify features that are actually used by the model" |
| Lovering ら | 最小記述長で測った取り出しやすさが，`f` の予測が性質に沿うことと相関する |
| Tamkin ら | プローブの性能が高い特徴と，微調整で重要な特徴が食い違う |

**まとめ（§5）．** "Absolute claims about representation quality seem difficult to make." 一方で，対照や指標の改善により相対的な主張はできる．因果の手法は "may reveal which properties are used by the original model"．元のモデルを改善した例は少ない——"Presently, there are few such successful examples"．その例の一つに，中間層へ音素の教師を掛けた Krishna らがある．

## 射程の外

- **自然言語処理モデルの解析についての総説**であり，VLM は扱わない．視覚への言及は，同じ考えがコンピュータビジョンのモデルでも提案された，の一文だけである．
- 新しい手法・実験・数値を提出しない．**個別の実験事実は，ここで紹介された原論文に当たること．**
- **学習による介入**（損失で表現を作り込む設定）は主題ではない．関連する記述は，`f` を更新し続ける Feder ら（§4.3）と，中間層に教師を掛けた Krishna ら（§5）の紹介にとどまる．

## ポスターとの対応

**ポスターはこの文献を引いていない．** 下の表は，ポスターの文の背景としての対応である．

| ポスターの文 | 原典の箇所 | 原文 | 読み |
| --- | --- | --- | --- |
| 読み取り精度 方法：𝒉 から署名を当てる線形分類器の正解率 | 要旨，§4.2 | "a classifier is trained to predict some linguistic property from a model's representations"，"Some studies advocate for using simple probes, such as linear classifiers" | probing の標準の形である．性能の指標としての正解率も §2 に "(e.g., accuracy)" とある．本研究の 𝒉 は長さ1に揃えたもので，10問以上ある署名に限っている（研究文書 §A.1） |
| （外した文）読める情報が使われるとは限らない | §4.3 | "it does not tell us whether this property is involved in predictions of f" | 「とは限らない」は §4.3 全体と整合する．「使われない」と書けば言い過ぎになる——Giulianelli らは使われている例を報告している |

## 学会で聞かれたら

1. **probing への批判の要点は？**
   比較の基準，指標，分類器の選択，そして方法が相関にとどまることです．加えて，データセットはタスクの代理にすぎないこと，調べる性質を先に決めねばならないことも挙げられています．（§1，§4）
2. **対照タスクとは何ですか？**
   Hewitt and Liang の対照で，プローブ用データのラベルを無作為にして，記憶でしか解けない課題を作ります．本来の課題との性能の差が selectivity で，精度が高くても selectivity が低ければプローブの記憶を疑います．（§4.1）
3. **相関と因果の問題とは？**
   プローブと元のモデルは別々に学習されるので，プローブが見つけた情報を元のモデルが使っているかは分かりません．使用を問うには，表現への介入が要ると整理されています．（§4.3，§5）
4. **読めることと使われることの関係は，もう決着していますか？**
   決着していません．Giulianelli らは使われている特徴を見つけられると結論し，Elazar らは常にではないと結論しています．「読めても使われるとは限らない」という言い方なら，この範囲に収まります．（§4.3）
5. **線形プローブでよいのですか？**
   単純なプローブを推す研究があり，線形のほうが selectivity が高い傾向も報告されています．一方で最も複雑なプローブを使うべきという主張もあり，統一された答えはありません．本研究の測定は線形だけです．（§4.1，§4.2，研究文書 §5.1）
6. **読めることと使われることの区別は既知です．本研究の新しさは何ですか？**
   区別そのものは既知の批判です．本研究は性質を明示的な損失で足し，ほぼ無代償で読み取れる形へ作り込んでも答えの決まり方に届かないことを，構成的な例で示します．なお Belinkov の総覧によれば，プローブの勾配で表現を変える介入は Giulianelli らが行っています．（研究文書 §2・§6，原典 §4.3）
7. **本研究の対照は，総覧のどれにあたりますか？**
   床は多数派の基準線で，言い回しプローブは単純な入力で学習した分類器との比較に近いものです．ablation のラベルの並べ替えはモデルの学習の側の対照で，プローブのラベルを無作為にする対照タスクとは別物です．研究文書に対照タスクの測定はありません．（§4.1，研究文書 §4.2・§A.1）
8. **読み取り精度 95.9% は高いと言えるのですか？**
   絶対的な主張は難しいと総覧も書いています．本研究が読むのは増分で，baseline 82.9%，ablation 81.0%，言い回しプローブ 69.7%，床 28.8% と比べて言っています．（§5，研究文書 §5.1）

---

**確認**：2026-08-22．ACL Anthology の書誌ページ（巻・号・頁・刊行月）と arXiv abs ページを参照し，要旨は逐語で照合した．

**照合**：2026-11-06．arXiv v4 の PDF の全12頁を pdftotext の文で読み，本文の論点を足した．書誌は ACL Anthology のページで確かめ直した．Computational Linguistics 版の PDF と逐語引用を突き合わせた．要旨の末尾は版で語が違う（arXiv は "This article"，刊行版は "This squib"）が，ここでは引いていない．ここで紹介した Giulianelli ら他の研究の本文は読んでいない——書いたのは Belinkov の記述である．研究文書についての答えは，研究文書の 2026-11-06 時点の記述による．
