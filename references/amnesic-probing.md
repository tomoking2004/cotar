# Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

**著者**：Yanai Elazar，Shauli Ravfogel，Alon Jacovi，Yoav Goldberg
**掲載**：Transactions of the Association for Computational Linguistics（TACL）2021．arXiv:2006.00995
**原文**：[arXiv](https://arxiv.org/abs/2006.00995)・[ACL Anthology](https://aclanthology.org/2021.tacl-1.10/)

**プローブの精度は，その情報が語予測に使われているかを予測しない——6つの性質で順位相関は見られなかった．**

## 主張

probing は「モデルが何を符号化しているか」を測るが，そこから**振る舞いについての結論を導くことはできない**．著者らは "the inability to infer behavioral conclusions from probing results" を指摘し，**何が符号化されているかではなく，どう使われているか**に焦点を移す方法を提案する．probing の欠点を，著者らは次のように書く——情報を取り出せることは示しても，"it provides no evidence for or against the actual use of this information by the model"（§1）．

中心にある直観は，**ある性質の有用性は，それを表現から取り除く因果的介入の影響を測れば分かる**というものである．

## 手法

### 情報の除去——INLP

**Iterative Nullspace Projection（INLP）** で，ある性質の**線形な**予測可能性を潰す．線形分類器（線形 SVM）を当てては，その分類器の零空間へ表現を射影する，を繰り返す——分類器が多数派クラスの精度しか出せなくなるまで続ける（実際には，開発データで多数派の精度から1点以内に入るまで）．

こうして作った**反実仮想の表現**（性質を除いたもの）を語予測の層に通し，振る舞いの変化を見る．**モデルの重みは変えない．** 敵対的学習は "requires changing the original encoding by retraining the model" ので採らない，と著者らは書く（§2.2）．

測る量は二つである．語予測の正解率（LM-Acc）と，介入の前後の語の分布の KL ダイバージェンス（LM-D_KL）である．

### 二つの対照

| 対照 | 何を確かめるか |
| --- | --- |
| **Rand** | **同じ本数のランダムな方向**を除く．性能低下が単なる階数の減少で説明できてしまわないかを見る．Rand の影響が Amnesic より小さければ重要な方向を除いたと結論し，同程度以上なら "there is no evidence for property usage" と結論する |
| **selectivity** | 除いた表現に正解の性質を32次元のベクトルとして連結し，後段を微調整する（他は凍結）．元の性能が戻れば損害は除いた性質で説明でき，戻らなければ意図より多くの情報を除いたことになる．Hewitt and Liang の selectivity とは別物である（脚注5） |

### 設定

- **モデル**：BERT（bert-base-uncased），全12層
- **除く性質**：品詞（粗 c-pos・細 f-pos），依存関係ラベル（dep），固有表現（ner），句境界（phrase start・phrase end）の6つ
- **主タスク**：語予測．§4 は単語を隠さない入力（非マスク）で，§5 以降は単語を隠した入力（マスク）で測る
- **データ**：UD Treebank・OntoNotes から10万トークン

## 結果

**probing の精度とタスクでの重要度は相関しない**——"the probe accuracy does not correlate with task importance as measured by our method (Spearman correlation of 8.5, with a p-value of 0.871)"（§4）．

**8.5 は原文の値である．** 順位相関は −1 から 1 の量なので，0.085 の意味と読むのが自然だが，そう書かれてはいない（TACL 版も 8.5）．表1の6性質で，プローブ精度と語予測の正解率の低下幅の順位相関を計算すると ρ = −0.086，p ≈ 0.87 となり，絶対値と p 値はほぼ一致する（こちらの計算．原文は計算の手順を書いていない）．**標本は6点と読める．** 相関が無いことを証明したのではなく，相関が見られなかったのである．

著者の言葉では "high probing performance does not entail this information is being used at a later part of the network"．

**語予測の正解率（表1．非マスク）**

| 性質 | 除いた方向の数 | probing 精度 | 除く前 | Rand | selectivity | Amnesic |
| --- | --- | --- | --- | --- | --- | --- |
| dep | 738 | 76.00 | 94.12 | 12.31 | 73.78 | **7.05** |
| f-pos | 585 | 89.50 | 94.12 | 56.47 | 92.68 | **12.31** |
| c-pos | 264 | 92.34 | 94.12 | 89.65 | 97.26 | **61.92** |
| ner | 133 | 93.53 | 94.00 | 92.56 | 96.06 | **83.14** |
| phrase start | 36 | 85.12 | 94.00 | 93.75 | 96.96 | **94.21** |
| phrase end | 22 | 83.09 | 94.00 | 93.86 | 96.93 | **94.32** |

**読める度合いの順と，除いたときの落ち幅の順は揃わない．** 最も低く読めた dep（76.00）が最も大きく落ち（−87.07），最も高く読めた ner（93.53）の落ち幅は −10.86 である．句境界は 85.12／83.09 と多数派（59.25／58.51）を大きく上回って読めるのに，除くと僅かに上がる（+0.21／+0.32）．

**句境界の「使われていない」には範囲がある．**

- 著者の言葉は "not important for the LM prediction at this part of the model" で，"they slightly harm it" と続く．層を問わない主張ではない．§7 では中間の層が句境界に効く．
- マスクの条件（表2）では句境界を除くと下がる（57.71 → 56.89／56.19）．Rand（57.46／57.27）より大きく下がる．

**Rand との比較．** dep・品詞・ner について著者は，Rand は元より低いが "substantially higher than the Amnesic scores" と書く．ただし dep の Rand は 12.31 まで落ちており，Amnesic との差は 5.26pt である．マスクの dep では Rand も Amnesic も 4.67 で差が無い．除いた方向が多いぶん，Rand も大きく効くのだろう（こちらの読み）．句境界では Rand は僅かに下がり，Amnesic は僅かに上がる．

**selectivity．** dep は 73.78 までしか戻らない——"some non-related features must have been coupled and removed with the dependency features"．

## 著者が明記した限界

1. **INLP が消せるのは線形な情報だけ**——"we remove only linearly-present information"．除いた情報は関心のある抽象的な性質の "an approximation" にすぎず，"one has to be cautious of causal interpretations of the results"．
2. **分類器は，正解と偶然相関する特徴に頼りうる**——その現れとして，ある層で除いても後の層でプローブ精度が上がることがあった（§7.1，§9）．
3. **性質どうしの重要度は比べにくい**——性質ごとに除く量が違うため．"The different erasure portion for different properties makes it hard to draw conclusions on which property is more important for the task of interest."
4. **非線形の除去は今後の課題**——"amnesic probing is not restricted to removing linear information"．手法が使えるようになれば INLP と差し替えられる，と書く．それで結論が変わるかは書かれていない．

## 射程の外

- **言語モデル単体**（BERT）であって，VLM ではない．
- **モデルを損失で作り変えて性質を入れることは扱わない．** 既にある表現から情報を除く方向の操作である．selectivity の対照だけは後段を微調整するが，これは性質を入れた学習の効果を見るためではない．
- 除く対象は品詞・依存・固有表現といった**言語学的な注釈**で，解き方の手順ではない．
- 測るのは語予測であって，質問応答の正解率ではない．

## ポスターとの対応

**ポスターはこの文献を引いていない．** 相対感度の目的にあった「読める情報が使われるとは限らない[5]」は，教授の指摘（「いらない」，2026-11-09）で文ごと外した．下の表のうち，外した文の行は口頭で話すときの対応，ほかの行はポスターの文の背景としての対応である．

| ポスターの文 | 原典の箇所 | 原文 | 読み |
| --- | --- | --- | --- |
| （外した文）読める情報が使われるとは限らない | §1 | "high probing performance does not mean that the probed information is used for predicting the main task" | 原文に戻せる．原典の "main task" は語予測である．口頭では「答えに」を外した一般の命題として言い，その範囲なら言い過ぎではない |
| 同上 | §4 | "high probing performance does not entail this information is being used at a later part of the network" | 「とは限らない」と同じ強さである |
| 相対感度 目的：答えが 𝒉 に依存する度合いの確認 | 要旨 | "the utility of a property for a given task can be assessed by measuring the influence of a causal intervention that removes it from the representation" | 問いは同じだが，測り方が違う．原典は除去の介入で，相対感度は第16層のまわりの一次の感度である．ポスターはこの文献を引かないので，ポスターの形でよい |
| 読み取り精度 方法：𝒉 から署名を当てる線形分類器の正解率 | §1，§4 | "to predict some desired information from the latent representations of the pre-trained model" | probing の定義に一致する．§4 の "the probing accuracy using a linear model" も線形である．ポスターのこの文に文献番号は無い |

## 学会で聞かれたら

### 論文そのもの

1. **amnesic probing とは何ですか？**
   ある性質を表現から取り除き，モデルの予測がどれだけ変わるかで，その性質が使われているかを測る方法です．何が符号化されているかではなく，どう使われているかを見ます．（要旨，§1）
2. **手順を説明してください．**
   線形分類器で性質を当て，その零空間へ表現を射影する，を多数派の精度しか出なくなるまで繰り返します．できた表現を語予測の層に通して，正解率と語の分布の変化を見て，Rand と selectivity の二つの対照で確かめます．（§2.2，§2.3，§3.3）
3. **INLP でなく敵対的学習ではだめなのですか？**
   敵対的学習はモデルを再学習して元の符号化を変えてしまい，元のモデルの振る舞いを調べる目的に合わないからです．情報を除ききれないという報告も挙げています．（§2.2）
4. **結論は何ですか？**
   プローブの精度は，除いたときの影響と相関しませんでした．だから高いプローブ精度から，その情報が後段で使われているとは言えない，という結論です．（要旨，§4）
5. **具体例はありますか？**
   非マスクの BERT で，依存関係ラベルはプローブ精度が 76% と6性質で最も低いのに，除くと語予測が 94% から 7% へ落ちます．句境界は 83〜85% で読めるのに，除いても下がりません．（表1）
6. **Rand 対照とは？**
   同じ本数のランダムな方向を除く対照です．Amnesic のほうが影響が大きければ重要な方向を除いたと結論し，Rand が同程度以上なら，使われている証拠は無いとします．（§2.3）
7. **selectivity 対照とは？**
   除いた性質の正解をベクトルとして連結し，後段を微調整して性能が戻るかを見ます．戻れば損害はその性質で説明でき，戻らなければ意図より多く除いたことになります．Hewitt and Liang の selectivity とは別物です．（§2.3，脚注5）
8. **相関の値はいくつですか？**
   原文は Spearman 8.5，p 値 0.871 と書いていて，0.085 の意味と読めます．表1の6性質から計算し直すと絶対値 0.086，p は約 0.87 でほぼ合うので，標本は6点と読めます．相関が無いと証明したのではなく，見られなかったという結果です．（§4，こちらの計算）
9. **著者が認めている限界は？**
   除けるのは線形に存在する情報だけで，因果の解釈には注意が要ると著者自身が書いています．性質どうしの重要度の比較も，除く量が違うので難しいとしています．（§9）

### 本研究との関係

10. **Elazar らは除く，本研究は足す，とはどう違うのですか？**
    Elazar らは学習済みのモデルを変えず，既にある性質を除いて語予測への影響を見ました．本研究は損失で署名を読み取れる形へ作り込んでから，答えがそれに頼るかを問います．読めることと使われることが別だという区別は同じで，向きが逆です．（原典 §2.2，研究文書 §2・§6）
11. **本研究でも署名を除いて答えの変化を見たのですか？**
    いいえ，除去の介入はしていません．言語モデルには別の層が補う自己修復が知られていて，壊しても出力が変わらないことは寄与が無いことを意味しないからです．代わりに，署名が答えを決める方向に載っているかと，第16層を動かしたときの相対感度を測りました．（研究文書 §A.4）
12. **Rand 対照は本研究のどこにあたりますか？**
    答えを決める方向 U_m に対する，同じ本数のランダムな方向 R_m です．5本引いて平均します．Elazar らは Rand の方向を除いて語予測を見ますが，本研究はその方向の中と外で読み取り精度を比べるので，使い方は同じではありません．（原典 §2.3，研究文書 §A.4）
13. **使われていないと言い切れるのですか？**
    言い切りません．相対感度は第16層のまわりの一次近似で，下がったのは2割前後でありゼロではありません．言えるのは「頼る度合いが下がった」までです．（研究文書 §5.3・§A.4）
14. **線形プローブの限界をどう考えますか？**
    読み取り精度が示すのは，線形に取り出せる形で載っている度合いです．線形でない経路については測っていません．Elazar らも線形な情報の除去にとどまると断っています．（研究文書 §5.1・§5.3，原典 §9）

---

**確認**：2026-08-22．arXiv abs ページ（書誌・要旨は逐語）と arXiv HTML 全文 v3 を参照し，数値・逐語引用・限界の記述を照合した．

**照合**：2026-11-06．arXiv v3 の PDF の全15頁（付録 A・B まで）を pdftotext の文で読み直した．表1・表2 は頁の画像でも確かめた．TACL 版の PDF と逐語引用，表1の値を突き合わせた．引用は両方の版に在る文字列に揃えた（dep の selectivity の引用だけは，arXiv 版の改行の分綴を TACL 版の綴りで読んだ）．確かめていないのは，図2〜6 の値の読み取りと，順位相関の算出手順である（原文に無い）．Giulianelli ら他の文献の本文は読んでいない．研究文書についての答えは，研究文書の 2026-11-06 時点の記述による．
