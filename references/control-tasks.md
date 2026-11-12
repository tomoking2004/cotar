# Designing and Interpreting Probes with Control Tasks

**著者**：John Hewitt，Percy Liang（Stanford University）
**掲載**：EMNLP-IJCNLP 2019（Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)，pp. 2733–2743，Hong Kong, China，2019年11月）．arXiv:1909.03368 v1，2019-09-08
**原文**：[ACL Anthology](https://aclanthology.org/D19-1275/)・[arXiv](https://arxiv.org/abs/1909.03368)

**プローブの高い精度は，表現が言語構造を符号化しているからか，プローブが課題を学んだだけか——単語型に無作為な出力を割り当てた control task を言語課題と並べて測り，言語課題の精度から control task の精度を引いた差（selectivity）で精度を読む．**

## 位置づけ

本研究は第16層の表現から署名を線形プローブで読み，その読み取り精度（82.9%→95.9%，研究文書 §5.1）を報告する．本論文は，プローブの精度がプローブ自身の記憶する力からどれだけ来るかを問う枠組みである．ただし，研究文書（`.claude/context.md`）は本論文を引いておらず，control task も selectivity も測っていない（検索した語：Hewitt，control task，selectiv）．研究文書がプローブに置く対照は，言い回しプローブ，書き出しを分離した分割，床の三つで（§5.1，§A.1），いずれも本論文の control task とは別の作りである．

## 主張

- プローブが言語課題で高い精度を得たとき，表現が言語構造を符号化していると結論できるのか，プローブが課題を学んだだけなのか，と問う（要旨，§1）．表現が損失の無い符号化であるかぎり，十分に表現力のあるプローブは十分な学習データがあればどんな課題でも学べる，と著者は述べる（§1）．
- 言語課題と**同じ出力空間を持ち，単語型に無作為な出力を結びつけた課題**（control task）を，言語課題を補うものとして提案する．良いプローブは selective——言語課題の精度が高く，control task の精度が低い——であるべきだとする（要旨）．
- ELMo 表現上のよく使われるプローブは selective でない（要旨）．広く使われるハイパーパラメタ設定では，MLP プローブの selectivity は非常に低い（§1）．"dropout, commonly used to control probe complexity, is ineffective for improving selectivity of MLPs"（要旨）．他の形の正則化は効く（要旨）．
- ELMo の第1層のプローブは第2層より品詞タグ付けの精度がわずかに高いが，第2層のプローブは selectivity がかなり大きい．これは，どちらの層が品詞をよく表すのかという問いを生む（要旨）．

## 手法

### control task

各単語型 v に control behavior C(v) を独立に無作為に割り当て，トークンの出力をその型の C(v) で決める（文脈によらない．§2，図1）．性質は二つ（§2）．

| 性質 | 内容 |
| --- | --- |
| structure | 単語トークンの出力は単語型の決定的な関数 |
| randomness | "The output for each word type is sampled independently at random." |

帰結として，control task の精度の上限は**評価集合のうち型が学習集合に現れるトークンの割合**（残りのトークンではチャンスの精度）で決まり，C(v) は単語型ごとに独立に記憶しなければならない（§2.3）．

| 言語課題 | control task |
| --- | --- |
| 品詞タグ付け（45 タグ） | 各型に 1:45 のいずれかを割り当てる．品詞タグの経験的トークン分布から標本化（§2.1，脚注2） |
| 依存関係の辺予測（各トークンの親の位置を当てる） | 各型に三つの振る舞いのいずれかを一様に割り当てる：自分に付く（y_i = i）／文頭のトークンに付く（y_i = 1）／文末の語に付く（y_i = T）．木であることを課さず各予測を独立に評価する．文の根は評価から除き，文外の ROOT トークンは置かない（§2.2） |

### selectivity

```
selectivity = 言語課題の精度 − control task の精度
```

図2の説明文："Selectivity is defined as the difference between linguistic task accuracy and control task accuracy"（文の途中までの引用）．

### プローブ（§3.1）

- 品詞：線形（softmax(Ah+b)），MLP-1（ReLU の隠れ層1枚），MLP-2（隠れ層2枚）
- 依存：bilinear，MLP-1，MLP-2
- 既定（§3.2）：階数・隠れ次元 1000，dropout 0，学習文 39832，weight decay 0，勾配ステップの上限 100000（約40エポック）．最適化は Adam

### 複雑さの制御（掃引した値，§3.2）

| 方法 | 値 |
| --- | --- |
| 階数・隠れ次元 | 品詞 {2, 4, 10, 45}；依存 {5, 10, 50, 100} |
| dropout | p ∈ {0.2, 0.4, 0.6, 0.8} |
| 学習例数 | {4000, 400} |
| L2 weight decay | {0.01, 0.1, 1.0, 10.0} |
| early stopping（勾配ステップの上限） | {50000, 25000, 12500, 6000, 3000, 1500} |

> 本文 §3.5 は "We tried 6 methods" と書くが，§3.2 に記述があるのは上の5つで，図4の説明文も "the 5 complexity control methods" とする．

## 実験設定

- **データ**：Penn Treebank（従来の構文解析の学習・開発・テストの分割）．依存は Stanford Dependencies に変換．精度は**開発集合**で報告（§3.3）
- **表現**：55億語で事前学習した ELMo．複雑さの掃引は第1 BiLSTM 層（ELMo1）のみ（§3，§3.4）．層の比較（§4）に ELMo2 と Proj0（ELMo の非文脈的な文字 CNN の単語埋め込みに，**学習していない** BiLSTM を掛けたもの）
- **言語**：英語の品詞タグ付けと依存関係の辺予測

## 結果

### 表1：言語課題の精度／control 精度／selectivity（ELMo1）

| 設定 | プローブ | 品詞 精度 | 品詞 Ctl | 品詞 Sel | 依存 精度 | 依存 Ctl | 依存 Sel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 既定 | 線形 | 97.2 | 71.2 | **26.0** | – | – | – |
| 既定 | bilinear | – | – | – | 89.0 | 82.4 | 6.6 |
| 既定 | MLP-1 | 97.3 | 92.8 | 4.5 | 92.3 | 93.0 | **−0.7** |
| 既定 | MLP-2 | 97.3 | 93.2 | 4.2 | 93.9 | 92.0 | 1.9 |
| dropout 0.4 | 線形 | 97.1 | 67.3 | 29.8 | – | – | – |
| dropout 0.4 | bilinear | – | – | – | 90.4 | 73.7 | 16.7 |
| dropout 0.4 | MLP-1 | 97.5 | 93.4 | 4.1 | 93.8 | 93.1 | 0.7 |
| dropout 0.4 | MLP-2 | 97.4 | 94.1 | 3.4 | 94.7 | 93.5 | 1.3 |
| control task で設計 | 線形 | 97.0 | 64.0 | **33.0** | – | – | – |
| control task で設計 | bilinear | – | – | – | 91.0 | 83.1 | 7.9 |
| control task で設計 | MLP-1 | 97.2 | 80.6 | **16.6** | 90.5 | 84.3 | 6.2 |
| control task で設計 | MLP-2 | 97.2 | 81.7 | 15.4 | 92.8 | 89.8 | 3.0 |

「設計」は selectivity を見て手で選んだ設定である．表1の説明文によれば，品詞はすべて階数10（隠れ次元10），依存は bilinear が weight decay 0.01，MLP-1 が weight decay 0.1，MLP-2 が階数50．

> §3.5 の本文は，この設定を「線形と MLP の品詞プローブにそれぞれ階数 10 と 45，bilinear の依存プローブに weight decay 0.01，MLP の依存プローブに weight decay 0.1」と書いており，表1の説明文と一致しない．また同じ §3.5 の本文は，既定の設定の依存について bilinear の selectivity を 16.7，MLP-2 を 1.3 と書くが，表1の既定の行は 6.6 と 1.9 である（16.7 と 1.3 は dropout 0.4 の行の値）．表は原文の表のとおり載せる．

**読み方**：

- **品詞の精度はどのプローブも 97.0–97.5 でほぼ同じ**だが，既定の control 精度は線形 71.2 に対し MLP 92.8–93.2．著者は，MLP の精度上の小さな利得はプローブの表現力の増加で説明できるかもしれないとする（§1）．
- **線形・bilinear も記憶する**——"Bilinear and linear probes themselves show a significant capacity for memorization."（§3.5）
- **dropout は MLP の selectivity を一貫しては上げない**——品詞タグ付けについて，p = 0.2 から 0.8 の広い範囲で "does not consistently lead to high-accuracy, high-selectivity MLP probes"（§3.5）．表1（p = 0.4）では MLP の selectivity が下がる場合があり（品詞 MLP-2 は 4.2→3.4），上がるのは依存の MLP-1（−0.7→0.7）だけだと本文は書く．依存では，p = 0.6 の dropout が MLP-2 の selectivity を上げ（MLP-1 は上げない），bilinear の selectivity をかなり上げる，とも書く．
- **隠れ次元の制限が効く**——"constraining the hidden state dimensionality of MLPs is an effective way to encourage selectivity" で，言語課題の精度の犠牲は小さい（表1の品詞 MLP-1：selectivity 4.5→16.6，精度 97.3→97.2）．
- 学習例数の削減は品詞では効くが，依存では言語課題の精度を大きく落とす．weight decay も適切な値なら効くが，どの値を使えばよいかは明らかでない——"finding selective MLP probes is non-trivial"．early stopping は試した範囲で品詞の selectivity にほとんど影響せず，依存の MLP では selectivity を上げる（§3.5）．

### 表2：層の比較（既定のハイパーパラメタ）

| 表現 | 品詞 線形 精度 | 品詞 線形 Sel | 品詞 MLP-1 精度 | 品詞 MLP-1 Sel | 依存 bilinear 精度 | 依存 bilinear Sel | 依存 MLP-1 精度 | 依存 MLP-1 Sel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proj0 | 96.3 | 20.6 | 97.1 | 1.6 | 79.9 | −4.3 | 86.5 | −9.0 |
| ELMo1 | **97.2** | 26.0 | 97.3 | 4.5 | 89.7 | 6.7 | 92.5 | −1.0 |
| ELMo2 | 96.6 | **31.4** | 97.0 | 8.8 | 84.5 | 6.2 | 89.5 | 1.4 |

- 著者："We find probes on ELMo2 to be strikingly more selective than those on ELMo1"（§4.2）．
- **精度の高い層が selective な層とは限らない**：品詞の線形プローブで ELMo1 は精度が 0.6 高く，ELMo2 は selectivity が 5.4 高い．著者は，プローブが単語の同一性を品詞予測の特徴として使い，その特徴が ELMo2 では ELMo1 より得にくい，という別の仮説を示す（§4.2）．
- Proj0 と ELMo2 は品詞精度が近いが，Proj0 は selectivity がはるかに低い（§4.2）．
- **注記**：本文は ELMo2 の優位を "consistent across all probes" とするが，表2の依存 bilinear では ELMo2 6.2 < ELMo1 6.7 である．また表2の ELMo1 依存の値（89.7／6.7，92.5／−1.0）は表1の既定（89.0／6.6，92.3／−0.7）と一致しない．原文のとおり載せる．

## 著者が明記した限界

独立した限界節はない．本文中の留保：

- **既定のプローブはどれも過大**——full-rank の重み行列，数百次元の隠れ状態，中程度の dropout のような既定を使うなら "all probes, whether linear, bilinear, or multi-layer perceptron, are over-parameterized"（§3.6）．
- **selective な MLP の設計は自明でない**（上記）．設計したプローブの設定は selectivity を見て手で選んだもので，複雑さの制御法について著者は "all worked except dropout and early stopping" と書き，ただし広い範囲のハイパーパラメタで効いたものは無いと添える（§3.5）．

## 射程の外

- **対象は ELMo の文脈表現と，言語学的な課題（品詞タグ付け・依存関係の辺予測）**である．VLM・GQA・解き方の手順を表すラベルは扱わない（全文を GQA，VQA，image，multimodal で検索し，該当なし）．
- **実験は英語の Penn Treebank・二つの課題に限られ，複雑さの掃引は ELMo1 のみである．** 著者はこれを限界としては挙げていない．
- **control task は単語型に出力を割り当てる，単語ごとの課題**である（§2 は課題を，文を単語ごとに一つの出力へ写す関数と定める）．文全体に一つのラベルを付ける設定の control task は定義されていない．
- **表現の側を学習し直す実験は無い．** 使うのは事前学習済みの ELMo と，学習していない BiLSTM で，学習するのはプローブだけである．
- selectivity は，言語課題の精度を**プローブが単語型から記憶する力**に照らして読むための量である（要旨）．読めた情報がモデル自身の出力に使われているかを調べる実験は，全文に見当たらなかった（検索した語：causal，intervention）．

---

**確認**：2026-11-13．arXiv abs ページ（版は v1 のみ，Comments は EMNLP 2019），arXiv の PDF v1（11頁，付録なし），ACL Anthology の書誌ページと PDF（D19-1275．本文は arXiv v1 とページ下の表記を除いて同一であることを抽出テキストの差分で確かめた）を取得して全文を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．表1・表2はページ画像でも行と列の対応を確かめた．下書き（2026-10-11 作成）から直した主な点：位置づけの「selectivity の語と control task の発想の出典」を外し，研究文書が本論文を引いていないことを書いた．「control task の精度が低いプローブだけを信頼する」「表現力で説明できてしまう」を原文の強さに直した．依存の辺予測の評価（文の根を除く，文外の ROOT は置かない）と，dropout の例外（どの課題のどの p か）を原文どおりにした．設計したプローブの設定と既定の依存の selectivity について，表1と §3.5 本文の食い違いを注記した．著者が限界として書いていない「ELMo1・英語 PTB・二課題」を射程の外へ移した．**読めていないもの**：図4（掃引の曲線）の個々の値は図からは読んでおらず，本文に書かれた記述だけを照合した．数式は pdftotext の抽出で記号が一部欠けるので，載せたのは本文とページ画像で読めた範囲に限る．参考文献の一覧は照合していない．
