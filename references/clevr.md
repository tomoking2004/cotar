# CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning

**著者**：Justin Johnson（Stanford University／Facebook AI Research），Bharath Hariharan（Facebook AI Research），Laurens van der Maaten（同），Li Fei-Fei（Stanford University），C. Lawrence Zitnick（Facebook AI Research），Ross Girshick（同）
**掲載**：CVPR 2017，pp. 2901–2910．arXiv:1612.06890 v1，2016-12-20（arXiv 版は 17 頁で，付録 A–D を含む．CVF Open Access の PDF は本文 10 頁で，付録を含まない）
**原文**：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html)・[arXiv](https://arxiv.org/abs/1612.06890)

**合成場面の各質問に，scene graph 上で実行できる functional program を付け，質問の型・関係の種類・構造などの軸でデータを切り分けて，VQA モデルの推論能力を診断する．**

## 位置づけ

GQA の著者は，GQA の作成にあたって CLEVR から着想を得たと書いている（GQA 論文 arXiv:1902.09506 v3 の本文："In creating GQA, we drew inspiration from the CLEVR task"）．本研究のタスク署名は GQA の functional program から演算子の系列を取り出したものなので（研究文書 §3.1），「質問に実行可能な program を付ける」という設計について，CLEVR は GQA のさらに前にある文献にあたる．CLEVR の論文自体は GQA（2019 年）より前のもので，GQA には触れていない．

## 主張

- 既存の VQA ベンチマークは，推論せずに正答できる強い偏りを持ち，複数の誤りの源が混ざっていて，モデルの弱点を特定しにくい（要旨）．
- 偏りを最小にし，各質問がどの種類の推論を要するかを詳細に注記した診断用データセットを作る（要旨）．
- それで当時の VQA モデルを分析し，次の弱点を挙げる（5 節）：**短期記憶を要する問**（属性の比較，整数の等しさ，same-attribute 関係，木構造の問），**長い推論の連鎖**，**空間関係**（絶対的な画像上の位置に頼る），**属性を分離した表現**（学習時に無い属性の組み合わせへ汎化しない）．

## 手法

### 場面

- "Every scene contains between three and ten objects with random shapes, sizes, materials, colors, and positions."
- 形は cube／sphere／cylinder，大きさ2種（small／large），材質2種（metal／rubber），色8種．
- 空間関係は left／right／behind／in front の4つで，カメラの視点ベクトルを地面に射影した向きで定める．
- 場面は，属性を注記した物体を節点とし，空間関係にある物体どうしを辺で結ぶ scene graph でも表せる．

### functional program

"Each question in CLEVR is associated with a functional program that can be executed on an image's scene graph, yielding the answer to the question."——**実行すると答えが出る．**

基本関数（付録 A．arXiv 版にある）は，それぞれ入力と出力の型が定められている．

| 関数 | 型 |
| --- | --- |
| `scene` | ∅ → ObjectSet |
| `unique` | ObjectSet → Object（入力が単元集合でなければ例外を上げ，その問を ill-posed とする） |
| `relate` | Object × Relation → ObjectSet |
| `count` | ObjectSet → Integer |
| `exist` | ObjectSet → Boolean |
| `filter_size`／`_color`／`_material`／`_shape` | ObjectSet × 値 → ObjectSet |
| `query_size`／`_color`／`_material`／`_shape` | Object → 値 |
| `AND`／`OR` | ObjectSet × ObjectSet → ObjectSet（積集合／和集合） |
| `same_size`／`_color`／`_material`／`_shape` | Object → ObjectSet（入力の物体自身は除く） |
| `equal_integer`／`less_than`／`greater_than` | Integer × Integer → Boolean |
| `equal_size`／`_color`／`_material`／`_shape` | 値 × 値 → Boolean |

例（付録 B）："What color is the cube behind the cylinder?" は

```
query_color(unique(filter_shape(cube, relate(behind, unique(filter_shape(cylinder, scene()))))))
```

**GQA の program との違い**：CLEVR の program は，この論文の中では**入れ子の関数式**として書かれ，関数名に属性の種別が入る（`filter_color`，`query_shape`）．本研究が使う GQA の program は，演算子・引数・依存先を持つステップの並びである（研究文書 §3.1．例 `select > relate > query`）．この対比はこちらで置いたもので，CLEVR の論文が述べたものではない．

### 質問の型・大きさ・構造

- **型は program の最も外側の関数**——"We categorize questions by question type, defined by the outermost function in the question's program"．
- 型の大分類（Figure 3，Figure 4）：exist，count，整数の比較（equal／less／more），属性の問い合わせ（size／color／material／shape），属性の比較（同4種）．
- **大きさ**："We define a question's size to be the number of functions in its program"．
- **構造**：chain 型と，二つの枝を論理 AND で結ぶ tree 型．
- 関係の種類：空間関係と，指定した属性の値が等しい物体どうしを結ぶ same-attribute 関係．

### 質問ファミリーと偏りの抑制

- **質問ファミリー**＝program の雛形1つ＋文の雛形複数．"CLEVR contains a total of 90 question families, each with a single program template and an average of four text templates."
- 文の雛形は，ファミリーごとに1〜2個を人手で書き，クラウドソーシングによる言い換えで増やす．形・色・材質には同義語を使う．
- 雛形の具体化は深さ優先探索で行い，場面の真の情報を使って，望ましくない問しか生まない範囲を刈る．ill-posed な問と退化した（degenerate）問を避ける．
- **ファミリーごとに答えの分布をほぼ一様にするため棄却サンプリング**を掛ける．ただし count の問では一様を強制せず "pushes towards a uniform distribution" にとどまる．

## 実験設定

**データ規模（Figure 3）**

| 分割 | 画像 | 質問 | 異なり質問 |
| --- | --- | --- | --- |
| train | 70,000 | 699,989 | 608,607 |
| val | 15,000 | 149,991 | 140,448 |
| test | 15,000 | 149,988 | 140,352 |
| 計 | 100,000 | 999,968 | 853,554 |

**比較したモデル**：Q-type mode（質問の型ごとに，訓練集合で最頻の答えを返す），LSTM（質問文のみ），CNN+BoW，CNN+LSTM，CNN+LSTM+MCB，CNN+LSTM+SA（soft spatial attention）．画像特徴は ImageNet 事前学習の ResNet-101（微調整なし，224×224，SA は conv4 の 14×14×1024）．Adam で学習し，ハイパーパラメタは val で選び，test は各モデル1回だけ評価．

**人間**：test の 5,500 問を Mechanical Turk で，3人の多数決．

## 結果

**型別の正解率（%，Figure 4．図の棒に添えられた数値を読んだ）**

| | Q-type mode | LSTM | CNN+BoW | CNN+LSTM | +MCB | +SA | 人間 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全体 | 41.8 | 46.8 | 48.4 | 52.3 | 51.4 | **68.5** | **92.6** |
| exist | 50.2 | 61.1 | 59.5 | 65.2 | 63.4 | 71.1 | 96.6 |
| count | 34.6 | 41.7 | 38.9 | 43.7 | 42.1 | 52.2 | 86.7 |
| 整数比較 equal | 51 | 63 | 50 | 57 | 57 | 60 | 79 |
| 整数比較 less | 52 | 73 | 54 | 72 | 71 | 82 | 87 |
| 整数比較 more | 50 | 71 | 49 | 69 | 68 | 74 | 91 |
| 問い合わせ size | 50 | 50 | 56 | 59 | 59 | 87 | 97 |
| 問い合わせ color | 12 | 13 | 32 | 32 | 32 | 81 | 95 |
| 問い合わせ material | 49 | 51 | 58 | 58 | 57 | 88 | 94 |
| 問い合わせ shape | 33 | 33 | 47 | 48 | 48 | 85 | 94 |

**属性の比較は，どのモデルもおよそ 50%**（"all models have an accuracy of approximately 50%"）——CNN+LSTM+SA も例外でない．答えは yes／no の二択である．

**型以外の切り口での診断（CNN+LSTM+SA，問い合わせ問）**

| 比較 | 正解率 |
| --- | --- |
| 空間関係1つの問 vs same-attribute 関係1つの問（4.3 節） | **93% vs 78%** |
| 空間関係2つの chain 型 vs 各枝に関係1つの tree 型（4.4 節） | **92% vs 74%** |

**組成的汎化（4.7 節）**：条件 A では cube を gray／blue／brown／yellow，cylinder を red／green／purple／cyan に限り，条件 B では両者の色を入れ替える（sphere はどちらも8色）．条件 A で学習し直したモデルを A と B の両方で評価すると，cube・cylinder の色を問う問で，CNN+LSTM+SA は **85%（A）から 51%（B）** に落ちる．cube・cylinder の材質を問う問では，CNN+LSTM+SA の差はより小さく（90% vs 81%），他のモデルには差がない．

## 著者が明記した限界

- **CLEVR の正解率そのものは目標ではない**："accuracy on CLEVR is not an end goal in itself"．CLEVR の世界を明示的に知る手作りの系はうまく働くかもしれないが，実世界へは汎化しない．
- **単独では使わない**："CLEVR should be used in conjunction with other VQA datasets"．
- count の問では答えの分布を一様に強制していない（棄却サンプリングは一様に「寄せる」だけ）．著者は，これが質問に条件づけた偏りを生むと書く．

## 著者が述べる含意

- 比較の弱さに対して："We plan to use CLEVR to study models with explicit short-term memory, facilitating comparisons between values"．
- 推論の型ごとに異なる構造を持つモデルへ："investigate methods that compile custom network architectures for different patterns of reasoning"．

## 射程の外

- **画像は合成**である（3〜10個の単純な立体．形3種・大きさ2種・材質2種・色8種，空間関係4つ）．実写画像は扱わない．
- **program は質問の生成と，型・関係・構造などによる分析に使われている．** 比較したモデルの学習に program を教師として使ったという記述は，全文に見当たらなかった（検索した語：program，supervis，train）．
- **評価は答えの正解率である．** モデルの内部表現を直接測る実験は無い．属性を分離した表現を学んでいないという議論（4.7 節，5 節）は，正解率の差からの推論である．
- **型は最も外側の関数1つで決まる．** program の形で問を束ねる単位としては質問ファミリー（program の雛形1つ）があるが，本研究の署名（演算子の系列の一致，研究文書 §3.1）にあたる単位は定義されておらず，ファミリー別の正解率も報告されていない．

---

**確認**：2026-11-13．arXiv abs ページ（v1 のみ），arXiv PDF v1（17頁，付録 A–D を含む全文．pdftotext でテキスト化し，Figure 3・Figure 4・Figure 9 と付録 A は頁の画像でも見た），CVF Open Access の書誌ページ（pp. 2901–2910）と PDF（10頁，付録なし）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．逐語引用は arXiv 版と CVF 版の両方に同じ文字列があることを確かめた．GQA については，GQA 論文（arXiv:1902.09506 v3）の CLEVR に触れた箇所だけを読んだ．下書き（2026-10-11 作成）から直した主な点：位置づけを GQA の著者の文に戻した．主張の弱点の列挙を 5 節の四項目に合わせた（「数える」「系統的な」は原文に無い）．材質の差を「小さな差」から「より小さい差」へ，85%→51% を A での評価と B での評価の比較と明記した．射程の外の「語彙・関係も閉じている」「署名にあたる単位は無い」を，原文に在る事実（属性と関係の数，質問ファミリー）に置き換え，「教師に使われていない」を「記述が見当たらなかった」の強さにした．著者の所属を一人ずつ書き，付録が arXiv 版にしか無いことを足した．**読めていないもの**：Figure 5–8 の棒・曲線の個々の値（本文に書かれた値だけを照合した），付録 D の例示ページの個々の問，GQA 論文の上記以外の部分，IEEE Xplore の最終出版版．
