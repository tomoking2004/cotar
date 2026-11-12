# Vision-Language Models Create Cross-Modal Task Representations

**著者**：Grace Luo，Trevor Darrell，Amir Bar（University of California, Berkeley）
**掲載**：ICML 2025（Proceedings of the 42nd International Conference on Machine Learning，PMLR 267:41157–41176）．arXiv:2410.22330 v2，2025-05-07（v1 は 2024-10-29）
**原文**：[PMLR](https://proceedings.mlr.press/v267/luo25c.html)・[arXiv](https://arxiv.org/abs/2410.22330)

**VLM は，テキストの例・画像の例・指示文という異なる仕様を，モダリティと形式に依らない共通のタスクベクトルへ写す．そのベクトルを別のモダリティの問いへ差し込むと，正しい答えをしばしば引き出せる．**

## 位置づけ

タスクの表現が，VLM の系列末尾近くの特定のトークン位置にある一本のベクトル（タスクベクトル）として，**仕様のモダリティをまたいで**共有されることを，patching という介入で測った文献である．

## 主張

- 自己回帰型 VLM は，概念的に同じ入力を**モダリティ（テキスト・画像）や形式（例・指示文）に依らない共通のタスクベクトル**へ写す（要旨）．
- テキストの例から取ったタスクベクトルを画像の問いへ patching すると，同じ例を画像の問いと一緒にプロンプトに入れるより良い——"although patching utilizes only a single vector, it significantly outperforms few-shot prompting"（§1）．
- タスクベクトルは**ベースの LLM から微調整後の VLM へ**移せる．**指示文だけからも**作れる（要旨）．

## 手法

### タスクベクトルの取り出し

Hendel ら（2023）に倣い，タスクベクトル `h^t_{l,txt}` は "is extracted from the raw output of the l-th model layer at the delimiter token between the last query-answer pair"（§3.2）．取り出すのは**層の生の出力**である．

### patching

取り出したベクトルを，別のモダリティの未見の問いの**対応する層とトークン位置**に差し込む．タスクはそれ以外の形では与えない（§3.2）．追加の学習は行わない（Figure 3 の説明文）．

### 仕様の種類

| 仕様 | 内容 |
| --- | --- |
| テキストの例 | テキストの入力–答えの対 |
| 画像の例 | 画像の入力–答えの対 |
| 指示文 | 生の文字列をそのまま（テンプレートなし） |
| アンサンブル | 指示文由来と例由来のベクトルの要素ごとの平均 |

### 層の選び方

**検証セットでのタスク平均の正解率**でモデルごとに patching する最良の層を選ぶ（§4 Implementation Details）．

## 実験設定

**モデル**（Table 8）

| モデル | 融合 | 言語側 | 視覚側 | 規模 | 層数 | 画像 ICL |
| --- | --- | --- | --- | --- | --- | --- |
| LLaVA-v1.5 | late | Vicuna | CLIP | 7B | 32 | 不可 |
| Mantis-Fuyu | early | Fuyu | Fuyu | 8B | 36 | 可 |
| Idefics2 | late | Mistral | SigLIP | 8B | 32 | 可 |

- **タスク**：6つ（Country-Capital，Country-Currency，Animal-Latin，Animal-Young，Food-Color，Food-Flavor）．タスクあたり検証 30・テスト 100．
- **例の数**：N = 5．例を与えるときのテンプレートは Todd ら（2024）．
- **評価**：最初に生成したトークンとラベルの完全一致．乱数種3つの平均．

## 結果

### テキストの例 → 画像の問い（6タスク平均，Table 2）

| モデル | テキスト例 Patch | テキスト例 Prompt | 画像例 Prompt | 画像例 Patch | 文脈なし |
| --- | --- | --- | --- | --- | --- |
| LLaVA-v1.5 | **0.32** | 0.18 | — | — | 0.00 |
| Mantis-Fuyu | **0.31** | 0.09 | 0.18 | 0.20 | 0.00 |
| Idefics2 | **0.51** | 0.18 | 0.43 | 0.31 | 0.01 |

下限として置かれた Random（例のなかの多数派の答え）は，モデルに依らない一行で，平均 0.14 である．

**6タスクの平均では，テキスト例のベクトル一本が，同じ例をプロンプトに入れる場合も，同じモダリティ（画像）の例を使う場合も上回る．** タスクごとに見ると上回らない組もある（例：Idefics2 の Country-Capital は テキスト例 Patch 0.61 に対し 画像例 Prompt 0.71）．

VQAv2 から作った拡張タスク3つの平均（Idefics2，Table 4）：テキスト例 Patch 0.69，テキスト例 Prompt 0.63，画像例 Prompt 0.52，画像例 Patch 0.36．

### 画像の例 → テキストの問い（付録 A.6・A.7，Table 12）

6タスク平均で，Image-Text Patch は Idefics2 で 0.42，Mantis-Fuyu で 0.25（Idefics2 の Text-Text Patch は 0.65）．著者は Table 12 について "text examples consistently outperform image examples regardless of the query modality" と述べる（A.7）．

### タスクベクトルの空間

t-SNE ではタスクベクトルが**モダリティではなくタスクでまとまる**（Figure 2b．§5，L=23）．ただし二つの食べ物タスク（Food-Color と Food-Flavor）はよく分かれない．**テキスト由来と画像由来のベクトル間の cosine 類似度は見当たらなかった**——全文で cosine を検索すると，現れるのは下の LLM と VLM の間の類似度（§4.2，Table 3，Figure 5）だけである．

### LLM → VLM の転移（Table 3）

ベースの LLM で作ったタスクベクトルを VLM の画像の問いに差し込む．対象は対応する LLM を持つ late-fusion の2モデルである．

| モデル | LLM と VLM のベクトルの平均 cosine 類似度 | VLM→VLM Patch | LLM→VLM Patch |
| --- | --- | --- | --- |
| LLaVA-v1.5 | 0.95 | 0.32 | 0.37 |
| Idefics2 | 0.89 | 0.51 | 0.52 |

cosine 類似度のランダム基準（対応しないベクトルの組すべての平均）は 0.58．"the inter-model setting performs 1-5% better than the VLM-only setting"（§4.2）．

### 指示文由来のベクトル（Idefics2，§4.3，Figure 6）

指示文 Patch は5例の Examples Patch に並ぶ．アンサンブル Patch は5例の Examples Patch を **18%** 上回る（絶対・相対の別は明記されない）．**この比較の正解率そのものは図でのみ示され，本文に値は無い．** この節に限り，正解率は大文字小文字を区別せずに数えている．

### 層の位置と表現の三相

- **最良層**（付録 A.5，Figure 10）："For the late-fusion models, the best task vector lies near the middle of the network."；"for the early-fusion model, the best task vector lies in the late-middle layers"．**patching に選んだ層の番号は，モデルごとには本文に見当たらなかった**（全文を読み，"best layer" と "Layer" を検索した．層の番号が出るのは分析用の L=18・L=23 と，図の軸・図中の札だけである）．
- **三相**（§5．Idefics2，最後の区切りトークンを logit lens で復号）："the input phase covers roughly the first 50% of layers, the task phase the next 40%, and the answer phase the last 10%"．本文の図（Figure 8）は Country-Capital のもので，全タスクは付録の Figure 14 にある．
  - input 相：*auf*（Idefics2 ではコロンに対応するトークン）へ復号される．
  - task 相：*headquarters* や *currency* のようなタスクの要約語へ復号される．
  - answer 相：*Rome* のような都市名へ復号される．

## 著者が明記した限界

- **なぜ共有されるかは説明できない**（§6）："In this work, we show that VLMs learn cross-modal task representations but lack a definitive explanation for why."．仮説として，言語と他の知覚表現空間の同型構造，表現の収束を促すものとしてのモデルとデータの規模を挙げる．

## 射程の外

- **評価モデルは LLaVA-v1.5・Mantis-Fuyu・Idefics2（7B・8B・8B）の3つ**．全文で SmolVLM を検索したが現れない．
- **タスクは6つの単純な写像**（国→首都，動物→学名など）と，VQAv2 から作った3つのタスクである．正誤は最初に生成したトークンだけで判定する．全文で GQA を検索したが現れない．
- **タスクベクトルは推論時の patching で使われる**．モデルを学習し直す操作は行っていない．
- タスクベクトルを取り出す入力は，例の列か，指示文（§4.5 では質問文を含む）である．画像と質問を合わせた一問の入力について，その末尾の表現がタスクごとにまとまるかを調べた分析は，全文を読んだ範囲で見当たらなかった．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴．Comments は "ICML 2025"，Journal-ref の欄は無い），arXiv PDF v2（20頁，付録 A.1–A.8 を含む全文），PMLR の書誌ページ（v267/luo25c）と PMLR 版 PDF（20頁）を pdftotext でテキスト化して読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．Table 8 は pdftotext で行が崩れたので，PDF の14頁を画像にして確かめた．PMLR 版と arXiv v2 は，語の単位で比べて，1頁脚注の回次（arXiv v2 は "41 st"，PMLR 版は "42 nd"）と謝辞の見出しの番号のほかに本文の違いは見つからなかった．回次は PMLR の書誌に従った．arXiv v1（20頁）は Table 4・三相の割合・L=23 を含まない別の版で，本文書の値はすべて v2 のものである．下書き（2026-10-11 作成）から直した主な点：射程の外の「単一トークン出力」を「最初に生成したトークンで判定」に直した（答えには *Canis lupus* のような複数語がある）．「例示を伴わない単一の質問の表現は問われていない」を，§4.5 が質問文からベクトルを作ることに合わせて書き直した．Random を各モデルの列から外し，共通の一行として書いた．「上回る」に「6タスクの平均では」と反例を添えた．「射影や学習を伴わない」を原文の語（raw output，without additional training）に直した．三相の引用を answer 相まで延ばし，input 相の復号先を *auf* と書いた．**読めていないもの**：Figure 2・6・8・10・13–15 の図中の値（本文に書かれた値だけを照合した）．arXiv v1 は数値の有無を検索しただけで，通読していない．引用されている先行研究（Hendel ら，Todd ら）の原典．
