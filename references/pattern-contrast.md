# Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems

**著者**：Zhongli Li（Tencent Cloud Xiaowei），Wenxuan Zhang（Peking University），Chao Yan（Peking University），Qingyu Zhou（Tencent Cloud Xiaowei），Chao Li（Tencent Cloud Xiaowei），Hongzhi Liu（Peking University），Yunbo Cao（Tencent Cloud Xiaowei）
**掲載**：Findings of the Association for Computational Linguistics: ACL 2022，pp. 2486–2496，2022年5月（Anthology ID 2022.findings-acl.195，DOI 10.18653/v1/2022.findings-acl.195）．arXiv:2110.08464（v1 2021-10-16，v2 2022-03-10．本要約は v2 による）
**原文**：[ACL Anthology](https://aclanthology.org/2022.findings-acl.195/)・[arXiv](https://arxiv.org/abs/2110.08464)

**算数文章題のソルバーでは，プロトタイプ式が同じ問題の表現は互いに近づき，その中心から離れた表現は誤答しやすい——そこで式の木構造が（部分木まで含めて）一致する問題を正例，節点数が同じで演算子の違う問題を負例とする三つ組のマージン損失を生成損失に足す．**

## 位置づけ

解き方の構造（プロトタイプ式の木）を対照学習の正例・負例の基準にした先行例．解き方の構造が一致する問題どうしの表現を近づける点で，本研究の「署名（functional program の演算子の系列）が一致する質問どうしの表現を近づける」（研究文書 §3.1・§3.2）と通じる．ただし違いが三つある．正例は木の完全な一致だけでなく部分木の一致も含む（本研究は署名の一致のみ，§3.1）．損失は SupCon ではなく三つ組のマージン損失である（本研究は SupCon，§3.2）．掛ける先は木デコーダの節点の埋め込みである（本研究は VLM の言語側の第16層の表現，§3.3）．

## 主張

- 著者は，既存のモデルが手順を文脈から暗記し浅い手がかりに頼るという近年の報告（"Recent work shows that existing models memorize procedures from context and rely on shallow heuristics to solve MWPs."）を受けて，原因は問題の型（パターン）の全体的な理解を欠くことだと論じる（要旨）．
- 事前分析：意味エンコーダ（BERT）は下位層で語彙的な意味を捉え，上位層で**プロトタイプ式**をまとめる．プロトタイプの中心から離れた表現は誤答しやすい傾向がある．
- 結論の一文（第6節）："In this paper, we find the neural network generates incorrect solutions due to the non-distinction of MWP patterns."
- プロトタイプ式の木構造で正例・負例を集めた対照学習を補助目的として加えると，Math23k（中国語）と MathQA（英語，加工版）の両方で，単言語・多言語の両設定において正解率が上がる（表3）．

## 手法

### モデル（BERT-TD）

- **encoder**：BERT-base（多言語設定では multilingual BERT）．
- **decoder**：Xie and Sun（2019）に倣う木デコーダ．根節点を [CLS] トークンの表現で初期化し，正解の式木を予測する節点確率の負の対数尤度で学習する．目標の式はポーランド記法の列 `[y_1, y_2, …, y_m]` に変換する．
- Xie and Sun（2019）の復号の仕方により，**各節点の埋め込みはその節点を根とする部分木全体の情報を表す**（"each node embedding represents the whole subtree information rooted in it"）．対照損失はこの節点埋め込みに掛ける．

### プロトタイプ式と木

- **プロトタイプ式**：原文は定義の文を置かず，例で示す——`n_1 + n_2`（要旨），`n_1 − n_2`（表1）．表1では問題文の数値も `n_1, n_2, …` に置き換えてある．
- Xie and Sun（2019）に倣いプロトタイプ式を木に変換し，**部分式・部分木も考慮する**（"consider its sub-equations and subtrees"）．

### 正例・負例の基準——「同一」ではなく「部分木まで含めた構造の一致」

| 役割 | 基準 | 使う埋め込み |
| --- | --- | --- |
| 基本問題 p | — | 式木の根節点 |
| **正例 p⁺** | **木または部分木が p と同じ構造**をもつ問題 | p と同じ構造をもつ木または部分木の**根節点** |
| **負例 p⁻**（hard negative） | **節点数が同じで，演算子節点の種類が異なる**木をもつ問題 | 式木の根節点 |

- 正例の定義（第4.1節）："The problem $p^+$ is taken as a positive example if its tree or subtree has the same structure as p"．
- 表1の例では，式が `n_1 − n_2` の問題に対し，`(n_1 − n_2)/n_3` のように**部分木に同じ構造を含む問題も正例**になる．その場合に引き寄せるのは**一致した木または部分木の根節点**である（第4.2節）："we find the root node of the tree or subtree containing the same structure as p, and pull its embedding closer to that of p"．
- 負例（第4.2節）："The root node embeddings of the problem p and its negative problem $p^-$ are picked for model to distinguish."
- 負例の基準は，第4.1節では "the same number of nodes but different operator node types"，第5.4.1節では "the same number of operators but different operator types" と書かれている（原文の表記のとおり）．
- 正例の基準は構造が同じか否かである．編集距離などの連続的な類似度を使う記述は見当たらなかった（検索した語：edit distance）．
- 単言語設定では三つ組を各データセットの内側で作る．**多言語設定では正例と負例を別の出典から取る**——中国語の Math23k の問題なら正例を MathQA から，負例を Math23k から集める．
- 1問あたりの正例の数，作った三つ組の総数は見当たらなかった（triple の出現箇所をすべて確認）．

### 損失——コサイン類似度のマージン損失，重み α = 5

節点埋め込みを `(e, e⁺, e⁻)` と書く．

```
L_cl = Σ_z max(0, η + sim(e, e⁻) − sim(e, e⁺))        (1)   z = (p, p⁺, p⁻)，sim はコサイン類似度，η はマージン
P(y|p) = Π_{k=1}^{m} P(y_k|p)                         (2)
L_eq = Σ_p −log P(y|p)                                (3)
L    = L_eq + α · L_cl                                (4)
```

- **SupCon/InfoNCE 型ではない**．式 (1) は三つ組ごとのヒンジ型の損失で，温度もソフトマックスも現れない（検索した語：temperature，softmax，InfoNCE）．
- **η = 0.2**，**α = 5**（表6で比べた α は 1 と 5）．
- 射影 head の記述は見当たらなかった（検索した語：project，head．head は参考文献の題の multi-head にのみ現れる）．損失は木デコーダの節点埋め込みに掛かる．

### 二段階学習

"However, not all problems have positive examples, such as those problems whose solution is one value without any operator." そこで二段に分ける．

1. **Stage I**：対照三つ組で学習する．第1節によれば，ここで式の生成損失と対照損失を同時に最適化する（"first jointly optimized by an equation generation loss and a contrastive learning loss on the collected examples"）．
2. **Stage II**：元のデータセットでさらに学習する．

Stage II で使う損失は明記されていない（第5.4.2節は "at stage II, training on all examples" とだけ書く）．

## 実験設定

**データ**（表2）

| データ | 学習 | 検証 | テスト |
| --- | --- | --- | --- |
| Math23k | 21,162 | 1,000 | 1,000 |
| MathQA | 29,837 | 4,475 | 2,985 |
| MathQA†（Tan et al. 2021 に倣い Math23k の対応物として加工） | 23,703 | 3,540 | 2,410 |

- 主結果の英語側は MathQA† で報告する．
- **評価**：Acc(eq)＝生成した式が注釈の式と一致する割合，Acc(ans)＝生成した式の計算値が正解値と一致する割合．
- **最適化**：AdamW，学習率 {5e-5, 1e-4}，各段のエポック数 {30, 50} を格子探索．バッチ 16，dropout 0.5，ビーム幅 3．
- 試行回数・乱数種・分散の記載は見当たらなかった（検索した語：seed，runs，variance，standard deviation）．表の値は各条件に一つである．
- **事前分析（第3節）**：BERT-TD の [CLS] 表現を，学習エポック（最上層）と BERT の層（最良モデル）ごとに t-SNE で可視化．加えて「語彙的な意味が似ているがプロトタイプが違う」20対と「プロトタイプが同じで題材や場面が違う」20対について，問題中の全語の表現の平均どうしのコサイン類似度を層ごとに比べる．プロトタイプごとに表現を平均して中心とし，表現と中心とのコサイン距離を幅 0.1 の区間に分けて区間ごとの正解率を求める．

## 結果

### 事前分析——同じプロトタイプは上位層で寄り，中心から離れると誤りやすい

- **エポック・層**：学習が進むほど，また層が深いほど，同じプロトタイプ式の問題がまとまる（図2．図の説明によれば正解率が最も高いのはエポック 43）．
- **語彙 vs プロトタイプ**：語彙的に似た対の類似度は下位層で高く，層が深くなるにつれて下がる．同じプロトタイプの対の類似度は層が深くなるにつれて上がる（図3）——"This demonstrates that lexical semantics affects problem-solving at lower layers,"．
- **誤答との関係**：区間ごとの正解率（図4．点線は多項式の最小二乗当てはめ）について，著者は "which suggests that the representations apart from centers tend to produce wrong solutions." と述べる．
- 区間ごとの正解率の数値は図4にのみあり，本文に数値はない．なお本文には "A higher cosine distance means the representation is closer to its center." とあり，コサイン距離の通常の定義と向きが逆である（原文のまま）．

### 主結果（表3，%）

| 設定 | モデル | Math23k Acc(eq) | Math23k Acc(ans) | MathQA† Acc(eq) | MathQA† Acc(ans) |
| --- | --- | --- | --- | --- | --- |
| 単言語 | GroupAttention | – | 69.5 | 63.3* | 70.4* |
| 単言語 | GTS | – | 75.6 | 68.9* | 71.3* |
| 単言語 | Graph2Tree | – | 77.4 | 70.0* | 72.0* |
| 単言語 | BERT-TD（CL なし） | 71.2 | 82.4 | 73.5 | 75.1 |
| 単言語 | BERT-TD（CL あり） | 71.8 | **83.2** | 74.4 | **76.3** |
| 多言語 | mBERT-TD（CL なし） | 67.8 | 80.5 | 72.0 | 73.5 |
| 多言語 | mBERT-TD（CL あり） | 70.9 | **83.9** | 74.2 | **76.3** |

\* は著者による公開コードに基づく再実装．**Acc(ans) の伸びは Math23k／MathQA† の順に，単言語で +0.8／+1.2（表からの差），多言語で +3.4／+2.8 ポイント（本文に明記）**．多言語設定の伸びのほうが大きい．

### 正例・負例の集め方（表4，Acc(ans) %）

表4の設定（単言語か多言語か）は表に書かれていない．基準線と最良の行の値は，表3の多言語（mBERT-TD）の行と一致する．

| 正例 | 負例 | Math23k | MathQA† |
| --- | --- | --- | --- |
| なし（基準線） | なし | 80.5 | 73.5 |
| Same（プロトタイプ式が完全一致） | 本手法（hard） | 82.3 | 75.5 |
| 本手法（部分木一致を含む） | Rand（残りから無作為） | 82.3 | 75.8 |
| 本手法（部分木一致を含む） | 本手法（hard） | **83.9** | **76.3** |

**正例を完全一致に絞ると，本手法の正例より Math23k で 1.6，MathQA† で 0.8 ポイント低い．** 負例を無作為にした場合も本手法より低い（Math23k で 1.6，MathQA† で 0.5 ポイント）．

### ハイパーパラメタ

**マージン η（表5，多言語，Acc(ans) %）**

| η | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
| --- | --- | --- | --- | --- | --- |
| Math23k | 82.6 | 83.7 | 83.4 | **83.9** | 81.8 |
| MathQA† | 76.1 | 76.2 | 76.1 | **76.3** | 76.0 |

**重み α（表6，Math23k，単言語，%）**

| 設定 | 段 | Acc(eq) | Acc(ans) |
| --- | --- | --- | --- |
| 基準線 | – | 71.2 | 82.4 |
| α = 1 | Stage I | 70.1 | 81.5 |
| α = 1 | Stage II | 70.5 | 83.0 |
| α = 5 | Stage I | 70.6 | 82.5 |
| α = 5 | Stage II | **71.8** | **83.2** |

**Stage I の時点では Acc(eq) がどちらの α でも基準線を下回り，Acc(ans) は α = 1 で下回り α = 5 で 0.1 ポイント上回る．Stage II を経ると Acc(ans) はどちらの α でも基準線を上回るが，Acc(eq) が基準線を上回るのは α = 5 だけである（α = 1 は 70.5 で基準線 71.2 より低い）．**

### 可視化と統計（第5.4.3節．図6〜8の凡例は mBERT／mBERT-TD）

- t-SNE（図6）について，著者は対照学習によって同じプロトタイプ式の表現がよりまとまると述べる．
- Calinski-Harabasz 指数（図7）は学習・テストの両集合で対照学習ありが高い——"method supports the model to gain higher clustering scores."（値は図のみ）．
- 距離区間ごとの式の正解率（図8）："We observe the accuracy increases in most intervals, which also verifies the effectiveness of contrastive learning."
- 事例（表7）：管の問題で，対照学習なしの `1 / ((1 / 30) + (1 / 90))` が，ありでは `1 / ((1 / 30) - (1 / 90))` になる．著者は "the contrastive learning method helps the model capture the quantitative relationships exactly" と述べる．

## 著者が明記した限界

- 限界の節も今後の課題の記述も見当たらなかった（検索した語：limitation，future．future は参考文献の題にのみ現れる．本文は第6節 Conclusion で終わり，付録は無い）．
- 本文から読める制約は一つ：演算子を含まない（解が一つの値だけの）問題には正例がない——"However, not all problems have positive examples, such as those problems whose solution is one value without any operator." これが二段階学習を採る理由とされる．

## 射程の外

- **対象は算数文章題の式生成**（Math23k・MathQA†）である．画像を伴う質問応答や VLM は扱わない（検索した語：image，VQA，GQA，vision，visual．vision は supervision と参考文献の会議名に，visual は t-SNE の visualization にのみ現れる）．
- **ラベルは出力（解の式）そのものの構造**であり，モデルは同じ式を生成するよう学習される．答えとは別の注釈（本研究の署名，研究文書 §3.1）で表現を揃え，それが答えの生成に使われるかを問う設定（研究文書 §1.2・§5.3）ではない．
- **損失は木デコーダの節点（木または部分木の根）の埋め込みに掛ける三つ組マージン損失**である．言語モデルの中間層の表現へ SupCon を掛ける設計（研究文書 §3.2・§3.3）や温度は現れない．
- 事前分析で層ごとに見るのは**表現の t-SNE とコサイン類似度**である．線形プローブや，表現への介入で予測への寄与を測る実験は見当たらなかった（検索した語：linear，classifier，causal．原文は第1節で "We probe the problem representations in BERT" と書くが，行っているのは可視化と類似度・距離の計算である）．
- 正例の基準は**木構造の一致（部分木を含む）**であり，演算子の系列の一致（本研究の署名，研究文書 §3.1）ではない．
- 比べた α は 1 と 5 のみで，分散の報告は見当たらなかった．

---

**確認**：2026-11-13．arXiv abs ページ（v1 2021-10-16，v2 2022-03-10，Comments "Accepted to ACL 2022"），arXiv PDF v2（11頁，付録なし），ACL Anthology の書誌ページ・BibTeX・PDF（11頁．本文は arXiv v2 と同じで，違いはページ番号とフッタのみ）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．表2〜7と図3・4・6〜8は PDF のページ画像でも確かめた．arXiv PDF v1（10頁）は表3〜6の数値が v2 と同じであることだけを確かめた（v1 の本文は v2 と文面が異なり，本要約の引用の一部は v1 に無い）．下書き（2026-10-11 作成）から直した主な点：冒頭の一文と主張を原文の強さに直した（暗記と浅い手がかりは近年の報告として引かれたもの，「寄り切らない問題ほど誤答」は「中心から離れた表現は誤答しやすい傾向」）．表6の読みを直した（Stage II 後も α = 1 の Acc(eq) は基準線より低い）．プロトタイプ式の「定義」は原文に定義文が無いので例示に改めた．部分木の正例の例を原文の表1の式に差し替えた．否定の記述を検索語つきの「見当たらなかった」に揃えた．本研究についての記述を研究文書のいまの節番号と語に合わせた．著者ごとの所属を PDF の所属番号から書いた．**読めていないもの**：図1・2・5の中身は細部まで照合していない（本文と図の説明に書かれたことだけを載せた）．arXiv v1 の本文全体，著者が公開したコード，引用先の論文（Xie and Sun 2019，Tan et al. 2021 など）は読んでいない．
