# Probing for the Usage of Grammatical Number

**著者**：Karim Lasri（Lattice〔École Normale Supérieure-PSL，CNRS，U. Sorbonne Nouvelle〕／University of Pisa），Tiago Pimentel（University of Cambridge），Alessandro Lenci（University of Pisa），Thierry Poibeau（Lattice），Ryan Cotterell（ETH Zürich）
**掲載**：ACL 2022（Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)，pp. 8818–8831，2022年5月，Dublin）．arXiv:2204.08831 v4，2024-05-22
**原文**：[ACL Anthology](https://aclanthology.org/2022.acl-long.603/)・[arXiv](https://arxiv.org/abs/2204.08831)

**プローブで抽出できる符号化と，モデルが予測に実際に使う符号化（functional encoding）を分ける——BERT は数の一致を解くのに数の線形な符号化に頼り，その符号化は名詞と動詞で別である．**

## 位置づけ

Amnesic probing（Elazar ら 2021）の除去介入を，**文法上の数という一つの性質と，主語–動詞の数の一致という一つの振る舞いの課題**に絞り，位置（手がかりの名詞／対象の動詞）と層ごとに当てた研究である．著者は probing を diagnostic・behavioral・causal の三つの枠組みに分け（§2），その組み合わせとして自分たちの立場を置く（§3）．

## 主張

- 診断プローブは**偽の（spurious）符号化**を拾いうる——"a spurious encoding of a linguistic property, i.e., an encoding that allows us to extract our target property from the representation, but which the model being probed may not actually use to make a prediction"（§1）．これに対し "we instead seek to find encodings that are actually used by a pre-trained model, which we term functional encodings."（§1）
- **usage-based probing**（§1，§3）：(1) 性質と，その性質の情報なしには解けない振る舞いの課題を選び，(2) その性質の符号化を因果的介入で除き，(3) 課題の性能が落ちるかを見る．抽出できることは，性質がモデルに使われるための必要条件だが十分条件ではない（§5.1）．
- BERT について："BERT relies on a linear encoding of grammatical number to produce the correct behavioral output"（要旨）．さらに，名詞と動詞で別々の符号化を使うこと，数の情報が名詞からその主辞の動詞へ渡る層を特定することを挙げる（要旨）．

## 手法

### 介入は除去（amnesic probing／INLP）

- 介入は **amnesic probing**（Elazar ら 2021）で，中身は Ravfogel ら（2020）の iterative null-space projection である（§5.2，脚注5）．**行ったのは除去と注意の切断であり，表現を反実仮想の値へ書き換える介入を行ったという記述は見当たらなかった**（検索した語：counterfactual，AlterRep．counterfactual は参考文献の題にしか現れない）．Ravfogel ら 2021（Counterfactual interventions …）は，脚注1で「似たパイプライン」として，§2 で causal probing の例として引かれるのみ．
- 768 次元の表現に線形プローブ（sigmoid）を当て，そのパラメタベクトルの零空間への射影を，性質が抽出できなくなるまで繰り返す．合成した射影を表現に掛け，(i) 後続の層への影響と (ii) 数の一致の課題の性能への影響を測る（§5.2）．
- 射影は**カテゴリごとに別々に学習**する：名詞，動詞，マスクした動詞，および名詞と動詞を合わせた mixed の四つ（§6）．
- **対照**：各層で同じ本数の**ランダムな方向**を除く（脚注10，付録D．Elazar らが提案した対照）．

### 注意の切断（§7.5，付録C）

(i) 対象（マスクした動詞）から手がかり（主語の名詞）への注意だけを切る，(ii) すべてのトークンから手がかりへの注意を切る．いずれも，ある層の全ヘッドに対して，層の範囲（第 i 層から第 j 層まで）で行う．

### 課題

**数の一致（NA）**（§4.1）：動詞をマスクした文脈で，正しい屈折形に誤った屈折形より大きい確率を割り当てれば成功とする．結果は NA の性能（accuracy）の低下として報告される．

## 実験設定

- **モデル**：bert-base-uncased（transformers ライブラリの実装．脚注6）．12層，各層12ヘッド（脚注14）．
- **データ**：Linzen ら（2016）の数の一致データ（Wikipedia から取った文．手がかりと対象の位置，それぞれの数のラベルつき）．文数の記載は見当たらなかった．著者はこのデータが数の一致の課題を代表すると仮定する（"this may not be true in general, however"，§6）．
- 介入前の NA 正解率は，数値としての記載が見当たらなかった．BERT がこのデータでよく解けることは Goldberg（2019）を引いて述べる（§6）．

## 結果

### 層ごとの介入（表1．付録に置かれている）

| 層 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 名詞：除いた方向の数 | 17 | 51 | 33 | 70 | 22 | 37 | 48 | 52 | 64 | 39 | 22 | 39 | 26 |
| 名詞：NA 低下 | 0.32 | 0.32 | 0.27 | 0.29 | 0.28 | 0.29 | 0.22 | 0.09 | 0.04 | 0.0 | 0.0 | 0.0 | 0.0 |
| 名詞：NA 低下（ランダム） | 0.06 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| マスク動詞：除いた方向の数 | 1 | 13 | 15 | 26 | 30 | 17 | 21 | 44 | 24 | 22 | 22 | 26 | 33 |
| マスク動詞：NA 低下 | 0.04 | 0.01 | 0.01 | 0.01 | 0.01 | 0.0 | 0.01 | 0.0 | 0.0 | 0.09 | 0.29 | 0.33 | 0.23 |
| マスク動詞：NA 低下（ランダム） | 0.03 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 |

- 同じ表の "Loss in Layers"（題では "the information loss resulting from amnesic interventions in each layer"）は，名詞で 0.37–0.49，マスク動詞で第0層が 0.0，第1層以降が 0.3–0.42 である．図3の題は情報の損失を，対象の位置で診断プローブが測った精度の低下（"the probing accuracy drop compared to when no intervention was performed"）と定めるが，表1の値の単位と，どの位置・層で測った値かは，表の題に書かれていない．
- **振る舞いへの影響は層で分かれる**："target interventions are only harmful after the 9th layer; while noun interventions only hurt up to the 8th layer"（§7.4）．表1でも，マスク動詞の第1–8層は Loss in Layers が 0.3–0.42 であるのに NA の低下は 0.0–0.01 である．著者はこれを "the cue passes its number information in the first 8 layers, and that the target stops acquiring number information in the last three layers" を示唆するものと読む（§7.4）．
- 著者は，情報の損失と課題の性能の低下が層をまたいで揃うことを，モデルが除かれた線形の情報を実際に使っていることの確認とする（§7.2）．
- 名詞から動詞への受け渡しを，序論は "this transfer occurs between BERT’s 3rd and 8th layers" とまとめる．一方 §7.4 の結論は「第8層より前の層で渡る」であり，"we concede that our analysis alone makes it difficult to pinpoint exactly which layers" と添える．

### 名詞と動詞の符号化は別物（§7.1，§7.3）

- プローブのパラメタベクトルの cosine 類似度（§7.1，図1．本文の定性的な記述）：名詞と動詞は最初の6層でむしろ逆向きに見え，それより後の層はおおむね直交する．マスクした動詞と動詞は，最初の数層では符号化を共有しないように見えるが，第6層以降は強く揃う．
- 介入で確かめた結果（§7.3）："using amnesic probes trained on nouns has no impact when applied to masked verbs"．動詞で学習したものを名詞に当てても影響がない．mixed で学習した射影は手がかり（名詞）には効くが，対象にはほとんど効かない．

### 注意の切断（§7.5，図4）

| 切る注意 | 効果 |
| --- | --- |
| 対象→手がかりのみ | 第6–10層を切れば，直接の注意についてほぼ最大の効果が出る．ただし低下は "relatively modest"（"≈ 10%"） |
| 全トークン→手がかり | "up to ≈ 40%"，第2–8層で最大 |

一つの層だけで注意を切っても，効果はほとんどない．著者はこれらから，対象は間のトークンからも情報を得られ，数は直接の注意だけで渡されるのではない，と述べる（§7.5）．

## 著者が明記した限界

独立した限界の節はない（Limitations・Discussion という見出しは見当たらなかった．あるのは倫理声明）．本文・脚注で次を認める．

- **層の特定は粗い**："we concede that our analysis alone makes it difficult to pinpoint exactly which layers"（§7.4）．
- **除去しすぎの可能性**："A potential criticism of amnesic probing is that it may remove more information than necessary."（脚注10）．著者は，カテゴリを入れ替えて当てた射影がほとんど影響しないことと，ランダムな方向の対照を，これへの答えとして挙げる．
- **データの代表性**：Linzen らのデータが数の一致の課題を代表するとの仮定は "may not be true in general"（§6）．

## 射程の外

- **言語モデル単体（BERT）**．VLM・画像入力・質問応答を扱う記述は見当たらなかった（検索した語：vision，visual，image，question answering，GQA）．
- **学習による介入を行わない．** 学習済みの BERT の表現から**線形に**抽出できる情報を除く解析であり，損失で性質を足す操作や微調整の記述は見当たらなかった（検索した語：fine-tun，finetun）．
- 対象の性質は文法上の数という形態統語的な性質（英語では単数／複数の二値）で，振る舞いはマスク位置での二つの屈折形の確率の比較である．
- 「使われ方」の測定は，除去後の NA の性能の低下と注意の切断である．ヤコビアンや勾配による感度の記述は見当たらなかった（検索した語：Jacobian，gradient）．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴・Comments），arXiv PDF v4（15頁，付録 A–E を含む全文），ACL Anthology の書誌ページと bib，ACL Anthology PDF の現行版（v2，14頁）と初版（v1，14頁）を取得し，v4 の全文を pdftotext のテキストで読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．逐語引用と表1は，arXiv v4 と Anthology の v1・v2 の三つで同じ文字列・同じ値であることを確かめ，表1・図3の題・脚注10・第1頁の所属は v4 のページ画像でも見た．下書き（2026-10-11 作成）から直した主な点：(1) cosine 類似度（図1）の記述の節を §7.3 から §7.1 へ，(2)「情報の損失の単位は明記なし」を，図3の題が精度の低下と定めていること・表1の題には無いことへ，(3)「全層でほぼ一様に生じている」「その位置・層の符号化は振る舞いに使われていない」という要約者の読みを外し，表の値と著者自身の読みへ，(4) Ravfogel ら 2021 の引かれる場所を「脚注と関連研究」から「脚注1と §2」へ，(5)「それらの混合」を「名詞と動詞を合わせた mixed」へ，(6) 否定の主張を，検索した語を添えた「見当たらなかった」の強さへ，(7) 著者と所属の対応を一人ずつに，(8) 表の値の太字（原文に無い強調）を外した．**読めていないもの**：arXiv の v1–v3（版の間の差は見ていない）．図1–6 の中の値は読み取っておらず，図については本文と題に書かれたことだけを載せた．
