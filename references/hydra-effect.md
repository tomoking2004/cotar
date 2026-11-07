# The Hydra Effect: Emergent Self-repair in Language Model Computations

**著者**：Thomas McGrath，Matthew Rahtz，János Kramár，Vladimir Mikulik，Shane Legg（Google DeepMind）
**掲載**：arXiv:2307.15771，2023-07-28
**原文**：[arXiv](https://arxiv.org/abs/2307.15771)

**ある注意層を除くと別の層が補う．ゆえに「除いても出力があまり変わらない」ことは「その部品が重要でない」ことを意味しない．**

## 主張

> 要旨（原文）："We investigate the internal structure of language model computations using causal analysis and demonstrate two motifs: (1) a form of adaptive computation where ablations of one attention layer of a language model cause another layer to compensate (which we term the Hydra effect) and (2) a counterbalancing function of late MLP layers that act to downregulate the maximum-likelihood token. Our ablation studies demonstrate that language model layers are typically relatively loosely coupled (ablations to one layer only affect a small number of downstream layers). Surprisingly, these effects occur even in language models trained without any form of dropout. We analyse these effects in the context of factual recall and consider their implications for circuit-level attribution in language models."

著者は名前についてこう断る——"we recognise that the use of the name 'Hydra' is not completely mythologically accurate: sometimes only one head grows in importance, and as we show in Section 4 the total effect decreases on average"．

## 手法

**除去は resample ablation（ゼロ除去ではない）**．最後のトークン位置で，注意層または MLP の層出力を，データセット中の**15本の別プロンプト**から取った活性で差し替える．理由は "probably the most principled, as every ablation is a naturally-occurring set of activations"——ゼロ除去と違い，どの除去も実際に生じうる活性の組になる．

**効果は相補的な二つの量で測る．**

- **直接効果（unembedding 基準）** Δ_unembed,l：unembedding を通した中心化 logit．因果推論でいう直接効果に対応する．**RMSNorm の正規化係数 σ は順伝播で得た値に固定する**——固定すると logit は層出力について線形になる．
- **総効果（除去基準）** Δ_ablate,l：層出力を差し替えたときの，最尤トークンの中心化 logit の変化．文脈におけるその層の総効果．

**この二つは食い違う**——"showing low correlation and Δunembed>Δablate for most prompts and layers, contrary to expectations"．著者はこの食い違いを，除去の効果を打ち消す下流の層の変化に帰す（§2.5）．

## 実験設定

- **モデル**：Chinchilla 系の **70億パラメタ**モデル（32層）．"was trained entirely without dropout, stochastic depth, or layer dropout"．
- **データ**：Counterfact——主語・関係・目的語の組から作った文脈 **1,209 件**．主語 s と関係 r の連結のみをプロンプトとして用いる．

## 結果

- **第23層（全32層）で，介入前後の変化の分散の 92% が補償で説明される．**
- **中間層では，Hydra 効果と MLP 効果の減少が合わさって，トークン logit の低下のおよそ 70% を回復する．**
- **補償は不完全**——"fitting a linear regression between direct effect and compensatory response gives a slope of less than one at all layers past layer 13"．

**層による違い**

| 層域 | 挙動 |
| --- | --- |
| 前段 | "early layer ablations have large total effects but almost no direct effect" |
| 中段〜後段 | 直接効果と補償応答が強く相関．**第23層が補償のピーク** |
| 最終盤 | "very late layers only have non-negligible direct effect"（下流の層が少ないので当然だ，と著者は添える） |

**後段 MLP の消去機能**："when the attention layer has a high positive impact they have a high negative impact and when the attention layer's Δunembed is reduced theirs is similarly attenuated"．第22層では MLP 側の応答の方が予測力が高くなり，**第23層ではほぼすべての応答が消去 MLP に生じる**．

## 著者が述べる含意

- **除去の効果が小さいことを，そのまま重要でないと読めない**：関連研究の節で "our findings caution against straightforwardly interpreting low effectiveness of ablations as meaning that a network component is unimportant" と述べる．
- **総効果で部品を並べる解析の自動化に課題が生じる**："if we prioritise network components for ablation according to their total effect, we will be using a measure that does not fully reflect the computational structure of the intact network"．ただし補償は通例 100% 未満なので，自動化にも手がかりは残ると添える．
- **unembedding 基準の解析も単純には読めない**：出力が消去 MLP に部分的に打ち消されるなら "it's no longer straightforward to interpret that output in terms of its direct effects on logits"．
- **帰属先が曖昧になる**："is the responsible component the attention layer that has the effect in the intact network, or the circuits that act to compensate following ablation?"
- **消去 MLP は入力語彙で意味づけられないかもしれない**："erasure MLPs may have no clear semantics in terms of the model's input, as they are responding to the language model's internal computations"．
- 過去の除去の研究（Morcos ら）に文脈を与える——"it is not enough simply to measure the total effect of an ablation without investigating downstream changes in the network"．

## 著者が明記した限界

- **粒度は層単位**："Although we have identified two new motifs, we have not investigated more deeply than individual layers"．
- **主張しないこと**："One thing we are not claiming is that neural networks are naturally forming causal models of their training data, or that networks learn to perform causal reasoning"．
- **機序は推測に留まる**：推論時には起きない除去への頑健さが何の得になるのかを問い，一つの可能性として "If gradient descent were to occasionally break network components then a kind of 'natural dropout' would occur during training. In this case it would be beneficial for networks to be robust to layers failing." を挙げるが，直後に "We emphasise that this is conjecture, however, and would need further research" と断る．原因が何かは今後の問いに挙げている（§6）．

## 射程の外

- **VLM は扱わない．** 対象はテキストの言語モデル1本．
- **測ったのは1モデル・1タスク**（Chinchilla 7B，Counterfact の事実想起）である．著者はこれを限界としては挙げず，GPT-2 Small の backup heads（Wang ら）と併せて，タスクとモデルをまたいで起きる可能性を述べる．訓練分布全体でどれだけ起きるかは今後の問いである．
- **学習時の介入は扱わない．** 本論文の操作はすべて**学習済みモデルへの推論時の除去**である．

## 学会で聞かれたら

本研究についての答えは研究文書（`.claude/context.md`）の範囲で書いた．

- **問：Hydra 効果とは何ですか．**
  答：ある注意層を除くと，後ろの別の層が効果を強めて補う現象です．7B の言語モデルの事実想起で示されました．後段の MLP が最尤トークンを押し下げる働きも併せて報告しています．（出所：原典 要旨）
- **問：どのくらい補うのですか．**
  答：中間の層では，Hydra 効果と MLP の働きの減少を合わせて，logit の低下のおよそ 70% を回復します．完全ではなく，第13層より後では回帰の傾きが 1 未満です．（出所：原典 §4，§6）
- **問：第16層を壊して，答えが変わるかを見ないのはなぜですか．**
  答：自己修復があるので，壊しても出力が変わらないことは，その層が寄与していないことを意味しないからです．そこで，署名が表現のどこに載っているかと，答えがその位置にどれだけ頼っているかを測りました．（出所：研究文書 §A.4，原典 §5）
- **問：第16層に頼る度合いが下がったのは，別の層が肩代わりしたからではないですか．**
  答：それは本研究では確かめていません．相対感度は第16層でしか測っていないので，他の層が代わりに担っているかは分かりません．（出所：研究文書 §A.5，§7）
- **問：Hydra 効果は，学習で依存が変わる場面にも当てはまりますか．**
  答：原典には書かれていません．原典が見たのは学習済みモデルへの推論時の除去で，本研究のような学習中の変化は扱っていません．（出所：原典 §1，§6）
- **問：相対感度は自己修復の影響を受けないのですか．**
  答：相対感度は表現を除かずに第16層のまわりで線形化した量で，層出力を差し替える原典の操作とは別です．ただし，自己修復の影響をどれだけ受けるかは，本研究では確かめていません．（出所：研究文書 §A.4）
- **問：では「使っていない」と言い切れますか．**
  答：言い切れません．相対感度は一次の量で，非線形の経路までは否定しません．下がったのも2割前後でゼロではないので，言えるのは「頼る度合いが下がった」までです．（出所：研究文書 §5.3）
- **問：本研究のモデルでも自己修復は起きますか．**
  答：本研究では確かめていません．原典の対象はテキストの言語モデル1本で，VLM は扱っていません．（出所：原典 §2.1）

---

**確認**：2026-08-09．arXiv abs ページと ar5iv による HTML 全文を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．

**再照合**：2026-11-06．arXiv の PDF v1（21頁，付録Aまでの全文）を pdftotext でテキスト化して読み，数式は ar5iv の HTML で補って，本文書の数値・引用・節番号を一件ずつ照合した．直した主な点：総効果の単位を予測確率から中心化 logit へ，「三つ組」を主語・関係・目的語の組へ直した．著者が限界として書いていない「単一モデル・単一タスク」を射程の外へ移し，dropout の説明を原文の問いに合わせ，引用の境界と大文字小文字を原文に揃えた．図の中の値は本文に書かれたものだけを照合した．
