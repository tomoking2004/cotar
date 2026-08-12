# The Hydra Effect: Emergent Self-repair in Language Model Computations

**著者**：Thomas McGrath，Matthew Rahtz，János Kramár，Vladimir Mikulik，Shane Legg（Google DeepMind）
**掲載**：arXiv:2307.15771，2023-07-28
**原文**：[arXiv](https://arxiv.org/abs/2307.15771)

**ある層を潰すと別の層が肩代わりする．ゆえに「潰しても出力が変わらない」ことは「その層が効いていない」ことを意味しない．**

## 主張

> 要旨（原文）："We investigate the internal structure of language model computations using causal analysis and demonstrate two motifs: (1) a form of adaptive computation where ablations of one attention layer of a language model cause another layer to compensate (which we term the Hydra effect) and (2) a counterbalancing function of late MLP layers that act to downregulate the maximum-likelihood token. Our ablation studies demonstrate that language model layers are typically relatively loosely coupled (ablations to one layer only affect a small number of downstream layers). Surprisingly, these effects occur even in language models trained without any form of dropout. We analyse these effects in the context of factual recall and consider their implications for circuit-level attribution in language models."

著者は名前についてこう断る——"we recognise that the use of the name 'Hydra' is not completely mythologically accurate: sometimes only one head grows in importance, and as we show in Section 4 the total effect decreases on average"．

## 手法

**除去は resample ablation（ゼロ除去ではない）**．データセット中の**15本の別プロンプト**から取った活性で差し替える．理由は "probably the most principled, as every ablation is a naturally-occurring set of activations"——ゼロ除去と違い，どの除去も実際に生じうる活性の組になる．

**効果は相補的な二つの量で測る．**

- **直接効果（unembedding 基準）** Δ_unembed,l：unembedding を通した中心化 logit．因果推論でいう直接効果に対応する．**RMSNorm の正規化係数 σ は順伝播で得た値に固定する**——固定すると logit は層出力について線形になる．
- **総効果（除去基準）** Δ_ablate,l：層出力を差し替えたときの予測確率の変化．文脈におけるその層の総効果．

**この二つは食い違う**——"showing low correlation and Δunembed>Δablate for most prompts and layers, contrary to expectations"．食い違いそのものが自己修復の指紋である．

## 実験設定

- **モデル**：Chinchilla 系の **70億パラメタ**モデル（32層）．"was trained entirely without dropout, stochastic depth, or layer dropout"．
- **データ**：Counterfact——三つ組から作った文脈 **1,209 件**．主語 s と関係 r の連結のみをプロンプトとして用いる．

## 結果

- **第23層（全32層）で，介入前後の変化の分散の 92% が補償で説明される．**
- **中間層では，Hydra 効果と MLP 効果の減少が合わさって，トークン logit の低下のおよそ 70% を回復する．**
- **補償は不完全**——"fitting a linear regression between direct effect and compensatory response gives a slope of less than one at all layers past layer 13"．

**層による違い**

| 層域 | 挙動 |
| --- | --- |
| 前段 | "early layer ablations have large total effects but almost no direct effect" |
| 中段〜後段 | 直接効果と補償応答が強く相関．**第23層が補償のピーク** |
| 最終盤 | "very late layers only have non-negligible direct effect (which makes sense as there are few downstream layers)" |

**後段 MLP の消去機能**："when the attention layer has a high positive impact they have a high negative impact and when the attention layer's Δunembed is reduced theirs is similarly attenuated"．第22層では MLP 側の応答の方が予測力が高くなり，**第23層ではほぼすべての応答が消去 MLP に生じる**．

## 著者が述べる含意

- **総効果で部品を並べる解析は危うい**："If we prioritise network components for ablation according to their total effect, we will be using a measure that does not fully reflect the computational structure of the intact network."
- **unembedding 基準の解析も単純には読めない**：出力が消去 MLP に部分的に打ち消されるなら "it's no longer straightforward to interpret that output in terms of its direct effects on logits"．
- **帰属先が曖昧になる**："Is the responsible component the attention layer that has the effect in the intact network, or the circuits that act to compensate following ablation?"
- **消去 MLP は入力語彙で意味づけられないかもしれない**："Erasure MLPs may have no clear semantics in terms of the model's input, as they are responding to the language model's internal computations."
- 除去に対する頑健性を報告した過去の研究は読み替えを要する——"it is not enough simply to measure the total effect of an ablation without investigating downstream changes in the network"．

## 著者が明記した限界

- **単一モデル・単一タスク**：Chinchilla 7B の Counterfact による事実想起に限られる．
- **粒度は層単位**："Although we have identified two new motifs, we have not investigated more deeply than individual layers"．
- **主張しないこと**："One thing we are not claiming is that neural networks are naturally forming causal models of their training data, or that networks learn to perform causal reasoning."
- **機序は推測に留まる**：dropout 無しでも起こる理由として "If gradient descent were to occasionally break network components then a kind of 'natural dropout' would occur during training. In this case it would be beneficial for networks to be robust to layers failing." を挙げるが，直後に "We emphasise that this is conjecture, however, and would need further research" と断る．

## 射程の外

- **VLM は扱わない．** 対象はテキストの言語モデル1本．
- **学習時の介入は扱わない．** 本論文の操作はすべて**学習済みモデルへの推論時の除去**である．

---

**確認**：2026-08-09．arXiv abs ページと ar5iv による HTML 全文を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．
