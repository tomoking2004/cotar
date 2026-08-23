# Designing and Interpreting Probes with Control Tasks

**著者**：John Hewitt，Percy Liang（スタンフォード大学）
**掲載**：EMNLP-IJCNLP 2019，pp. 2733–2743，香港
**原文**：[ACL Anthology](https://aclanthology.org/D19-1275/)

**プローブが高精度で当てられたとき，表現が構造を持つのか，プローブが課題を覚えただけなのかは，精度だけでは分けられない．**

## 主張

probing は「表現が言語構造を符号化している」根拠として使われるが，"does this mean that the representations encode linguistic structure or just that the probe has learned the linguistic task?" という問いが残る．

著者らは **control task**——語の型に無作為な出力を対応づけた課題——を，本来の言語課題と並べて置くことを提案する．control task は構成上，プローブ自身にしか学習できない（表現の側に手がかりが無い）．したがって良いプローブとは**選択的（selective）**なもの，すなわち**言語課題では高精度，control task では低精度**を示すものである．

**selectivity ＝ 言語課題の精度 − control task の精度**．これが，プローブが語の型から丸暗記する能力に照らして，言語課題の精度を読む枠を与える．

## 手法

英語の**品詞タグ付け**と**依存関係の辺の予測**について control task を構成し，ELMo の表現の上で一般的なプローブを評価する．

## 結果

- **ELMo に対する一般的なプローブは選択的ではない．**
- **dropout はプローブの複雑さを抑える手段として広く使われているが，MLP の selectivity を上げるには効かない**．他の正則化は効く．
- **ELMo の第1層は品詞タグ付けの精度がやや高いが，第2層のほうが selectivity は大幅に高い**——どちらの層が品詞をよく表現しているのか，という問いが立つ．

## 著者の推奨

言語課題と並べて control task を置き，プローブが表現の構造を捉えているのか，語の型に紐づく見かけの規則を覚えているのかを確かめる．

## 射程の外

- **静的・文脈化埋め込み（ELMo）**の時代の研究であり，VLM も生成型 LLM も扱わない．
- **学習による介入を行わない．** 既にある表現を測る側の話で，損失で構造を作り込む場合は扱わない．
- control task は**ラベルを無作為化する**が，無作為化するのは**語の型に対する出力の割り当て**である．バッチ内でラベルを置換する形とは構成が異なる．
- **selectivity は「読めるか」を正す道具であって，「使われているか」は測らない．** 忠実性の問いには答えない．

---

**確認**：2026-08-22．ACL Anthology の書誌ページを参照し，要旨は逐語で照合した．**本文 PDF は精読していない**ため，本文の細部（control task の構成手続き・実験設定の詳細）は要旨と Anthology の記載に依る．
