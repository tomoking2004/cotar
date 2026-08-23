# Probing Classifiers: Promises, Shortcomings, and Advances

**著者**：Yonatan Belinkov（テクニオン）
**掲載**：Computational Linguistics 48(1)，pp. 207–219，MIT Press，2022-03（squib）．arXiv:2102.12452（2021）
**原文**：[ACL Anthology](https://aclanthology.org/2022.cl-1.7/)・[arXiv](https://arxiv.org/abs/2102.12452)

**probing を批判的に総覧した論考——この枠組みの約束と，限界と，その後の前進．**

## 位置づけ

**本研究がこの文献を引くのは，個別の主張のためではなく，probing という枠組みそのものへの批判が一本の総説にまとまっている場所として**である．新しい実験を提案する論文ではなく，Computational Linguistics 誌の squib（短い論考）である．

## 主張

probing classifier の考え方は単純である——"a classifier is trained to predict some linguistic property from a model's representations"——，そして極めて広く使われてきた．しかし著者は "recent studies have demonstrated various methodological limitations of this approach" として，この枠組みを批判的に総覧し，約束（promises）・欠点（shortcomings）・前進（advances）を整理する．

## 射程の外

- **言語モデルの解析についての総説**であり，VLM は扱わない．
- 新しい手法・実験・数値を提出しない．**個別の主張の根拠として引くべき文献ではない．**
- **学習による介入**（損失で表現を作り込む設定）は扱わない．

---

**確認**：2026-08-22．ACL Anthology の書誌ページ（巻・号・頁・刊行月）と arXiv abs ページを参照し，要旨は逐語で照合した．本文は精読していない——本ノートは書誌と要旨の範囲にとどまり，本文が catalogue する個別の欠点・前進の一覧はここに写していない．引用に際して個別の論点が要るときは，本文に当たること．
