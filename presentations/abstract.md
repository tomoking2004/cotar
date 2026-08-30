# PCSJ/IMPS 2026 講演申込——題目とアブストラクト（教授チェック用）

2026-08-31 時点．研究の全体は [.claude/context.md](../.claude/context.md) にある．

## 学会と締切

- **PCSJ/IMPS 2026**（<https://www.pcsj-imps.org/>）．2026-11-16（月）〜18（水），御殿場高原ホテル．一般講演はすべてオンサイトのポスター発表（横 180cm × 縦 90cm）．
- **講演申込：2026-09-04（金）**．原稿（A4 2ページ）：2026-10-16（金）．参加申込：2026-11-09（月）．
- 申込は電子情報通信学会の研究会システム（<https://ken.ieice.org/ken/program/index.php?tgid=IEICE-PCSJ-IMPS>）から．**申込フォームのアブストラクト欄の字数制限は未確認**（サイトが外部からの取得を拒む）——長短2版を用意したので，フォームの上限に合わせて選ぶ．

**登録は，題目とアブストラクトを教授に見せて確認を受けてから，本人が行う．** 勝手に登録しない（教授の指示）．申込フォームの他の項目（下の分野・キーワード）も本人が決めて入力する．

## 申込フォームに入れる内容（案）

| 項目 | 案 |
| --- | --- |
| 題目（和文） | 視覚言語モデルにおけるタスク類似性に基づく内部表現整合学習に関する検討 |
| 題目（英文） | A Study of Internal Representation Alignment Based on Task Similarity in Vision-Language Models |
| 著者 | 中野 友晴（東京都立大学）．共著者（指導教員）の記載は教授に確認する |
| 発表分野 | IMPS「画像認識/解析」（サイト掲載の分野のうち最も近い．「映像処理応用」も候補） |
| キーワード（和） | 視覚言語モデル，内部表現，対照学習，線形プローブ，視覚的質問応答 |
| キーワード（英） | vision-language model, internal representation, contrastive learning, linear probe, visual question answering |
| アブストラクト | 下の和文（長版）．字数が収まらなければ短版 |

## 書き方の方針

**数値を書かない．** 計り直し（整合の強さ・層の追加測定）や方針の追加（表現の頑健性の測定）があっても取り消しにならない主張——三つの主結果の向きと対照の構成——だけを書く．具体的な数値は10月の2ページ原稿とポスターに置き，そこで測定時点を添える．

主結果は三つ——(1) 整合した層で署名が線形にずっと読み取りやすくなる，(2) 正解率はほとんど変わらない，(3) それでも答えの生成はその構造の利用を伴わない（整合した層への依存が下がる；複数の乱数種で一貫，ラベル並べ替えでは起きない）．(3) の「読める ≠ 使われる」は本研究のもう一つの主結果であって，付随的な限定ではない．

**補助的な検証（研究文書の付録A）には触れない．** 質問の形式の残差化・演算子の組成・頻度層別・層を変えた試走は，発表で述べず，質問されたときに答える材料として持つ．

---

## 和文（長版）

視覚言語モデル（VLM）の頑健性は通常，入力を揺らしても出力が変わらないことで測られる．しかし近年，出力が同じでも内部表現は大きく動きうることが示され，内部表現そのものの性質は別途問う必要がある．本研究はこれに隣接する問いとして，同じ解き方で解ける質問どうしの内部表現が互いに近いか，そして明示的に近づけたとき何が起きるかを扱う．GQA の各質問に付く functional program の演算子列を「タスク署名」と呼び，署名が一致する質問の中間層表現を近づける補助損失を，通常の学習に加える．整合の有無とラベルの意味だけを変えた三つの条件を複数の乱数種で学習し，表現の側と出力の側の両方で評価した．その結果，整合を課すと，整合した層でタスク署名が線形に顕著に読み取りやすくなる一方，質問応答の正解率はほとんど変わらないことを確認した．この読み取りやすさの向上は，質問文の言い回しでも，ラベルの無作為な並べ替えでも説明されない．しかし，読み取りやすくなった構造は答えの生成での利用を伴わない——整合した条件では答えが整合した層に頼る度合いがむしろ下がり（複数の乱数種で一貫し，ラベルを並べ替えた条件では起きない），読める構造は答えを決める方向にも集中しない．表現から性質が線形に読み取れることと，モデルがそれを答えの生成に使うことは別である——この区別は従来，既にある性質を表現から除く介入で論じられてきたが，本研究は性質を損失で足す方向からも両者が一致しないことを構成的に示す．整合が入力の揺らぎに対する表現の頑健性を高めるかは，今後の課題である．

## 和文（短版）

視覚言語モデルの内部表現は，出力が同じでも入力の揺らぎで大きく動きうることが報告されている．本研究は，同じ解き方で解ける質問どうしの内部表現が近いかを問い，GQA の functional program の演算子列（タスク署名）が一致する質問の中間層表現を近づける補助損失を通常の学習に加えた．その結果，署名は線形に顕著に読み取りやすくなる一方で正解率はほとんど変わらず，しかも読み取りやすくなった構造は答えの生成での利用を伴わない——「読み取れること」と「使われること」は一致しない——ことを，ラベルの意味だけを変えた対照条件との比較で示す．

## 英訳（必要なら）

The robustness of vision-language models (VLMs) is usually measured by whether the output stays fixed under input perturbations. Yet recent work shows that internal representations can shift substantially even when the output does not, so the representations themselves must be examined separately. As an adjacent question, we ask whether questions solvable by the same procedure have internal representations that are close to one another, and what happens when they are explicitly pulled together. We take the operator sequence of each GQA question's functional program as its "task signature," and add an auxiliary loss that pulls together the mid-layer representations of questions sharing a signature. Training three conditions that differ only in whether alignment is applied and in what the labels mean, across several random seeds, we evaluate on both the representation side and the output side. We find that alignment makes the task signature markedly more linearly decodable at the aligned layer, while answer accuracy remains almost unchanged. This gain in decodability is not explained by question wording or by randomly shuffling the labels. However, the added decodability is not accompanied by use in answer generation: under alignment the model's answer relies less on the aligned site (consistently across seeds, and not under the shuffled-label control), and the decodable structure does not concentrate on the answer-deciding directions. Decodability and use are distinct: this distinction has so far been argued by removing existing properties from representations, and we show constructively that the two also fail to coincide when a property is added by a loss. Whether alignment improves representational robustness to input perturbations is left for future work.
