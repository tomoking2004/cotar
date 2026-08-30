# PCSJ/IMPS 2026 講演申込——入力項目

2026-08-31 時点．講演申込の前に，題目とアブストラクトの確認をお願いするための一覧．確認後に中野が申込サイトで登録する．

## 学会と締切

| | |
| --- | --- |
| 学会 | PCSJ/IMPS 2026（画像符号化シンポジウム／映像メディア処理シンポジウム）．<https://www.pcsj-imps.org/> |
| 会期・会場 | 2026-11-16（月）〜18（水），御殿場高原ホテル |
| 発表形態 | 一般講演はすべてオンサイトのポスター発表（横 180cm × 縦 90cm） |
| 講演申込〆切 | **2026-09-04（金）** |
| 原稿〆切 | 2026-10-16（金）．A4 2ページ．申込後に通知されるアップロード用 URL から提出 |
| 参加申込〆切 | 2026-11-09（月） |
| 申込フォーム | 電子情報通信学会 研究会発表申込システム <https://ken.ieice.org/ken/program/index.php?tgid=IEICE-PCSJ-IMPS> の「発表申込」 |

## 申込フォームの入力項目

フォームの順に並べた．研究会ごとに変わる選択肢（講演の分類・分野）は，実際のフォームの選択肢に合わせる．

| | 項目 | 入力する内容 |
| --- | --- | --- |
| 1 | 申込み研究会 | 画像符号化シンポジウム／映像メディア処理シンポジウム（2026-11-16〜18，御殿場） |
| 2 | 発表の形態 | 現地会場におけるプレゼンテーション（一般講演はすべてオンサイトのポスター） |
| 3 | 本文の言語 | 日本語（英文タイトルあり） |
| 4 | 書誌情報の公開 | 日本語／英語の書誌情報（タイトル・著者・所属）を入力して公開する |
| 5 | 講演の分類 | 一般講演．分野を選ぶ欄があれば IMPS「画像認識/解析」 |
| 6 | タイトル（和文） | 視覚言語モデルにおけるタスク類似性に基づく内部表現整合学習に関する検討 |
| 7 | タイトル（和文サブタイトル） | なし |
| 8 | タイトル（英文） | A Study of Internal Representation Alignment Based on Task Similarity in Vision-Language Models |
| 9 | タイトル（英文サブタイトル） | なし |
| 10 | 第1著者 | 氏名：中野 友晴／Tomoharu Nakano，フリガナ：ナカノ トモハル．所属：東京都立大学／Tokyo Metropolitan University，略称：都立大（部署・学科は書かない） |
| 11 | 第2著者以降 | 指導教員を連名にするかは先生の指示に従う．連名にする場合は第1著者と同じ形式で入力する |
| 12 | 講演者 | 第1著者 |
| 13 | 講演者は学生ですか | 講演者は学生である【学生】 |
| 14 | 所属学会 | 電子情報通信学会の会員なら会員番号を，入会手続中ならその欄を．いずれでもなければチェックしない |
| 15 | 発表概要 | 下の短版（196 字）．欄の案内は「100〜200 文字程度（英文の場合は 200 語以下），最大 800 文字」なので，中版（281 字）・長版（651 字）も入る |
| 16 | 連絡先名前 | 中野 友晴 |
| 17 | 住所 | 郵便番号・住所・東京都立大学の学部（研究科）・研究室（本人が入力） |
| 18 | TEL／携帯TEL／FAX | 本人が入力（携帯と FAX は任意） |
| 19 | Email-1 | 講演者（本人）の大学のアドレス．ピリオドを含む Gmail は登録不可 |
| 20 | Email-2 | 指導教員のアドレス（学生は必須）．必要なら共著者のアドレスを Email-5 まで |
| 21 | お知らせメール受信の同意 | 本人が選ぶ |
| 22 | 使用機器 | ポスター発表につき該当なし |
| 23 | 備考 | なし |
| 24 | 原稿の著作権譲渡の同意 | 著作権規程を確認してチェックし，同意者氏名に「中野 友晴」（連名者がいれば全員の同意を得て氏名を併記） |
| 25 | 連名者の同意 | チェック（連名者がいる場合は申込前に同意を得る） |
| 26 | アンケート（懇親会参加予定） | 本人が選ぶ |

## 発表概要（和文・短版，196 字）——フォームに入れる版

視覚言語モデルの内部表現は，出力が同じでも入力の揺らぎで動きうる．本研究は，GQA の functional program の演算子列（タスク署名）が一致する質問の中間層表現を近づける補助損失を学習に加え，正解率をほとんど変えないまま署名を線形に顕著に読み取りやすくできる一方，その構造をモデルが答えの生成に使うようになるわけではないことを，ラベルの意味だけを変えた対照条件との比較で示す．

## 発表概要（和文・中版，281 字）

視覚言語モデルの内部表現は，出力が同じでも入力の揺らぎで大きく動きうることが報告されている．本研究は，同じ解き方で解ける質問どうしの内部表現が近いかを問い，GQA の functional program の演算子列（タスク署名）が一致する質問の中間層表現を近づける補助損失を通常の学習に加えた．その結果，正解率をほとんど変えないまま署名を線形に顕著に読み取りやすくでき，しかも読み取りやすくなった構造をモデルが答えの生成に使うようになるわけではない——「読み取れること」と「使われること」は一致しない——ことを，ラベルの意味だけを変えた対照条件との比較で示す．

## 発表概要（和文・長版，651 字）

視覚言語モデル（VLM）の頑健性は通常，入力を揺らしても出力が変わらないことで測られる．しかし近年，出力が同じでも内部表現は大きく動きうることが示され，内部表現そのものの性質は出力とは別に問う必要がある．本研究はこれに隣接する問いとして，同じ解き方で解ける質問どうしの内部表現は互いに近いか，そして明示的に近づけたとき何が起きるかを扱う．GQA の各質問に付く functional program の演算子列を「タスク署名」と呼び，署名が一致する質問の中間層表現を近づける補助損失を通常の学習に加える．整合の有無とラベルの意味だけを変えた三つの条件を複数の乱数種で学習し，表現の側と出力の側の両方で評価した．その結果，質問応答の正解率をほとんど変えないまま，整合した層でタスク署名を線形に顕著に読み取りやすくできる．この読み取りやすさの向上は，質問文の言い回しでも，ラベルの無作為な並べ替えでも説明されない．しかし，読み取りやすくなった構造をモデルが答えの生成に使うようになるわけではない——整合した条件では，答えが整合した層に頼る度合いはむしろ下がる（複数の乱数種で一貫し，ラベルを並べ替えた条件では起きない）．表現から性質が線形に読み取れることと，モデルがその性質を答えの生成に使うことは別である．この区別は従来，既にある性質を表現から除く介入で論じられてきたが，本研究は性質を損失で足す方向からも両者が一致しないことを示す．整合が入力の揺らぎに対する表現の頑健性を高めるかは，今後の課題とする．

## Abstract (English, 270 words)

The robustness of vision-language models (VLMs) is usually measured by whether the output stays fixed under input perturbations. Recent work, however, shows that internal representations can shift substantially even when the output does not, so the representations themselves must be examined apart from the output. As an adjacent question, we ask whether questions solved in the same way have internal representations close to one another, and what happens when they are explicitly pulled together. We take the operator sequence of each GQA question's functional program as its "task signature," and add to ordinary training an auxiliary loss that pulls together the mid-layer representations of questions sharing a signature. Training three conditions that differ only in whether alignment is applied and in what the labels mean, across several random seeds, we evaluate both the representations and the answers. Alignment can make the task signature markedly more linearly decodable at the aligned layer while leaving answer accuracy almost unchanged; the gain in decodability is explained neither by the wording of the questions nor by randomly shuffling the labels. Yet the model does not come to use the structure it has made decodable: under alignment the answer comes to rely less on the aligned layer, consistently across seeds and not under the shuffled-label control. Linear decodability of a property and its use in generating the answer are distinct. This distinction has so far been argued by removing existing properties from representations; we show that the two also fail to coincide when a property is added by a loss. Whether alignment improves the robustness of representations to input perturbations is left for future work.

## 備考

- **数値を書いていない．** 測り直し（整合の強さ・層の追加測定）や方針の追加（表現の頑健性の測定）があっても取り消しにならない主張——三つの結果の向きと，対照条件の構成——だけを書いた．数値は10月の2ページ原稿とポスターに置き，測定時点を添える．
- 三つの結果とは，(1) 整合した層でタスク署名が線形に読み取りやすくなる，(2) 正解率はほとんど変わらない，(3) 読み取りやすくなった構造をモデルが使うようになるわけではなく，整合した層への依存はむしろ下がる，である．向きと対照の構成は研究文書の要約と同じ言葉で書き，強さを変えていない．
- 補助的な検証（研究文書の付録A）には触れない．質問されたときに答える材料として持つ．
- 賞（優秀論文賞・学生論文賞・ベストポスター賞）は運営委員・実行委員の推薦と聴講者の意見で選ばれ，申込は要らない．学生論文賞の対象になるのは「講演者は学生である」の回答による．
- 入力項目は，同じ研究会発表申込システムの別の研究会のフォームから読み取った．PCSJ/IMPS のフォーム自体は外部からの自動取得を拒むので，選択肢の細部は登録時に画面で確かめる．
