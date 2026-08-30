# PCSJ/IMPS 2026 講演申込——入力項目

2026-08-31 時点．講演申込の前に，題目と発表概要の確認をお願いするための一覧．確認後に中野が申込サイトから登録する．

## 学会と締切

| | |
| --- | --- |
| 学会 | PCSJ/IMPS 2026（画像符号化シンポジウム／映像メディア処理シンポジウム）．<https://www.pcsj-imps.org/> |
| 会期・会場 | 2026-11-16（月）〜18（水），御殿場高原ホテル |
| 発表形態 | 一般講演はすべてオンサイトのポスター発表（横 180cm × 縦 90cm） |
| 講演申込〆切 | **2026-09-04（金）** |
| 原稿〆切 | 2026-10-16（金）．A4 2ページ．PCSJ/IMPS サイトのテンプレートで作成し，申込後に通知される URL から提出する |
| 参加申込〆切 | 2026-11-09（月） |
| 申込フォーム | 電子情報通信学会 研究会発表申込システム <https://ken.ieice.org/ken/program/index.php?tgid=IEICE-PCSJ-IMPS> の「発表申込」 |

## 申込フォームの入力項目

フォームの順に並べた．（本人）は登録時に本人が埋める欄，【教授に確認】と【不明】は未確定の箇所．

| | 項目 | 入力する内容 |
| --- | --- | --- |
| 1 | 申込み研究会 | 画像符号化シンポジウム／映像メディア処理シンポジウム（PCSJ-IMPS）．開催回：2026年11月16日(月)-11月18日(水) 御殿場高原ホテル |
| 2 | 発表の形態 | 現地会場におけるプレゼンテーション |
| 3 | 本文の言語 | 日本語（英文タイトルあり） |
| 4 | 書誌情報の公開 | 日本語/英語の書誌情報（タイトル/著者/所属）を入力して公開する |
| 5 | 講演の分類 | 一般講演（実験） |
| 6 | タイトル（和文） | 視覚言語モデルにおけるタスク類似性に基づく内部表現整合学習に関する検討 |
| 7 | タイトル（和文サブタイトル） | なし |
| 8 | タイトル（英文） | A Study of Internal Representation Alignment Based on Task Similarity in Vision-Language Models |
| 9 | タイトル（英文サブタイトル） | なし |
| 10 | 第1著者 | 中野 友晴（ナカノ トモハル）／Tomoharu Nakano．所属は東京都立大学（略称：都立大）／Tokyo Metropolitan University（英語略称は【不明】） |
| 11 | 第2著者以降 | 【教授に確認】指導教員を連名にするか |
| 12 | 講演者 | 第1著者 |
| 13 | 講演者は学生ですか | 講演者は学生である【学生】 |
| 14 | 所属学会 | （本人）会員なら会員番号 |
| 15 | 発表概要 | 下の短版（226 字） |
| 16 | 連絡先名前 | 中野 友晴 |
| 17 | 住所 | （本人） |
| 18 | TEL／携帯TEL／FAX | （本人）携帯と FAX は任意 |
| 19 | Email-1 | （本人）大学のアドレス |
| 20 | Email-2〜5 | （本人）Email-2 は指導教員のアドレス，3〜5 は空欄 |
| 21 | お知らせメール受信の同意 | （本人） |
| 22 | 使用機器 | デフォルトのまま（PCプロジェクタにチェックが入った状態） |
| 23 | 備考 | なし |
| 24 | 原稿の著作権譲渡の同意 | チェックし，同意者氏名に「中野 友晴」 |
| 25 | 連名者の同意 | チェック |

## 発表概要（和文・短版，226 字）

視覚言語モデルの学習に，GQA の functional program の演算子列（タスク署名）が一致する質問の中間層表現を近づける補助損失を加える．すると，正解率はほとんど変わらないまま，署名を線形に読み取れる度合いが大きく上がる．しかし，モデルがその構造を答えの生成に使うようになるわけではない——答えがその層に頼る度合いは，むしろ下がる．いずれも，ラベルの意味だけを変えた対照条件では起きない．「読み取れること」と「使われること」は一致しない．

## 発表概要（和文・中版，299 字）

視覚言語モデルの内部表現は，出力が同じでも入力の揺らぎで大きく動きうることが報告されている．本研究は，同じ解き方で解ける質問どうしの内部表現は近いかを問い，GQA の functional program の演算子列（タスク署名）が一致する質問の中間層表現を近づける補助損失を，通常の学習に加える．すると，正解率はほとんど変わらないまま，署名を線形に読み取れる度合いが大きく上がる．しかし，モデルがその構造を答えの生成に使うようになるわけではない——答えがその層に頼る度合いは，むしろ下がる．いずれも，ラベルの意味だけを変えた対照条件では起きない．「読み取れること」と「使われること」は一致しない．

## 発表概要（和文・長版，658 字）

視覚言語モデル（VLM）の頑健性は通常，入力を揺らしても出力が変わらないことで測られる．しかし近年，出力が同じでも内部表現は大きく動きうることが示され，内部表現そのものの性質は出力とは別に問う必要がある．本研究はこれに隣接する問いとして，同じ解き方で解ける質問どうしの内部表現は互いに近いか，そして明示的に近づけたとき何が起きるかを扱う．GQA の各質問に付く functional program の演算子列を「タスク署名」と呼び，署名が一致する質問の中間層表現を近づける補助損失を通常の学習に加える——これを整合と呼ぶ．整合の有無とラベルの意味だけを変えた三つの条件を複数の乱数種で学習し，表現の側と出力の側の両方で評価した．その結果，質問応答の正解率をほとんど変えないまま，整合した層でタスク署名を線形に読み取れる度合いを大きく上げられる．この向上は，質問文の言い回しでも，ラベルの無作為な並べ替えでも説明されない．しかし，読み取れるようになった構造をモデルが答えの生成に使うようになるわけではない——整合した条件では，答えが整合した層に頼る度合いはむしろ下がる（複数の乱数種で一貫し，ラベルを並べ替えた条件では起きない）．表現から性質が線形に読み取れることと，モデルがその性質を答えの生成に使うことは別である．この区別は従来，既にある性質を表現から除く介入で論じられてきたが，本研究は性質を損失で足す方向からも両者が一致しないことを示す．整合が入力の揺らぎに対する表現の頑健性を高めるかは，今後の課題とする．

## 発表概要（英文，275 語）

The robustness of vision-language models (VLMs) is usually measured by whether the output stays fixed under input perturbations. Recent work, however, shows that internal representations can shift substantially even when the output does not, so the representations themselves must be examined apart from the output. As an adjacent question, we ask whether questions solved in the same way have internal representations close to one another, and what happens when they are explicitly pulled together. We take the operator sequence of each GQA question's functional program as its "task signature," and add to ordinary training an auxiliary loss that pulls together the mid-layer representations of questions sharing a signature — we call this alignment. Training three conditions that differ only in whether alignment is applied and in what the labels mean, across several random seeds, we evaluate both the representations and the answers. Alignment can make the task signature markedly more linearly decodable at the aligned layer while leaving answer accuracy almost unchanged; the gain in decodability is explained neither by the wording of the questions nor by randomly shuffling the labels. Yet the model does not come to use the structure it has made decodable: under alignment the answer comes to rely less on the aligned layer, consistently across seeds and not under the shuffled-label control. Linear decodability of a property and its use in generating the answer are distinct. This distinction has so far been argued by removing existing properties from representations; we show that the two also fail to coincide when a property is added by a loss. Whether alignment improves the robustness of representations to input perturbations is left for future work.

## 備考

- 発表概要に数値は書かない．測り直しや測定の追加があっても取り消しにならない主張——三つの結果の向きと，対照条件の構成——だけを書き，数値は10月の原稿とポスターに置く．
- 三つの結果とは，(1) 整合した層で，タスク署名を線形に読み取れる度合いが大きく上がる，(2) 正解率はほとんど変わらない，(3) 読み取れるようになった構造をモデルが答えの生成に使うようになるわけではなく，整合した層への依存はむしろ下がる，である．向きも強さも，研究文書の要約と同じ言葉で書いた．
- 補助的な検証（研究文書の付録A）には触れない．質問されたときに答える材料として持つ．
- 発表概要の欄の案内は「100〜200 文字程度（英文の場合は 200 語以下），最大 800 文字」．この欄は幹事がプログラム編成の参考にするだけで，公開されない．英文の版は 200 語を超えるのでフォームには入れず，10月の原稿の材料として持つ．
- 講演の分類は，フォーム冒頭の注意書きが一般講演の中からの選択を求める．選択肢は理論／シミュレーション／実験／試作・実用化の報告／サーベイ・解説／その他／テーマ１〜３で，学習を回して測定する本研究には（実験）が最も近い．分野を選ぶ欄はない．
- 非会員でも発表できる．使用機器の欄は PCプロジェクタが最初からチェックされている．注意書きにはポスター発表では使えないとある（会場の提供はポスターボードのみ）が，デフォルトのまま提出する．著作権譲渡の同意は形式上のチェックで，第二種研究会のため権利は著者に残る（いずれもフォーム冒頭の注意書きによる）．
- 賞（優秀論文賞・学生論文賞・ベストポスター賞）は推薦で選ばれ，申込は要らない．学生論文賞の候補になるのは，【学生】を選んだ一般講演のみ．
- フォームに入れる発表概要は申込時点のもの．データベース用アブストラクトは，原稿の提出後に，確認メールに記載の別フォームから登録する．
- 入力項目と選択肢は，PCSJ/IMPS の実フォーム（2026-08-31 取得）で確認した．確認は入力画面まで（その先の確認画面は見ていない）．
