# 文献の要約の目録

1本につき1ファイルである．要約の文は原典に照らして確かめた表現で書き，いつ何に照らしたかを各ファイルの末尾の「確認」に残す．原典について書くときは，要約を経由せず原典の本文に戻る（要約は要約者の読みである）．

要約は二つに分かれる．研究文書（`.claude/context.md`）が参考文献に挙げる論文と，挙げていないが関連する論文である．後者は 2026-10-11 に書いた下書きを，2026-11-13 に1本ずつ原典の全文で確かめ直して取り込んだもので，研究文書との関わりは各要約が自分で書いている．関わりの多くは背景にとどまり，研究文書の主張の根拠にはなっていない．

個別の要約を作っていない文献の調べは [novelty-check.md](novelty-check.md) にある．

## 研究文書が引く論文

| 要約 | 論文 |
| --- | --- |
| [hidden-instability](hidden-instability.md) | Wani ら，Same Answer, Different Representations: Hidden instability in VLMs |
| [task-format](task-format.md) | Sahoo ら，Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States |
| [amnesic-probing](amnesic-probing.md) | Elazar ら，Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals |
| [probing-classifiers](probing-classifiers.md) | Belinkov，Probing Classifiers: Promises, Shortcomings, and Advances |
| [supcon](supcon.md) | Khosla ら，Supervised Contrastive Learning |
| [gqa](gqa.md) | Hudson & Manning，GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering |
| [vlsi](vlsi.md) | Lee ら，VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models |
| [hydra-effect](hydra-effect.md) | McGrath ら，The Hydra Effect: Emergent Self-repair in Language Model Computations |
| [global-workspace](global-workspace.md) | Gurnee ら，Verbalizable Representations Form a Global Workspace in Language Models |
| [smolvlm](smolvlm.md) | Marafioti ら，SmolVLM: Redefining small and efficient multimodal models |

## 研究文書が引いていない，関連する論文

一行は，各要約を原典に照らした照合の報告から取った．「関わる節」は研究文書の節番号で，関わりの中身と限界は各要約にある．

### 補助損失を掛けた表現が，答えに使われるか

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [cosine-misleads](cosine-misleads.md) | latent を視覚標的へ寄せる補助損失の5変種で，cosine 整合度は V\*Bench 正解率と負に相関し（r=−0.94），latent を壊しても正解率は約4ポイント以内しか動かない（Zhang ら，arXiv 2026） | §5.3，§A.4 |
| [latent-bypass](latent-bypass.md) | latent 視覚推論の4手法は，latent を無情報なトークンに置き換えても正解率がほぼ変わらず，予測 latent は oracle から離れて潰れる（Viveiros ら，NeurIPS 2026 ポスター） | §5.3 |
| [interchange-intervention](interchange-intervention.md) | 因果モデルの変数に対応づけた表現を入れ替え，出力が因果モデルの反実仮想と一致するよう学習する IIT を提案し，3課題で評価した（Geiger ら，ICML 2022） | §2，§5.3 |
| [viral](viral.md) | LLaVA-1.5 の第16層の視覚トークンを DINOv2 の特徴へ cosine で整合する損失を足し，視覚中心のベンチマークが上がると報告した（Yoon ら，arXiv 2025） | §3.3，§A.5 |

### 解き方の構造を教師にする

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [program-supervision](program-supervision.md) | VQA の分類 head 直前の表現から GQA の program を予測させる正則化で，oracle transfer と GQA／GQA-OOD の正解率が上がる（Kervadec ら，NeurIPS 2021） | §1.2，§3 |
| [pattern-contrast](pattern-contrast.md) | 算数文章題で，式の木構造が部分木まで含めて一致する問題を正例とする三つ組の対照損失を足し，正解率が上がることを示した（Li ら，Findings of ACL 2022） | §3.1，§3.2 |
| [clevr](clevr.md) | 合成画像の各質問に実行可能な functional program を付け，型・関係・構造別に VQA モデルの弱点を診断するデータセット（Johnson ら，CVPR 2017） | §3.1 |
| [gqa-ood](gqa-ood.md) | GQA を文脈ごとに稀な答え（tail）と頻出の答え（head）に分けて測ると，評価した VQA モデルはどれも tail の正解率が head より低い（Kervadec ら，CVPR 2021） | §A.3 |

### 読み取れることと，使われること

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [probing-paradigm](probing-paradigm.md) | NLI の文符号化器で，学習タスクに要らない性質も同程度にプローブで読め，プローブ精度はモデルが頼る情報を示さないとした（Ravichander ら，EACL 2021） | §2，§6 |
| [feature-representations](feature-representations.md) | 二つの特徴がラベルを冗長に予測すると，モデルは一方を優先して表現し，その選好は学習前のモデルで最も線形に読み出せた特徴を反映する（Hermann & Lampinen，NeurIPS 2020） | §2，§6 |
| [grammatical-number](grammatical-number.md) | BERT は数の一致を解くのに数の線形な符号化に頼り，その符号化は名詞と動詞で別であることを，除去介入で示した（Lasri ら，ACL 2022） | §5.3，§A.4 |
| [localization-editing](localization-editing.md) | Causal Tracing による事実の局在は，編集の成否をほとんど説明せず，編集層の選択のほうがはるかによく説明する（Hase ら，NeurIPS 2023） | §5.3，§6 |
| [leace](leace.md) | 線形に守られる条件はクラス平均の一致と同値であり，それを満たす最小二乗の意味で最小のアフィン編集を閉形式で与える（Belrose ら，NeurIPS 2023） | §A.4 |

### プローブの測り方

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [linear-probes](linear-probes.md) | 各層の特徴に，モデルの学習に影響しない線形分類器（プローブ）を当て，線形分離可能性が深い層ほどほぼ単調に増すことを観察した（Alain & Bengio，ICLR 2017 Workshop） | §5.1，§A.1 |
| [control-tasks](control-tasks.md) | 単語型に無作為な出力を割り当てた control task を提案し，ELMo 上のよく使われるプローブ，特に MLP は selectivity が低いと示した（Hewitt & Liang，EMNLP 2019） | §5.1，§A.1 |
| [mdl-probing](mdl-probing.md) | プローブを精度でなくラベルの記述長（MDL）で評価すると，control task やランダム初期化モデルとの差が出て，設定や乱数種に対して安定すると示した（Voita & Titov，EMNLP 2020） | §5.1，§A.1 |

### 介入・感度で「使われる」を測る

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [distributed-alignment](distributed-alignment.md) | 凍結したモデルの表現に回転を学習し，部分空間への入れ替え介入で高水準の因果変数との対応を探す DAS を提案した（Geiger ら，CLeaR 2024） | §6 |
| [subspace-illusion](subspace-illusion.md) | 部分空間の活性パッチは，出力から切り離された方向と休眠方向の和でも効き，特徴の在りかを誤認させうると GPT-2 の IOI と事実想起で示した（Makelov ら，ICLR 2024） | §5.3，§A.4 |
| [atp-star](atp-star.md) | 活性パッチの勾配近似 AtP が attention の飽和と効果の打ち消しで偽陰性を生むことを示し，修正版 AtP\* と偽陰性の上界の診断法を提案した（Kramár ら，arXiv 2024） | §5.3，§A.4 |
| [self-repair](self-repair.md) | 個々の attention head の除去でも事前学習分布全体で自己修復が在り，不完全で noisy である．一因は LayerNorm の尺度の変化と疎な Anti-Erasure ニューロン（Rushing & Nanda，ICML 2024） | §A.4 |
| [tuned-lens](tuned-lens.md) | 凍結した言語モデルの各層にアフィンのプローブを学習し，隠れ状態を語彙上の分布として読む．logit lens より偏りと perplexity が低い（Belrose ら，arXiv 2023） | §A.4 |
| [linear-representation](linear-representation.md) | 線形表現を出力側と文脈側の二通りに定式化し，causal inner product のもとで両者が統一されることを示し，LLaMA-2 で確かめた（Park ら，ICML 2024） | §A.4 |

### タスクの表現が載る位置

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [task-vectors](task-vectors.md) | ICL は例示を中間層の単一のタスクベクトルへ圧縮し，それを差し込むだけで通常の ICL の正解率の約80〜90%を保つと示した（Hendel ら，Findings of EMNLP 2023） | §3.3 |
| [function-vectors](function-vectors.md) | ICL で少数の attention head がタスクの表現を運び，その平均出力の和を例示のない入力に足すとタスクの実行が引き起こされる（Todd ら，ICLR 2024） | §3.3 |
| [cross-modal-task-vectors](cross-modal-task-vectors.md) | VLM はテキスト例・画像例・指示文を共通のタスクベクトルへ写し，それを別モダリティの問いへ差し込むと正答をしばしば引き出す（Luo ら，ICML 2025） | §3.3 |
| [cross-modal-flow](cross-modal-flow.md) | LLaVA 系列の GQA 質問応答で注意を遮断し，画像の情報が下層から中間層で二度質問の位置へ移り，そこから最終位置へ移ることを示した（Zhang ら，CVPR 2025） | §3.3 |

### 対照損失と表現の幾何

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [supcon-finetuning](supcon-finetuning.md) | RoBERTa-Large の微調整で，交差エントロピーと [CLS] 上の教師あり対照項を重み付き平均し，GLUE の少数ショットで精度を改善した（Gunel ら，ICLR 2021） | §3.2，§3.3 |
| [supcon-geometry](supcon-geometry.md) | 交差エントロピーと教師あり対照損失は，各クラスが正則単体の頂点に崩壊するとき最小になると証明し，最適化の振る舞いは異なると示した（Graf ら，ICML 2021） | §3.2，§3.3 |
| [alignment-uniformity](alignment-uniformity.md) | 対照損失は負例数無限の極限で alignment と uniformity を最適化すると証明し，両指標の直接最適化で同等以上の下流性能を得た（Wang & Isola，ICML 2020） | §3.2 |
| [dimensional-collapse](dimensional-collapse.md) | 対照型の自己教師学習でも埋め込みが低次元に潰れる次元崩壊が起き，射影 head が表現空間をそれから守ることを示した（Jing ら，ICLR 2022） | §3.3 |
| [anisotropy](anisotropy.md) | ELMo・BERT・GPT-2 の文脈化表現はほぼ全層で異方的で，無作為な語どうしの cos の平均を基準に差し引いて文脈化の度合いを測る（Ethayarajh，EMNLP 2019） | §4.4 |
| [clip](clip.md) | バッチ内のどの画像–テキスト対が実在するかを当てる対称な対照損失で4億対から学び，ImageNet の zero-shot 精度 76.2% を得た（Radford ら，ICML 2021） | §3.2 |

### 実装と統計

| 要約 | 論文が示したこと | 関わる節 |
| --- | --- | --- |
| [schedule-free](schedule-free.md) | 学習率スケジュールを使わず，補間と平均の3系列で，停止時刻を事前に定めずに cosine スケジュールに並ぶか上回る最適化法を示した（Defazio ら，NeurIPS 2024）．実装が使う最適化器の元の論文で，研究文書は最適化器に触れていない | — |
| [ml-variance](ml-variance.md) | ベンチマークの性能は複数の乱数源で揺れ，データの bootstrap が最大の分散源であると示し，乱数源の無作為化と P(A>B) による判定を勧めた（Bouthillier ら，MLSys 2021） | §4.3 |

## 下書きにあって，取り込まなかったもの

- **SmolVLM**：上の [smolvlm](smolvlm.md) が，2026-11-06 に PDF の全文で照合し直した新しい版である．
- **REPA（Yu ら），INLP（Ravfogel ら），Layer by Layer（Skean ら），Kornblith らの損失と転移の論文**：研究文書との関わりが，別の論文を介した間接のものか背景にとどまる．原典との照合もしていない．要約が要るようになったら，原典の全文から書く．
