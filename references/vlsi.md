# VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models

**著者**：Byung-Kwan Lee，Ryo Hachiuma，Yu-Chiang Frank Wang，Yong Man Ro，Yueh-Hua Wu（NVIDIA／KAIST／国立台湾大学）
**掲載**：CVPR 2025，pp. 29545–29557．arXiv:2412.01822 v2
**原文**：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_VLsI_Verbalized_Layers-to-Interactions_from_Large_to_Small_Vision_Language_Models_CVPR_2025_paper.html)・[arXiv](https://arxiv.org/abs/2412.01822)

**大きい VLM の層ごとの推論の進み方を，小さい VLM に写す——ただし特徴どうしを突き合わせるのではなく，自然言語を挟んで写す．**

## 主張

> 要旨（原文）："The recent surge in high-quality visual instruction tuning samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. However, scaling VLMs to improve performance using larger models brings significant computational challenges, especially for deployment on resource-constrained devices like mobile platforms and robots. To address this, we propose VLsI: Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes, which prioritizes efficiency without compromising accuracy. VLsI leverages a unique, layer-wise distillation process, introducing intermediate "verbalizers" that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. This approach mitigates the training instability often encountered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones."

## 手法

### verbalizer

**対象の中間層それぞれに取り付ける**（大小どちらのバックボーンにも）．構成は **verb-FFN** と，**バックボーン VLM の language head** の二つ．**第1段で学習されるのは verb-FFN だけで，バックボーンの重みは固定される**——"Since the weights in both backbone VLMs are fixed, the gradient updates for each verbalizer at each layer remain independent."

対象とする中間層は，小 VLM が "2nd, 6th, 10th, ..., and 26th layers"，大 VLM が "2nd, 6th, 10th, ..., and 78th layers" である（小 VLM の LLM は28層，大 VLM は80層）．

### 層の対応づけ

- **順序の保存**："Order Preservation—the matched layer j (large-backbone VLM) of layer i (small-backbone VLM) should be deeper than the matched layer k of layer i−1, ensuring j>k"（対応は必ず深い方へ進む）．
- **多項サンプリングによる探索**：verbalizer の語彙確率分布どうしの **KL divergence に反比例する分布**から対応先を引く．`T ← scale / (kld-list.max − kld-list.min + ϵ)`，`p-list ← Softmax(−kld-list / T)`．
- **文章は生成しない**：層ごとの言語情報には language head の語彙確率を使う．生成は各層で行うには計算が重すぎるからである（§3.2）．

### 学習の三段

原文はこれを "a three-step process" と呼び，段の名は "the verbalization step"，"the interaction step"，"the SFT step" である．

| 段 | 何をするか | 損失 |
| --- | --- | --- |
| **1. Verbalization** | 大小両方の対象中間層に verbalizer を付け，中間埋め込みを言語空間へ写す | 自己回帰損失（autoregressive loss） |
| **2. Interaction** | KL divergence に基づく多項サンプリングで層を対応づけ，対応先の語彙確率分布を合わせる | 対応づいた層の verbalizer どうしの KL divergence |
| **3. SFT** | 小バックボーン VLM 全体を visual instruction データで微調整 | 自己回帰損失 |

**第3段は不可欠**——SFT を外した ablation（表4(b)）では，どの指標も大きく落ち，貪欲復号ではバックボーンのままより低い（7B の MM-Vet は Qwen2-VL-7B が 62.0，SFT なしの VLsI が 60.5，SFT ありが 75.8）．

**データだけの効果でもない**——同じデータで SFT だけをした Qwen2-VL-7B の MM-Vet は 63.7 にとどまる（表4(c)）．著者はこの比較で，向上が 290万件の微調整の効果だけではないことを確かめたと述べる（付録D）．

### 自然言語を挟む理由——原文が言っている範囲

**ここは誤読しやすいので，原文の言い回しをそのまま置く．**

- 要旨："This approach mitigates the training instability often encountered in **output imitation** and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones."
- 関連研究："we leverage natural language in order to make small VLMs mimic the reasoning progression of large VLMs across layers. We hope that incorporating natural language will facilitate smoother communication between large and small VLMs, **alleviating the complexities of feature alignment**."

- 付録D（表4(d) の補足の表）：verbalizer の language head を外し，第1段を飛ばして，大小の verb-FFN の隠れ次元を揃えて L2 で合わせる ablation がある．MM-Vet／MMMU は 63.3〜64.6／53.5〜55.9 で，language head ありの 75.8／69.3 に届かず，最終層だけの蒸留（65.2〜66.5／56.8〜57.5）も下回る．著者はこう読む——"These findings suggest that directly imitating outputs from the large-backbone VLM, without verbalization from vL-Head, introduces instability and can lead to suboptimal results"．

つまり——**本編で「学習の不安定」が帰されているのは output imitation であり，特徴の対応づけ（feature alignment）に対して用いられている語は "complexities"（煩雑さ）である．不安定さを language head 抜きの模倣に帰す文は，付録Dにある．その文の主語も "directly imitating outputs" であり，根拠に示されるのは最終の性能である．学習の不安定さを直接測った図表は付録Dに無い．**

## 実験設定

- **バックボーン**：Qwen2-VL（§4.1）．
- **規模**："Qwen2-1.5B and Qwen2-7B each contain 28 layers, and Qwen2-72B consists of 80 layers"——教師は 72B（80層），生徒は 2B と 7B（各28層）．表4(a) と図4では LLaVA-OV を教師にした組も試す．
- **データ**：**290万件**の visual instruction tuning サンプル．
- **学習**：LLM 部分に LoRA（rank・alpha とも64）を当て，NVIDIA A100 80GB 8枚で学習する．

**評価の範囲**（ベンチマークは表ごとに異なる）

| 表 | ベンチマーク |
| --- | --- |
| 表1（7B）・表2（2B） | QBench，AI2D，ChartQA，POPE，HallusionBench，MME，MathVista，MMB，MMB-Chinese，MM-Vet，MMMU |
| 表3(b)（横断比較） | MM-Vet，MM-Vet-v2，MMMU，MMStar，AI2D，SEED-2-Plus，MathVista，BLINK，CV-Bench，LLaVA-Wilder |
| 表4（ablation） | MMB，BLINK，MM-Vet，MMMU |

## 結果

**VLsI-2B（対 Qwen2-VL-2B）**：AI2D 60.2→89.0，ChartQA 73.5→85.8，MathVista 43.0→68.4，MMB 74.9→81.7，MM-Vet 49.5→64.8，MMMU 41.1→51.4．

**VLsI-7B（対 Qwen2-VL-7B）**：AI2D 77.5→87.3，ChartQA 83.0→86.1，MathVista 58.2→74.7，MMB 83.0→86.3，MM-Vet 62.0→75.2，MMMU 54.1→69.3．（VLsI-7B の MM-Vet は，表1では 75.2，表3(b)・表4では 75.8 である．原文の中で値が揃っていない．）

全体としては——"demonstrating significant performance gains of 11.0% (2B model) and 17.4% (7B model) over GPT-4V"．

## 著者が明記した限界

- **tokenizer の制約**："the large and small-backbone VLMs must share the same tokenizer and token index order"．
- **今後**："We will explore more general ways that accommodate different tokenizers and token index orders, potentially expanding VLsI's applicability and scalability."

## 射程の外

- **外部の大型モデルを教師とする cross-model 蒸留**である．同一モデル内での自己整合や，異なるタスク間の表現の関係は扱わない．
- **290万件の visual instruction tuning データと 72B の教師を要する**——資源要求は小さくない．
- **GQA はどの評価表にも現れない．** 本文と付録の全文に GQA の語が無い．

## 学会で聞かれたら

本研究についての答えは研究文書（`.claude/context.md`）の範囲で書いた．

- **問：VLsI はどんな研究ですか．**
  答：大きい VLM の層ごとの推論の進み方を，小さい VLM へ蒸留する手法です．各層に verbalizer を付けて語彙の確率分布へ写し，大小の層を対応づけて KL divergence で合わせます．（出所：原典 §3）
- **問：自然言語を挟むとは，文章を生成して教えるのですか．**
  答：いいえ．生成は計算が重いので避けていて，合わせるのは language head が出す語彙の確率分布です．（出所：原典 §3.2）
- **問：なぜ中間の特徴を直接合わせないのですか．**
  答：原典の付録に，language head を外して L2 で合わせた比較があり，MM-Vet は 64.6 以下で，language head ありの 75.8 に届きません．著者は，言語を介さない模倣は不安定さを招くと読んでいますが，示しているのは最終の性能で，不安定さを直接測った図表はありません．（出所：原典 付録D）
- **問：VLsI の向上は，データが多いからではないですか．**
  答：著者はそれを確かめています．同じデータで SFT だけをした Qwen2-VL-7B の MM-Vet は 63.7 で，VLsI-7B の 75.8 に届きません．（出所：原典 表4(c)，付録D）
- **問：VLsI では正解率が上がったのに，本研究ではなぜ上がらないのですか．**
  答：VLsI は 72B の教師を使う蒸留で，本研究は教師を持たない同一モデル内の自己整合なので，設定が違います．どの違いが効いたかは本研究では確かめていません．本研究で測ったのは，誤答した問でも署名が高い精度で読み取れることまでです．（出所：原典 §4，研究文書 §2，§A.3）
- **問：途中の層に損失を掛けると効く，という根拠は VLsI で足りますか．**
  答：足りません．研究文書でも傍証にとどめていて，中核の主張の支柱にはしていません．VLsI でも，language head を介さず L2 で合わせた場合は最終層だけの蒸留より低く，本研究の損失は表現を直接近づけるので，この差が本研究にどう効くかは確かめていません．（出所：原典 付録D，研究文書 §2）
- **問：VLsI の GQA の数値と比べられますか．**
  答：比べられません．VLsI の本文と付録に GQA は出てきません．（出所：原典の全文検索）

---

**確認**：2026-08-09．arXiv HTML 全文 v2 を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．掲載巻号は CVPR Open Access の書誌（`Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 29545-29557`）で確認．「自然言語を挟む理由」節は，序論と関連研究を対象に該当表現を探した結果に基づく．

**再照合**：2026-11-06．arXiv の PDF v2（pdftotext で本文と付録の全文をテキスト化）と HTML v2（表の値）を読み，本文書の数値・引用・節番号を一件ずつ照合した．CVPR Open Access の書誌の頁も見直した．"instability" と GQA は全文を検索した．直した主な点：不安定さに触れた付録Dの文を足し，「本文中に存在しない」の断定を改めた（前回は序論と関連研究しか探していなかった）．引用元を序論から要旨へ，損失の名を原文の autoregressive loss へ直し，連続していない引用を除いた．付録C の生成例と付録E の生成品質は表と図の値を照合していない．
