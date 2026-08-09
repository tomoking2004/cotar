# VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models

**著者**：Byung-Kwan Lee，Ryo Hachiuma，Yu-Chiang Frank Wang，Yong Man Ro，Yueh-Hua Wu（NVIDIA／KAIST／国立台湾大学）
**掲載**：CVPR 2025，pp. 29545–29557．arXiv:2412.01822 v2
**原文**：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_VLsI_Verbalized_Layers-to-Interactions_from_Large_to_Small_Vision_Language_Models_CVPR_2025_paper.html)・[arXiv](https://arxiv.org/abs/2412.01822)

**大きい VLM の層ごとの推論の進み方を，小さい VLM に写す——ただし特徴どうしを突き合わせるのではなく，自然言語を挟んで写す．**

## 主張

> 要旨（原文）："The recent surge in high-quality visual instruction tuning samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. However, scaling VLMs to improve performance using larger models brings significant computational challenges, especially for deployment on resource-constrained devices like mobile platforms and robots. To address this, we propose VLsI: Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes, which prioritizes efficiency without compromising accuracy. VLsI leverages a unique, layer-wise distillation process, introducing intermediate "verbalizers" that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. This approach mitigates the training instability often encountered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones."

## 手法

### verbalizer

**中間層それぞれに取り付ける**（大小どちらのバックボーンにも）．構成は **verb-FFN** と，**バックボーン VLM の language head** の二つ．**学習されるのは verb-FFN だけで，バックボーンの重みは固定される**——"Since the weights in both backbone VLMs are fixed, the gradient updates for each verbalizer at each layer remain independent."

対象とする中間層は："For intermediate target layers, we select i_s: 2nd, 6th, 10th, …, and 26th layers, and i_t: 2nd, 6th, 10th, …, and 78th layers"（小 VLM の LLM は28層，大 VLM は80層）．

### 層の対応づけ

- **順序の保存**："Order Preservation—the matched layer j (large-backbone VLM) of layer i (small-backbone VLM) should be deeper than the matched layer k of layer i−1, ensuring j>k"（対応は必ず深い方へ進む）．
- **多項サンプリングによる探索**：verbalizer の語彙確率分布どうしの **KL divergence に反比例する分布**から対応先を引く．`T ← scale / (kld-list.max − kld-list.min + ϵ)`，`p-list ← Softmax(−kld-list / T)`．

### 学習の三段

原文の言い方で "the three-step process: (1) the verbalization step…(2) the interaction step…(3) the SFT step"．

| 段 | 何をするか | 損失 |
| --- | --- | --- |
| **1. Verbalization** | 大小両方の対象中間層に verbalizer を付け，中間埋め込みを言語空間へ写す | 自己回帰の交差エントロピー |
| **2. Interaction** | KL divergence に基づく多項サンプリングで層を対応づけ，対応先の語彙確率分布を合わせる | 対応づいた層の verbalizer どうしの KL divergence |
| **3. SFT** | 小バックボーン VLM 全体を visual instruction データで微調整 | 最終出力上の自己回帰の交差エントロピー |

**第3段は不可欠**——SFT を外した ablation では，表4の各指標が大きく落ちる．

### 自然言語を挟む理由——原文が言っている範囲

**ここは誤読しやすいので，原文の言い回しをそのまま置く．**

- 序論："This approach mitigates the training instability often encountered in **output imitation** and goes beyond typical final-layer tuning by aligning the small VLMs' layer-wise progression with that of the large ones."
- 関連研究："we leverage natural language in order to make small VLMs mimic the reasoning progression of large VLMs across layers. We hope that incorporating natural language will facilitate smoother communication between large and small VLMs, **alleviating the complexities of feature alignment**."

つまり——**「学習の不安定」が帰されているのは output imitation であり，特徴の対応づけ（feature alignment）に対して用いられている語は "complexities（煩雑さ）" である．「教師の生の中間特徴を直接突き合わせると学習が不安定になる」と述べた文は，本文中に存在しない．**

## 実験設定

- **バックボーン**："we select Qwen2-VL as our backbone VLM"．
- **規模**："Qwen2-1.5B and Qwen2-7B each contain 28 layers, and Qwen2-72B consists of 80 layers"——教師は 72B（80層），生徒は 2B と 7B（各28層）．
- **データ**：**290万件**の visual instruction tuning サンプル．

**評価の範囲**（ベンチマークは表ごとに異なる）

| 表 | ベンチマーク |
| --- | --- |
| 表1（7B）・表2（2B） | QBench，AI2D，ChartQA，POPE，HallusionBench，MME，MathVista，MMB，MMB-Chinese，MM-Vet，MMMU |
| 表3(b)（横断比較） | MM-Vet，MM-Vet-v2，MMMU，MMStar，AI2D，SEED-2-Plus，MathVista，BLINK，CV-Bench，LLaVA-Wilder |
| 表4（ablation） | MMB，BLINK，MM-Vet，MMMU |

## 結果

**VLsI-2B（対 Qwen2-VL-2B）**：AI2D 60.2→89.0，ChartQA 73.5→85.8，MathVista 43.0→68.4，MMB 74.9→81.7，MM-Vet 49.5→64.8，MMMU 41.1→51.4．

**VLsI-7B（対 Qwen2-VL-7B）**：AI2D 77.5→87.3，ChartQA 83.0→86.1，MathVista 58.2→74.7，MMB 83.0→86.3，MM-Vet 62.0→75.2，MMMU 54.1→69.3．

全体としては——"demonstrating significant performance gains of 11.0% (2B model) and 17.4% (7B model) over GPT-4V"．

## 著者が明記した限界

- **tokenizer の制約**："the large and small-backbone VLMs must share the same tokenizer and token index order"．
- **今後**："We will explore more general ways that accommodate different tokenizers and token index orders, potentially expanding VLsI's applicability and scalability."

## 射程の外

- **外部の大型モデルを教師とする cross-model 蒸留**である．同一モデル内での自己整合や，異なるタスク間の表現の関係は扱わない．
- **290万件の visual instruction tuning データと 72B の教師を要する**——資源要求は小さくない．
- **GQA はどの評価表にも現れない．**

---

**確認**：2026-08-09．arXiv HTML 全文 v2 を参照し，本文の数値・逐語引用を一件ずつ原文と照合した．掲載巻号は CVPR Open Access の書誌（`Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 29545-29557`）で確認．「自然言語を挟む理由」節は，序論と関連研究を対象に該当表現を探した結果に基づく．
