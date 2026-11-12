# 新規性の調べ——個別の要約を作っていない文献

本研究の主張のうち，どこまでが既に報告されているかを見るために集めた文献の控えである．2026-10-11 の下書きを，2026-11-13 に公開ページで確かめ直して取り込んだ．個別の要約がある論文は [README.md](README.md) の目録にあり，ここには要約を作っていない 21 本を置く．

## 読むときの注意

- **21 本は要旨の水準でしか確かめていない．** 書誌（題・全著者・版）は arXiv の abs ページか公式の書誌ページで確かめ，一文ごとに要旨の文に戻せることを確かめた．本文まで読んだのは「読んだ範囲」に PDF と書いた 7 本の該当箇所だけである．論文に書くときは，原典の本文に当たり直す．
- **「前例は見つからなかった」は，2026-10-11 の下書きの探索の報告である．** 探索の範囲と検索語は記録が無く，2026-11-13 にはやり直していない．被引用を辿る探索もしていない．無いことの証明ではない．
- **研究文書（`.claude/context.md`）は，ここに挙げた文献を引いていない．** 研究文書が先行研究として挙げるのは Belinkov の総覧と Elazar らである（§2，§6）．

## 既に報告されていること

| 主張 | 報告している文献 |
| --- | --- |
| 線形に読めることと，使われることは一致しない | Elazar ら（[amnesic-probing](amnesic-probing.md)），Ravichander ら（[probing-paradigm](probing-paradigm.md)），Belinkov の総覧（[probing-classifiers](probing-classifiers.md)）．VLM と視覚モデルについては C 節 |
| VLM で，補助損失で教師した latent トークンを壊したり置き換えたりしても，正解率がほとんど変わらない | Zhang ら（[cosine-misleads](cosine-misleads.md)），Viveiros ら（[latent-bypass](latent-bypass.md)） |
| 中間変数を分類器で読めるよう表現ごと学習させた比較群（MULTI）は，分布シフト下の振る舞いの正解率が，STANDARD の群と同じく 0.00% だった | Geiger ら 2022 の MNIST-PVR（[interchange-intervention](interchange-intervention.md) の表1．入れ替え介入の test は MULTI 20.50%，STANDARD 20.60%，IIT 94.85%）．比較群の一結果で，論文の問いではない |
| GQA の program を，VQA モデルの表現への補助教師にする | Kervadec ら 2021（[program-supervision](program-supervision.md)）．program 全体（演算・引数・依存）を予測させる損失で，表現が答えに使われるかを確かめる介入は全文に見当たらなかった |
| 解き方の構造が一致する問題を，対照損失の正例にする | Li ら 2022（[pattern-contrast](pattern-contrast.md)）．算数文章題の式の木構造で，三つ組の損失 |

## 下書きの探索では前例が見つからなかったこと

- program の演算子列（署名）を supervised contrastive loss のラベルにし，VLM の中間層の表現に掛けること（研究文書 §3）．
- 補助損失で署名の読み取り精度が上がるのに，その層への相対感度が下がること．整合なしの群とラベルを並べ替えた群の両方と比べ，並べ替えでは下がらないこと（研究文書 §5.1，§5.3）．
- VLM で，ヤコビアンにもとづく感度で層への依存を測ること．Jacobian lens を使う研究は，下書きの探索ではテキストの言語モデルと反復型の構造のものだけが見つかった（C 節の末尾の 2 本）．

最も近い Zhang ら（cosine-misleads）と本研究の違いは，要約の「射程の外」に原文に照らして書いてある．要点は四つである．対象が挿入した latent トークンであること．補助損失が視覚標的への回帰（MSE）であること．補助損失を最初から掛けない群が見当たらないこと．学習時の標的やラベルを無作為にした群が見当たらないこと．

## A．概念や量を教師した表現が，使われるか

| 書誌 | 確かめた内容 | 読んだ範囲 |
| --- | --- | --- |
| Anita Mahinpei，Justin Clark，Isaac Lage，Finale Doshi-Velez，Weiwei Pan. "Promises and Pitfalls of Black-Box Concept Learning Models." arXiv:2106.13314 v1（2021-06-24） | 概念表現が soft な場合，学習された概念表現は定義した概念以外の情報を符号化し，下流の分類器はその情報を予測に使う．joint でない学習でも漏れは起こり，自然な緩和策でも防ぎきれない（要旨 "concept representations learned by these models encode information beyond the pre-defined concepts"，"natural mitigation strategies do not fully work"） | abs，PDF v1 の §1・§3 |
| Marton Havasi，Sonali Parbhoo，Finale Doshi-Velez. "Addressing Leakage in Concept Bottleneck Models." Advances in NeurIPS 35（NeurIPS 2022），pp. 23386–23397 | soft な概念確率を使い，sequential または joint で学習した CBM では，概念確率を通じて情報が漏れ，概念への介入も損なわれる（本文 "Leakage occurs in soft CBMs when trained sequentially or jointly"）．この文は Mahinpei らを引いた背景の文で，この論文自身の新しい結果ではない | 書誌ページ，PDF の要旨と「Leakage」の段落 |
| Chih-Ting Liao，Xin Cao. "The Objective Decides: When a Learned Dynamics Model Uses a Conserved Quantity." arXiv:2607.03728 v1（2026-07-04） | 学習済みの動力学モデルで，エネルギーなどの保存量は R²≈1 で線形に読めるのに，読み出した方向を別の状態の値で上書きしても，順伝播はほぼ変わらない（τ≈0）．訓練目的がその量に報いると，同じ方向が因果的に効くようになる（τ→+1） | abs のみ |
| Haoming Wang，Wei Gao. "Uncovering and Shaping the Latent Representation of 3D Scene Topology in Vision-Language Models." arXiv:2605.07148 v1（2026-05-08） | Dirichlet energy にもとづく潜在正則化を，単純な合成データでの 500 ステップの SFT に加えると，シーントポロジーの理解を含む空間タスクで，標準の SFT と競合手法を最大 12.1% 上回る | abs のみ |

## B．VQA で，program や推論の構造を学習に使う

| 書誌 | 確かめた内容 | 読んだ範囲 |
| --- | --- | --- |
| Yash Kant，Abhinav Moudgil，Dhruv Batra，Devi Parikh，Harsh Agrawal. "Contrast and Classify: Training Robust VQA Models." ICCV 2021，pp. 1604–1613．arXiv:2010.06087 v2 | 同じ正解を持つ標本と質問の言い換えを正例とし，言い換えの正例に高い重みを与える Scaled Supervised Contrastive Loss を，交差エントロピーと併せて最適化する．損失は，視覚と言語を融合した表現 h を射影した z = g(h) に掛ける．VQA val は 66.31（逆翻訳増強＋交差エントロピー）→ 66.98 | abs，書誌ページ，PDF v2 の §1・§3・§4.2・§4.4・§5.2・表1・表2 |
| Chenchen Jing，Yunde Jia，Yuwei Wu，Xinyu Liu，Qi Wu. "Maintaining Reasoning Consistency in Compositional Visual Question Answering." CVPR 2022，pp. 5099–5108 | GQA の正解のシーングラフと functional program から部分質問を作り，答えの予測が一貫しないことを罰する制約を足す | 書誌ページ，PDF の要旨・§3.4・§4.1 |

## C．読めることと使われることの食い違い

**VLM，または視覚モデル．** 要旨の範囲では，観察か事後の介入が中心である．Liao らは，学習した低ランクの編集も扱う．

- Runyu Zhou，Qi Zhang，Yisen Wang. "Diagnosing Visual Ignorance in Vision-Language Models." arXiv:2606.06890 v2（2026-09-27）．正答に必要な視覚証拠は中間層から既に読み出せるが，デコーダの最後の数層に至るまで出力にほとんど影響しない．（abs のみ）
- Chih-Ting Liao，Fei Shen，Xin Cao，Tat-Seng Chua. "Decodable Is Not Grounded: A Vision-Ablation Arbiter for VLM Spatial Reasoning." arXiv:2606.31257 v1（2026-06-30）．空間推論で，steering で確かめた線形プローブでも，VLM が画像に基づいているものを系統的に過大評価しうる．対照は，画像を灰色の無地に置き換えるものである．（abs のみ）
- Zeyu Wang，Xinming Xu. "Knowing Isn't Always Saying: When Do Spatial Encodings Reach Answers in Vision-Language Models?" arXiv:2608.22916 v1（2026-08-24）．abs の Comments に EMNLP 2026 Main Conference に採択とある．10 の VLM で direction patching を行い，答えの logit への因果的影響は中〜深層でのみ現れる．（abs のみ）
- Farhad Nooralahzadeh，Omid Rohanian，Yi Zhang，Jonathan Fürst，Kurt Stockinger. "Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts." arXiv:2604.09364 v2（2026-04-13）．見たものを報告できず誤答するモデルも，正答するモデルと同じ強さで視覚証拠を符号化している（著者の語で Encoding-Grounding Dissociation）．（abs のみ）
- Stephanie Fu，Tyler Bonnen，Devin Guillory，Trevor Darrell. "Hidden in plain sight: VLMs overlook their visual representations." arXiv:2506.08008 v1（2025-06-09）．深度推定や対応点などの視覚中心のベンチマークで，VLM は自身の視覚エンコーダより大幅に劣る．（abs のみ）
- Benlin Liu，Amita Kamath，Madeleine Grunde-McLaughlin，Winson Han，Ranjay Krishna. "Visual Representations inside the Language Model." arXiv:2510.04819 v1（2025-10-06）．abs の Comments に COLM 2025 に採択とある．BLINK の Art Style の問の 33.3% で，言語モデルの中にある知覚の情報が出力に現れない．（abs のみ）
- Lianghuan Huang，Yingshan Chang. "Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs." arXiv:2510.09794 v1（2025-10-10）．物体の数え上げで微調整した ViT で，中間層の物体トークンは読みにくいのに因果的影響が強く，最終層の物体トークンは正確に読めるのに機能しない．（abs のみ）
- Hongxuan Wu，Yukun Zhang，Xueqing Zhou. "How Vision Becomes Language: A Layer-wise Information-Theoretic Analysis of Multimodal Reasoning." arXiv:2602.15580 v1（2026-02-17）．LLaVA-1.5-7B と LLaVA-1.6-7B を，6 つの GQA の推論タスクで，層ごとの Partial Information Decomposition により分析する．（abs のみ）

**テキストの言語モデルほか．**

- Devesh Tiwari，Camille Davis，Shivank Sinha，Talia Weaver，Aditya Shah，Maheep Chaudhary. "Decodability is Not Causality: Dissociating Probe Readouts from Behavioral Drivers via SAE Decomposition." arXiv:2609.18080 v1（2026-09-16）．Gemma2-9B-Instruct の真偽プローブで，プローブに沿う特徴の順位と，振る舞いへの勾配感度が高い特徴の順位は，弱くしか重ならない（約 12%，Spearman ρ = 0.10）．（abs のみ）
- Fabrizio Davide，Francesco Collova. "Knowing, and Saying It Only When Asked: LLM Endognostics and the Schizognosis of Minerva-7B." arXiv:2609.22219 v1．Jacobian lens を用いる．第10層の線形プローブは精度 77% だが，その方向を直交 ablation しても回復率は 0% である（原文 "a 0% recovery rate"）．abs ページの投稿日は 2026-09-02 で，番号の並びと合わない．（abs のみ）
- Wenlong Wang，Fergal Reid. "Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?" arXiv:2609.01924 v1（2026-09-01）．Jacobian lens を反復型の構造に拡張する．（abs のみ）

## D．言語モデルや VLM の表現に，補助損失を掛けた微調整

| 書誌 | 確かめた内容 | 読んだ範囲 |
| --- | --- | --- |
| Andy Zou，Long Phan，Justin Wang，Derek Duenas，Maxwell Lin，Maksym Andriushchenko，Rowan Wang，Zico Kolter，Matt Fredrikson，Dan Hendrycks. "Improving Alignment and Robustness with Circuit Breakers." arXiv:2406.04313 v4（2024-07-12）．掲載先は確かめていない | 有害な過程に関わる表現を，元の表現と直交する方向へ最適化する（余弦類似度に ReLU を掛けた損失，LoRA）．言語モデルの実験（Mistral-7B-Instruct-v2，Llama-3-8B-Instruct）では第10層と第20層を，マルチモーダルの実験（LLaVA-NeXT-Mistral-7B）では第16層を対象にする | abs，PDF v4 の §3・Algorithm 1・付録 C.1〜C.3.1 |
| Luke Bailey，Alex Serrano，Abhay Sheshadri，Mikhail Seleznyov，Jordan Taylor，Erik Jenner，Jacob Hilton，Stephen Casper，Carlos Guestrin，Scott Emmons. "Obfuscated Activations Bypass LLM Latent-Space Defenses." arXiv:2412.09565 v2（2025-02-08） | 有害性を分類するプローブに対し，著者の攻撃は，recall を 100% から 0% に下げつつ jailbreak 率 90% を保てることが多い（"often"）．複雑な課題（SQL のコードを書く）では，難読化がモデルの性能を下げる | abs（PDF v2 は SQL の記述だけ） |
| Nihal Jain，Dejiao Zhang，Wasi Uddin Ahmad，Zijian Wang，Feng Nan，Xiaopeng Li，Ming Tan，Ramesh Nallapati，Baishakhi Ray，Parminder Bhatia，Xiaofei Ma，Bing Xiang. "ContraCLM: Contrastive Learning For Causal Language Model." ACL 2023，pp. 6436–6459．arXiv:2210.01185 v2（2023-05-02．v1 の題は "ContraGen: Effective Contrastive Learning For Causal Language Model"） | デコーダの最終の隠れ層の表現に，トークン単位と系列単位の対照損失を，因果言語モデルの損失に足して掛ける | abs（v1・v2），ACL Anthology の BibTeX，PDF v2 の §3.3・付録 A |
| Lingfang Li，Procheta Sen，Shubham Das，Danushka Bollegala. "Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models." arXiv:2609.21113 v2（2026-09-23）．abs の Comments に AACL-IJCNLP 2026 に採択とある | EAP で特定した課題に関わる成分の層ごとの分布は，微調整で表現が最も大きく変わる層と，ほぼ相関しない | abs のみ |

---

**確認**：2026-11-13．A〜D 節の 21 本について，arXiv の abs ページ（19 本．題・全著者と順・版の履歴・Comments と Journal-ref）と公式の書誌ページ（Havasi らは proceedings.neurips.cc，Kant らと Jing らは openaccess.thecvf.com，Jain らは aclanthology.org）を取得し，書誌と一文ごとの内容を照合した．21 本とも実在し，書誌は下書きと一致した．数値（12.1%，33.3%，約 12%，ρ = 0.10，77%，0%，66.31 → 66.98，100% → 0%，90%）と逐語引用は原文と一致した．下書き（2026-10-11 作成）から直した主な点：Mahinpei らの「概念に教師をかけた場合も含め」（戻せる原文が無い）を原文の言い方に，Liao & Cao の「動かない」を「ほぼ変わらない」に，Kant らの正例の定義に質問の言い換えと重みを足し，Zou らの第10層・第20層が言語モデルの実験の設定であること（マルチモーダルは第16層）を明記し，Jain らの arXiv の最新版を v2 と確かめ，C 節の「性質を学習で足してはいない」という一括の言い切りを外した．「既に報告されていること」の表は，照合済みの各要約に照らして書き直した（下書きの一覧は，Geiger ら 2022 の IIT の 93.93% を入れ替え介入の値のように書いていたが，これは振る舞いの test の値で，入れ替え介入の test は 94.85% である）．発表と執筆への含意の節は，判断であって文献の事実ではないので取り込んでいない．**読めていないもの**：「abs のみ」の 14 本の本文．Kant らの要旨が書く 0.78% がどの分割の値か．Zou らの掲載先．下書きの探索そのもの（やり直していない）．
