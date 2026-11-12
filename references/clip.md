# Learning Transferable Visual Models From Natural Language Supervision

**著者**：Alec Radford，Jong Wook Kim，Chris Hallacy，Aditya Ramesh，Gabriel Goh，Sandhini Agarwal，Girish Sastry，Amanda Askell，Pamela Mishkin，Jack Clark，Gretchen Krueger，Ilya Sutskever（全員 OpenAI．Radford と Kim は equal contribution）
**掲載**：ICML 2021（Proceedings of the 38th International Conference on Machine Learning，PMLR 139:8748–8763）．arXiv:2103.00020 v1，2021-02-26（版は v1 のみ）．**本ノートの節番号・逐語引用・数値は arXiv v1（48頁）のものである**——PMLR の本文（16頁）は短縮版で，節番号が異なり，下の引用の一部を含まない（末尾の確認行を参照）
**原文**：[PMLR](https://proceedings.mlr.press/v139/radford21a.html)・[arXiv](https://arxiv.org/abs/2103.00020)

**バッチ内の N×N 組のうちどの画像–テキスト対が実際に起きたかを当てる対称な対照損失で，4億対から転移可能な視覚表現を学ぶ．**

## 位置づけ

本研究の整合損失は，温度 τ を学習可能なパラメタとし，初期値を 0.07 とする（研究文書 §3.2）．学習可能な温度と，0.07 相当という初期値は，本論文（CLIP）の設定にもある．ただし研究文書はこの形を「こちらで決めた」と書いており，CLIP を出どころとして引いてはいない．本ノートは対照損失の実装上の選択に絞る．

## 主張

- どのキャプションがどの画像に対応するかを予測する事前学習の後，自然言語で学習済みの視覚概念を参照する（または新しい概念を記述する）ことで，下流タスクへ zero-shot 転移できる（要旨）．
- bag-of-words 符号化を予測する基準線の予測目的を対照目的に替えると，ImageNet への zero-shot 転移の効率がさらに **4 倍**改善した——"observed a further 4x efficiency improvement in the rate of zero-shot transfer to ImageNet"（2.3 節）．
- 最良の CLIP モデルは ImageNet の zero-shot 精度を **11.5%（Visual N-Grams）から 76.2%** に上げ，元の ResNet-50 の性能に並ぶ（3.1.3 節，Table 1）．

## 手法

### 対照目的を選んだ理由（2.3 節）

最初は画像のキャプションを予測する目的を試したが，効率よく規模を広げるのが難しかった（6300万パラメタの Transformer 言語モデルは，同じテキストの bag-of-words 符号化を予測する基準線より，ImageNet のクラス認識の学習が 3 倍遅い）．著者は，画像の対照表現学習で "contrastive objectives can learn better representations than their equivalent predictive objective"（Tian ら 2019）と報告されていることなどを踏まえ，テキストの正確な語ではなく，どのテキストが全体としてどの画像と対になるかだけを予測する課題を試した．

### 損失：対称 InfoNCE（2.3 節）

N 対のバッチで "predict which of the N × N possible (image, text) pairings across a batch actually occurred"．実在する N 対の埋め込みの cosine 類似度を最大化し，誤った N²−N 対の類似度を最小化する．

**"We optimize a symmetric cross entropy loss over these similarity scores."** 著者の知るかぎり，このバッチ構成と目的は Sohn（2016）の multi-class N-pair loss として最初に導入され，"was popularized for contrastive representation learning by Oord et al. (2018) as the InfoNCE loss"．

Figure 3 の擬似コード（抜粋）：

```python
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# t - learned temperature parameter
logits = np.dot(I_e, T_e.T) * np.exp(t)
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

**二つの軸方向の交差エントロピーを平均する**．ラベルは `np.arange(n)` なので，正例は各行・各列に1つ（対角）である．

### 温度（2.3 節・2.5 節）

| 項目 | 設定 | 原文 |
| --- | --- | --- |
| 学習可能 | 訓練中に直接最適化．ハイパーパラメタとして調整しない | "to avoid turning as a hyper-parameter"（原文の綴りのまま．2.3 節） |
| パラメタ化 | **log でパラメタ化した乗算スカラー**（擬似コードでは logits に `np.exp(t)` を掛ける） | "log-parameterized multiplicative scalar"（2.3 節） |
| 初期値 | **0.07 相当**（Wu ら 2018） | "the equivalent of 0.07 from (Wu et al., 2018)"（2.5 節） |
| 上限 | **logits の倍率が 100 を超えないよう clip** | "clipped to prevent scaling the logits by more than 100"（2.5 節） |

換算すると，初期の倍率は 1/0.07 ≈ 14.3，上限 100 は τ ≥ 0.01 にあたる（この二つは本ノートの換算で，原文の数値ではない）．**clip は学習の不安定を防ぐために必要だった**——"which we found necessary to prevent training instability."　付録 F の Table 18 は，この上限を "Maximum temperature 100.0" と表記している．

### 射影：線形のみ（2.3 節・2.4 節）

- 表現と対照埋め込み空間のあいだの非線形射影（Bachman ら 2019 が導入し，Chen ら 2020b が広めたと著者が書くもの）を**使わない**——"do not use the non-linear projection between the representation and the contrastive embedding space"．
- **各エンコーダの表現から multi-modal 埋め込み空間へは線形射影だけ**（擬似コードの `W_i`，`W_t`）．擬似コードでは射影後に L2 正規化する．
- **線形と非線形で学習効率に差は認められなかった**——"We did not notice a difference in training efficiency between the two versions"．著者は，非線形射影が "may be co-adapted with details of current image only in self-supervised representation learning methods" と推測する．
- テキスト側は Transformer 最上層の [EOS] トークンの活性を特徴とし，layer norm の後に線形射影する（2.4 節）．

## 実験設定

- **データ**：インターネット上の公開の出どころから集めた **4 億**の（画像，テキスト）対（2.2 節）．
- **モデル**：ResNet-50，ResNet-101，RN50x4，RN50x16，RN50x64 と ViT-B/32，ViT-B/16，ViT-L/14．ViT-L/14 は 336 px でさらに 1 エポック事前学習（ViT-L/14@336px）（2.5 節）．
- **初期化**：画像エンコーダを ImageNet の重みで，テキストエンコーダを事前学習の重みで初期化せず，**ゼロから学習**（2.3 節）．
- **増強**：リサイズした画像からの無作為な正方形切り出しのみ（2.3 節）．
- **バッチ**："We use a very large minibatch size of 32,768."
- **最適化**：Adam（gain と bias 以外の全重みに decoupled weight decay），cosine schedule で学習率を減衰，**32 エポック**．
- **大バッチの工夫**：mixed precision，gradient checkpointing，Adam 統計量の半精度，テキストエンコーダの重みの半精度（確率的丸め），埋め込み類似度の計算を GPU ごとに分割（sharded）．
- **計算量**：RN50x64 は V100 592 枚で 18 日，最大の ViT は V100 256 枚で 12 日．

## 結果

| 項目 | 値 |
| --- | --- |
| 予測目的から対照目的への切替による zero-shot 転移の効率改善（2.3 節，Figure 2） | **4 倍** |
| ImageNet zero-shot 精度（最良の CLIP モデル．3.1.3 節，Table 1） | **76.2%**（Visual N-Grams の 11.5% から） |
| MNIST（手書き数字）の zero-shot 精度（6 節） | 88% |

本ノートでは 30 超のデータセットでの zero-shot・linear probe の詳細な結果は扱わない．

## 著者が明記した限界

いずれも 6 節（Limitations）から．

- **計算量**："we estimate around a 1000x increase in compute is required for zero-shot CLIP to reach overall state-of-the-art performance."
- **分布外への汎化**："CLIP only achieves 88% accuracy on the handwritten digits of MNIST."
- **データ効率**："CLIP also does not address the poor data efficiency of deep learning."

## 射程の外

- **対象は画像–テキスト対の事前学習**である．事前学習済みモデルの微調整中に対照損失を補助損失として掛ける設定は，本文に見当たらなかった（検索した語：auxiliary，fine-tun）．"auxiliary" は，テキストエンコーダに言語モデリングを補助目的として加える余地を今後の課題とする一文（2.4 節）にだけ現れる．
- **擬似コードの正例は，対になった画像と文の1対だけ**である．同じラベルを共有する複数の正例を使う教師あり対照学習への言及は見当たらなかった（検索した語：supervised contrastive，SupCon，Khosla，multiple positive，same label．Khosla は ImageNet の文献の共著者としてだけ現れる）．
- **損失は線形射影・L2 正規化後の埋め込みに掛けられる**（Figure 3）．言語モデルの中間層の hidden state に直接掛ける設計への言及は見当たらなかった（検索した語：intermediate，hidden．hidden は参考文献の題に1件だけ）．

---

**確認**：2026-11-13．arXiv abs ページ（版は v1 のみ，2021-02-26），arXiv PDF v1（48頁，付録 A–F を含む）を pdftotext でテキスト化したもの，PMLR の公式書誌ページ（題・著者・巻・頁）を取得し，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．通読したのは 1 節・2 節・3.1.1–3.1.3 節・6 節・付録 F で，3.1.4 節以降の評価・7–9 節・付録 A–E は通読せず，全文への語の検索（上の「検索した語」と temperature，0.07，4x など）で当たった．Table 1 と Table 18 はページ画像で値を確かめた．PMLR の本文 PDF（16頁）は主な引用の有無だけを検索で見た：PMLR 版は短縮されており，節番号が異なる（対照目的の節は 2.2），"InfoNCE" の一文・"0.07"・"32,768"・MNIST 88% の文・データ効率の文は本文に無い．下書き（2026-10-11 作成）から直した主な点：位置づけの「本研究の温度設定（log スケール・上限 100 を含む）は CLIP 式である」を，研究文書 §3.2 に在る事実（学習可能・初期値 0.07・こちらで決めた）に限った．節番号・引用が arXiv v1 のものであることを掲載行に明記した．「Tian らを根拠に選ばれた」を原文の強さ（先行の報告を踏まえて試した）に，「学習が遅い」を「効率よく規模を広げるのが難しかった」に，「画像→テキストとテキスト→画像」を擬似コードどおり「二つの軸方向」に，11.5% の出どころを Visual N-Grams に直した．射程の外の否定3件を「見当たらなかった（検索した語）」の強さに直した．Table 18 の "Maximum temperature" の表記を足した．**読めていないもの**：PMLR の補足資料（supplementary）．arXiv v1 の 3.1.4 節以降・7–9 節・付録 A–E は通読していない（語の検索のみ）．Figure 2 の値は本文とキャプションに書かれたものだけを照合した．
