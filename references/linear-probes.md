# Understanding intermediate layers using linear classifier probes

**著者**：Guillaume Alain，Yoshua Bengio（Mila, University of Montreal）
**掲載**：ICLR 2017 Workshop Track（ICLR 2017 公式サイトの Workshop Poster Sessions に W13，4月24日午後として掲載）．arXiv:1610.01644 v4，2018-11-22（初版 2016-10-05）
**原文**：[arXiv](https://arxiv.org/abs/1610.01644)

**中間層ごとに，モデルから独立に学習した線形分類器（プローブ）を当てれば，各層の特徴が分類にどれだけ適しているかを，モデルの学習に影響を与えずに測れる．**

## 位置づけ

本研究は第16層の表現から署名を当てる線形分類器をプローブと呼び，その正解率を読み取り精度として報告する（研究文書 §5.1，§A.1）．本論文は，中間層の特徴に当てる線形分類器を "probes" と名づけ，モデルに影響を与えない計測器として扱った論文である．研究文書の参考文献にこの論文は挙がっていない（2026-11-13 時点）．

対応が成り立つのは「中間層の特徴に線形分類器を当て，その成績でその層を特徴づける」という型までである．原典のプローブは softmax(W h_k + b) を交差エントロピーで学習し，当てるラベルはモデルが分類するクラスそのものである．本研究のプローブは長さ1に揃えた表現に W だけを当てはめ（§A.1），当てるラベルは署名である．

## 主張

- 各層の特徴を監視し，分類にどれだけ適しているかを測るために，**線形分類器（"probes"）**を当てる．プローブは "trained entirely independently of the model itself"（要旨）．用途として，層の役割の理解と，問題の診断を挙げる（要旨）．
- 実験から，"the linear separability of features increase monotonically along the depth of the model"（要旨）．
- 単調性は学習目標が直接課したものではない：交差エントロピー損失は最終層に直接強い圧力をかけ（"applies a lot of pressure directly on the last layer"），"Any degree of linear separability in the intermediate layers happens only as a by-product."（§3.1）

## 手法

### 線形分類器プローブ

層 k の特徴 h_k から D クラスを予測する線形分類器を，交差エントロピーで学習する（§3.2）．

```
f_k : H_k → [0,1]^D,   h_k ↦ softmax(W h_k + b)
```

- 損失は L_k^train（および L_k^valid，L_k^test）．softmax 交差エントロピーによる線形分類器の学習は**凸**な問題なので，局所解の問題を避けられる（§3.2）．
- 多層のプローブは凸性を失う．著者は "as of now we feel that it is premature to start using multi-layer probes" と述べる（§6）．

### プローブはモデルを変えない

- "These probes do not affect the model training."（§3.2）
- プローブからモデルへの逆伝播は止める（`tf.stop_gradient`，あるいはプローブのパラメタをモデルのパラメタと別に管理する．§3.2）．付録 A はこれをダイオード記号で表す．
- 計測器としての比喩："We suggest that the reader think of those probes as thermometers used to measure the temperature simultaneously at many different locations."（§1）
- "we make sure that we never influence the model itself by taking measurements with probes"（§1）．

### 動機：情報量ではなく線形分離可能性を測る

- 前向きに伝わるにつれて情報は増えない——"New information is never added as we propagate forward in a model."（§3.1）．著者はこれをデータ処理不等式で示す．
- 一方で，深い層の特徴ほど線形分類器でよく予測できることを実験で観察する．著者はここから，ニューラルネットワークの要は情報量ではなく，計算上役立つ表現への蒸留にあるとする（§3.1）．プローブが測るのは，その層の特徴の線形分離可能性の度合いである（§3.2）．

### 実務上の工夫

- 大きなモデルでは特徴の次元を減らす（§3.4）：無作為な部分集合（常に同じもの．Inception v3 で使用），学習した射影（"This is probably a worse idea than it sounds"——射影行列自体が大きな記憶域を要しうるため），空間方向のプーリング（特徴数がチャネル数になる．ResNet-50 で使用）．
- 監視する量としては L_k^train より L_k^valid のほうが意味がありそうだとする（ただし実験の組み方によっては常に追えるとは限らないとも書く．§3.3）．最適解に厳密に届かなくてもよい——"we are probably going to use only a convenient approximation of the minimum."（§3.3）
- 多くの実験では，交差エントロピーではなく分類誤差を報告する（§3.3）．

## 実験設定

| 実験 | モデル・データ | プローブの置き方 |
| --- | --- | --- |
| MNIST 畳み込み網（§3.5） | tensorflow/models リポジトリの MNIST 畳み込みモデル（畳み込み2層＋全結合2層．一方は最終の分類器） | 各畳み込み・活性化関数・プーリングの前後．検証集合 10^4 件に基づく early stopping．報告はテスト予測誤差 |
| ResNet-50（§4.1） | Keras の作者のリポジトリにある ImageNet 事前学習済みモデル | input_1 と add_1〜add_16．空間プーリングで特徴数をチャネル数に縮約（§3.4）．報告は検証予測誤差 |
| Inception v3（§4.2，付録 C） | ImageNet（1000 クラス）．本文はミニバッチ 308230 時点（約2週間の学習）のみを報告 | 各プローブに無作為な 1000 特徴（挿入点によっては 426,320 特徴ある）．図の凡例は probe training (prediction) error |
| 病的な skip 接続（§5.1） | MNIST，全結合 128 層×128 ユニット，モデルの前半を完全に飛ばす長い skip 接続 | 図6(a) は入力 X と H_0〜H_127 にプローブを描く |

## 結果

### 単調性：ResNet-50 の検証予測誤差（図4(a)の表）

太字はこの要約が付けた（原表には無い）．

| 層 | 形状 | 誤差 |
| --- | --- | --- |
| input_1 | (224, 224, 3) | 0.99 |
| add_1 | (28, 28, 256) | 0.94 |
| add_2 | (28, 28, 256) | 0.89 |
| add_3 | (28, 28, 256) | 0.88 |
| add_4 | (28, 28, 512) | 0.87 |
| add_5 | (28, 28, 512) | 0.82 |
| add_6 | (28, 28, 512) | 0.79 |
| add_7 | (28, 28, 512) | 0.76 |
| add_8 | (14, 14, 1024) | **0.77** |
| add_9 | (14, 14, 1024) | 0.69 |
| add_10 | (14, 14, 1024) | 0.67 |
| add_11 | (14, 14, 1024) | 0.62 |
| add_12 | (14, 14, 1024) | 0.57 |
| add_13 | (14, 14, 1024) | 0.51 |
| add_14 | (7, 7, 2048) | 0.41 |
| add_15 | (7, 7, 2048) | 0.39 |
| add_16 | (7, 7, 2048) | **0.31** |

著者："the relationship between depth and validation prediction error is almost perfectly monotonic"（図4の説明文）．表の値は**厳密な単調ではない**——add_7→add_8 で 0.76→0.77 と一度だけ増える（この増加を著者は本文で論じていない）．著者はこの性質が通常の学習の結果として生じるもので，プローブの挿入によるものではないと添える（同）．

### MNIST：無作為初期化でも層が役立つ

学習開始時（無作為初期化，学習なし）でも，プローブのテスト予測誤差は "goes from 8% to 2% simply using those random features"（図3の説明文）．最大の効果は最初の ReLU から．学習後（10 エポック）について，著者は "the test prediction error is improving at every layer (with the exception of a minor kink on fc1_preact)" と書く．

### Inception v3

著者は図5の色の滑らかな変化を，線形分離可能性が深い層ほど単調に増すことの表れとして読む（§4.2．図の説明文では "almost perfectly monotonic"）．数値の表は無い．

### 診断：死んだ区間

skip 接続を足すと 128 層のモデルは通常の SGD で学習できるようになるが，プローブを見るとモデルの半分は使われないままである——"The weights are not zero, but there is no useful signal passing through that segment. The skip connection left a dead segment and skipped over it."（§5.1）

### 結論

"the level of linear separability increases monotonically as we go to deeper layers"．"This is purely an indirect consequence of enforcing this constraint on the last layer."（§7）

## 著者が明記した限界

- **プローブは最適とは限らない**——"we are probably going to use only a convenient approximation of the minimum."（§3.3）．直観を得る目的には許容できるとする．
- **大きなモデルでは次元削減が要る**（§3.4）．Inception v3 の無作為 1000 特徴について著者は，一部の層が不公平に有利になり，"There was no simple "fair" solution." と述べる．ただし 17 個のプローブのうち 13 個は 100,000 を超える特徴を持つので "things were relatively comparable" とも書く（付録 C）．
- **特徴が多すぎるとプローブが過学習しうる**．その場合プローブの測定は意味を失うので避けたいとし，次元削減がこの懸念に役立つとする（§3.4）．多層プローブは凸性を失う（§6）．
- **単調性の理由**は，さらに深く調べる価値があるとする（§6）．

## 著者が述べる含意

- 新しい方向として，"One completely new direction would be to train a model in a way that actively discourages certain internal layers to be useful to linear classifiers."（§6）．著者は，そのときプローブはもはや非侵襲ではなくなると添える．
- 他の構造（ResNet-101，ResNet-151，VGG-16，VGG-19，RNN）や，GAN・敵対的事例が使われる設定への適用を挙げる（§6）．
- 別のラベルへの適用：プローブを別のラベル集合に使うことは可能だと述べる——"Note that it would be entirely possible to use linear classifier probes on a different set of labels."（§3.6）．実験はしていない．

## 射程の外

- **実験の対象は画像分類**（MNIST・ImageNet の畳み込み網と，MNIST の全結合網）で，プローブが当てるラベルはモデルが分類するクラスである（§1）．言語モデル・VLM の実験は無い（全文で language，VLM，multimodal を検索し，該当なし）．モデルの出力と異なるラベルを中間層から読むことは，可能だと述べるにとどまる（§1，§3.6）．
- **線形分離可能性が深さとともに単調に増す**ことを，著者は最終層にかかる圧力の副産物として説明する（§3.1，§7）．中間層の表現を線形に読み取りやすくする損失を中間層に掛ける実験は無い（Inception v3 は学習に寄与する補助分岐を持ち，著者はその予測誤差を主出力と比べるが，これはモデルに元からある分岐である．著者が将来の方向として挙げるのは逆向き——中間層を線形分類に役立たなくする学習——である）．
- プローブで読めることと，その特徴がモデルの出力に使われることとを区別する議論は見当たらなかった（検索した語：caus，rely，relies，unused）．§5.1 は，プローブで有用な信号が読めない区間を "unused" と読んでいる．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴 v1〜v4．Comments と Journal-ref の欄は無い），PDF v4（13頁，付録 A〜C を含む全文．pdftotext でテキスト化し，6・7・9頁は画像でも確認），ICLR 2017 公式サイトの Workshop Poster Sessions のページ（iclr.cc/archive）を読み，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．下書き（2026-10-11 作成）から直した主な点：掲載の根拠を公式サイトの一覧に替え，確かめられなかった workshop 版についての記述（4頁，arXiv 版との内容の違い）と OpenReview へのリンクを外した．「位置づけ」を研究文書の記述に合わせ，「出典」「語が遡る」を外した．「最終層だけに圧力」を原文の強さに，学習した射影の評価と MNIST の学習後の記述を原文の引用に直した．「別のラベルは扱わない」を，原文が可能性を述べている（§1，§3.6）ことに合わせて直した．二つに割れていた温度計の引用を一続きに戻した．**読めていないもの**：OpenReview の workshop 版（forum ページ・PDF とも閲覧確認で取得できず）．arXiv v1〜v3（v4 との差は見ていない）．図3・図5・図6・図9 の図中の値（説明文と本文に書かれたものだけを照合した）．
