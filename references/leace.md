# LEACE: Perfect linear concept erasure in closed form

**著者**：Nora Belrose（EleutherAI），David Schneider-Joseph（EleutherAI），Shauli Ravfogel（Bar-Ilan University），Ryan Cotterell（ETH Zürich），Edward Raff（Booz Allen Hamilton），Stella Biderman（EleutherAI，Booz Allen Hamilton）
**掲載**：Advances in Neural Information Processing Systems 36（NeurIPS 2023，Main Conference Track），pp. 66044–66063．arXiv:2306.03819 v4，2025-04-03（初版 v1 は 2023-06-06）
**原文**：[arXiv](https://arxiv.org/abs/2306.03819)・[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)

**分類課題が線形に守られるのは，どのクラスの平均も等しいとき，かつそのときに限る．その条件を満たすアフィン変換のうち，元の特徴からの平均二乗距離を最小にするものが閉形式で求まる．**

## 主張

LEACE（LEAst-squares Concept Erasure）は "a closed-form method which provably prevents all linear classifiers from detecting a concept" であり，しかも埋め込みの変化を，広いクラスのノルムで測ってできるだけ小さくする（要旨）．さらに，ネットワークの全層から対象の概念の情報を消す **concept scrubbing** を提案し，言語モデルが品詞情報にどれだけ頼っているかの測定と，BERT 埋め込みの性別バイアスの低減に適用する（要旨）．

以下の節番号・数値・引用は arXiv v4 による．v4 は，NeurIPS 版・v3 が "representation" と書く箇所の多くを "embedding" と書く．本文書の引用にその語は含まれない．

## 手法

### 設定と定義（§2）

$X \in \mathbb{R}^d$ は一次モーメントが有限な確率ベクトル（入力データ），$Z$ は $k$ クラスの one-hot ラベル（各クラスの確率は正）である．

| 定義 | 内容 |
| --- | --- |
| 2.1 guardedness | $X$ が，予測器のクラス $V$ と損失のクラスについて $Z$ を守る：そのクラスのどの損失についても，$X$ が「最良の予測器の期待損失」を最大にする．言い換えると，条件付き分布 $P(X \mid Z)$ が，$X$ から $Z$ を予測するうえで最悪の分布の一つである |
| 2.2 trivially attainable loss | 定数予測器で得られる最小の期待損失 |
| 2.3 linear guardedness | $V$ を線形予測器 $\eta(x) = b + Wx$（$b \in \mathbb{R}^k$，$W \in \mathbb{R}^{k \times d}$），損失のクラスを第一引数について凸な非負の損失とした guardedness |

### 線形消去の必要十分条件（§3）

論文は次の5条件が同値であると述べる．

1. $X$ が $Z$ を線形に守る．
2. すべての凸な損失について，trivially attainable loss が $(X, Z)$ 上で最適である．
3. **クラス条件付き平均が全体平均に等しい**：$\mathbb{E}[X \mid Z = i] = \mathbb{E}[X]$．
4. $X$ のどの成分も $Z$ のどの成分とも共分散が 0 である（相互共分散行列 $\Sigma_{XZ}$ が零行列）．
5. $X$ 上のどの線形分類器も，$Z$ に関して statistical parity を満たす．

| 定理 | 向き | 内容 |
| --- | --- | --- |
| 3.1 | 3 → 2 | クラス条件付き平均がどれも全体平均に等しければ，線形予測について凸な損失では trivially attainable loss を下回れない（Jensen の不等式による） |
| 補題 3.2・定理 3.3 | 2 → 3 | 補題 3.2：損失の偏導関数が有界で，正解でないクラスについての偏導関数が消えず，クラスに依らないとき，$W = 0$ の定数予測器が線形予測器のなかで期待損失を最小にするなら，クラス条件付き平均は全体平均に等しい．定理 3.3 は，交差エントロピーがこの条件を満たすことを示す |
| 3.4 | 3 ⇔ 4 | クラス条件付き平均の一致は $\Sigma_{XZ} = 0$ と同値 |

本文は，条件 1・2・5 の同値は付録 B・C にあると述べる．付録 B が示すのは 1 と 2 の同値（凸な損失・線形予測器の仮定なし），付録 C が示すのは 3 と 5 の同値である（定理 C.2）．必要の向き（2 → 3）は補題 3.2 の条件を満たす損失について証明されており，本文がその条件を確かめているのは交差エントロピーである．十分の向き（定理 3.1）は凸な損失について成り立つ．

### 最小編集の閉形式（§4）

**定理 4.1.** アフィン写像 $r(x) = Px + b$ について，$r(X)$ が $Z$ を線形に守るのは，$\Sigma_{XZ}$ の列がすべて $P$ の零空間に含まれるとき，かつそのときに限る（$\mathrm{Cov}(r(X), Z) = P\Sigma_{XZ}$ による）．

**定理 4.2.** 中心化した $X, Z$（二次モーメント有限）と，半正定値行列 $M$ が定める（退化しうる）内積 $\langle x, y \rangle_M = x^{\top} M y$ について，

$$
\min_P \ \mathbb{E}\lVert PX - X \rVert_M^2 \quad \text{s.t.} \quad \mathrm{Cov}(PX, Z) = 0
$$

の解は

$$
P^{*} = I - W^{+} P_{W\Sigma_{XZ}} W,
\qquad
W = (\Sigma_{XX}^{1/2})^{+},
\qquad
P_{W\Sigma_{XZ}} = (W\Sigma_{XZ})(W\Sigma_{XZ})^{+}
$$

である．$W$ は白色化の変換，$P_{W\Sigma_{XZ}}$ は $W\Sigma_{XZ}$ の列空間への直交射影行列，$^{+}$ は Moore–Penrose 擬似逆行列である．**一つの消去関数が，こうしたノルムのすべてを同時に最小にする**（§4.1）．付録 E.1 は，解が一意でないのは，分散が 0 の $X$ の成分に対応する列と，$M$ の重みが 0 の成分に対応する行だけだと述べる．

**定理 4.3**（中心化しない場合）はバイアスを $b^{*} = \mathbb{E}[X] - P^{*}\mathbb{E}[X]$ とする．まとめると

$$
r_{\mathrm{LEACE}}(x) = x - W^{+} P_{W\Sigma_{XZ}} W \bigl(x - \mathbb{E}[X]\bigr)
$$

——平均を引いて白色化し，$X$ と $Z$ の相関を担う部分空間へ射影し，白色化を戻して，その値を $x$ から引く．

**消える次元．** 論文は消える次元の一般式を書かない．付録 E.1 の証明は $\mathrm{rank}(\Sigma_{XZ})$ を $\ell$ と置き，白色化した $X$ のうち $Z$ と相関する成分を $\ell$ 個とする．§6 の対照（ランダム消去）も「相互共分散行列と同じ階数」の部分空間を使う．実験では，18 種の品詞で階数が 17 減る（§5.3．v1・v2 は 18 と書き，v3・NeurIPS 版・v4 は 17 と書く）．二値の性別については，v4 の末尾に付いた頁の図2の説明文が "Our method is rank 1" と書く．なお，one-hot の $Z$ では成分の和が定数なので $\Sigma_{XZ}$ の階数は高々 $k - 1$ になるが，これは原典の記述ではなく，要約者が導いたものである．

### 斜交射影が最小二乗で最適（§4.2）

$P^{*}$ は2回掛けても1回と同じ（$(P^{*})^2 = P^{*}$）だが，$r_{\mathrm{LEACE}}$ は一般には直交射影ではなく，**斜交射影**である．$I - W^{+} P_{W\Sigma_{XZ}} W$ が対称になるのは，$P_{W\Sigma_{XZ}}$ と $W$ が可換な特別な場合だけである．線形に守るための必要十分条件 $P\Sigma_{XZ} = 0$ は $P$ の零空間への制約であって値域への制約ではないので，値域は平均二乗距離が最小になるよう自由に選べる．そのため，直交射影は一般には最小二乗の意味で最適でない．論文は，先行研究が消去関数を直交射影とすべきだと仮定してきたと述べ（§4.2），RLACE（Ravfogel ら）について "While they seek an orthogonal projection, we empirically showed that their solution is very close to ours." と書く（付録 A）．付録 F の例（$d = 2$，$k = 1$）では，平均二乗の編集距離は直交射影で 2，斜交射影で 1 である．

**連続な $Z$（§4.3）.** 損失を通常の最小二乗 $\lVert \eta - z \rVert_2^2$ に限れば，$Z$ が $\mathbb{R}^k$ の任意の値を取る場合にも同じアフィン消去器が得られる（"While not a focus of this work" と断る）．付録 I は，各 $x$ のラベル $z$ を使える場合の Oracle LEACE を与える——$X$ を $Z$ へ回帰した最小二乗の残差（を適切にずらしたもの）である．

### concept scrubbing（§6）

1層だけでなく，すべての層の活性から，概念についての線形な情報を消す（Algorithm 1）．

1. 入力を埋め込み，$H_1$ を得る．
2. 各層 $l$ について順に：$H_l$ と $Z$ から LEACE のパラメタ $(P, b)$ を当てはめて保存し，$H_l$ を消去してから層 $f_l$ に通して $H_{l+1}$ を得る．
3. 全層のパラメタを返す．

消去を掛けるのは，各 transformer ブロックの入力の，正規化（LayerNorm または RMSNorm）の直後である（§6.1）．**順に当てはめる**——ある層での介入は後続の層の隠れ状態の分布を変えるので，介入しないモデルで全層を独立に当てはめて一度に掛けると，対象の概念を消し切れないことがある．各層の隠れ状態をキャッシュするやり方は，学習時に大きなメモリまたはディスク（実験では最大 500GB）を要する．キャッシュせずに再計算すれば，総計算量は層数 $\ell$ について $O(\ell)$ から $O(\ell^2)$ に増える．

## 実験設定

| 節 | モデル | データ | 消す概念 | 比較対象 |
| --- | --- | --- | --- | --- |
| §5.1 内在評価 | BERT（凍結），最終層の [CLS] 埋め込み | De-Arteaga らの経歴データ（二値の性別と職業の注釈つき） | 二値の性別 | INLP，RLACE（図2） |
| §5.2 下流の公平性 | 同上＋職業を予測するロジスティック回帰 | 同上 | 二値の性別 | 元の埋め込み |
| §5.3 amnesic probing の再検証 | bert-base-uncased（1回の実験で介入するのは1層） | English Universal Dependencies（粗い品詞 18 種） | 品詞 | INLP（20 反復），零空間の階数がラベル空間と同じ（18）のランダムな直交射影 |
| §6 concept scrubbing | Pythia 160M・1.4B・6.9B・12B，LLaMA 7B・13B・30B | Pile の validation split（Pythia），LLaMA 事前学習コーパスの RedPajama による再現（LLaMA）．当てはめ用と評価用に $2^{22}$ トークンずつ | 品詞（spaCy のモデルで自動付与した Universal Dependency タグ） | ランダム消去，full-rank SAL |

## 結果

### 内在評価と公平性（§5.1・§5.2）

消した情報をロジスティック回帰で読み戻せるかを見る．"our method is the only to achieve random accuracy (perfect erasure) with a small edit"．RLACE は迫るが INLP は迫らない（"although RLACE (but not INLP) comes close"）．論文は LEACE が約2桁速く，勾配による最適化を要しないと述べる（何に対して2桁かは，その文に明記がない）．図2の個々の点の値は，本文に数値がない．

| 職業の分類（[CLS]，性別を消去） | 元の埋め込み | 消去後の埋め込み |
| --- | --- | --- |
| 職業の正解率 | 79.3% | 77.3% |
| TPR gap | 0.198 | 0.084 |
| 職業ごとの TPR gap と，その職業の女性の割合の相関 | 0.867 | 0.392 |

消去後の埋め込みで主課題の分類器を当てはめ直すと，正解率は 78.1% とわずかに上がり，TPR gap は 0.158 に増える．

### amnesic probing の再検証（§5.3）

| | 減る階数 |
| --- | --- |
| INLP（20 反復） | 360 |
| LEACE | 17 |

INLP を 20 反復するのは，INLP が概念を十分に消さないからだと論文は述べる——"even after 20 iterations, classification accuracy is above majority accuracy."

LEACE では BERT の言語モデルの精度と損失は第8層まで穏やかにしか変わらず，最大の低下は第11層に出る．INLP の効果は第6層で最大だが，"Since it removes hundreds of dimensions, it is difficult to attribute this effect to the erasure of the concept." 著者はこれを "These results suggest that the causal effect of the POS concept on the language model is concentrated in layer 11." と読む．

arXiv v4 の PDF の末尾には，本文と別に番号を振った1頁が付く（NeurIPS 版と v3 には無い）．その図3の説明文は，投稿した版の図3が "No Intervention" の MLM 精度を 0.928 でなく 0.892 と誤って載せ，そのためランダムな部分空間を除いたほうが BERT の MLM 精度がよいかのように報告していた，と述べる．

### concept scrubbing（§6.2，表1．自己回帰言語モデルのパープレキシティ．単位は bits per UTF-8 byte）

| 条件 | LLaMA 7B | 13B | 30B | Pythia 160M | 1.4B | 6.9B | 12B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No intervention | 0.69 | 0.66 | 0.62 | 0.90 | 0.70 | 0.64 | 0.62 |
| Random erasure | 0.69 | 0.66 | 0.62 | 0.99 | 0.72 | 0.66 | 0.63 |
| **LEACE** | **1.73** | **1.84** | **1.96** | **2.79** | **2.25** | **3.57** | **3.20** |
| SAL | 3.24 | 3.26 | 3.16 | 3.53 | 3.44 | 4.17 | 4.69 |
| unigram entropy | 2.90 | 2.90 | 2.90 | 2.66 | 2.66 | 2.66 | 2.66 |

ランダム消去は，相互共分散行列と同じ階数のランダムな線形部分空間へ直交射影する対照である（ミニバッチごとに部分空間を引き直す）．論文は，ランダムな部分空間を消しても言語モデルの性能はほとんど変わらず，品詞情報を消すとすべてのモデルでパープレキシティが大きく増えると述べ，これを "We find strong evidence that autoregressive language models heavily rely on linearly encoded part-of-speech information." と読む．SAL（Shao ら）はバイアス項を持たず，特徴間の相関を補正しない点で LEACE と異なる．SAL は試したすべてのモデルでパープレキシティをより大きく増やし，著者はこれを，SAL のほうが他の有用な特徴への巻き添えが大きいためと読む．INLP については "INLP often requires the deletion of several dozen dimensions before linear guarding is achieved" とし，巻き添えのために複数の層へは掛けられず，Elazar らが1層の介入にとどまったのは INLP の限界のためだと述べる（§6）．

## 著者が明記した限界

- **線形に限る**："In this work, we focused exclusively on linear concept erasure due to its simplicity and tractability."
- **非線形の消去についての予想**："We conjecture that it is intractable to nondestructively edit X so as to prevent a general nonlinear adversary from recovering Z, unless the data generating process for X is known in detail."
- **concept scrubbing の検証は途上**："Much work remains to be done to validate concept scrubbing." 続けて "we'd like to see experiments that target concepts much narrower than part-of-speech, and use behavioral metrics to determine whether scrubbing changes the network in the ways we'd intuitively expect" と書く．
- **学習に組み込んだ場合の回避**：そうした実験がうまくいけば concept scrubbing を事前学習や finetuning に組み込むのが次の一歩だとしたうえで，勾配による最適化器が，守る属性をまったく非線形な形で符号化して制約を「回避」できるかは今後を見なければ分からない，とする．
- concept scrubbing が実用で満足な結果を出さないなら，学習の手続きを変えるようなモデル固有の手法を改善するほうが有望かもしれない，とする．

## 位置づけ

研究文書（`.claude/context.md`）は LEACE に触れていない．本研究は性質を除く介入をしていない（研究文書 §A.4）ので，本研究は LEACE を使っていない．関わるのは，研究文書が引く amnesic probing（Elazar ら．§A.4，→ [amnesic-probing.md](amnesic-probing.md)）の側である：本論文は §5.3 で，amnesic probing の実験を INLP の代わりに LEACE でやり直し，INLP は 20 反復で階数を 360 減らすのに対し LEACE は 17 しか減らさないと報告する．

## 射程の外

- 実験の対象は**言語モデル**（BERT・Pythia・LLaMA）である．全文に VLM・GQA・質問応答の語は見当たらなかった（検索した語：GQA，VLM，multimodal，image，question）．
- **学習済みの表現に事後に掛ける消去**である．著者は LEACE を，finetuning もパラメタの検査もせずに入力や隠れ状態を編集する手法として述べ（§1），学習時に掛ける敵対的な手法とは区別する（付録 A）．実験で消去を掛けるのは，公開の学習済みモデル（BERT・Pythia・LLaMA）の表現である．
- 消す概念は二値の性別と品詞である．
- 保証は**線形な**予測器に対するものに限る．非線形については上の予想を述べるにとどまる．
- 測った量は，性別の読み戻しの精度，職業分類の正解率と TPR gap，BERT の MLM の精度と損失，自己回帰言語モデルのパープレキシティである．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴 v1〜v4・要旨），arXiv PDF v4（22頁．本文10頁，参考文献，付録 A〜J，末尾の訂正の1頁），NeurIPS Proceedings の書誌ページ（巻・頁・DOI）と同 PDF（20頁），arXiv PDF v1・v2・v3 を取得した．v4 は pdftotext で全文を読み，崩れた箇所（§4.1・§4.2 の式，表1，$2^{22}$，図4，末尾の頁）はページ画像で確かめて，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．NeurIPS 版と v1〜v3 は，本文書の数値と逐語引用が版で変わっていないかを検索で確かめた（変わっていたのは LEACE が減らす階数の 18→17 だけ）．下書き（2026-10-11 作成）から直した主な点：太字の一文と定義 2.1 を原文の言い方へ／条件 1・2・5 の同値について付録 B・C が実際に示す範囲を明記／「直交射影になるのは可換な場合に限る」を「対称になるのは」へ／「平均は保たれる」「低いほどよい」を削除／「RLACE より約2桁速い」の比較相手を原文どおり不明記に／ランダム消去からの帰結を著者の読みとして引用に／"…" で切っていた引用を全文に／勾配による回避を学習へ組み込む文脈に／位置づけの「道具の候補」を研究文書に合わせて削除／「後段の層が非線形に復元することは保証の外（著者が明記）」と「損失で構造を作り込んだ表現は扱わない」を削除／著者の所属を追加．**読めていないもの**：付録 E の二つの証明と付録 I の証明は，式の導出を一行ずつは検算していない（定理の主張と結論だけを照合した）．図1〜図4の点の値は読み取っていない．リンク先の inlp.md は，照合の時点でリポジトリの references/ に無い．
