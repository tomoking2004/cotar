# Explorations of Self-Repair in Language Models

**著者**：Cody Rushing（テキサス大学オースティン校），Neel Nanda（論文に所属の記載なし）
**掲載**：Proceedings of the 41st International Conference on Machine Learning（ICML 2024），PMLR 235，pp. 42836–42855，2024．arXiv:2402.15390 v1 2024-02-23／v2 2024-05-26
**原文**：[arXiv](https://arxiv.org/abs/2402.15390)・[PMLR](https://proceedings.mlr.press/v235/rushing24a.html)・[OpenReview](https://openreview.net/forum?id=5ZwEifshyo)

**attention head を1本ずつ除くと後段が直接効果の一部を埋め戻す自己修復は，学習分布全体で複数のモデルに起きる——ただし不完全で，プロンプトごとにばらつき，その一部は最終 LayerNorm の尺度の変化という副作用で説明される．**

## 位置づけ

既存ノート [hydra-effect.md](hydra-effect.md)（McGrath ら）の続編にあたる．本論文の序論によれば，McGrath らが自己修復を調べたのは Counterfact データセットの上，層全体の除去に限られていた．本論文はこれを**個々の head・事前学習の分布全体・複数モデル**へ広げ，自己修復が不完全で noisy であることと，その一部が LayerNorm の尺度の変化から生じることを示した．McGrath らが述べた MLP Erasure は，本論文では疎な Erasure ニューロンの単位で調べられ，上流の head を除くとそのニューロンの打ち消しが弱まることが，Anti-Erasure として自己修復に現れる．

本研究（研究文書 §A.4）が「壊して出力の変化を見る」方法を採らない根拠として引くのは McGrath らであり，研究文書は本論文を引いていない．本論文は，除去が部品の役割を測る道具として信頼できないこと（§5.1）と，除去がモデルを分布の外へ連れ出すこと（§5.2）を，head 単位・事前学習の分布全体で述べており，同じ向きの材料になる．ただし本論文の対象はテキストの言語モデルの attention head と次トークンの logit であり，本研究のモデル・層・量にそのまま当てはまるかは，本論文からは言えない．

## 主張

先行研究の自己修復（ある部品を除くと後段の部品が振る舞いを変えて補う現象．McGrath らの Hydra effect など）は，狭いデータ分布や，より大きな部品の除去で示されてきた（§6）．本論文は，**個々の attention head を除いたときにも，事前学習の分布全体で，複数のモデル系列・規模で自己修復が在る**ことを示す．そのうえで次の三点を主張する．

1. **自己修復は不完全である**——要旨の言葉では "self-repair is imperfect, as the original direct effect of the head is not fully restored"．
2. **自己修復は noisy である**——修復の度合いはプロンプトによって大きく異なり，元の効果を超えて補正しすぎることもある（要旨）．
3. **寄与する機序として二つを挙げる**——要旨の言葉では "changes in the final LayerNorm scaling factor" と "sparse sets of neurons implementing Anti-Erasure"．

さらに，自己修復を予測する枠組みとして Iterative Inference 仮説を論じる（要旨："a framework that predicts self-repair"）．

## 手法

### 直接効果と自己修復の定義（§2.1）

最終残差ストリームは各部品（attention 層・head・MLP 層・ニューロン・埋め込み）の出力の和 $r_k = \sum_i C^i_k$ に分解できる．logit は正規化 LN（LayerNorm または RMSNorm）を通して unembedding $W_U$ を掛けたもので，平均が 0 になるよう中心化して扱う．LN を線形近似すると，正解の次トークン $x_{k+1}$ の logit は $r_k$ について線形になり，部品 $i$ の**直接効果**（direct effect，DE）を

$$
\mathrm{DE}^i_k = W[x_{k+1}]^{\top} C^i_k
$$

と定める——後段の部品に使われる中間表現を作る働きは含めず，その部品が正解の次トークンを直接押し上げる分だけを測る．

除去は **resample ablation**：head の出力を，事前学習分布から引いた別のトークン列での出力に差し替える．自己修復は，正解 logit の変化 $\Delta\mathrm{logit} = \mathrm{logit}_{\mathrm{ablated}} - \mathrm{logit}_{\mathrm{clean}}$ と，head 自身の直接効果の変化 $\Delta\mathrm{DE}_{\mathrm{head}}$ の差で定める（式 1）：

$$
\text{self-repair} = \Delta\mathrm{logit} - \Delta\mathrm{DE}_{\mathrm{head}}
$$

resample ablation の後の直接効果はほぼ 0 なので，ほとんどの head で $\Delta\mathrm{DE}_{\mathrm{head}} \approx -\mathrm{DE}_{\mathrm{head}}$ である．論文の例では，DE = 0.5 の head を除いて logit 差が −0.2 にとどまれば，自己修復は $-0.2-(-0.5)=0.3$ と近似できる．自己修復されない head は $\Delta\mathrm{logit} \approx -\mathrm{DE}_{\mathrm{head}}$ となり，自己修復される head は $-1 < \Delta\mathrm{logit}/\mathrm{DE}_{\mathrm{head}} < 0$，Figure 2 では $y=-x$ の線と横軸のあいだに入る（§2.2）．元の直接効果が正で自己修復が負になる場合は，自己修復ではなく下流の breakage と考えるべきだとし，本論文はその検討に重点を置かない．

### 上位2%のトークンに絞る（§2.2）

著者は，意味のある大きさの直接効果が疎であることを観察したうえで，**各 head の直接効果が上位 2% のトークン**に絞って自己修復を平均する．雑音ではなく，head がモデルの中で果たす役割の自己修復を見るためである．

## 実験設定

| | |
| --- | --- |
| 主実験（§2.2） | Pythia-1B，The Pile（同モデルの学習に使われたデータ）の 100 万トークン．各 head を1本ずつ resample ablation |
| トークン単位のばらつき（§2.3） | Pythia-410M，The Pile の 5,000 トークン．著者が選んだ4本：L22H11・L20H6・L23H1・L20H14 |
| LayerNorm（§3） | Pythia-160M の L11H2（The Pile の 100 万トークン）ほか |
| Anti-Erasure（§4） | 最終 MLP 層より前にあり，The Pile 上の平均直接効果が大きい head（Figure 8，付録 C の Figure 12–15）：Llama-7b L30H8，GPT-2 Small L9H11，Pythia-160m L7H8，Pythia-410m L17H4．加えて Pythia-160M L10H11（§4.1・§4.2.2） |

## 結果

### 自己修復は在る，しかし不完全（§2.2）

直接効果と $\Delta\mathrm{logit}$ を head ごとに平均して並べると（Figure 2），$y=-x$ の線と横軸のあいだに入る head が多数ある．**最終層の head にもある**——"even for heads in the last layer (which don't have attention or MLP layers downstream of it"（脚注は Pythia-1B が parallel attention のモデルであると添える）．一方，自己修復される head も分布全体で完全に修復されるわけではなく，除くと小さいが目に見える logit 差が残る．

### 自己修復は noisy（§2.3）

平均では自己修復されているように見える head でも，トークンごとには自己修復の量が元の直接効果よりはるかに小さいことも，はるかに大きいこともある．元の直接効果より大きい場合，head を除いたほうがモデルの性能が上がる．直接効果と $\Delta\mathrm{logit}$ にはっきりした相関の見られない head もあり（L23H1・L20H14），著者は，直接効果が大きいのに強い相関の無い L20H14 のような head について，直接効果がほぼ完全に自己修復されているかもしれないと述べる．この相関の欠如は McGrath らが層の除去で報告したもので，個々の head の水準でも成り立つ．

### LayerNorm による自己修復（§3）

最終 LayerNorm を前後の線形層の重みに畳み込むと，残差ストリームのノルムに比例する尺度 $S$ で割る非線形が残る．parallel attention のモデルの最終層の head を除く単純化した場合について，除去で $S$ が $S'$ に変わると

$$
\Delta\mathrm{logit} = \left(\frac{S}{S'} - 1\right)\mathrm{logit} + \frac{S}{S'}\,\Delta\mathrm{DE}_{\mathrm{head}}
$$

となる（§3.1，導出は付録 B）．第1項は既にある正解 logit に掛かるので，$S > S'$ なら正解 logit が増幅され，自己修復として現れる．$S > S'$ となる理由について著者は，ありうる直観的な説明として，head の出力が残差ストリームと揃っていればそれを除くとノルムが大きく縮み，差し替えた出力は残差ストリームと揃いにくいのでノルムを戻さない，と述べる．

- **Pythia-160M の L11H2 で，直接効果の上位 2% のトークンのうち，尺度の比（除去前／除去後）が 1 を超えるものは，resample ablation で 90.66%，mean ablation で 98.75%，zero ablation で 100%** である（§3.2）．zero ablation は残差のノルムを最も大きく変え（§5.2・付録 D），直接効果が正の head を除いたのに logit 差が上がることさえある（この例の head を本文は Pythia-160M の L11H10，Figure 5 の注は L11H0 とする．原文の表記のとおり）．本文は，この head が最後の attention 層にあり Pythia-160M が parallel attention のモデルなので，自己修復を担う部品は LayerNorm だけだと述べる．
- **寄与の大きさ**：全体の自己修復から，下流の head と MLP 層の直接効果の変化の和を引いた残りを LayerNorm に帰す（式 2）．これをトークンごとに head の元の直接効果に対する割合として，0–100% に切り詰めて測ると，LayerNorm は多くの層で平均して head の**直接効果の約 30%** を説明する（§3.2）．§5.2 は同じ数字を "only explains up to around 30% of the self-repair on average" と書く（§3.2 は直接効果に対する割合，§5.2 は自己修復に対する割合と書いており，原文の言い方が揃っていない）．著者はこれを精密な量ではなく方向を示す指標とし，切り詰めない割合は極端な値になると付録 A で断っている．

### Anti-Erasure を担う疎なニューロン（§4）

後段の MLP の部品が前段の部品の出力した重要な方向を部分的に打ち消すことを **Erasure** と呼ぶ（McGrath らの MLP Erasure）．**Anti-Erasure** は，上流の部品を除くと，それに因果的に依存していた下流の Erasure も除かれることによる自己修復である．論文の例では，上流が $x$，下流が $-0.7x$ を同じ logit に寄与していたなら，上流を除くと logit の変化は $0.3x$ にとどまり，$0.7x$ 分の自己修復が生じる．

- Pythia-160M の L10H11 では，最終 MLP 層が最もよく自己修復する上位 2% の事例で，最も自己修復するニューロンは多くのトークンで，除去前の直接効果が負で，除去後にはその負が小さくなる（§4.1，Figure 7）．
- **Llama-7B の L30H8 では，最もよく自己修復する1個のニューロンが，40% を超える事例で L30H8 の直接効果の 10% を修復する**（§4.2，Figure 8．事例は head の直接効果が上位 2% のトークン）．
- **どのニューロンが修復するかはプロンプトで変わる**（§4.2.2）：Pythia-160M の L10H11 が最もよく自己修復される上位 2% のトークンで，どの単一のニューロンも，修復上位 10 個に入るのはトークンの最大 16% にとどまる．
- **Anti-Erasure がすべてではない**（§4.2.1・付録 C）：著者は，最終層のニューロンのおよそ 60% は直接効果の変化が無視でき，残る 40% は，小さく互いに打ち消し合うが絶対値の合計では無視できない変化を持つ，と見ている（"We suspect that"）．Pythia-410M（L17H4 を除去）では，修復上位 1,000 個のニューロンが，最終 MLP 層全体の直接効果の変化の絶対値の合計の約 50% を占める（付録 C，Figure 15）．

### Iterative Inference 仮説（§5.3）

著者は，層が複雑な回路の中で後続の全層を助けるという上からの過程に対し，"many model components are more of a bottom-up process, where each layer treats the input as a guess for the final output" という見方を置く——各層は入力を最終出力の推測とみなし，それと真の次トークンとの誤差を縮めようとする．この見方では自己修復は意外でない：ある head が担う処理を，同じ信号を読む下流の head が，前者を除いたときに代わりに完了させうるからである．

- **支持する材料**（§5.3.1）：copy suppression（McDougall ら）の知見と整合すること．head の出力を，その head に入る残差ストリームに足し戻すと直接効果が下がる self-repressing head（GPT2-Medium の L21H1．抑制は足し戻す出力の倍率に比例する．付録 F）．IOI 課題の Backup Name Mover head（Name Mover の読む残差ストリームを Backup Name Mover の読む残差ストリームへ移すと，後者も名前の移送を行い始める）．
- **反する材料**：previous token head や induction head は自己修復されないように見える（著者は anecdotally と断る）．また先行研究は，層をまたいでは容易に存在しない複雑な回路の存在を強調しており，IOI 課題の S-inhibition head がその一つである．著者は，モデルの多くの head は「反復的な」処理を行っていないかもしれないと結ぶ．
- 著者自身はこの仮説を推測的な枠組みとし，示すのは予備的な証拠にとどまる（序論）．

### 著者が述べる含意（§5.1–5.2）

- **除去は信頼できない道具になる**（§5.1）．自己修復が大きいと，直接効果と除去時の logit の変化は大きく食い違いうる．直接効果が大きいのに完全に自己修復される部品は，無関係なのか，重要だが補われているのかが定まらない．ただし回路の解析に要るのは部品が重要か否かの判別であって除去の効果の正確な値ではなく，重要度は裾が非常に重いので，推定した効果がかなりの割合で下がっても，どの部品が重要かは変わらない，と著者は述べる．それでも，自己修復の度合いが部品ごとに大きく違えば，境界上の部品の取捨を誤りうる．
- **除去はモデルを分布の外へ連れ出す**（§5.2）．著者は当初，モデルが意図的に自己修復していると想定していたが，LayerNorm による自己修復は，モデルの数学的な性質と，除去が残差ストリームのノルムを変えることとの相互作用による副作用に見える，と述べる．対処として，zero ablation を避けること，除去のあいだ LayerNorm を固定すること，部品全体を差し替える代わりに path patching を使うことを挙げる．

## 著者が明記した限界

- **測ったのは直接効果の自己修復だけ**であり，attention head の他の「振る舞い」の指標での自己修復は扱っていない．§3・§4 の機序がそうした別の形の自己修復にどこまで一般化するかは分からない（§7．著者は "A strong limitation" と書く）．
- **LayerNorm の約 30% は方向を示す指標**であって精密な割合ではない（§3.2・付録 A）．
- **Iterative Inference 仮説は推測的**であり，示したのは予備的な証拠である（序論）．
- sparse な Anti-Erasure は，MLP 層の自己修復の全体像の一部でしかない（§4.2.1・付録 C）．

## 射程の外

- **テキストの言語モデル**（Pythia・GPT-2・Llama）であって，VLM ではない．全文に VLM・image・multimodal の語は無く，vision は関連研究での他論文への言及1か所だけである．
- **学習時の介入は見当たらない．** 操作はすべて学習済みモデルへの推論時の操作（主に除去）である．全文に fine-tun・contrastive の語は無い．
- 除去するのは個々の **attention head** であり，MLP のニューロンは除去の対象ではなく，head を除いたときの直接効果の変化を測る側である．残差ストリームを直接動かす操作は，§5.3.1 の検証（head の出力の足し戻し，残差ストリームの移し替え）で行われる．
- 測る量は**正解の次トークンの logit**（直接効果）であって，質問応答の正解率ではない（全文に accuracy・question answering の語は無い）．
- GQA・SmolVLM の語は全文に無い．

---

**確認**：2026-11-13．arXiv abs ページ（題・著者・版の履歴・Comments「ICML 2024」），arXiv の PDF v2（20頁．本文 10頁，参考文献，付録 A–H）と v1（17頁），PMLR の書誌ページ（巻・頁・編者・OpenReview へのリンク）を取得した．v2 を pdftotext でテキスト化して全文を読み，式と Figure 2・4・5 は 3頁・5頁を画像にして確かめ，本文書の書誌・数値・逐語引用・節番号を一件ずつ照合した．本文書の数値は v2 のものである．v1 は全文を読まず，本文書の数値と head 番号の文字列が v1 にも在ることを検索で確かめた（90.66／98.75%，around 30%，over 40%，16%，roughly 60%，top 1000，1 million，5000，各 head 番号）．序論の要約の文言は v1 と v2 で異なる（v2 は "possibly 30%"，v1 は "Up to 30%"）が，本文書はこの文言を引いていない．本研究についての記述は研究文書（`.claude/context.md`）§A.4 と照合した．下書き（2026-10-11 作成）から直した主な点：位置づけの「本研究の根拠を補強する」を，研究文書が本論文を引いていないことと対象の違いを明記した形に改めた．付録 C・§4.2.1 の 60%／40% を，著者の見立て（"We suspect that"）の強さに直した．90.66% などの母数（直接効果の上位 2% のトークン）を補った．約 30% が §3.2 では直接効果に対する割合，§5.2 では自己修復に対する割合と書かれていることを明記した．S-inhibition head の記述，「重要な部品の集合はあまり変わらない」，「反復的な処理を行わない head も多いだろう」を原文の強さに直した．射程の外の「除去の単位は head（と MLP ニューロン）」を，ニューロンは除去されていないので改めた．「最終層の head の修復が LayerNorm を調べる動機になる」は原文に無いので外した．**読めていないもの**：OpenReview のページと PMLR 版の PDF は取得していない（PMLR の書誌ページにあるリンクの宛先だけを確かめた）．v1 の全文．図の中の値（Figure 1・6・8・11–22 の点や線）は本文と注に書かれたものだけを照合し，図から読み取ってはいない．
