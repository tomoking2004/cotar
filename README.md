# cotar

**「視覚言語モデルにおけるタスク類似性に基づく内部表現整合学習に関する検討」の実装．**（`cotar` はリポジトリ名でありパッケージ名．手法や研究の名前ではない．）

GQA の各質問には functional program が付いている．その**演算子列**（例 `select > relate > query`）を**タスク署名**と呼び，署名の一致をもって**タスク類似性**とする．署名が一致するタスクどうしの中間層表現を supervised contrastive loss で引き寄せる——これが**内部表現整合**である．言語モデリング損失への補助項として足すだけで，掴むのは生の hidden state そのもの．

問いと結果は研究文書にある．**この README は「どう動かすか」だけを扱う．**

| どこに | 何が |
| --- | --- |
| [.claude/context.md](.claude/context.md) | **研究文書**．問い・手法・検証方法・結果・答え・次の一手．これ1本で研究が分かる |
| [.claude/context-philosophy.md](.claude/context-philosophy.md) | 上を**どう書くか**の原則．研究の中身には依存しない |
| [references/](references/) | 依拠する論文の要約．1本につき1ファイル |
| [presentations/](presentations/) | 対外発表の成果物．発表した時点の記録であって，最新の結果ではない |
| [snapshots/](snapshots/) | 各 run が残した記録．重みを除いた写しなので commit できる |
| [analyses/](analyses/) | run をまたいで測った結果．研究文書の数値の出どころ |

## 準備

PyTorch は GPU の世代ごとに index が違うので**先に**入れる．

```bash
conda create -n cotar -c conda-forge python=3.12 && conda activate cotar
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e .
```

`pip install -e .` が `cotar` をパスに通すので，どこからでも `python scripts/train.py` が動く．flash-attn は任意（Linux のみ）——無ければ自動で SDPA に落ちる．学習ループの [train4all](https://github.com/tomoking2004/train4all) は**リリースタグに固定**してある（どのタグかは `pyproject.toml`）——`main` を追うと，指標の並び順のような出力の形が実験の途中で変わりうる．

データセットは [GQA 公式](https://cs.stanford.edu/people/dorarad/gqa/download.html) から取り，`cfg.datasets_root/gqa/` に置く（`images/`, `questions/`, `eval/`）．`train_balanced_questions.json` は 813MB あるので，これを読むスクリプトは立ち上がりに数分かかる．

**パスの起点は2つある．** リポジトリが持つのはコード・文書・snapshots・analyses——版管理に載せる物である．`cfg.work_root`（既定は `~/TMU/Master/study/cotar`）が持つのはデータセットと `runs/`——数十GBあって，版管理にも同期フォルダにも置けない物である．どちらがどちらかは `cotar/config.py` が答える．

## 走らせる

```bash
python scripts/train.py                    # 3群 × 3 seed を回す（GPU・約26時間）
python scripts/sweep.py                    # 整合の強さ／層を振る（GPU・1点あたり約3時間）
python scripts/generate.py                 # モデルが読み込めて受け答えするかだけを見る
python scripts/audit_dataset.py            # データ側の数値（分割・署名分布・含意対・答えの語彙）
python scripts/summarize_runs.py           # run 側の数値（表現の幾何・損失・正解率・対にした差と区間）
python scripts/summarize_sweep.py          # 読み取り精度と正解率を，整合の強さごとに並べる
python scripts/probe_signature.py          # 署名が線形に読めるか
python scripts/probe_operators.py          # プローブが見ていない演算子の組み合わせが読めるか
python scripts/stratify_by_frequency.py    # 正解率を署名の頻度で層別し直す
python scripts/probe_bypass.py             # 整合した構造が出力の読む方向に載っているか（§4.5 第1段）
python scripts/probe_bypass_pullback.py    # その方向を第16層へ引き戻して問い直す（§4.5 第2段）
```

**モデルを動かすのは4本**——上の3本と `probe_bypass_pullback.py`（ヤコビ積を取るのでモデルを走らせる）である．残る7本は保存済みのファイルを読み直すだけなので，ノートPCで何度でも回せる．それぞれ自分と同じ名前の JSON を `analyses/` に書く（`probe_signature.py` → `analyses/probe-signature.json`）．どの実験を読むかは `cotar/analysis/experiment.py` の `TIMESTAMP` が決める．`probe_bypass.py` と `probe_bypass_pullback.py` の2本は学習済みの重みを要求する（後述）．

`train.py` と `sweep.py` は run の建て方を共有する——`cotar/training/run.py` の `Settings` と `run_training` である．既定値が報告した実験の値なので，**sweep は変える設定だけを書き，残りが同じであることが目で見える．** どちらも冒頭の `DEBUG = True` で全分割を切り詰めた短い試走になり，数分で配線を確かめられる．

`train.py` は `SEEDS`（既定 42・43・44）の各 seed で baseline → proposal → ablation を順に通す．run ごとにモデル・loader・trainer を作り直すので，同じ seed の3群は同じバッチ列を見る（サンプラの epoch カウンタを次の群に持ち越さない）．seed が変わればその並びも初期化も変わるので，**群間の差を run ごとのばらつきと区別できる**．

制約する層は `Settings.layers`．`(16,)` なら第16層の1本，`(8, 16, 24)` なら3本を同時に整合する——各層で独立に SupCon を計算して**平均**するので，層を増やしても `align_weight` の意味は変わらない（温度は全層で共有）．

1 run あたり RTX 5090 で約2時間53分（`train_balanced` を1エポック＝29,441 バッチ——全94.3万問のうち，同署名の相方がいない884問はペアを作れないので落ち，残る 942,116 件ぶんを引き終えるまでが1エポック）．**3群 × 3 seed ＝9 run で約26時間．** どちらも `snapshots/20260727-002344_*/log.txt` の実時間．

## 次の一手を走らせる

**研究文書 §7 の五つのうち，いま走らせられるのは §7.5 だけである．** 残る四つにはまだスクリプトが無い——§7.1（依存の大きさ）は `probe_bypass_pullback.py` が正規直交化のときに捨てている長さを取り出すもので，同じ逆伝播で足りる．§7.2（表現の頑健性）は重みを，§7.3（稀な署名を表現の側で）は保存済みの表現を要し，§7.4（別のラベル）は新しい学習を要する．

**迂回の判定（研究文書 §4.5）は2段とも済んでいる**——`probe_bypass.py` と `probe_bypass_pullback.py` が `analyses/` に結果を書いているので，走らせ直す必要はない．

```bash
# 0. 重みが在るか（§7.1 と §7.2 はこれを要求する．無ければ即座に落ちて場所を告げる）
ls "$(python -c 'from cotar.config import cfg; print(cfg.runs_root)')"/20260727-002344_*/checkpoints/best.pth

# 1. §7.5 整合の強さを振る — まず DEBUG = True で数分の試走．配線を確かめてから本走
python scripts/sweep.py
#    print された timestamp を summarize_sweep.py の SWEEP_TIMESTAMP に入れてから
python scripts/summarize_sweep.py
```

**手順1は重みを要さない．** `sweep.py` は $\lambda = 0.03 / 0.3 / 1.0$ を seed 42 の1本で回す（既に測ってある $\lambda = 0$ と $0.1$ は走らせ直さない）ので約9時間．`summarize_sweep.py` は snapshots だけを読むのでノートPCで回せる——sweep を走らせる前でも，既存の2点だけを表に出して配線を確かめられる．

層を振るときは `sweep.py` の末尾を `VARIANTS = LAYER_VARIANTS` に変える．**強さと層を同時に振らないこと**——一つの曲線に二つの説明が混ざる．

## 結果を読む

結果は `cfg.runs_root/<timestamp>_<arm>_seed<seed>/` に落ちる．9 run は同じ timestamp を共有するので，1回の実験がひとまとまりに並ぶ．

| | |
| --- | --- |
| `config.json` | trainer に渡した引数だけ．cotar 由来のもの（arm・align_weight・init_scale）は常に，train4all 由来のもの（seed・batch_size…）は既定から変えたときだけ載る |
| `metrics/eval.json` | 群の identity（arm・seed・align_weight・align_pairing・phase）＋testdev の全指標 |
| `metrics/predictions.json` | 公式形式の予測．公式評価器で再採点できる |
| `metrics/representations.pt` | **testdev の表現そのもの**（`(N, L, H)` float32）＋層番号・署名・質問ID．同署名／異署名の cosine 差は 0.02 程度しかなく，半精度ではそれが丸めに沈むので単精度で置く |
| `metrics/epoch_metrics.json`・`metrics/step_metrics.json`・`plots/` | 学習中に記録した指標の推移．JSON と，同じものを描いた PNG |
| `checkpoints/best.pth` | val accuracy が最良のエポックの重み．extras にモデルの checkpoint・層・attention 実装・loader の設定 |
| `log.txt` | 実行の記録．冒頭が環境バナーで，マシンに続けて結果を決めるライブラリの版が載る（`transformers`・`train4all`・`schedulefree`，そして入っていれば `flash-attn`） |
| `dashboard.html`・`dashboard_data.json` | 学習中のライブ表示と，それが描くデータ．Environment パネルはバナーと同じものを読む |

**同じものの写しが，`checkpoints/` だけを抜いて `snapshots/` の同じ run 名の下にも落ちる．** 重みを落とせば1 run は数十MB——上の表のうち `checkpoints/best.pth` 以外は全部こちらに在るので，git がそのまま追跡でき，push すれば結果は GitHub からも読める．エポックごとと最終評価の直後に，変わったファイルだけを原子的に置き換えて更新するので，走っている最中に掴んでも写しは丸ごと揃っている．

**逆に言えば，重みは clone に付いてこない．** 本走9 run の `checkpoints/best.pth` は，それを回したマシン（`log.txt` の環境バナーによれば Ubuntu 26.04・RTX 5090）の `runs/` にしかない．`probe_bypass.py` と `probe_bypass_pullback.py` はこれを要求するので，そのマシンで走らせるか，重みを先に持ってくる．

`snapshots/` には2つのバッチが在る．研究文書が報告するのは `20260727-002344` の9 run（3群 × 3 seed）だけで，分析スクリプトもこれしか読まない．`20260714-033208` の3 run は seed を振る前の単一 seed の実験で，[presentations/](presentations/) のポスターの裏付けとして残してある——**別の実験なので，数値が研究文書と一致しなくて当然である．**

再現に要る情報は3箇所に分かれる——**実験を決める trainer の引数が `config.json`，モデルと loader の設定が `best.pth` の extras，マシンとライブラリの版が `log.txt` の環境バナー**．`config.json` を引数だけに保つので `Trainer(vlm, processor, **config)` がそのまま通り，trainer が受け取らないもの（`vlm` の引数・loader の設定）は extras に回り，**この checkout の外が決めるもの**はバナーが引き受ける．ソースが決めているもの（重みの fp32・bf16 autocast 等）はどこにも無い．この checkout のコードが答えるからで，二重に持たない．

**上の表と，この3箇所が語るのは，この checkout がこれから走らせる run が残すものである．** 報告する9 run は 2026-07-27 に，いまのコードより前の版で走っており，記録が三点だけ違う——`config.json` には当時 `align_pairing` も載っていた（`arm` から導かれるので，いまは載せない．この1キーがあるぶん，9 run の `config.json` は `Trainer(vlm, processor, **config)` にそのままは通らない），`best.pth` の extras は `vlm` と `layers` の2つだけで，モデルの checkpoint・attention 実装・loader の設定は入っていない，`log.txt` の環境バナーにライブラリの版が無い（版を記録するようにしたのが run の後である）．どれも測定値ではないので，研究文書の数値には一つも効かない．

決定指標は `eval.json` の `official_gqa.accuracy`——GQA 公式評価器そのものが `testdev_balanced` に出した数字で，`binary`／`open`／`distribution` と型別内訳が付く．3群の横並びは `summarize_runs.py`（全 testdev の平均・spread・対にした差と区間）と `stratify_by_frequency.py`（署名の頻度で層別したもの）が出す．**区間は研究文書 §3.3 の式そのままで，実装は `analysis/statistics.py` にしか無い．**

`intra_sim`／`inter_sim`／`separation`／`separation_d` は**操作確認**で，決定指標ではない（理由は研究文書）．複数層を制約したときは，学習中は層をまたぐ平均と `separation_d/L16` のような層別が並び，`eval.json` では `representation_stability` が `L16`／`L24` と入れ子になる（単一層ならどちらも平のまま）．`representation_stability` はseed で決まる4,000件の抜き取りを一度に計算したもの——類似度行列が件数の2乗で膨らむので上限を置いてある（`MAX_STABILITY_SAMPLES`）．同じ4つの名前が `eval.json` の `test_metrics` にも並ぶが，そちらはバッチごとに計算してエポックで平均した別の量である．

損失は3つに分かれる．群間比較に使えるのは `lm_loss` だけで，**`loss` ではない**——`loss` は目的関数で，整合する群ではそこに補助項が乗っている．学習後に `train_eval`（train を eval モードで測ったスライス）と `val` を並べて出すので，その2数字の開きが素の過学習量になる．

## 構成

```
cotar/
  config.py          cfg — デバイスと，2つの根（リポジトリ／作業領域）から引くパス
  types.py           VLM / VLMProcessor のプロトコル．モデル実装との唯一の契約
  pairwise.py        pairwise_cosine / pairwise_equal（損失と指標の共有土台）
  training/          ── 実験を回す
    run.py             Settings / run_training — 1 run の建て方．train.py と sweep.py が共有
    trainer.py         Trainer — train4all の BaseTrainer を実装．3群の定義（Arm）はここ
    losses.py          supervised_contrastive_loss
    metrics.py         Evaluator — run が測る数値はすべてここ．公式 GQA 評価器もここから
    logit_scale.py     LogitScale — 学習可能な温度
  analysis/          ── 済んだ実験を読む（GPU も学習も要らない）
    experiment.py      報告する実験はどれで，その成果物をどう読むか（重みの読み口もここ）
    probing.py         線形プローブと，質問文だけを入力にした対照（コードでは surface）
    subspaces.py       部分空間の中と外で読み，同じ幅のランダムな対照と突き合わせる
    statistics.py      平均±標準偏差・対にした差・信頼区間——3 seed のまとめ方の唯一の定義
  models/            SmolVLM + SmolVLMProcessor（プロンプト構築・ラベルマスク・表現のプール）
  data/
    gqa.py             GQADataset / MPerSignatureSampler / task_signature / build_gqa_dataloader
    _gqa_eval.py       GQA 公式評価器（上流＋局所パッチ．読み方は冒頭のヘッダ）
  utils/             JSON の読み書き・run ディレクトリ名
scripts/
  train.py                  3群を全 seed で本走．回す範囲は冒頭の定数，run の中身は Settings
  sweep.py                  報告した実験が固定した設定を1つだけ振る（整合の強さ・層）
  generate.py               モデルが読み込めて受け答えするかだけを見る最小の確認
  audit_dataset.py          研究文書が語るデータ側の数値を一度に出す（本走に依存しない）
  summarize_runs.py         9 run の記録から研究文書が語る run 側の数値を一度に出す
  summarize_sweep.py        整合の強さごとに読み取り精度と正解率を並べ，交換率を出す
  probe_signature.py        保存済みの表現から署名が線形に読めるかを測る
  probe_operators.py        プローブが見ていない演算子の組み合わせが読めるかを測る
  stratify_by_frequency.py  済んだ実験を署名の頻度で層別し直す
  probe_bypass.py           整合した構造が出力の読む方向に載っているかを測る（要 checkpoints）
  probe_bypass_pullback.py  その方向を第16層へ引き戻して同じ判定を掛け直す（要 checkpoints・GPU）
```

**`cotar/` は，実験を回す側と，済んだ実験を読む側に分かれる．** `train.py` は `training/` を，分析の7本は `analysis/` を使う——だから import がそのスクリプトの立ち位置を語る（`generate.py` はモデルを読むだけなのでどちらも要らない）．`analysis/` は GPU も学習済みモデルも要求しないので，ノートPCで完結する．

**機構は，この研究の中身に依存しない．** 別の研究に持っていくとき差し替わるのは3つだけ——**ファイルがどこにあるか**（`config.py`），**1 run をどう建てるか**（`training/run.py`），**どの実験を報告するか**（`analysis/experiment.py` の `TIMESTAMP`）．残りはすべてパスを引数で受け取り，実験を名指ししない．だから `cotar/` で `cfg` を import するのは後の2つだけで，機構は一つも import しない——`data`・`models`・`types`・`pairwise` と，`training/` の残り（`trainer.py`・`losses.py`・`metrics.py`・`logit_scale.py`）のどこにも `cfg` は無い．`cotar/__init__.py` が何も再輸出しないのも同じ理由で，`import cotar` がマシン固有のパスを束縛しないためである．

`cfg` 自身も2種類を分けて持つ——**マシン固有**（データセット・`runs/`・デバイス）と，**チェックアウト相対**（`snapshots/`・`analyses/`）．**済んだ実験を読む側が触るのは後者だけ**で，マシンのことは一度も訊かない．

**`scripts/` は走らせるものだけを置く．** 複数のスクリプトが分け合うもの——どの実験を読むか，成果物の在処，プローブの当てはめ——は `cotar/` 側にある．スクリプトが自分で決めてよいのは，自分だけの設定と，出力の並べ方だけである．

**置くのは，学習・検証と，環境が整っているかの最小の確認に要るコードだけ．** ほかのスクリプトの出力に含まれる情報を再計算するだけのものは持たない．ルート直下のディレクトリ名は小文字の複数形の名詞にする．

**run が測る数値は `metrics.py` だけを見ればいい．** デコードも採点も表現の比較も公式評価器もそこにあり，`Evaluator.measure()` が1バッチの，`Evaluator.report()` がエポック全体の測定値を返す．Trainer が足すのは自分が組んだ目的関数の内訳（`lm_loss`・`align_loss`・`temperature`）だけで，モデルを測ることはしない．**`analysis/` が測るのはこれの例外ではない**——問うのは run が残したもの（表現・予測・重み）であって，走っているモデルではない．

**モデルは `models/` を足すだけで差し替わる．** `types.py` の `VLM` / `VLMProcessor` プロトコルが唯一の契約で，`VLMOutput["representation"]`——整合する層と位置でプールした `(B, L, H)`——を返せば，loader も損失も指標も Trainer もそのまま動く．

**整合を表現の一部だけに掛けたくなったときの差し込み口は `Trainer._project`**（現状 identity——表現をそのまま返す）．入れるなら**固定の**射影にすること——学習可能にすると射影の側を動かすだけで損失を満たせてしまい，仮説が反証不能になる．

**まだ使っていないレバーが二つある．** 層を選ぶ側は実装済みで（`Settings.layers`，`sweep.py` の `LAYER_VARIANTS` が待っている），振る用意はあるが一度も走らせていない．位置を増やす側は未実装で，`SmolVLM._pool` の一般化が要る（現状は最終プロンプト位置の決め打ち）．

## 研究文書の記述は，コードのどこか

節番号は研究文書のもの．

| 研究文書 | 実装 |
| --- | --- |
| $\mathcal{L}_{\mathrm{align}}$（§2.2） | `losses.supervised_contrastive_loss`——対角を `-inf` にした `pairwise_cosine(features) * logit_scale` に `logsumexp` を取り，正例の平均を引く |
| 温度 $\tau$（§2.2） | `LogitScale` が $1/\tau$ を log スケールで保持し 100 で clamp．初期値は `logit_scale.INIT_SCALE = 1/0.07` |
| $\mathcal{L}_{\mathrm{LM}}$ のマスク（§2.2） | `SmolVLMProcessor._mask_labels`——プロンプト位置と `attention_mask == 0` の位置を `IGNORE_INDEX` に落とす |
| 掛ける位置（§2.3） | `SmolVLM._pool`——`hidden_states[row, prompt_lens[row] - 1]` |
| プローブが読む表現（§4.1） | 損失が使うのと**同一のテンソル**．`Trainer` が `set_cache("representation", …)` した値を `Evaluator.measure` がそのまま貯め，`representations.pt` に `(N, 1, H)` で落ちる．**評価ローダも `with_labels=True`**（`training/run.py` の共有 kwargs）で右詰め——左詰めだと `prompt_lens` が作られず，`_pool` の位置が意味を失う．生成は `decode_answers` が質問だけを左詰めに詰め直した別テンソルで行うので，表現には触れない |
| $s_{\mathrm{pooled}}$（§3.4） | `metrics.cohens_d`——`torch.var` は不偏なので，文書の式どおり $n-1$ で割った分散から作る |
| $\mathcal{S}$・$\mathcal{D}$（§3.4） | `metrics.representation_stability`——対角を除いた類似度行列**全体**から取るので順序つき対になる．平均と分散は変わらない |
| プローブ（§4.1・§4.2） | `analysis/probing.py`——sklearn は入れていないので torch の線形層＋AdamW で解く．当てはめ方と打ち切りの根拠はモジュール冒頭．研究文書の**言い回しプローブ**が `surface_matrices`（コードでは一貫して `surface`）で，`words`／`words_and_pairs` の高いほうを採る |
| **形式の残差化**（§4.1） | `probing.format_matrix` が形式の3特徴を標準化して並べ，`residualize` が ridge の当てはめを引く．**署名ラベルは回帰に入らない**ので，プローブの学習側・評価側を分ける前の全行で当てはめてよい．残差を再正規化しない理由は関数の docstring にある |
| 表現の幾何・損失・正解率の要約，崩壊値 $\log(B-1)$，1エポックのバッチ数と実行時間（§3.1・§3.4・§5.3） | `summarize_runs.py`——9 run の `eval.json`・`step_metrics.json`・`log.txt` を読み直すだけ．新しく測るものは無く，文書がしている算術をここでする |
| $\bar{x}$・$s$・$\bar{d}$・信頼区間（§3.3） | `analysis/statistics.py`——$t_{2,\,0.975}$ は文書と同じ閉じた式で求める（表を引かない）．区間が $0$ を跨ぐか否かも保存する |
| 頻度層別と層ごとの区間（§5.4） | `stratify_by_frequency.py` |
| 整合の強さと交換率（§7.5） | `sweep.py` が点を作り，`summarize_sweep.py` が並べる——読み取り精度は §5.1 と同じ当てはめで測り直すので，既存の2点が §5.1・§5.3 の数値を再現することが，両者が同じものを測っている証拠になる |
| 迂回の判定・第1段（§4.5） | `probe_bypass.py`——checkpoint の出力層から答えの**先頭トークン**の行を抜き，中心化した主成分 $m$ 本を $U$ とする．表現を $U$ の中と外へ射影して同じプローブを掛け，**中と外の両方**を同次元のランダム直交基底と比べる．部分空間の道具は `analysis/subspaces.py` にあり，第2段と共有する |
| 迂回の判定・第2段（§4.5） | `probe_bypass_pullback.py`——$U$ の各方向をベクトル–ヤコビ積で第16層へ引き戻し（`SmolVLM.readout_from_site` が site と読み出しを微分可能なまま返す），直交化して $U$ の代わりに使う．引き戻す前後の cos も測る——1 に近ければ第1段の近似は正しかったことになる（実際は 0.2 台だった） |
| 分割の排他性・署名の分布・対を組めない問・含意対との一致・答えの語彙・演算子が自分の名を質問文に置く率（§2.1・§3.1・§4.5・§6.3） | `audit_dataset.py`——学習も GPU も要らず，質問ファイルだけから全部を一度に出す．含意対は**対の数**（`unordered`）で数える．GQA は同じ関係を両側から並べるので，訪問回数（`ordered`）で数えると往復を二度数えて率が数ポイント動く．研究文書が引くのは前者．**食い違いの中身**（yes/no 型と自由回答型を組にした割合，最も多い署名の組）も同じ関数が出す——率だけでは「署名が誤っている」としか読めないので |

**研究文書の数値には，例外なくそれを出すスクリプトがある．** 手で数えて本文にだけ書いた数値は，書いた時点から誰にも検算できない．出どころは主張の種類で決まっていて，新しく何かを主張するならまずその場所に関数を足す．

| 主張の種類 | 出す場所 |
| --- | --- |
| データについて（分割・署名・演算子・語彙） | `audit_dataset.py` |
| 9 run の記録について（幾何・損失・正解率・実行時間） | `summarize_runs.py` |
| 保存済みの表現に新しく問うこと | `probe_*.py`・`stratify_by_frequency.py` |
| 3 seed のまとめ方（平均・spread・区間） | `analysis/statistics.py`——上のどれもがここを通る |

**要約統計を各スクリプトで書き直さない．** 平均・標準偏差・対にした差・信頼区間は `analysis/statistics.py` にしかない——二つのスクリプトが別々に区間を作れば，同じ文書の中で同じ括弧が二つの意味を持つ．
