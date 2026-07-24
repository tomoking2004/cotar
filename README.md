# cotar

**「視覚言語モデルにおけるタスク類似性に基づく内部表現整合学習に関する検討」の実装．**（`cotar` はリポジトリ名でありパッケージ名．手法や研究の名前ではない．）

推論過程が同じタスクは，VLM の内部表現も似るべきか——そして似せる制約を学習に足せば，性能は上がるか．

GQA の各質問には functional program が付いている．その**演算子列**（例 `select > relate > query`）を**タスク署名**と呼び，署名の一致をもって**タスク類似性**とする．署名が一致するタスクどうしの中間層表現を supervised contrastive loss で引き寄せる——これが**内部表現整合**．言語モデリング損失への補助項として足すだけで，射影 head は挟まない——生の hidden state を直接掴むからこそ，仮説は反証可能なままになる．

研究の問い・確定事項・未決の余白は [.claude/context.md](.claude/context.md) にある．**この README は「どう動かすか」だけを扱う．**

## 構成

```
cotar/
  config.py        cfg — デバイス・パス（datasets/ と runs/ の位置）
  types.py         VLM / VLMProcessor のプロトコル．モデル実装との唯一の契約
  trainer.py       Trainer — train4all の BaseTrainer を実装．3群の定義（Arm）はここ
  losses.py        supervised_contrastive_loss
  metrics.py       Evaluator — 報告する数値はすべてここ．公式 GQA 評価器もここから叩く
  pairwise.py      pairwise_cosine / pairwise_equal（losses と metrics の共有土台）
  modules/         LogitScale — 学習可能な温度
  models/          SmolVLM + SmolVLMProcessor（プロンプト構築・ラベルマスク・表現のプール）
  utils/           JSON の読み書き・run ディレクトリ名（timestamp と arm）
  data/
    gqa.py         GQADataset / MPerSignatureSampler / task_signature / build_gqa_dataloader
    _gqa_eval.py   GQA 公式評価器（上流＋局所パッチ．読み方は冒頭のヘッダ）
scripts/
  train.py         3群を本走．設定は全て冒頭の定数
  analyze_signatures.py  署名分布を数える（学習不要）
  generate.py      モデルが喋るかだけ見る最小スクリプト
```

**学習が要るのは `train.py` だけ．** 残る2本は学習不要で，ノートPCで何度でも回せる．

**指標は `metrics.py` だけを見ればいい．** デコードも採点も表現の比較も公式評価器もそこにあり，`Evaluator.measure()` が1バッチの全指標を，`Evaluator.report()` がエポック全体の全指標を返す．Trainer は何も測らない．

**モデルは `models/` を足すだけで差し替わる．** `types.py` の `VLM` / `VLMProcessor` プロトコルが唯一の契約で，`VLMOutput["representation"]`——整合する層と位置でプールした `(B, L, H)`——を返せば，loader も損失も指標も Trainer もそのまま動く．

**部分空間 `U` の差し込み口は `Trainer._project`**（現状 identity）．

## 準備

PyTorch は GPU の世代ごとに index が違うので**先に**入れる．

```bash
conda create -n cotar -c conda-forge python=3.12 && conda activate cotar
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e .
```

`pip install -e .` が `cotar` をパスに通すので，どこからでも `python scripts/train.py` が動く．flash-attn は任意（Linux のみ）——無ければ自動で SDPA に落ちる．学習ループの [train4all](https://github.com/tomoking2004/train4all) は**リリースタグに固定**してある（どのタグかは `pyproject.toml`）——`main` を追うと，指標の並び順のような出力の形が実験の途中で変わりうる．

データセットは [GQA 公式](https://cs.stanford.edu/people/dorarad/gqa/download.html) から取り，`cfg.datasets_root/gqa/` に置く（`images/`, `questions/`, `eval/`）．

## 走らせる

```bash
python scripts/train.py  # 3群を回す（GPU）
```

`train.py` は `cfg.seed`（既定42）で baseline → proposal → ablation を順に通す．run ごとにモデル・loader・trainer を作り直すので，**3群は同じバッチ列を見る**（サンプラの epoch カウンタを次の群に持ち越さない）．

制約する層は `train.py` の `LAYERS`．`(16,)` なら層16の1本，`(8, 16, 24)` なら3本を同時に整合する——各層で独立に SupCon を計算して**平均**するので，層を増やしても `ALIGN_WEIGHT` の意味は変わらない（温度は全層で共有）．

1 run あたり RTX 4090 で約2時間（train_balanced の94.2万問・1エポック——全94.3万問のうち，同署名の相方がいない884問はペアを作れないので落ちる）．**3群＝約6時間．**

## 結果を読む

結果は `cfg.runs_root/<timestamp>_<arm>/` に落ちる．3群は同じ timestamp を共有するので，1回の実験がひとまとまりに並ぶ．

| | |
| --- | --- |
| `metrics/eval.json` | 群の identity（arm・seed・align_weight・align_pairing・phase）＋testdev の全指標 |
| `metrics/predictions.json` | 公式形式の予測．公式評価器で再採点できる |
| `metrics/representations.pt` | **testdev の表現そのもの**（`(N, L, H)` float32）＋層番号＋署名＋質問ID |
| `checkpoints/best.pth` | val accuracy が最良のエポックの重み |
| `dashboard.html` | 学習中のライブ表示 |

読むのは `eval.json` の `official_gqa.accuracy`——GQA 公式評価器そのものが testdev_balanced に出した数字で，`binary`／`open`／`distribution` と型別内訳が付く．3群を横に並べてベースラインと比べる．横並びの比較スクリプトも有意差検定も今は無い．

`intra_sim`／`inter_sim`／`separation`／`separation_d` は**操作確認**——学習で直接最適化している量なので，上がっても仮説の支持証拠にはならない．複数層を制約したときは学習中は `separation_d/L16` のように層別に並び，`eval.json` では `representation_stability` が `L16`／`L24` と入れ子になる（単一層なら平のまま）．

`lm_loss` は診断用で，**`loss` ではない**——`loss` は目的関数で，整合する群ではそこに補助項が乗っており，群間で同じ量ではない．学習後に `train_eval`（train を eval モードで測ったスライス）と `val` を並べて出すので，その2数字の開きが素の過学習量になる．
