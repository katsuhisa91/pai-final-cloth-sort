# 番号付き布片の仕分け — Physical AI 応用編1 最終課題

手書きの番号が書かれた 5 cm 角の布片を、番号どおりの置き場へ運ぶロボットのパイプラインです。
「撮影 → 布片の検出 → 番号の読み取り(自前学習した CNN) → 計画 → pick & place → カメラで検証」を
dora-rs の dataflow としてつなぎ、仮想アームで動かしています。

- 提出動画: `runs/submission/PAI最終課題_katsuhisa91.mp4`(約 2 分、1920x1080。説明スライド + dora 版の実行 67 秒、等速)
- 実行部分のみ: `runs/seed1/cloth_sort.mp4`(単一プロセス版、約 70 秒)、`runs/dora/cloth_sort_dora.mp4`(dora 版)
- 報告書: `report/report.pdf`(元は `report/report.html`)
- 実機は未接続。アームは `RobotArm` インターフェースの仮想実装(`VirtualArm`)で、実機対応はこの 1 クラスを足す形

## 動かし方

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH=$PWD

# 1. 番号読み取りモデルを学習(合成データ、CPU で 1 分程度)
.venv/bin/python -m cloth_sort.ml.train --out models

# 2. 単一プロセスで実行し、フレームと動画を作る(ffmpeg が必要)
.venv/bin/python -m cloth_sort.pipeline --seed 1 --n 12 --out runs/seed1 --video

# 3. dora-rs で分散実行(sim / brain / recorder の 3 ノード)
./run_dora.sh
./run_dora.sh graph     # 構成図

# 4. 提出用の動画(スライド + dora 版の実行フレーム)を作る
.venv/bin/python tools/make_submission_video.py --frames runs/dora/frames --log runs/dora/log.jsonl

# テスト
.venv/bin/python -m pytest -q
```

## 構成

```
cloth_sort/
  config.py            作業台・置き場のレイアウト(mm)
  sim/world.py         2D シミュレーション(布片・アーム・俯瞰カメラ)
  sim/synth.py         手書き風番号の描画と学習データ合成(世界と学習で同じ描画器を使う)
  sim/fonts.py         学習用フォントと「見たことのない筆跡」用フォントの分離
  perception/detect.py 背景差分による布片検出(古典的手法)
  perception/model.py  1 文字 CNN
  perception/reader.py インク抽出 → 文字分割 → CNN → 番号文字列と確信度
  ml/train.py          データ生成 → 学習 → 未見フォントで評価 → 重みと指標を保存
  planning/planner.py  番号 → 置き場、読めないものは要確認トレイへ、近い順に実行
  robot/executor.py    RobotArm インターフェース、VirtualArm、PickPlace 状態機械
  pipeline.py          単一プロセスの実行(動画・ログ生成)
  nodes/               dora ノード(sim / brain / recorder)
  viewer.py            動画フレームの描画
dataflow.yml           dora のノード構成(つなぎ方)
tests/                 pytest
```

## 設計の要点

- 読み取りは「1 文字 CNN + 連結成分で文字分割」。語彙は 0〜9 と「-」なので、学習で見ていない番号(例: 1-2)も読める
- 確信度がしきい値未満の布片は止めずに要確認トレイへ逃がす。誤った置き場に置くより、人が後で見るほうが安い
- 置いた直後にカメラで読み直して計画と照合する(自己検証)。ログは JSONL で残し、再学習データの種にできる
- ロボット側(sim)と AI 側(brain)を別プロセスに分け、dora の Arrow メッセージで接続。実機化はロボット側だけの差し替え

## 講義の技術スタック・推奨との対応

AI アルゴリズムそのものではなく「動かし続けるための基盤」を扱う講座でした。
本リポジトリが講義のどの内容に乗っていて、どこを意図的に変えたかを整理します。

| 講義で扱った内容 | 本リポジトリでの対応 |
|---|---|
| MLOps・RobOps(第1・2回: データ収集 → 学習 → 評価 → 運用のループ) | 実行ログを JSONL で残し、置いた後にカメラで読み直して計画と照合(自己検証)。低確信度は止めずに要確認トレイへ逃がし、人の確認結果を再学習データにできる構造 |

### 講義の想定と違う点(意図して外したもの)

- そのため把持や接触の物理は検証できていない。実機化の前段には MuJoCo か Gazebo(どちらも講義で紹介された OSS)でのアーム動作の確認が必要

## ライセンスと注意

Apache-2.0 で公開しています(`LICENSE`)。
