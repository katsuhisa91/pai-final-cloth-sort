# 番号付き布片の仕分け — Physical AI 応用編1 最終課題

手書きの番号が書かれた 5 cm 角の布片を、番号どおりの置き場へ運ぶロボットのパイプラインです。
「撮影 → 布片の検出 → 番号の読み取り(自前学習した CNN) → 計画 → pick & place → カメラで検証」を
dora-rs の dataflow としてつなぎ、MuJoCo の物理シミュレーション上で Franka Panda(吸着ツール付き)を動かしています。

- 提出動画: `runs/submission/PAI最終課題_katsuhisa91.mp4`(約 2 分半、1920x1080、等速。説明スライド + dora 版の実行)
- 実行部分のみ: `runs/mj4/cloth_sort.mp4`(単一プロセス版、seed 4)。dora 版の実行は `./run_dora.sh` で `runs/dora/frames` に出る(提出動画の中盤がそれ)
- 報告書: `report/report.pdf`(元は `report/report.html`)
- 提出の流れ(public 化、YouTube、omnicampus): `SUBMISSION.md`
- 実機は未接続。アームは `RobotArm` インターフェースの MuJoCo 実装(`MujocoArm`)で、実機対応はこの 1 クラスを足す形

## 動かし方

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH=$PWD

# 1. 番号読み取りモデルを学習(合成データ、CPU で 5 分程度)
.venv/bin/python -m cloth_sort.ml.train --out models

# 2. 単一プロセスで実行し、フレームと動画を作る(ffmpeg が必要)
.venv/bin/python -m cloth_sort.pipeline --backend mujoco --seed 4 --n 12 --out runs/mj4 --video
.venv/bin/python -m cloth_sort.pipeline --backend 2d --seed 1 --n 12 --out runs/seed1 --video   # 軽量な 2D 版

# 3. dora-rs で分散実行(sim / brain / recorder の 3 ノード。既定で MuJoCo 版)
./run_dora.sh
./run_dora.sh graph     # 構成図

# 4. 提出用の動画(スライド + dora 版の実行フレーム)を作る
.venv/bin/python tools/make_submission_video.py --frames runs/dora/frames --log runs/dora/log.jsonl

# テスト
.venv/bin/python -m pytest -q
```

MuJoCo の描画は OpenGL(macOS では CGL)を使うので、描画が必要なコマンドはサンドボックス等で
ウィンドウサーバに繋げない環境だと失敗します。物理計算とテストは描画なしで動きます。

## 構成

```
cloth_sort/
  config.py            作業台・置き場のレイアウト(mm)。アームの土台位置は MuJoCo で到達範囲を実測して決めた
  sim/world.py         2D シミュレーション(布片・レイアウト・俯瞰画像)。MuJoCo 版もここの布片と筆跡を使う
  sim/mujoco_world.py  MuJoCo 版: 作業台と布片(薄い剛体)、Panda + 吸着ツール、IK、俯瞰カメラと斜め視点
  sim/assets/panda/    Franka Panda のモデル(mujoco_menagerie、Apache-2.0)に吸着ツールを足したもの
  sim/synth.py         手書き風番号の描画と学習データ合成(世界と学習で同じ描画器を使う)
  sim/fonts.py         学習用フォントと「見たことのない筆跡」用フォントの分離
  perception/detect.py 背景差分による布片検出(古典的手法)
  perception/model.py  1 文字 CNN
  perception/reader.py インク抽出 → 文字分割 → CNN → 番号文字列と確信度
  ml/train.py          データ生成 → 学習 → 未見フォントで評価 → 重みと指標を保存
  planning/planner.py  番号 → 置き場、読めないものは要確認トレイへ、近い順に実行
  robot/executor.py    RobotArm インターフェース、VirtualArm(2D)、PickPlace 状態機械
  pipeline.py          単一プロセスの実行(動画・ログ生成)。--backend で 2d / mujoco を選ぶ
  nodes/               dora ノード(sim / brain / recorder)
  viewer.py            動画フレームの描画(3D 視点 + 俯瞰カメラ + 状態パネル)
dataflow.yml           dora のノード構成(つなぎ方)
tools/                 提出動画の生成、読み取りの手試し、シーンの確認
tests/                 pytest(19 件)
```

## 設計の要点

- 読み取りは「1 文字 CNN + 連結成分で文字分割」。語彙は 0〜9 と「-」なので、学習で見ていない番号(例: 1-2)も読める
- 確信度がしきい値未満の布片は止めずに要確認トレイへ逃がす。誤った置き場に置くより、人が後で見るほうが安い
- 置いた直後にアームを視野外へ退避させ、カメラで読み直して計画と照合する(自己検証)。読み直しの確信度が低いものは NG ではなく「要確認」。ログは JSONL で残し、再学習データの種にできる
- ロボット側(sim)と AI 側(brain)を別プロセスに分け、dora の Arrow メッセージで接続。実機化はロボット側だけの差し替え
- MuJoCo 版のアームは、ツール先端の目標位置と向き(真下)からヤコビアンの減衰最小二乗で関節角を解き、位置サーボ(重力補償つき)で追従させる。目標点は速度制限つきで動かす
- 吸着は「カップが布片の上面に触れたら weld 拘束で固定、離すと解放」の模擬。布片は 1.2 mm 厚の剛体で、重力・接触・落下は物理計算に任せる。薄い布を指で挟む接触の物理は扱っていない

## 講義の技術スタック・推奨との対応

AI アルゴリズムそのものではなく「動かし続けるための基盤」を扱う講座でした。
本リポジトリが講義のどの内容に乗っていて、どこを意図的に変えたかを整理します。

| 講義で扱った内容 | 本リポジトリでの対応 |
|---|---|
| MLOps・RobOps(第1・2回: データ収集 → 学習 → 評価 → 運用のループ) | 実行ログを JSONL で残し、置いた後にカメラで読み直して計画と照合(自己検証)。低確信度は止めずに要確認トレイへ逃がし、人の確認結果を再学習データにできる構造 |

### 講義の想定と違う点(意図して外したもの)

- 把持は吸着の模擬(近接したら weld 拘束)で、布の変形や指で挟む接触は扱っていない。実機化のときは吸着ハンドを使うか、把持の物理を別途検証する必要がある

## ライセンスと注意

Apache-2.0 で公開しています(`LICENSE`)。Franka Panda の MJCF とメッシュは mujoco_menagerie 由来(Apache-2.0)です。
