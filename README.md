# 番号付き布片の仕分け — Physical AI 応用編1 最終課題

手書きの番号が書かれた 5 cm 角の布片を、番号どおりの置き場へ運ぶロボットのパイプラインです。
「撮影 → 布片の検出 → 番号の読み取り(自前学習した CNN) → 計画 → pick & place → カメラで検証」を
dora-rs の dataflow としてつなぎ、MuJoCo の物理シミュレーション上で Franka Panda を動かしています。
布は変形体(MuJoCo の flex)で、2 本指ハンド(ゴムパッド付き)でつまむ把持そのものを物理に任せています。

- 提出動画: `runs/submission/PAI最終課題_katsuhisa91.mp4`(約 2 分半、1920x1080。説明スライド + 単一プロセス版の実行(2 倍速、画面に明記))
- 実行部分のみ: `runs/cloth8/cloth_sort.mp4`(単一プロセス版、seed 4、等速)。dora 版の実行は `./run_dora.sh` で `runs/dora/frames` に出る(同じ seed で搬送中の落下が 4 枚。把持の成否は物理の微小な差で変わる)
- 報告書: `report/report.pdf`(元は `report/report.html`)
- 提出の流れ(public 化、YouTube、omnicampus): `SUBMISSION.md`
- 実機は未接続。アームは `RobotArm` インターフェースの MuJoCo 実装(`GripperArm`)で、実機対応はこの 1 クラスを足す形
- 3 つのバックエンドを切り替えられる: `cloth`(変形体の布 + 2 本指、既定)/ `mujoco`(剛体の布片 + 吸着の模擬)/ `2d`(自前の 2D、軽量)

## 動かし方

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH=$PWD

# 1. 番号読み取りモデルを学習(合成データ、CPU で 5 分程度)
.venv/bin/python -m cloth_sort.ml.train --out models

# 2. 単一プロセスで実行し、フレームと動画を作る(ffmpeg が必要)
.venv/bin/python -m cloth_sort.pipeline --backend cloth --seed 4 --n 8 --out runs/cloth8 --video    # 変形体 + 2 本指(30 分ほど)
.venv/bin/python -m cloth_sort.pipeline --backend mujoco --seed 4 --n 12 --out runs/mj4 --video    # 剛体 + 吸着(2 分ほど)
.venv/bin/python -m cloth_sort.pipeline --backend 2d --seed 1 --n 12 --out runs/seed1 --video      # 2D(数秒)

# 3. dora-rs で分散実行(sim / brain / recorder の 3 ノード。既定は cloth)
./run_dora.sh
./run_dora.sh graph     # 構成図

# 4. 提出用の動画(スライド + dora 版の実行フレーム、実行部分は 2 倍速)を作る
.venv/bin/python tools/make_submission_video.py --frames runs/cloth8/frames --log runs/cloth8/log.jsonl --summary runs/cloth8/summary.json --fig runs/cloth8/top_detect.png --run-speed 2

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
  sim/mujoco_world.py  MuJoCo 版(剛体): 作業台と布片(薄い剛体)、Panda + 吸着ツール、IK、俯瞰カメラと斜め視点
  sim/mujoco_cloth_world.py  MuJoCo 版(変形体): 布を flex で作り、Panda の 2 本指ハンドでつまむ。GripperArm(閉じる・放す・再試行)
  sim/assets/panda/    Franka Panda のモデル(mujoco_menagerie、Apache-2.0)。吸着ツール版と、ゴムパッド付き 2 本指ハンド版
  sim/synth.py         手書き風番号の描画と学習データ合成(世界と学習で同じ描画器を使う)
  sim/fonts.py         学習用フォントと「見たことのない筆跡」用フォントの分離
  perception/detect.py 背景差分による布片検出(古典的手法)
  perception/model.py  1 文字 CNN
  perception/reader.py インク抽出 → 文字分割 → CNN → 番号文字列と確信度
  ml/train.py          データ生成 → 学習 → 未見フォントで評価 → 重みと指標を保存
  planning/planner.py  番号 → 置き場、読めないものは要確認トレイへ、近い順に実行
  robot/executor.py    RobotArm インターフェース、VirtualArm(2D)、PickPlace 状態機械(再試行・搬送中の落下検知)
  pipeline.py          単一プロセスの実行(動画・ログ生成)。--backend で cloth / mujoco / 2d を選ぶ
  nodes/               dora ノード(sim / brain / recorder)
  viewer.py            動画フレームの描画(3D 視点 + ハンド追従カメラ + 俯瞰カメラ + 状態パネル)
dataflow.yml           dora のノード構成(つなぎ方)
tools/                 提出動画の生成、読み取りの手試し、シーンの確認
tests/                 pytest(22 件)
```

## 設計の要点

- 読み取りは「1 文字 CNN + 連結成分で文字分割」。語彙は 0〜9 と「-」なので、学習で見ていない番号(例: 1-2)も読める
- 確信度がしきい値未満の布片は止めずに要確認トレイへ逃がす。誤った置き場に置くより、人が後で見るほうが安い
- 置いた直後にアームを視野外へ退避させ、カメラで読み直して計画と照合する(自己検証)。読み直しの確信度が低いものは NG ではなく「要確認」。ログは JSONL で残し、再学習データの種にできる
- ロボット側(sim)と AI 側(brain)を別プロセスに分け、dora の Arrow メッセージで接続。実機化はロボット側だけの差し替え
- MuJoCo 版のアームは、ツール先端の目標位置と向き(真下)からヤコビアンの減衰最小二乗で関節角を解き、位置サーボ(重力補償つき)で追従させる。目標点は速度制限つきで動かす
- 布は MuJoCo の flex(2D シェル、9×9 格子、厚さ 3 mm、ヤング率 1e5 のフェルト程度)。重力・接触・曲げは物理計算
- 把持は Panda の 2 本指ハンド + ゴムパッド(摩擦 2.0)で、物理に任せる。平らに置かれた布を平行グリッパでつまむために、左指のパッドを 2.5 mm 長い「唇」にして机面まで押し込み、閉じる間はハンド全体を動かして押さえ側の指を世界に対して止める(折り目が押さえ指の前で立ち、2 枚重ねでつままれる)。持ち上げ 3 cm/s、搬送 6 cm/s を超えると抜ける
- 置くときは置き場の上で放して落とす。布は折れて落ちるので番号は読み直せず、置いた後の検証は「置き場の領域に布があるか」を背景差分で確かめる方式に変えた。つかみ損ねは撮り直して最大 2 回再試行し、搬送中に落としたら失敗として人に回す
- 剛体 + 吸着版(`--backend mujoco`)も残してある。こちらは置いた後に番号を読み直す検証ができる

## 講義の技術スタック・推奨との対応

AI アルゴリズムそのものではなく「動かし続けるための基盤」を扱う講座でした。
本リポジトリが講義のどの内容に乗っていて、どこを意図的に変えたかを整理します。

| 講義で扱った内容 | 本リポジトリでの対応 |
|---|---|
| MLOps・RobOps(第1・2回: データ収集 → 学習 → 評価 → 運用のループ) | 実行ログを JSONL で残し、置いた後にカメラで読み直して計画と照合(自己検証)。低確信度は止めずに要確認トレイへ逃がし、人の確認結果を再学習データにできる構造 |

### 講義の想定と違う点(意図して外したもの)

- 把持は物理に任せたが、置き方は妥協した。布は折れたまま置き場に落ちるので、平らに並べる工程(実機なら治具か人手)は別に要る。剛体 + 吸着版はその限界を持たない代わりに「つかめるか」を検証していない。両方を残して、どちらの限界も報告書に書いた

## ライセンスと注意

Apache-2.0 で公開しています(`LICENSE`)。Franka Panda の MJCF とメッシュは mujoco_menagerie 由来(Apache-2.0)です。
