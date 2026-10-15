"""模倣学習でアームの動きを学ぶ(LeRobot の ACT)。

- env.py        : 布 1 枚の pick & place を 1 エピソードとして回す環境(観測・行動の定義)
- collect.py    : 手続き的な把持(GripperArm + PickPlace)を教師にデモを集め、LeRobot のデータセットにする
- train.py      : lerobot-train(ACT)を既定の設定で起動する
- policy_arm.py : 学習した方策でアームを動かす(上位の計画・検証はそのまま)
- evaluate.py   : 学習した方策と手続き的な把持を、学習に使っていないシードで比べる
"""
