"""模倣学習まわり。描画(カメラ画像)は使わず、場面生成・状態・行動の受け渡し・方策実行器を確かめる。"""

import numpy as np
import pytest

from cloth_sort import config
from cloth_sort.learning.env import ACTION_NAMES, DT, STATE_NAMES, PickPlaceEnv, make_layout, run_expert
from cloth_sort.learning.policy_arm import LearnedArm, PolicyPickPlace
from cloth_sort.robot.executor import Phase


def test_make_layout_is_deterministic_and_target_is_in_source_area():
    a, sa = make_layout(123)
    b, sb = make_layout(123)
    assert [p.center for p in a.pieces] == [p.center for p in b.pieces]
    assert sa.place_xy == sb.place_xy and sa.place_name == sb.place_name
    x0, y0, x1, y1 = config.SOURCE_AREA
    assert x0 < sa.pick_xy[0] < x1 and y0 < sa.pick_xy[1] < y1
    assert sa.target_id == 0 and a.pieces[0].placed_at is None
    names = {s.label for s in config.board_slots()} | {"review"}
    assert sa.place_name in names


def test_state_and_action_vectors():
    env = PickPlaceEnv()
    spec = env.reset(7)
    s = env.state()
    assert s.shape == (len(STATE_NAMES),) and s.dtype == np.float32
    assert abs(s[3] - 0.018) < 0.004                       # 指は開いている
    a = env.last_command()
    assert a.shape == (len(ACTION_NAMES),) and a[3] == pytest.approx(1.0)
    # 手先を 5 cm 横へ動かす行動を 1 秒ぶん与えると、その方向へ動く
    target = s[:3].copy()
    target[0] += 0.05
    for _ in range(int(1.0 / DT)):
        env.apply(np.array([*target, 1.0], dtype=np.float32))
    assert abs(env.world.tip_pos[0] - target[0]) < 0.01


class _ReplayController:
    """記録した行動をそのまま返す(方策の差し替え口を確かめる用)。"""

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def reset(self):
        self.i = 0

    def act(self, top, wrist, state, task):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a


@pytest.mark.slow
def test_replayed_expert_actions_drive_policy_executor_to_done():
    """教師の行動列を PolicyPickPlace 経由で再生すると同じ結果になる(行動の定義と実行器のつなぎの確認)。"""
    seed = 10005
    env = PickPlaceEnv()
    env.reset(seed)
    tr = run_expert(env, images=False)
    if not tr.success:
        pytest.skip("このシードでは教師が失敗した(把持は物理任せ)")
    actions = np.stack(tr.action)
    assert actions.shape[1] == len(ACTION_NAMES) and actions[:, 3].min() < 0.1 < actions[:, 3].max()

    env2 = PickPlaceEnv()
    spec = env2.reset(seed)
    arm = LearnedArm.__new__(LearnedArm)
    arm.world, arm.inner, arm.controller = env2.world, env2.arm, _ReplayController(actions)
    env2.observe = lambda: type("O", (), {"top": None, "wrist": None, "state": env2.state()})()
    pp = PolicyPickPlace(arm, spec.pick_xy, spec.place_xy, spec.place_name)
    pp.env.observe = env2.observe          # 描画を使わない
    assert pp.phase is Phase.POLICY         # 退避位置から始まる
    for _ in range(len(actions) + 200):
        pp.step(DT)
        if pp.finished:
            break
    assert pp.phase is Phase.DONE
    assert env2.world.pieces[spec.target_id].placed_at == spec.place_name
    assert env2.placed()
