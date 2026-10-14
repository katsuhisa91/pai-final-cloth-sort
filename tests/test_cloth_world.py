"""変形体の布 + 2 本指ハンド。描画は使わず、物理と制御だけを確かめる(把持は時間がかかるので 1 枚)。"""

import numpy as np
import pytest

from cloth_sort import config
from cloth_sort.robot.executor import Phase, PickPlace
from cloth_sort.sim.mujoco_cloth_world import GripperArm, MujocoClothWorld, OPEN_CTRL


def test_cloth_rests_flat_on_bench():
    w = MujocoClothWorld.random_scene(2, seed=3)
    before = [p.center for p in w.pieces]
    w.step(0.5)
    for p, c in zip(w.pieces, before):
        v = w.cloth_vertices(p.id)
        assert 0.0 < v[:, 2].mean() < 0.004                        # 机に載っている(厚さ 3 mm)
        assert v[:, 2].max() - v[:, 2].min() < 0.003                # 平ら
        assert abs(p.center[0] - c[0]) < 3 and abs(p.center[1] - c[1]) < 3


def test_gripper_opens_and_tool_points_down():
    w = MujocoClothWorld.random_scene(1, seed=1)
    w.step(0.3)
    arm = GripperArm(w)
    for _ in range(200):
        if arm.move_to((300, 200), 1.0, 0.05):
            break
    assert abs(w.data.qpos[7] - OPEN_CTRL * 0.04 / 255) < 0.003      # 指が開いている
    zaxis = w.data.site_xmat[w.model.site("tool_tip").id].reshape(3, 3)[:, 2]
    assert zaxis[2] < -0.98


@pytest.mark.slow
def test_pinch_grasp_lifts_cloth_or_reports_failure():
    """つまみ上げは物理任せなので成功を保証しない。成功なら布が持ち上がり、失敗なら FAILED で止まる。"""
    w = MujocoClothWorld.random_scene(1, seed=4)
    w.step(0.4)
    arm = GripperArm(w)
    p = w.pieces[0]
    slot = config.board_slots()[0]
    pp = PickPlace(arm, p.center, slot.center, slot.label)
    lifted = False
    for _ in range(1500):
        phase = pp.step(0.05)
        if phase is Phase.CARRY:
            lifted = lifted or w.cloth_vertices(p.id)[:, 2].mean() > 0.02
        if pp.finished:
            break
    assert pp.finished
    if pp.phase is Phase.DONE:
        assert lifted and p.placed_at == slot.label
        assert abs(p.center[0] - slot.center[0]) < 40 and abs(p.center[1] - slot.center[1]) < 40
    else:
        assert pp.phase is Phase.FAILED and p.placed_at is None
        assert pp.lost_in_transit or pp.grip_retries == pp.max_grip_retries
