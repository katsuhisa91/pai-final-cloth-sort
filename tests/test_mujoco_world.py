"""MuJoCo 版の世界とアーム。描画(OpenGL)は使わず、物理と制御だけを確かめる。"""

import numpy as np
import pytest

from cloth_sort import config
from cloth_sort.robot.executor import Phase, PickPlace
from cloth_sort.sim.mujoco_world import (CONTACT_Z, LIFT_Z, MujocoArm, MujocoWorld, mm_to_world,
                                         world_to_mm)


def test_mm_world_roundtrip():
    for xy in [(0, 0), (600, 300), (1200, 600), (123.4, 567.8)]:
        x, y = mm_to_world(xy)
        back = world_to_mm(x, y)
        assert abs(back[0] - xy[0]) < 1e-6 and abs(back[1] - xy[1]) < 1e-6


def test_pieces_rest_on_bench_after_settling():
    w = MujocoWorld.random_scene(5, seed=2)
    before = [p.center for p in w.pieces]
    w.step(0.5)
    for p, c in zip(w.pieces, before):
        z = w.data.xpos[w.model.body(f"piece_{p.id}").id][2]
        assert 0.0 < z < 0.003                                    # 机の上に載っている
        assert abs(p.center[0] - c[0]) < 3 and abs(p.center[1] - c[1]) < 3   # 落下で横ずれしない


def test_arm_reaches_target_with_tool_pointing_down():
    w = MujocoWorld.random_scene(2, seed=1)
    w.step(0.3)
    arm = MujocoArm(w)
    for _ in range(200):
        if arm.move_to((300, 200), 1.0, 0.05):
            break
    goal = np.array([*mm_to_world((300, 200)), LIFT_Z])
    assert np.linalg.norm(w.tip_pos - goal) < 0.004
    zaxis = w.data.site_xmat[w.model.site("tool_tip").id].reshape(3, 3)[:, 2]
    assert zaxis[2] < -0.98                                        # ツールの z 軸が真下


def test_suction_only_near_piece_and_piece_follows_tool():
    w = MujocoWorld.random_scene(2, seed=4)
    w.step(0.3)
    arm = MujocoArm(w)
    p = w.pieces[0]
    for _ in range(300):
        if arm.move_to(p.center, 1.0, 0.05):
            break
    assert not arm.grip()                                          # 上空では吸着しない
    for _ in range(100):
        if arm.move_to(p.center, 0.0, 0.05):
            break
    assert arm.grip() and p.held and w.held_piece == p.id
    for _ in range(100):
        if arm.move_to(p.center, 1.0, 0.05):
            break
    z = w.data.xpos[w.model.body(f"piece_{p.id}").id][2]
    assert z > LIFT_Z - 0.03                                       # 一緒に持ち上がる
    arm.release("7")
    w.step(0.5)
    z = w.data.xpos[w.model.body(f"piece_{p.id}").id][2]
    assert z < 0.003 and p.placed_at == "7" and not p.held        # 離すと落ちて机に載る


def test_pick_place_parks_outside_camera_view():
    w = MujocoWorld.random_scene(1, seed=9)
    w.step(0.3)
    arm = MujocoArm(w)
    p = w.pieces[0]
    slot = config.board_slots()[2]
    pp = PickPlace(arm, p.center, slot.center, slot.label)
    for _ in range(3000):
        if pp.step(0.05) is Phase.DONE:
            break
    assert pp.phase is Phase.DONE
    assert abs(p.center[0] - slot.center[0]) < 8 and abs(p.center[1] - slot.center[1]) < 8
    assert p.placed_at == slot.label
    gx, gy = w.gripper_xy
    assert gy > config.BENCH_H_MM                                  # 待機姿勢ではツールが机の外(カメラに写らない)
    park = w.model.key("park").qpos[:7]
    assert np.all(np.abs(w.data.qpos[:7] - park) < 0.05)           # 関節が待機姿勢に戻っている
