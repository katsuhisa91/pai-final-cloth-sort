from cloth_sort import config
from cloth_sort.planning.planner import Destination, Observation, Planner
from cloth_sort.robot.executor import Phase, PickPlace, VirtualArm
from cloth_sort.sim.world import World


def test_planner_routes_each_case():
    p = Planner(min_confidence=0.6)
    obs = [
        Observation((100, 100), "3", 0.95),     # 正常
        Observation((200, 100), "", 0.0),       # 読めない
        Observation((300, 100), "7", 0.2),      # 低確信度
        Observation((400, 100), "99", 0.9),     # 置き場に無い
        Observation((500, 100), "3", 0.9),      # 重複
    ]
    tasks = p.plan(obs, gripper_xy=(0, 0))
    by_pick = {t.pick_xy: t for t in tasks}
    assert by_pick[(100, 100)].destination is Destination.SLOT
    assert by_pick[(100, 100)].place_xy == dict((s.label, s.center) for s in config.board_slots())["3"]
    for xy in [(200, 100), (300, 100), (400, 100), (500, 100)]:
        assert by_pick[xy].destination is Destination.REVIEW
    assert p.review_count == 4
    # 要確認トレイの置き位置は重ならない
    review_xy = [t.place_xy for t in tasks if t.destination is Destination.REVIEW]
    assert len(set(review_xy)) == 4


def test_planner_nearest_first_and_filled_slots_persist():
    p = Planner()
    obs = [Observation((500, 500), "1", 0.9), Observation((100, 100), "2", 0.9)]
    tasks = p.plan(obs, gripper_xy=(90, 90))
    assert tasks[0].label == "2"
    p.mark_done(tasks[0])
    again = p.plan([Observation((100, 100), "2", 0.9)], gripper_xy=(0, 0))
    assert again[0].destination is Destination.REVIEW and "使用済み" in again[0].reason


def test_pick_place_moves_piece_to_slot():
    w = World.random_scene(2, seed=7)
    arm = VirtualArm(w)
    piece = w.pieces[0]
    target = config.board_slots()[4].center
    pp = PickPlace(arm, piece.center, target, "5")
    for _ in range(2000):
        if pp.step(0.05) is Phase.DONE:
            break
    assert pp.phase is Phase.DONE
    assert piece.center == target and piece.placed_at == "5"
    assert w.gripper_z == 1.0 and not w.gripper_closed


def test_pick_place_fails_after_retries_when_nothing_to_grip():
    w = World.random_scene(1, seed=7)
    arm = VirtualArm(w)
    pp = PickPlace(arm, (300, 300), (700, 130), "1", max_grip_retries=1)
    for _ in range(2000):
        if pp.finished:
            break
        pp.step(0.05)
    assert pp.phase is Phase.FAILED
    assert pp.grip_retries == 1


def test_virtual_arm_respects_speed_limit():
    w = World.random_scene(1, seed=7)
    arm = VirtualArm(w)
    start = w.gripper_xy
    arm.move_to((start[0] + 1000, start[1]), 1.0, dt=0.1)
    moved = w.gripper_xy[0] - start[0]
    assert abs(moved - arm.limits.xy_speed_mm_s * 0.1) < 1e-6
