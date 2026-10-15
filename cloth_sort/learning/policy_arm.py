"""学習した方策(LeRobot の ACT)でアームを動かす。

上位(検出・番号読み取り・計画・検証)と下位(IK + 位置サーボ)はそのままで、
「目標が与えられてから、つまんで運んで放すまでの手先の動き」を方策に置き換える。
PickPlace(手続き的な状態機械)と同じ顔(step / finished / phase / grip_retries / lost_in_transit)を持つので、
pipeline.py と dora の sim ノードはどちらでも動く。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloth_sort.learning.env import MAX_EPISODE_S, TASK, EpisodeSpec, PickPlaceEnv
from cloth_sort.robot.executor import Phase
from cloth_sort.sim.mujoco_cloth_world import GripperArm, MujocoClothWorld

GRIP_CLOSED = 0.3      # 指の指令がこれ以下なら「閉じた」
GRIP_OPEN = 0.7        # これ以上なら「開いた」
RETREAT_Z = 0.05       # 放したあと手先がこの高さまで上がったらエピソード終了
NEAR_PLACE_M = 0.08    # 置く点からこの距離以内で開いたときだけ「置いた」とみなす(つかみ損ねて開き直すのと区別する)
SETTLE_S = 1.0         # 開いてから布が落ち切るまで待つ時間(判定はその後)


def episode_done(closed_once: bool, grip: float, tip_xyz, place_xyz, steps: int) -> bool:
    """方策のエピソード終了判定: 一度閉じたあと、置く点の近くで開いて手先が上がったら終わり。"""
    if not (closed_once and grip >= GRIP_OPEN and float(tip_xyz[2]) >= RETREAT_Z and steps > 40):
        return False
    dx, dy = float(tip_xyz[0]) - float(place_xyz[0]), float(tip_xyz[1]) - float(place_xyz[1])
    return (dx * dx + dy * dy) ** 0.5 <= NEAR_PLACE_M


class ACTController:
    """学習済み ACT を読み込み、観測 → 行動(手先目標 xyz + 指)を返す。毎周期推論し、時間方向にアンサンブルする。"""

    def __init__(self, path: str | Path, device: str | None = None, ensemble_coeff: float | None = 0.01) -> None:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.torch = torch
        path = str(path)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        if ensemble_coeff is not None:
            cfg.temporal_ensemble_coeff = ensemble_coeff
            cfg.n_action_steps = 1
        self.policy = ACTPolicy.from_pretrained(path, config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def act(self, top: np.ndarray, wrist: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        t = self.torch
        batch = {
            "observation.images.top": t.from_numpy(top).permute(2, 0, 1).float() / 255.0,
            "observation.images.wrist": t.from_numpy(wrist).permute(2, 0, 1).float() / 255.0,
            "observation.state": t.from_numpy(state).float(),
            "task": task,
        }
        with t.inference_mode():
            batch = self.pre(batch)
            action = self.policy.select_action(batch)
            action = self.post(action)
        return action[0].cpu().numpy().astype(np.float32)


class LearnedArm:
    """GripperArm を包み、1 枚ぶんの実行を方策に任せる。pose / park などは GripperArm に委譲する。"""

    def __init__(self, world: MujocoClothWorld, controller: ACTController) -> None:
        self.world = world
        self.inner = GripperArm(world)
        self.controller = controller

    @property
    def pose(self):
        return self.inner.pose

    @property
    def limits(self):
        return self.inner.limits

    def park(self, dt: float) -> bool:
        return self.inner.park(dt)

    def move_to(self, xy, z, dt):
        return self.inner.move_to(xy, z, dt)

    def release(self, placed_at):
        self.inner.release(placed_at)

    def pick_place(self, pick_xy, place_xy, place_name: str) -> "PolicyPickPlace":
        return PolicyPickPlace(self, pick_xy, place_xy, place_name)


class PolicyPickPlace:
    """1 枚ぶんを方策で実行する。退避位置から始め(必要なら先に退避)、放して上がったら退避して完了。"""

    max_grip_retries = 0

    def __init__(self, arm: LearnedArm, pick_xy, place_xy, place_name: str) -> None:
        self.arm = arm
        self.pick_xy = tuple(pick_xy)
        self.place_xy = tuple(place_xy)
        self.place_name = place_name
        self.grip_retries = 0
        self.lost_in_transit = False
        self.steps = 0
        self.closed_once = False
        self.timed_out = False
        self._settle = 0.0
        self._last_action = None
        self.env = PickPlaceEnv()
        target = arm.world.piece_at(self.pick_xy, tol_mm=45)
        target_id = target.id if target else 0
        spec = EpisodeSpec(0, len(arm.world.pieces), target_id, self.pick_xy, self.place_xy, place_name)
        self.env.attach(arm.world, spec, arm.inner)
        near_park = bool(np.all(np.abs(arm.world.data.qpos[:7] - arm.inner.park_q) < 0.05))
        self.phase = Phase.POLICY if near_park else Phase.PARK
        self._pre_park = not near_park
        arm.controller.reset()

    @property
    def finished(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED)

    def step(self, dt: float) -> Phase:
        if self.phase is Phase.PARK:
            if self.arm.park(dt):
                if self._pre_park:
                    self._pre_park = False
                    self.env.arm._cmd[:] = self.arm.world.tip_pos
                    self.phase = Phase.POLICY
                else:
                    self.phase = Phase.DONE
            return self.phase
        if self.phase is Phase.POLICY:
            o = self.env.observe()
            a = self.arm.controller.act(o.top, o.wrist, o.state, TASK)
            self.env.apply(a, dt)
            self.steps += 1
            grip = float(a[3])
            if grip <= GRIP_CLOSED:
                self.closed_once = True
            self._last_action = a
            if episode_done(self.closed_once, grip, self.arm.world.tip_pos, self.env.goal_vector()[2:4], self.steps):
                self.phase = Phase.RELEASE           # 開いた。布が落ち切るまで待ってから判定する
            elif self.steps * dt > MAX_EPISODE_S:
                self.timed_out = True
                self._fail()
            return self.phase
        if self.phase is Phase.RELEASE:
            self.env.apply(self._last_action, dt)
            self._settle += dt
            if self._settle >= SETTLE_S:
                if self.env.placed():
                    self.env.mark_placed()
                    self.phase = Phase.PARK
                else:
                    self._fail()
        return self.phase

    def _fail(self) -> None:
        self.arm.inner.release(None)
        self.lost_in_transit = self.closed_once
        self.phase = Phase.FAILED
