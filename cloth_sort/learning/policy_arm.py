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


def episode_done(closed_once: bool, grip: float, tip_xyz, place_xyz, steps: int) -> bool:
    """方策のエピソード終了判定: 一度閉じたあと、置く点の近くで開いて手先が上がったら終わり。"""
    if not (closed_once and grip >= GRIP_OPEN and float(tip_xyz[2]) >= RETREAT_Z and steps > 40):
        return False
    dx, dy = float(tip_xyz[0]) - float(place_xyz[0]), float(tip_xyz[1]) - float(place_xyz[1])
    return (dx * dx + dy * dy) ** 0.5 <= NEAR_PLACE_M


class ACTController:
    """学習済み ACT を読み込み、観測 → 行動(手先目標 xyz + 指)を返す。毎周期推論し、時間方向にアンサンブルする。"""

    ENSEMBLE_COEFF = -0.05    # 時間方向の平均は新しい予測を重く(ACT 既定 0.01 は古い予測を重くし、止まる位置が遅れた)

    def __init__(self, path: str | Path, device: str | None = None, ensemble_coeff: float | None = ENSEMBLE_COEFF) -> None:
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
        else:
            cfg.temporal_ensemble_coeff = None       # 100 ステップぶんを予測してそのまま実行(5 秒ごとに観測し直す)
            cfg.n_action_steps = cfg.chunk_size
        self.policy = ACTPolicy.from_pretrained(path, config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self.state_dim = int(cfg.input_features["observation.state"].shape[0])
        self.state_mode = "relative" if self.state_dim == 12 else "basic"
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

    def __init__(self, world: MujocoClothWorld, controller: ACTController, place_bias_mm: float | None = None) -> None:
        self.world = world
        self.inner = GripperArm(world)
        self.controller = controller
        self.place_bias_mm = PLACE_BIAS_MM if place_bias_mm is None else place_bias_mm

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
        return PolicyPickPlace(self, pick_xy, place_xy, place_name, self.place_bias_mm)


RETRY_RADIUS_MM = 150.0   # つかみ損ねた布がこの範囲に残っていれば撮り直して再試行(手続き版の relocate と同じ発想)
PLACE_BIAS_MM = 22.0      # 方策が進行方向に止まり遅れる系統誤差の補正量 [mm]。評価とは別の 20 場面(seed 21000〜)で測った平均(sd 15)


def corrected_place(pick_xy, place_xy, bias_mm: float) -> tuple[float, float]:
    """置く点を、進行方向(つまむ点 → 置く点)の手前へ bias_mm だけずらす(キャリブレーション)。"""
    dx, dy = place_xy[0] - pick_xy[0], place_xy[1] - pick_xy[1]
    n = (dx * dx + dy * dy) ** 0.5
    if bias_mm == 0.0 or n < 1e-6:
        return tuple(place_xy)
    return (place_xy[0] - bias_mm * dx / n, place_xy[1] - bias_mm * dy / n)


class PolicyPickPlace:
    """1 枚ぶんを方策で実行する。退避位置から始め(必要なら先に退避)、放して上がったら退避して完了。

    置けていなければ、布が元の位置の近くに残っている限り最大 max_grip_retries 回まで撮り直して再試行する
    (手続き版と同じ上位の立て直し。方策そのものは変えない)。
    """

    max_grip_retries = 2

    def __init__(self, arm: LearnedArm, pick_xy, place_xy, place_name: str, place_bias_mm: float | None = None) -> None:
        self.arm = arm
        self.pick_xy = tuple(pick_xy)
        self.place_xy = tuple(place_xy)
        self.place_name = place_name
        self.place_bias_mm = PLACE_BIAS_MM if place_bias_mm is None else place_bias_mm
        self.grip_retries = 0
        self.lost_in_transit = False
        self.timed_out = False
        self.env = PickPlaceEnv()
        target = arm.world.piece_at(self.pick_xy, tol_mm=45)
        self.target_id = target.id if target else 0
        self._attach()
        near_park = bool(np.all(np.abs(arm.world.data.qpos[:7] - arm.inner.park_q) < 0.05))
        self.phase = Phase.POLICY if near_park else Phase.PARK
        self._pre_park = not near_park

    def _attach(self) -> None:
        spec = EpisodeSpec(0, len(self.arm.world.pieces), self.target_id, self.pick_xy, self.place_xy, self.place_name,
                           goal_place_xy=corrected_place(self.pick_xy, self.place_xy, self.place_bias_mm))
        self.env.attach(self.arm.world, spec, self.arm.inner)
        self.env.state_mode = getattr(self.arm.controller, "state_mode", "basic")
        self.steps = 0
        self.closed_once = False
        self._settle = 0.0
        self._last_action = None
        self.arm.controller.reset()

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
                self.arm.inner._cmd[:] = a[:3]       # 以後の放す動作(振り落としを含む)は手続き版の release_step に任せる
            elif self.steps * dt > MAX_EPISODE_S:
                self.timed_out = True
                self._retry_or_fail()
            return self.phase
        if self.phase is Phase.RELEASE:
            # 放す動作そのものは手続き版と共通: 全開にして落ち着くまで待ち、布が指に引っかかっていれば横に振って落とす
            inner = self.arm.inner
            if inner.holding_id is None:
                inner.holding_id = self.target_id
            if inner.release_step(dt, None) is None:
                return self.phase
            if self.env.placed():
                self.env.mark_placed()
                self.phase = Phase.PARK
            else:
                self._retry_or_fail()
        return self.phase

    def _retry_or_fail(self) -> None:
        """置けていない。布が元の位置の近くにあれば撮り直して再試行、遠くに落ちていれば失敗(人が拾う)。"""
        self.arm.inner.release(None)
        if self.grip_retries < self.max_grip_retries and not self.timed_out:
            x, y, z = self.env.cloth_centroid_mm()
            if z < 0.02 and ((x - self.pick_xy[0]) ** 2 + (y - self.pick_xy[1]) ** 2) ** 0.5 <= RETRY_RADIUS_MM:
                self.grip_retries += 1
                self.pick_xy = self.arm.inner.relocate(self.pick_xy) if getattr(self.arm.world, "background", None) is not None else (x, y)
                self._attach()
                self.phase = Phase.PARK           # 退避位置に戻ってから、方策をもう一度最初から回す
                self._pre_park = True
                return
        self.lost_in_transit = self.closed_once
        self.phase = Phase.FAILED
