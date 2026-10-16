"""模倣学習用の環境: 布 1 枚の pick & place を 1 エピソードとして回す。

観測(policy が見るもの)
- 俯瞰カメラ(256×128。AI 側のカメラを小さくしたもの)
- 手首カメラ(256×128。ハンドに固定した eye-in-hand)
- 状態 8 次元: 手先位置 xyz [m]、指の開き [m]、目標(つまむ点 xy、置く点 xy)[m]
  目標は上位(検出 + 番号読み取り + 計画)が決める。方策は「どう動かすか」だけを学ぶ

行動(policy が出すもの)
- 手先の目標位置 xyz [m] と、指の指令(0 = 閉、1 = 開)。20 Hz
- 関節角への変換(IK)と位置サーボは手続き的な把持と同じものを使う。実機で言えばモータドライバ相当

教師は手続き的な把持(GripperArm + PickPlace)。1 エピソードは退避位置から始まり、放して上空へ戻るまで。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from cloth_sort import config
from cloth_sort.robot.executor import Phase, PickPlace
from cloth_sort.sim.mujoco_cloth_world import OPEN_CTRL, GripperArm, MujocoClothWorld
from cloth_sort.sim.mujoco_world import NQ, mm_to_world
from cloth_sort.sim.world import World

IMG_W, IMG_H = 256, 128
FPS = 20
DT = 1.0 / FPS
STATE_NAMES = ["tip_x", "tip_y", "tip_z", "finger", "goal_pick_x", "goal_pick_y", "goal_place_x", "goal_place_y"]
# "relative": 目標と手先の差(目標 − 手先)も足した 12 次元。置く位置のずれを減らす狙い(方策は差を 0 にすればよい)
STATE_NAMES_RELATIVE = STATE_NAMES + ["pick_dx", "pick_dy", "place_dx", "place_dy"]
ACTION_NAMES = ["cmd_x", "cmd_y", "cmd_z", "grip"]
TASK = "numbered cloth piece: pick it up by the edge and drop it on the target"
PLACE_TOL_MM = 50.0        # 置き場の半ピッチ。この範囲に布の重心があれば「置けた」
MAX_EPISODE_S = 45.0


@dataclass
class EpisodeSpec:
    seed: int
    n_pieces: int
    target_id: int
    pick_xy: tuple[float, float]
    place_xy: tuple[float, float]
    place_name: str
    goal_place_xy: tuple[float, float] | None = None    # 方策に渡す置く点(系統誤差の補正後)。None なら place_xy


@dataclass
class Observation:
    top: np.ndarray          # (IMG_H, IMG_W, 3) uint8
    wrist: np.ndarray        # (IMG_H, IMG_W, 3) uint8
    state: np.ndarray        # (8,) float32


REVIEW_SHARE = 0.30     # 要確認トレイ行きの割合。最初の 160 エピソードは 0.15 で集めた(トレイ行きが弱かったので増やした)


def make_layout(seed: int, review_share: float = REVIEW_SHARE) -> tuple[World, EpisodeSpec]:
    """布 1〜3 枚の場面。1 枚目が運ぶ対象。残りは仕分け途中の盤面を模して、半分は置き場に置いておく。"""
    rng = np.random.default_rng(seed)
    n = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
    layout = World.random_scene(n, seed)
    slots = config.board_slots()
    if rng.random() < 1.0 - review_share:      # 0.15 のとき最初の収集・評価と同じ場面になる
        s = slots[int(rng.integers(len(slots)))]
        place_xy, name = s.center, s.label
    else:
        place_xy, name = config.review_tray_position(int(rng.integers(0, 7))), "review"
    used = {name}
    for p in layout.pieces[1:]:
        if rng.random() < 0.5:
            free = [s for s in slots if s.label not in used]
            s = free[int(rng.integers(len(free)))]
            used.add(s.label)
            p.center = (s.center[0] + float(rng.uniform(-8, 8)), s.center[1] + float(rng.uniform(-8, 8)))
            p.angle_deg = float(rng.uniform(-30, 30))
            p.placed_at = s.label
    target = layout.pieces[0]
    return layout, EpisodeSpec(seed, n, target.id, target.center, place_xy, name)


class PickPlaceEnv:
    """1 エピソード = 1 枚。reset(seed) で場面を作り、observe() / apply(action) で回す。"""

    def __init__(self) -> None:
        self.world: MujocoClothWorld | None = None
        self.arm: GripperArm | None = None
        self.spec: EpisodeSpec | None = None
        self.time_s = 0.0
        self._renderers: dict[str, mujoco.Renderer] = {}

    # ---------- 場面 ----------
    def reset(self, seed: int, review_share: float = REVIEW_SHARE) -> EpisodeSpec:
        layout, spec = make_layout(seed, review_share)
        self.world = MujocoClothWorld.from_layout(layout)
        self.attach(self.world, spec)
        self.start_from_park()
        self.world.step(0.4)                      # 布が机に落ち着くまで
        spec.pick_xy = self.world.pieces[spec.target_id].center
        return spec

    def attach(self, world: MujocoClothWorld, spec: EpisodeSpec, arm: GripperArm | None = None) -> None:
        """既存の世界(パイプライン実行中の机)にこの環境の観測・行動を重ねる。"""
        self.world = world
        self.spec = spec
        self.arm = arm or GripperArm(world)
        self.time_s = 0.0
        self._renderers = {}

    def start_from_park(self) -> None:
        d = self.world.data
        d.qpos[:NQ] = self.arm.park_q
        d.qvel[:NQ] = 0.0
        d.ctrl[:NQ] = self.arm.park_q
        d.ctrl[NQ] = OPEN_CTRL
        mujoco.mj_forward(self.world.model, d)
        self.arm._cmd[:] = self.world.tip_pos
        self.arm._vel = 0.0

    # ---------- 観測 ----------
    def _render(self, camera: str) -> np.ndarray:
        r = self._renderers.get(camera)
        if r is None:
            r = self._renderers[camera] = mujoco.Renderer(self.world.model, height=IMG_H, width=IMG_W)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        r.update_scene(self.world.data, camera=camera)
        return r.render().copy()

    def goal_vector(self) -> np.ndarray:
        place = self.spec.goal_place_xy if self.spec.goal_place_xy is not None else self.spec.place_xy
        return np.array([*mm_to_world(self.spec.pick_xy), *mm_to_world(place)], dtype=np.float32)

    state_mode: str = "basic"          # "basic"(8 次元)か "relative"(12 次元)。読み込んだ方策に合わせる

    def state(self) -> np.ndarray:
        tip = self.world.tip_pos
        finger = float(self.world.data.qpos[NQ])
        basic = np.array([tip[0], tip[1], tip[2], finger, *self.goal_vector()], dtype=np.float32)
        return with_relative_goal(basic) if self.state_mode == "relative" else basic

    def observe(self) -> Observation:
        return Observation(self._render("top"), self._render("wristcam"), self.state())

    # ---------- 行動 ----------
    def last_command(self) -> np.ndarray:
        """直前の周期にアームへ与えた指令(手続き的な把持を教師にするときに読む)。"""
        return np.array([*self.arm._cmd, float(self.world.data.ctrl[NQ]) / OPEN_CTRL], dtype=np.float32)

    def apply(self, action: np.ndarray, dt: float = DT) -> None:
        """方策の行動を 1 周期ぶん実行する。IK + 位置サーボは手続き的な把持と共通。"""
        self.arm._cmd[:] = action[:3]
        self.arm._vel = 0.0
        self.world.data.ctrl[NQ] = float(np.clip(action[3], 0.0, 1.0)) * OPEN_CTRL
        self.arm._servo(dt)
        self.time_s += dt

    # ---------- 判定 ----------
    def cloth_centroid_mm(self) -> tuple[float, float, float]:
        v = self.world.cloth_vertices(self.spec.target_id)
        c = v.mean(axis=0)
        from cloth_sort.sim.mujoco_world import world_to_mm
        x, y = world_to_mm(float(c[0]), float(c[1]))
        return x, y, float(c[2])

    def placed(self) -> bool:
        x, y, z = self.cloth_centroid_mm()
        px, py = self.spec.place_xy
        return abs(x - px) <= PLACE_TOL_MM and abs(y - py) <= PLACE_TOL_MM and z < 0.02

    def mark_placed(self) -> None:
        p = self.world.pieces[self.spec.target_id]
        p.held = False
        p.placed_at = self.spec.place_name
        self.world.held_piece = None
        self.world.gripper_closed = False


def with_relative_goal(basic: np.ndarray) -> np.ndarray:
    """8 次元の状態に(目標 − 手先)の xy を足して 12 次元にする。記録済みのデータにも後から適用できる。"""
    basic = np.asarray(basic, dtype=np.float32)
    tip = basic[..., 0:2]
    rel = np.concatenate([basic[..., 4:6] - tip, basic[..., 6:8] - tip], axis=-1)
    return np.concatenate([basic, rel], axis=-1).astype(np.float32)


@dataclass
class Trajectory:
    spec: EpisodeSpec
    top: list[np.ndarray] = field(default_factory=list)
    wrist: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    success: bool = False
    grip_retries: int = 0
    final_phase: str = ""
    sim_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.action)


NOISE_PHASES = {Phase.APPROACH, Phase.CARRY, Phase.RETREAT}     # 精度が要らない工程だけ外乱を入れる


def run_expert(env: PickPlaceEnv, record: bool = True, images: bool = True, on_step=None,
               noise_m: float = 0.0, noise_seed: int = 0) -> Trajectory:
    """手続き的な把持で 1 エピソード回し、(観測, 行動) を記録する。images=False なら描画せず状態と行動だけ残す。

    noise_m > 0 なら DART 方式: 接近・搬送・退避の間、実際に動かす手先目標に外乱(標準偏差 noise_m [m]、
    低域通過)を足し、記録する行動は教師のきれいな指令のまま。方策が少しずれた状態からの戻り方を学べる。
    """
    spec = env.spec
    pp = PickPlace(env.arm, spec.pick_xy, spec.place_xy, spec.place_name)
    tr = Trajectory(spec)
    rng = np.random.default_rng(noise_seed)
    noise = np.zeros(3)
    env.arm.cmd_offset = np.zeros(3)
    while True:
        if noise_m > 0:
            target = rng.normal(0.0, noise_m, 3) if pp.phase in NOISE_PHASES else np.zeros(3)
            noise = 0.9 * noise + 0.1 * target                 # ゆっくり変わる外乱
            env.arm.cmd_offset = noise
        if record:
            if images:
                o = env.observe()
                tr.top.append(o.top)
                tr.wrist.append(o.wrist)
                tr.state.append(o.state)
            else:
                tr.state.append(env.state())
        phase = pp.step(DT)
        env.time_s += DT
        env.world.time_s = env.time_s
        tr.phases.append(phase.value)
        if record:
            tr.action.append(env.last_command())
        if on_step is not None:
            on_step()
        if phase in (Phase.PARK, Phase.DONE, Phase.FAILED) or env.time_s > MAX_EPISODE_S:
            break
    env.arm.cmd_offset = np.zeros(3)
    tr.grip_retries = pp.grip_retries
    tr.final_phase = phase.value
    tr.sim_seconds = env.time_s
    tr.success = phase is not Phase.FAILED and env.placed()
    return tr
