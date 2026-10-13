"""MuJoCo 版の作業台: Franka Panda + 吸着ツールで布片を運ぶ 3D 物理シミュレーション。

2D 版(world.py)と同じレイアウト(config)・同じ布片(Piece)・同じ筆跡を使い、
物理(重力・接触・落下)と描画(俯瞰カメラ・斜め視点)を MuJoCo に任せる。

- アーム: mujoco_menagerie の franka_emika_panda(Apache-2.0)に吸着ツールを足したもの。
  関節はヤコビアンの減衰最小二乗で逆運動学を解き、位置サーボで動かす。
- 把持: 吸着を「カップが布片の上面に触れたら weld 拘束で固定、離すと解放」で模擬する。
  薄い布を指で挟む接触の物理は扱わない(ここは正直に明記する)。
- カメラ: 俯瞰カメラは 1 px = 1 mm になるよう画角を決め、2D 版と同じ検出・読み取りに流す。
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from cloth_sort import config
from cloth_sort.sim.world import Piece, World

ASSETS = Path(__file__).parent / "assets" / "panda"
PANDA_XML = ASSETS / "panda_suction.xml"

# 世界座標: ロボット土台(link0)が原点、z は机面 = 0。机の中心は土台から +Y 側。
BENCH_C = (0.0, (config.ARM_BASE[1] - config.BENCH_H_MM / 2) / 1000.0)   # 土台から見た机中心
PIECE_HALF_T = 0.0006           # 布片の半分の厚さ(1.2 mm の薄い剛体として扱う。厚いと側面が俯瞰画像に写る)
LIFT_Z = 0.15                   # 搬送高さ(gripper_z = 1.0 のとき)
CONTACT_Z = 2 * PIECE_HALF_T + 0.002   # カップ先端の「接触」高さ(gripper_z = 0.0 のとき)
TOP_CAM_H = 1.5
TEX_SCALE = 4                   # 布片テクスチャの解像度(px / mm)
NQ = 7


def mm_to_world(xy: tuple[float, float]) -> tuple[float, float]:
    return ((xy[0] - config.BENCH_W_MM / 2) / 1000.0 + BENCH_C[0],
            (config.BENCH_H_MM / 2 - xy[1]) / 1000.0 + BENCH_C[1])


def world_to_mm(x: float, y: float) -> tuple[float, float]:
    return ((x - BENCH_C[0]) * 1000.0 + config.BENCH_W_MM / 2,
            config.BENCH_H_MM / 2 - (y - BENCH_C[1]) * 1000.0)


def _inline_panda() -> str:
    """panda_suction.xml の中身を取り出す(<mujoco> と <compiler> を外して親モデルに埋め込む)。"""
    body = PANDA_XML.read_text()
    body = re.sub(r"<mujoco[^>]*>", "", body, count=1).replace("</mujoco>", "")
    body = re.sub(r"<compiler[^>]*/>", "", body, count=1)
    # 重力補償はコンパイル時に数えられるので XML 側で指定する(実行時に body_gravcomp を書いても効かない)
    return re.sub(r'<body name="(link[1-7]|attachment|suction_tool)"', r'<body name="\1" gravcomp="1"', body)


def _lookat_xyaxes(pos: np.ndarray, target: np.ndarray) -> str:
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return " ".join(f"{v:.5f}" for v in (*right, *up))


def _quat_z(angle_deg: float) -> str:
    a = math.radians(angle_deg) / 2
    return f"{math.cos(a):.6f} 0 0 {math.sin(a):.6f}"


@dataclass
class MujocoWorld:
    """2D 版 World と同じ属性(pieces / gripper_xy / gripper_z / time_s など)を持つ MuJoCo 版。"""

    layout: World
    model: mujoco.MjModel
    data: mujoco.MjData
    time_s: float = 0.0
    held_piece: int | None = None
    gripper_closed: bool = False
    _renderers: dict | None = None

    # ---------- 生成 ----------
    @classmethod
    def random_scene(cls, n_pieces: int, seed: int, labels: list[str] | None = None) -> "MujocoWorld":
        layout = World.random_scene(n_pieces, seed, labels)
        return cls.from_layout(layout)

    @classmethod
    def from_layout(cls, layout: World) -> "MujocoWorld":
        tex_dir = Path(tempfile.mkdtemp(prefix="cloth_sort_tex_"))
        layout.bench_texture().convert("RGB").save(tex_dir / "bench.png")
        for p in layout.pieces:
            # 2d テクスチャは箱の全面に同じ画像が貼られ、上面は画像どおりの向きになる(検証済み)
            layout.piece_texture(p, TEX_SCALE).convert("RGB").save(tex_dir / f"piece_{p.id}.png")
        xml = cls._build_xml(layout, tex_dir)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:NQ] = model.key("home").qpos[:NQ]
        data.ctrl[:NQ] = data.qpos[:NQ]
        mujoco.mj_forward(model, data)
        w = cls(layout=layout, model=model, data=data)
        w._sync_pieces()
        return w

    @staticmethod
    def _build_xml(layout: World, tex_dir: Path) -> str:
        bw, bh = config.BENCH_W_MM / 2000.0, config.BENCH_H_MM / 2000.0
        cx, cy = BENCH_C
        fovy = math.degrees(2 * math.atan(bh / TOP_CAM_H))
        iso_pos = np.array([cx + 1.35, cy + 1.45, 1.15])
        iso_axes = _lookat_xyaxes(iso_pos, np.array([cx - 0.05, cy - 0.15, 0.12]))
        pieces_xml, assets_xml, eq_xml = [], [], []
        for p in layout.pieces:
            x, y = mm_to_world(p.center)
            assets_xml.append(
                f'<texture name="tex_piece_{p.id}" type="2d" file="piece_{p.id}.png"/>\n'
                f'<material name="mat_piece_{p.id}" texture="tex_piece_{p.id}" texuniform="false" '
                f'specular="0.05" shininess="0.05"/>')
            pieces_xml.append(
                f'<body name="piece_{p.id}" pos="{x:.4f} {y:.4f} {PIECE_HALF_T + 0.002:.4f}" quat="{_quat_z(p.angle_deg)}">\n'
                f'  <freejoint name="free_piece_{p.id}"/>\n'
                f'  <geom name="geom_piece_{p.id}" type="box" size="0.025 0.025 {PIECE_HALF_T}" '
                f'material="mat_piece_{p.id}" mass="0.005" friction="0.9 0.005 0.0001" '
                f'solref="0.004 1" solimp="0.95 0.99 0.001"/>\n'
                f'</body>')
            eq_xml.append(f'<weld name="weld_piece_{p.id}" body1="suction_tool" body2="piece_{p.id}" active="false"/>')
        return f"""
<mujoco model="cloth_sort">
  <compiler meshdir="{ASSETS / 'assets'}" texturedir="{tex_dir}" angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast" impratio="10" cone="elliptic"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.3 0.3 0.3" specular="0.02 0.02 0.02"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>
  {_inline_panda()}
  <asset>
    <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.30 0.30 0.32" rgb2="0.36 0.36 0.38" width="256" height="256"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="12 12" texuniform="true" reflectance="0.05"/>
    <texture name="tex_bench" type="2d" file="bench.png"/>
    <material name="mat_bench" texture="tex_bench" texuniform="false" specular="0.1" shininess="0.1"/>
    <material name="mat_leg" rgba="0.35 0.33 0.30 1"/>
    <material name="mat_pedestal" rgba="0.28 0.29 0.33 1"/>
    {chr(10).join(assets_xml)}
  </asset>
  <worldbody>
    <light name="sun" pos="{cx + 0.4} {cy - 0.6} 2.2" dir="-0.2 0.3 -1" directional="true" diffuse="0.45 0.45 0.45"
      specular="0.05 0.05 0.05" castshadow="true"/>
    <light name="fill" pos="{cx - 1.0} {cy + 0.8} 1.6" dir="0.5 -0.4 -0.8" directional="true" diffuse="0.15 0.15 0.15"
      castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="mat_floor"/>
    <geom name="bench_top" type="box" size="{bw} {bh} 0.02" pos="{cx} {cy} -0.02" material="mat_bench"
      friction="0.9 0.005 0.0001"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx - bw + 0.04} {cy - bh + 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx + bw - 0.04} {cy - bh + 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx - bw + 0.04} {cy + bh - 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx + bw - 0.04} {cy + bh - 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom name="pedestal" type="cylinder" size="0.095 0.375" pos="0 0 -0.375" material="mat_pedestal" contype="0" conaffinity="0"/>
    <camera name="top" pos="{cx} {cy} {TOP_CAM_H}" xyaxes="1 0 0 0 1 0" fovy="{fovy:.4f}"/>
    <camera name="iso" pos="{' '.join(f'{v:.4f}' for v in iso_pos)}" xyaxes="{iso_axes}" fovy="40"/>
    {chr(10).join(pieces_xml)}
  </worldbody>
  <equality>
    {chr(10).join(eq_xml)}
  </equality>
  <keyframe>
    <key name="park" qpos="0 -0.785 0 -2.356 0 1.571 0.785 {' '.join(f'{v:.4f} {w:.4f} {PIECE_HALF_T + 0.002:.4f} {_quat_z(p.angle_deg)}' for p, (v, w) in ((p, mm_to_world(p.center)) for p in layout.pieces))}"/>
  </keyframe>
</mujoco>
"""

    # ---------- 状態 ----------
    @property
    def pieces(self) -> list[Piece]:
        return self.layout.pieces

    @property
    def rng(self):
        return self.layout.rng

    @property
    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.model.site("tool_tip").id]

    @property
    def gripper_xy(self) -> tuple[float, float]:
        x, y, _ = self.tip_pos
        return world_to_mm(float(x), float(y))

    @property
    def gripper_z(self) -> float:
        return (float(self.tip_pos[2]) - CONTACT_Z) / (LIFT_Z - CONTACT_Z)

    def _sync_pieces(self) -> None:
        for p in self.pieces:
            x, y, _ = self.data.xpos[self.model.body(f"piece_{p.id}").id]
            p.center = world_to_mm(float(x), float(y))

    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self._sync_pieces()

    # ---------- 把持(吸着の模擬) ----------
    def attach_under_tool(self) -> int | None:
        """カップ先端の直下(xy 20 mm 以内、高さ 12 mm 以内)に布片があれば weld で固定する。"""
        self.gripper_closed = True
        tip = self.tip_pos
        for p in self.pieces:
            if p.held:
                continue
            bid = self.model.body(f"piece_{p.id}").id
            pos = self.data.xpos[bid]
            top_z = pos[2] + PIECE_HALF_T
            if math.hypot(tip[0] - pos[0], tip[1] - pos[1]) <= 0.020 and abs(tip[2] - top_z) <= 0.012:
                self._set_weld(p.id, True)
                p.held = True
                self.held_piece = p.id
                return p.id
        return None

    def release(self, placed_at: str | None = None) -> None:
        self.gripper_closed = False
        if self.held_piece is None:
            return
        p = self.pieces[self.held_piece]
        self._set_weld(p.id, False)
        p.held = False
        p.placed_at = placed_at
        self.held_piece = None
        self.step(0.15)       # 置いた布片が落ち着くまで少し進める

    def _set_weld(self, piece_id: int, active: bool) -> None:
        eq = self.model.equality(f"weld_piece_{piece_id}").id
        if active:
            b1 = self.model.body("suction_tool").id
            b2 = self.model.body(f"piece_{piece_id}").id
            p1, q1 = self.data.xpos[b1], self.data.xquat[b1]
            p2, q2 = self.data.xpos[b2], self.data.xquat[b2]
            q1c = np.zeros(4)
            mujoco.mju_negQuat(q1c, q1)
            rel_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_pos, p2 - p1, q1c)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, q1c, q2)
            self.model.eq_data[eq, :3] = 0.0
            self.model.eq_data[eq, 3:6] = rel_pos
            self.model.eq_data[eq, 6:10] = rel_quat
            self.model.eq_data[eq, 10] = 1.0
        self.data.eq_active[eq] = 1 if active else 0

    def piece_at(self, xy: tuple[float, float], tol_mm: float = 30) -> Piece | None:
        return self.layout.piece_at(xy, tol_mm)

    # ---------- 描画 ----------
    def _renderer(self, name: str, width: int, height: int) -> mujoco.Renderer:
        if self._renderers is None:
            self._renderers = {}
        key = (name, width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def render_camera(self, noise: bool = True, include_pieces: bool = True) -> np.ndarray:
        """俯瞰カメラ画像(BENCH_H_MM × BENCH_W_MM × 3, uint8)。1 px = 1 mm。"""
        r = self._renderer("top", config.BENCH_W_MM, config.BENCH_H_MM)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        hidden: list[tuple[int, float]] = []
        if not include_pieces:
            for p in self.pieces:
                gid = self.model.geom(f"geom_piece_{p.id}").id
                hidden.append((gid, float(self.model.geom_rgba[gid, 3])))
                self.model.geom_rgba[gid, 3] = 0.0
        r.update_scene(self.data, camera="top")
        img = r.render().copy()
        for gid, a in hidden:
            self.model.geom_rgba[gid, 3] = a
        if noise:
            g = np.random.default_rng(int(self.time_s * 1000) + 7)
            arr = img.astype(np.float32) * g.uniform(0.96, 1.04) + g.normal(0, 1.5, img.shape)
            img = np.clip(arr, 0, 255).astype(np.uint8)
        return img

    def render_3d(self, width: int = 1180, height: int = 660) -> Image.Image:
        r = self._renderer("iso", width, height)
        r.update_scene(self.data, camera="iso")
        return Image.fromarray(r.render().copy())

    def render_scene(self) -> Image.Image:
        """2D 版との互換用(ビューアは render_3d / render_camera を直接使う)。"""
        return Image.fromarray(self.render_camera(noise=False))


class MujocoArm:
    """RobotArm プロトコルの MuJoCo 実装。目標を速度制限つきで動かし、IK で関節目標を作って位置サーボへ渡す。"""

    def __init__(self, world: MujocoWorld, xy_speed_mm_s: float = 600.0, z_speed_per_s: float = 3.5,
                 dwell_s: float = 0.2) -> None:
        self.world = world
        self.xy_speed_mm_s = xy_speed_mm_s
        self.z_speed_per_s = z_speed_per_s
        self.dwell_s = dwell_s
        m = world.model
        self._site = m.site("tool_tip").id
        self._lo, self._hi = m.jnt_range[:NQ, 0].copy(), m.jnt_range[:NQ, 1].copy()
        self._target_quat = np.array([0.0, 1.0, 0.0, 0.0])   # ツールを真下向きに
        self._cmd = np.array([*mm_to_world(world.gripper_xy), float(world.tip_pos[2])])
        self._stall = 0.0
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    # 2D 版 VirtualArm と同じ "limits" 風の属性(PickPlace が dwell を参照する)
    @property
    def limits(self):
        return self

    @property
    def pose(self) -> tuple[float, float, float]:
        return (*self.world.gripper_xy, self.world.gripper_z)

    def move_to(self, xy: tuple[float, float], z: float, dt: float) -> bool:
        goal = np.array([*mm_to_world(xy), CONTACT_Z + z * (LIFT_Z - CONTACT_Z)])
        # 速度制限つきで指令点を進める
        d = goal[:2] - self._cmd[:2]
        dist = float(np.linalg.norm(d))
        step = self.xy_speed_mm_s / 1000.0 * dt
        self._cmd[:2] = goal[:2] if dist <= step else self._cmd[:2] + d / dist * step
        dz = goal[2] - self._cmd[2]
        zstep = self.z_speed_per_s * (LIFT_Z - CONTACT_Z) * dt
        self._cmd[2] = goal[2] if abs(dz) <= zstep else self._cmd[2] + math.copysign(zstep, dz)
        # IK → 位置サーボ → 物理を dt ぶん進める
        q = self._ik(self._cmd)
        self.world.data.ctrl[:NQ] = q
        self.world.step(dt)
        err = float(np.linalg.norm(self.world.tip_pos - goal))
        cmd_done = bool(np.allclose(self._cmd, goal))
        if cmd_done:
            self._stall += dt
        else:
            self._stall = 0.0
        if cmd_done and (err < 0.004 or self._stall > 1.5):
            self._stall = 0.0
            return True
        return False

    def park(self, dt: float) -> bool:
        """関節空間で待機姿勢(土台の上で腕を畳んだ Panda の ready 姿勢)へ戻る。カメラの視野にどのリンクも入らない。

        xy の目標点を IK で解く退避だと、肘が机の上に張り出して置いた布片に被ることがあるため、関節で指定する。
        """
        m, d = self.world.model, self.world.data
        home = m.key("park").qpos[:NQ]
        q = d.qpos[:NQ]
        step = 2.0 * dt                                   # 関節速度の上限 2 rad/s(Panda の仕様内)
        d.ctrl[:NQ] = q + np.clip(home - q, -step, step)
        self.world.step(dt)
        self._cmd[:] = [*self.world.tip_pos]              # 次の move_to が現在位置から始まるように
        done = bool(np.all(np.abs(d.qpos[:NQ] - home) < 0.02))
        if done:
            self._stall += dt
        else:
            self._stall = 0.0
        if done or self._stall > 1.5:
            self._stall = 0.0
            return True
        return False

    def grip(self) -> bool:
        return self.world.attach_under_tool() is not None

    def release(self, placed_at: str | None) -> None:
        self.world.release(placed_at)

    def _ik(self, target: np.ndarray, iters: int = 12) -> np.ndarray:
        m, d = self.world.model, self.world.data
        q = d.qpos[:NQ].copy()
        q_saved = d.qpos.copy()
        quat = np.zeros(4)
        q_conj = np.zeros(4)
        q_err = np.zeros(4)
        rot = np.zeros(3)
        for _ in range(iters):
            d.qpos[:NQ] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self._site]
            mujoco.mju_mat2Quat(quat, d.site_xmat[self._site])
            mujoco.mju_negQuat(q_conj, quat)
            mujoco.mju_mulQuat(q_err, self._target_quat, q_conj)
            mujoco.mju_quat2Vel(rot, q_err, 1.0)
            err = np.concatenate([target - pos, 0.5 * rot])
            if np.linalg.norm(err[:3]) < 5e-4 and np.linalg.norm(rot) < 1e-2:
                break
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self._site)
            J = np.vstack([self._jacp[:, :NQ], self._jacr[:, :NQ]])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), self._lo, self._hi)
        d.qpos[:] = q_saved
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return q
