"""MuJoCo 版その 2: 変形体の布を Panda の 2 本指ハンド(ゴムパッド)でつまむ。

布は MuJoCo の flex(2D シェル、9×9 格子、厚さ 3 mm、フェルト程度のかたさ)。把持は物理に任せる。
平らに置かれた布を平行グリッパでつまむために、次の 2 つを入れている(いずれも実機でも使える工夫)。

- 左指のゴムパッドを 2.5 mm 長くした「唇」を机面まで押し込み、布の縁を押して折り目を作る
- 閉じる間、押さえ側の指が世界に対して止まるようにハンド全体を動かす(折り目が押さえ指の前で立つ)

持ち上げ・搬送は布が抜けないようゆっくり(3〜6 cm/s)。置くときは置き場の上で放して落とす。
布は折れて落ちるので、置いた後に番号を読み直す検証はできない。置き場に布があるかをカメラで確認する。
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
from cloth_sort.sim.mujoco_world import (ASSETS, BENCH_C, NQ, TEX_SCALE, TOP_CAM_H, _lookat_xyaxes, _quat_z,
                                         mm_to_world, world_to_mm)
from cloth_sort.sim.world import Piece, World

PANDA_HAND_XML = ASSETS / "panda_hand.xml"
CLOTH_COUNT = 9                 # 格子の点数(9×9)
CLOTH_RADIUS = 0.0015           # 布の半分の厚さ
CLOTH_YOUNG = 1.0e5             # 実験で把持できた範囲の中央(3e4〜1.5e5 で保持、4e5 は折り目が開いて抜ける)
TIMESTEP = 0.0005
OPEN_CTRL = 115.0               # 指 1 本 18 mm 開く → パッド面のすき間 約 23 mm
PRESSER_TRAVEL = OPEN_CTRL * 0.04 / 255.0   # 閉じるときに押さえ指が動く量(ハンドをこの分だけ逆に動かす)
LIP_H = -0.0012                 # 唇パッドの底を机面から 1.2 mm 押し込む(ゴムが机に沿う想定)
LIFT_Z = 0.09                   # 搬送高さ(gripper_z = 1.0)。布は 5 cm ぶら下がる
PLACE_Z = (0.058 - LIP_H) / (LIFT_Z - LIP_H)   # 置くときの高さ(正規化)。布の下端が机のすぐ上
GRASP_EDGE_MM = 25.0            # 布の中心から縁(ロボット側)までの距離
CLOSE_T = 1.2
SETTLE_T = 0.3
RELEASE_T = 0.4
DROP_T = 0.8
SHAKE_T = 1.0                   # 開いても布が指に引っかかっているとき、横に 3 cm 振って戻す時間
# 「敷く」置き方(斜め敷き): ぶら下がった布の下端が机に触れる高さで水平に進み始め、進みながらゆっくり降りる。
# 下端が机との摩擦で後ろに留まり、布は手の後ろに順に敷かれる(ベルトコンベアのように)。進む距離は降りる高さの数倍にする
LAY_Z0 = 0.050                  # 敷き始めの手先高さ(布の下端が机に触れる直前)
LAY_Z1 = 0.004                  # 敷き終わりの手先高さ(唇が机のすぐ上)
LAY_TRAVEL = 0.12               # 水平に進む距離
LAY_T = 4.0                     # 敷く時間(3 cm/s)
LAY_APPROACH_T = 1.0
LAY_DIR = (0.0, -1.0)           # 進む向き(世界座標)。ロボット側へ進み、布はその反対側(机の奥)に敷かれる
SHAKE_AMP = 0.03
MAX_SHAKES = 2


def _inline(path: Path) -> str:
    body = path.read_text()
    body = re.sub(r"<mujoco[^>]*>", "", body, count=1).replace("</mujoco>", "")
    body = re.sub(r"<compiler[^>]*/>", "", body, count=1)
    return re.sub(r"<option[^>]*/>", "", body, count=1)     # 積分器は親モデル側で決める(flex は discrete が必要)


def cloth_grid(count: int, size: float) -> tuple[str, str, str]:
    """count×count の格子。点は中心原点、texcoord は画像の左上が (0,0)、右下が (1,1)。"""
    pts, tex, tri = [], [], []
    for j in range(count):
        for i in range(count):
            x = -size / 2 + size * i / (count - 1)
            y = size / 2 - size * j / (count - 1)
            pts += [x, y, 0.0]
            tex += [i / (count - 1), j / (count - 1)]
    for j in range(count - 1):
        for i in range(count - 1):
            a = j * count + i
            b, c, dd = a + 1, a + count, a + count + 1
            tri += [a, c, b, b, c, dd]
    return (" ".join(f"{v:.5f}" for v in pts), " ".join(str(v) for v in tri), " ".join(f"{v:.4f}" for v in tex))


@dataclass
class MujocoClothWorld:
    layout: World
    model: mujoco.MjModel
    data: mujoco.MjData
    time_s: float = 0.0
    held_piece: int | None = None
    gripper_closed: bool = False
    verify_mode: str = "presence"      # 置いた後の検証は「置き場に布があるか」
    _renderers: dict | None = None

    @classmethod
    def random_scene(cls, n_pieces: int, seed: int, labels: list[str] | None = None) -> "MujocoClothWorld":
        return cls.from_layout(World.random_scene(n_pieces, seed, labels))

    @classmethod
    def from_layout(cls, layout: World, young: float = CLOTH_YOUNG) -> "MujocoClothWorld":
        tex_dir = Path(tempfile.mkdtemp(prefix="cloth_sort_flex_"))
        layout.bench_texture().convert("RGB").save(tex_dir / "bench.png")
        for p in layout.pieces:
            layout.piece_texture(p, TEX_SCALE).convert("RGB").save(tex_dir / f"piece_{p.id}.png")
        model = mujoco.MjModel.from_xml_string(cls._build_xml(layout, tex_dir, young))
        data = mujoco.MjData(model)
        home = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
        data.qpos[:NQ] = home
        data.ctrl[:NQ] = home
        data.ctrl[NQ] = OPEN_CTRL
        mujoco.mj_forward(model, data)
        w = cls(layout=layout, model=model, data=data)
        w._sync_pieces()
        return w

    @staticmethod
    def _build_xml(layout: World, tex_dir: Path, young: float) -> str:
        bw, bh = config.BENCH_W_MM / 2000.0, config.BENCH_H_MM / 2000.0
        cx, cy = BENCH_C
        fovy = math.degrees(2 * math.atan(bh / TOP_CAM_H))
        iso_pos = np.array([cx + 1.35, cy + 1.45, 1.15])
        iso_axes = _lookat_xyaxes(iso_pos, np.array([cx - 0.05, cy - 0.15, 0.12]))
        p_str, e_str, t_str = cloth_grid(CLOTH_COUNT, config.PIECE_MM / 1000.0)
        assets_xml, cloth_xml = [], []
        for p in layout.pieces:
            x, y = mm_to_world(p.center)
            assets_xml.append(
                f'<texture name="tex_piece_{p.id}" type="2d" file="piece_{p.id}.png"/>\n'
                f'<material name="mat_piece_{p.id}" texture="tex_piece_{p.id}" texuniform="false" specular="0.05" shininess="0.05"/>')
            cloth_xml.append(
                f'<flexcomp name="cloth_{p.id}" type="direct" dim="2" point="{p_str}" element="{e_str}" texcoord="{t_str}"\n'
                f'  radius="{CLOTH_RADIUS}" mass="0.005" pos="{x:.4f} {y:.4f} {CLOTH_RADIUS + 0.002:.4f}" quat="{_quat_z(p.angle_deg)}"\n'
                f'  material="mat_piece_{p.id}" rgba="1 1 1 1">\n'
                f'  <contact selfcollide="none" contype="2" conaffinity="1" friction="2 0.05 0.01" solref="0.002 1" solimp="0.99 0.999 0.0002" condim="3"/>\n'
                f'  <elasticity young="{young}" poisson="0.3" thickness="0.001" damping="0.002" elastic2d="both"/>\n'
                f'  <edge damping="0.001"/>\n'
                f'</flexcomp>')
        return f"""
<mujoco model="cloth_sort_flex">
  <compiler meshdir="{ASSETS / 'assets'}" texturedir="{tex_dir}" angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="{TIMESTEP}" integrator="discrete" impratio="10" cone="elliptic"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.3 0.3 0.3" specular="0.02 0.02 0.02"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>
  {_inline(PANDA_HAND_XML)}
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
      friction="0.4 0.005 0.0001"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx - bw + 0.04} {cy - bh + 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx + bw - 0.04} {cy - bh + 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx - bw + 0.04} {cy + bh - 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom type="box" size="0.02 0.02 0.355" pos="{cx + bw - 0.04} {cy + bh - 0.04} -0.395" material="mat_leg" contype="0" conaffinity="0"/>
    <geom name="pedestal" type="cylinder" size="0.095 0.375" pos="0 0 -0.375" material="mat_pedestal" contype="0" conaffinity="0"/>
    <camera name="top" pos="{cx} {cy} {TOP_CAM_H}" xyaxes="1 0 0 0 1 0" fovy="{fovy:.4f}"/>
    <camera name="iso" pos="{' '.join(f'{v:.4f}' for v in iso_pos)}" xyaxes="{iso_axes}" fovy="40"/>
    {chr(10).join(cloth_xml)}
  </worldbody>
  <contact>
    <!-- 唇パッドは机の上を滑る(机との摩擦は低く)。布との摩擦はゴムの値のまま -->
    <pair geom1="rubber_left" geom2="bench_top" friction="0.05 0.05 0.001" solref="0.01 1" solimp="0.5 0.9 0.01"/>
    <pair geom1="rubber_right" geom2="bench_top" friction="0.05 0.05 0.001" solref="0.01 1" solimp="0.5 0.9 0.01"/>
  </contact>
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
        return (float(self.tip_pos[2]) - LIP_H) / (LIFT_Z - LIP_H)

    def flex_id(self, piece_id: int) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, f"cloth_{piece_id}")

    def cloth_vertices(self, piece_id: int) -> np.ndarray:
        f = self.flex_id(piece_id)
        a, n = self.model.flex_vertadr[f], self.model.flex_vertnum[f]
        return self.data.flexvert_xpos[a:a + n]

    def _sync_pieces(self) -> None:
        for p in self.pieces:
            c = self.cloth_vertices(p.id).mean(axis=0)
            p.center = world_to_mm(float(c[0]), float(c[1]))

    def step(self, seconds: float) -> None:
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self._sync_pieces()

    def rubber_contacts(self, piece_id: int) -> int:
        """ゴムパッドとこの布の接触数。"""
        f = self.flex_id(piece_id)
        rub = {self.model.geom("rubber_left").id, self.model.geom("rubber_right").id}
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in rub and c.flex[1] == f) or (c.geom2 in rub and c.flex[0] == f):
                n += 1
        return n

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
        r = self._renderer("top", config.BENCH_W_MM, config.BENCH_H_MM)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        saved = self.model.flex_rgba.copy()
        if not include_pieces:
            self.model.flex_rgba[:, 3] = 0.0
        r.update_scene(self.data, camera="top")
        img = r.render().copy()
        self.model.flex_rgba[:] = saved
        if noise:
            g = np.random.default_rng(int(self.time_s * 1000) + 7)
            arr = img.astype(np.float32) * g.uniform(0.96, 1.04) + g.normal(0, 1.5, img.shape)
            img = np.clip(arr, 0, 255).astype(np.uint8)
        return img

    def render_3d(self, width: int = 1180, height: int = 660) -> Image.Image:
        r = self._renderer("iso", width, height)
        r.update_scene(self.data, camera="iso")
        return Image.fromarray(r.render().copy())

    def render_hand(self, width: int = 560, height: int = 280) -> Image.Image:
        """手首カメラ(ハンドに固定、指先を見下ろす)。実機の eye-in-hand に相当。AI は使わず、可視化のみ。"""
        r = self._renderer("wristcam", width, height)
        r.update_scene(self.data, camera="wristcam")
        return Image.fromarray(r.render().copy())

    def render_scene(self) -> Image.Image:
        return Image.fromarray(self.render_camera(noise=False))


class GripperArm:
    """RobotArm プロトコルの 2 本指版。つまむ・放すは複数周期にまたがる(grip_step / release_step)。"""

    def __init__(self, world: MujocoClothWorld) -> None:
        self.world = world
        m = world.model
        self._site = m.site("tool_tip").id
        self._lo, self._hi = m.jnt_range[:NQ, 0].copy(), m.jnt_range[:NQ, 1].copy()
        self._target_quat = np.array([0.0, 1.0, 0.0, 0.0])   # ハンド z を真下、ハンド +y(左指 = 唇)を世界 -y へ
        self._cmd = np.array([*world.tip_pos])
        self._vel = 0.0
        self._stall = 0.0
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self.holding_id: int | None = None
        self._grip_t: float | None = None
        self._release_t: float | None = None
        self._shake_t0: float | None = None
        self._shakes = 0
        self._resettle_until = 0.0
        self.dwell_s = 0.0
        self.place_z = PLACE_Z
        self.park_q = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        self.cmd_offset = np.zeros(3)       # 実際に IK へ渡す目標に足す外乱(模倣学習のデータ収集で使う。通常は 0)
        self.place_mode = "drop"            # "drop": 置き場の上で放して落とす / "lay": 平らに敷いてから放す
        self._lay_t: float | None = None
        self.lay_params = dict(z0=LAY_Z0, z1=LAY_Z1, travel=LAY_TRAVEL, t=LAY_T)   # 実験で変えられる

    @property
    def limits(self):
        return self

    @property
    def pose(self) -> tuple[float, float, float]:
        return (*self.world.gripper_xy, self.world.gripper_z)

    # つまむ点: 布の中心からロボット側の縁(y_mm が大きい側)へ寄せる
    def grasp_point(self, pick_xy: tuple[float, float]) -> tuple[float, float]:
        return (pick_xy[0], pick_xy[1] + GRASP_EDGE_MM)

    def relocate(self, pick_xy: tuple[float, float]) -> tuple[float, float]:
        """つかみ損ねた布は動いていることがあるので、カメラで撮り直して近くの布の位置を取り直す。"""
        from cloth_sort.perception.detect import detect_pieces
        bg = getattr(self.world, "background", None)
        if bg is None:
            return pick_xy
        image = self.world.render_camera()
        r = 90
        region = (int(pick_xy[0] - r), int(pick_xy[1] - r), int(pick_xy[0] + r), int(pick_xy[1] + r))
        dets = detect_pieces(image, bg, region=region, min_area_mm2=600, max_area_mm2=20000)
        if not dets:
            return pick_xy
        best = min(dets, key=lambda d: (d.center[0] - pick_xy[0]) ** 2 + (d.center[1] - pick_xy[1]) ** 2)
        return best.center

    def _speeds(self) -> tuple[float, float, float]:
        """(xy 速度 m/s, z 速度 m/s, xy 加速度 m/s²)。布を持っているときは抜けないようにゆっくり・なめらかに。"""
        if self.holding_id is not None:
            return 0.05, 0.03, 0.05
        return 0.6, 0.5, 3.0

    def move_to(self, xy: tuple[float, float], z: float, dt: float) -> bool:
        goal = np.array([*mm_to_world(xy), LIP_H + z * (LIFT_Z - LIP_H)])
        vxy, vz, acc = self._speeds()
        d = goal[:2] - self._cmd[:2]
        dist = float(np.linalg.norm(d))
        # 台形速度: 加速度制限つきで上限まで上げ、目標の手前で減速する(布を揺らさない)
        self._vel = min(vxy, self._vel + acc * dt, math.sqrt(2 * acc * dist) + 1e-4)
        step = self._vel * dt
        if dist <= step:
            self._cmd[:2] = goal[:2]
            self._vel = 0.0
        else:
            self._cmd[:2] = self._cmd[:2] + d / dist * step
        dz = goal[2] - self._cmd[2]
        zstep = vz * dt
        self._cmd[2] = goal[2] if abs(dz) <= zstep else self._cmd[2] + math.copysign(zstep, dz)
        self._servo(dt)
        err = float(np.linalg.norm(self.world.tip_pos - goal))
        cmd_done = bool(np.allclose(self._cmd, goal))
        self._stall = self._stall + dt if cmd_done else 0.0
        if cmd_done and (err < 0.004 or self._stall > 1.5):
            self._stall = 0.0
            return True
        return False

    def _servo(self, dt: float) -> None:
        q = self._ik(self._cmd + self.cmd_offset)
        self.world.data.ctrl[:NQ] = q
        self.world.step(dt)

    def grip_step(self, dt: float) -> bool | None:
        """閉じる → 落ち着く → 接触で判定。終わるまで None、終わったら成否。"""
        d = self.world.data
        if self._grip_t is None:
            self._grip_t = 0.0
            self._grip_y0 = float(self._cmd[1])
        self._grip_t += dt
        t = self._grip_t
        if t <= CLOSE_T:
            f = min(1.0, t / CLOSE_T)
            d.ctrl[NQ] = OPEN_CTRL * (1 - f)
            self._cmd[1] = self._grip_y0 + PRESSER_TRAVEL * f      # 押さえ指を世界に対して止める
            self._servo(dt)
            return None
        if t <= CLOSE_T + SETTLE_T:
            self._servo(dt)
            return None
        self._grip_t = None
        self.world.gripper_closed = True
        best, best_n = None, 0
        for p in self.world.pieces:
            if p.held or p.placed_at:
                continue
            n = self.world.rubber_contacts(p.id)
            if n > best_n:
                best, best_n = p, n
        if best is None:
            return False
        best.held = True
        self.holding_id = best.id
        self.world.held_piece = best.id
        return True

    MIN_HOLD_CONTACTS = 6     # これより少ない接触は角をかすめただけで、搬送中に抜ける(実機なら力覚で見る)

    def holding(self, strict: bool = False) -> bool:
        """布が机から離れて指についてきているか。strict は持ち上げ直後の品質判定(接触が少ない把持は搬送中に抜ける)。

        搬送中は接触数が揺れるので、宙に浮いていて接触が 1 つでもあれば保持とみなす。
        """
        if self.holding_id is None:
            return False
        v = self.world.cloth_vertices(self.holding_id)
        n = self.world.rubber_contacts(self.holding_id)
        need = self.MIN_HOLD_CONTACTS if strict else 1
        ok = float(v[:, 2].mean()) > 0.015 and n >= need
        if not ok:
            self._drop()
        return ok

    def _drop(self) -> None:
        if self.holding_id is not None:
            self.world.pieces[self.holding_id].held = False
        self.holding_id = None
        self.world.held_piece = None

    def lay_step(self, dt: float, place_xy: tuple[float, float]) -> bool:
        """つまんだ布を置き場に平らに敷く。終わったら True(そのあと release_step で放す)。

        布は手先の後ろに敷かれるので、敷き終わりの手先位置は置き場の中心から進行方向に 25 mm 先。
        """
        P = self.lay_params
        if self._lay_t is None:
            self._lay_t = 0.0
            d = np.array(LAY_DIR)
            c = np.array(mm_to_world(place_xy))
            half = (config.PIECE_MM / 2) / 1000.0
            self._lay_start = np.array([*self._cmd[:3]])
            self._lay_to = np.array([*(c + half * d), P["z1"]])
            self._lay_from = np.array([*(self._lay_to[:2] - P["travel"] * d), P["z0"]])
        self._lay_t += dt
        t = self._lay_t
        if t <= LAY_APPROACH_T:
            self._cmd[:] = self._lay_start + (self._lay_from - self._lay_start) * (t / LAY_APPROACH_T)
        else:
            f = min(1.0, (t - LAY_APPROACH_T) / P["t"])
            self._cmd[:] = self._lay_from + (self._lay_to - self._lay_from) * f
        self._vel = 0.0
        self._servo(dt)
        if t >= LAY_APPROACH_T + P["t"] + 0.3:
            self._lay_t = None
            return True
        return False

    def cloth_hangs(self, piece_id: int | None = None) -> bool:
        """開いたのに布が指に引っかかって宙にあるか(ゴムパッドとの接触が残り、布が机から浮いている)。"""
        pid = self.holding_id if piece_id is None else piece_id
        if pid is None:
            return False
        v = self.world.cloth_vertices(pid)
        return float(v[:, 2].mean()) > 0.02 and self.world.rubber_contacts(pid) > 0

    def release_step(self, dt: float, placed_at: str | None) -> bool | None:
        """指を開いて布を落とし、落ち着くまで待つ。布が指に引っかかって落ちなければ、横に振って落とす(最大 2 回)。"""
        d = self.world.data
        if self._release_t is None:
            self._release_t = 0.0
            self._shake_t0 = None
            self._shakes = 0
            self._resettle_until = RELEASE_T + DROP_T
        self._release_t += dt
        t = self._release_t
        d.ctrl[NQ] = OPEN_CTRL * min(1.0, t / RELEASE_T)
        if self._shake_t0 is None and t >= self._resettle_until and self._shakes < MAX_SHAKES and self.cloth_hangs():
            self._shake_t0 = t
            self._shake_x0 = float(self._cmd[0])
            self._shakes += 1
        if self._shake_t0 is not None:
            f = (t - self._shake_t0) / SHAKE_T
            if f < 1.0:
                self._cmd[0] = self._shake_x0 + SHAKE_AMP * math.sin(2 * math.pi * f)
            else:
                self._cmd[0] = self._shake_x0
                self._shake_t0 = None
                self._resettle_until = t + DROP_T
        self._servo(dt)
        if t < self._resettle_until or self._shake_t0 is not None:
            return None
        self._release_t = None
        self.world.gripper_closed = False
        if self.holding_id is not None:
            p = self.world.pieces[self.holding_id]
            p.held = False
            p.placed_at = placed_at
        self.holding_id = None
        self.world.held_piece = None
        return True

    def release(self, placed_at: str | None) -> None:
        """単発の解放(再試行で使う)。指を開くだけ。"""
        self.world.data.ctrl[NQ] = OPEN_CTRL
        self.world.gripper_closed = False
        self._grip_t = None
        self._drop()

    def park(self, dt: float) -> bool:
        m, d = self.world.model, self.world.data
        q = d.qpos[:NQ]
        step = 2.0 * dt
        d.ctrl[:NQ] = q + np.clip(self.park_q - q, -step, step)
        d.ctrl[NQ] = OPEN_CTRL
        self.world.step(dt)
        self._cmd[:] = [*self.world.tip_pos]
        self._vel = 0.0
        done = bool(np.all(np.abs(d.qpos[:NQ] - self.park_q) < 0.02))
        self._stall = self._stall + dt if done else 0.0
        if done or self._stall > 1.5:
            self._stall = 0.0
            return True
        return False

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
