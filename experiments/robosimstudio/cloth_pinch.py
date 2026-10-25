"""5 cm 角の布を Panda の 2 本指でつまんで運ぶ(RoboSimStudio / Newton VBD)。

本体(MuJoCo flex)と同じ手順・同じ速度で動かし、布の挙動と把持の成否を比べるための実験。
RoboSimStudio の布は GPU(CUDA)でしか指の摩擦でつまめない(CPU では SuperDex に切り替わり、
つまんだ縁が持ち上がらない)。Colab の T4 で動かした。

    # RoboSimStudio の checkout で(uv sync 済み)
    uv run python cloth_pinch.py --trials 10 --out out/pinch --video 3

1 回の試行は「縁の上へ → 指先を机まで下ろす → 握りながら 3 mm 上げる → 3 cm/s で 3 cm 持ち上げる
→ 6 cm/s で 15 cm 運ぶ → 机から 3 cm の高さで放す」。布は物理だけで指に保持される(固定はしない)。
布の位置と向きは試行ごとに少しずらす。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import robosimstudio as rss

CELL = "bench_075"
SIDE = 0.05  # 布片の一辺 [m]。本体と同じ 5 cm 角
CELLS = 10  # 一辺あたりの格子数(5 mm 格子)
CENTER = (0.0, 0.0)  # 布の中心(作業面の座標)[m]
EDGE_IN = 0.006  # つまむとき、ツール中心を布の縁から内側へ寄せる量 [m]。外側の指は机、内側の指は布に載る
SQUEEZE = 0.95  # 握りの強さ(0〜1)。--grip で変える。Panda では 0.95 で指の隙間 5 mm(垂れた裾 4 mm 用の値)
SQUEEZE_RISE = 0.003  # 握りながら上げる量 [m]
LIFT = 0.03  # 持ち上げる高さ [m]
LIFT_SPEED = 0.03  # [m/s]。本体と同じ
CARRY = 0.15  # 運ぶ距離 [m]
CARRY_SPEED = 0.06  # [m/s]。本体と同じ
RELEASE_Z = 0.03  # 放す高さ(机から)[m]
HOLD_RADIUS = 0.020  # ツール中心からこの距離にある粒子を「つまんでいる」とみなす [m]

# フェルトに近い布: 厚さ 3 mm、300 g/m^2。膜(membrane)で作る。厚みを持つ板(slab)にすると、
# Panda の指(メッシュ + 符号付き距離場)が布をすり抜けて押しても動かなかった(YAM の指なら当たる)
FELT = rss.replace(
    rss.materials.TERRY_MEMBRANE,
    thickness=0.003,
    areal_density=0.3,
    particle_radius=0.0015,  # 膜の慣例で厚さの半分。接触の厚みは 3 mm
)


def build(center_xy, yaw):
    cuda = wp.is_cuda_available()
    # 標準の設定に、指の摩擦で布をつまむための 2 つ(片方向の連成、面全体の接触)だけを足す。
    # towel デモのようにソルバーの設定を上書きすると、指と布の接触が効かなくなった
    physics = rss.replace(
        rss.GENERIC,
        coupling=rss.coupler("one_way"),
        # Panda の当たり判定はメッシュなので、布の面との接触には符号付き距離場(CUDA のみ)が要る
        collision=rss.CollisionSpec(full_surface_contact=True, force_sdf=cuda),
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="cloth_pinch")
    scene.add(rss.robot("panda"))
    c, s = math.cos(yaw), math.sin(yaw)
    corner = np.array(center_xy) - np.array([c - s, s + c]) * SIDE / 2  # 回転した正方形の原点の角
    scene.add(rss.sheet(
        name="patch", size=(SIDE, SIDE), cells=(CELLS, CELLS),
        at=(float(corner[0]), float(corner[1]), 0.002),
        quat=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
        material=FELT, color=(0.80, 0.78, 0.70),
    ))
    return scene.build(control_mode="joint_position")


# ---------- 腕の操作(RoboSimStudio の tests/fixtures/grasp.py と同じく、アームパネル経由で動かす) ----------

def open_session(sim, port):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    session = StudioSession(sim, config=StudioConfig(port=port, panels="drive", speed="max", live=False))
    session.begin_run()
    return session


class Runner:
    def __init__(self, sim, session, camera, frames_every):
        self.sim, self.session = sim, session
        self.arm = next(iter(session.arms))
        self.panel = session.arms[self.arm]
        self.hz = float(sim.request.physics.control_hz)
        self.camera, self.every, self.frames, self.steps = camera, frames_every, [], 0
        self.ids = np.asarray(sim["patch"].entity.particles, dtype=np.int64)

    def tick(self, n=1):
        done = 0
        while done < n:
            if self.session.tick():
                done += 1
                self.steps += 1
                if self.every and self.steps % self.every == 0:
                    self.frames.append(self.sim.render(self.camera)["rgb"])

    def q(self):
        return self.sim.state.particle_q.numpy().astype(np.float64)[self.ids]

    def tool(self):
        pos, quat = self.panel.ik.ee_pose(self.sim.state)
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)

    def glide(self, target, quat, speed, settle=1.0, tol=0.004):
        """ツール中心を target へ speed [m/s] で直線移動し、追いつくまで待つ。"""
        start = np.array(self.panel.cursor_pos, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        legs = max(1, math.ceil(np.linalg.norm(target - start) / (speed / self.hz)))
        for k in range(1, legs + 1):
            self.panel.cursor_pos, self.panel.cursor_quat = start + (target - start) * k / legs, quat
            self.panel.dragging = k < legs
            self.tick()
        self.panel.dragging = False
        for _ in range(round(settle * self.hz)):
            if np.linalg.norm(self.tool()[0] - target) < tol:
                break
            self.tick()
        return float(np.linalg.norm(self.tool()[0] - target))

    def wait(self, seconds):
        self.tick(round(seconds * self.hz))

    def grip(self, amount):
        self.panel.grip_request = float(amount)


def quat_mul(a, b):
    """xyzw の積 a*b。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def finger_axis(sim):
    """2 本の指の中心を結ぶ水平方向(閉じる方向)の単位ベクトル。"""
    labels = [str(label) for label in sim.model.body_label]
    body_q = sim.state.body_q.numpy()
    left = next(i for i, l in enumerate(labels) if l.endswith("leftfinger"))
    right = next(i for i, l in enumerate(labels) if l.endswith("rightfinger"))
    d = body_q[right][:3] - body_q[left][:3]
    d[2] = 0.0
    return d / np.linalg.norm(d)


def finger_bottom(sim):
    """指の当たり判定の一番低い点の z [m](メッシュは頂点で測る)。"""
    import newton

    from robosimstudio.core.math import quat_to_mat

    model = sim.model
    labels = [str(label) for label in model.body_label]
    body_q = sim.state.body_q.numpy().astype(np.float64)
    bodies = model.shape_body.numpy()
    kinds = model.shape_type.numpy()
    xforms = model.shape_transform.numpy().astype(np.float64)
    scales = model.shape_scale.numpy().astype(np.float64)
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)
    low = np.inf
    for shape in range(model.shape_count):
        b = bodies[shape]
        if b < 0 or "finger" not in labels[b]:
            continue
        rot_b = quat_to_mat(body_q[b][3:7])
        centre = body_q[b][:3] + rot_b @ xforms[shape][:3]
        rot = rot_b @ quat_to_mat(xforms[shape][3:7])
        if int(kinds[shape]) == int(newton.GeoType.BOX):
            local = corners * scales[shape]
        elif getattr(model.shape_source[shape], "vertices", None) is not None:  # メッシュ(SDF 付きも)
            local = np.asarray(model.shape_source[shape].vertices, dtype=np.float64) * scales[shape]
        else:
            continue
        low = min(low, float((centre + (rot @ local.T).T)[:, 2].min()))
    return low


def hull_area(xy):
    """2 次元の凸包の面積(Andrew の単調連鎖)。"""
    pts = sorted(map(tuple, np.round(xy, 6)))
    if len(pts) < 3:
        return 0.0

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    poly = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def trial(seed, port, video_dir=None, squeeze=SQUEEZE):
    rng = np.random.default_rng(seed)
    center = np.array(CENTER) + rng.uniform(-0.02, 0.02, 2)
    yaw = float(rng.uniform(-0.3, 0.3))
    sim = build(center, yaw)
    surface = float(rss.get_cell(CELL).surface_z)
    session = open_session(sim, port)
    cam = None
    if video_dir is not None:
        mid = np.array([*rss.get_cell(CELL).surface_frame[:2], surface]) + np.array([center[0], center[1] + CARRY / 2, 0.0])
        cam = rss.CameraSpec(name="view", width=960, height=540, fov_y=40.0,
                             eye=tuple(mid + np.array([0.45, -0.35, 0.30])), look_at=tuple(mid),
                             up=(0.0, 0.0, 1.0), near=0.01, far=5.0)
    r = Runner(sim, session, cam, frames_every=2 if cam else 0)
    try:
        r.wait(0.5)  # 布を落ち着かせる
        rest = r.q()
        area0 = hull_area(rest[:, :2])
        _, home_quat = r.tool()
        # 指の閉じる方向を布の辺に直交させる(閉じる方向 = 布の x 軸)
        axis = finger_axis(sim)
        turn = math.atan2(math.sin(yaw), math.cos(yaw)) - math.atan2(axis[1], axis[0])
        quat = quat_mul(np.array([0.0, 0.0, math.sin(turn / 2), math.cos(turn / 2)]), home_quat)
        ex = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        world_center = rest.mean(axis=0)
        grasp = world_center + ex * (SIDE / 2 - EDGE_IN)
        r.glide(grasp + np.array([0, 0, 0.10]), quat, 0.25)
        tip_off = r.tool()[0][2] - finger_bottom(sim)  # ツール中心から指先までの高さ
        grasp[2] = surface + tip_off + 0.001
        r.glide(grasp, quat, 0.05)
        # 腕は目標の 3〜4 mm 手前で止まり、指が厚さ 3 mm の布の上で浮く。指先の実際の高さを測って追い込む
        for _ in range(3):
            gap = finger_bottom(sim) - (surface + 0.0005)
            if abs(gap) < 0.0008:
                break
            grasp[2] -= gap
            r.glide(grasp, quat, 0.02, settle=1.5, tol=0.001)
        miss = finger_bottom(sim) - surface  # 指先と机の隙間 [m]
        r.grip(squeeze)
        r.wait(0.2)
        r.glide(grasp + np.array([0, 0, SQUEEZE_RISE]), quat, 0.01)
        r.wait(1.0)
        hand = r.tool()[0]
        q = r.q()
        pinched = np.flatnonzero(np.linalg.norm(q - hand, axis=1) < HOLD_RADIUS)
        top = grasp + np.array([0, 0, SQUEEZE_RISE + LIFT])
        r.glide(top, quat, LIFT_SPEED)
        q = r.q()
        lifted = float(q[:, 2].mean() - rest[:, 2].mean())
        held_top = float((np.linalg.norm(q[pinched] - r.tool()[0], axis=1) < 0.025).mean()) if len(pinched) else 0.0
        # 運ぶ: 布の中心を +y 方向へ CARRY だけ動かす
        dest = top + np.array([0.0, CARRY, 0.0])
        r.glide(dest, quat, CARRY_SPEED)
        q = r.q()
        carried = bool(q[:, 2].min() > surface + 0.005)  # 運び終えた時点で布が机から浮いている
        r.glide(np.array([dest[0], dest[1], surface + tip_off + RELEASE_Z]), quat, 0.03)
        r.grip(0.0)
        r.wait(1.0)
        r.glide(r.tool()[0] + np.array([0, 0, 0.08]), quat, 0.1)
        r.wait(1.5)
        end = r.q()
        target = world_center[:2] + np.array([0.0, CARRY])
        res = {
            "seed": seed, "grip": squeeze, "yaw": yaw, "fingertip_gap_mm": 1000 * miss,
            "pinched_particles": int(len(pinched)), "lift_m": lifted, "held_at_top": held_top,
            "carried": carried,
            "landed_offset_mm": float(1000 * np.linalg.norm(end[:, :2].mean(axis=0) - target)),
            "area_ratio": hull_area(end[:, :2]) / area0,
            "z_spread_mm": float(1000 * (end[:, 2].max() - end[:, 2].min())),
            "finite": bool(np.isfinite(end).all()),
            "sim_seconds": r.steps / r.hz,
        }
        res["success"] = bool(carried and res["landed_offset_mm"] < 50.0)
        if video_dir is not None and r.frames:
            write_mp4(r.frames, Path(video_dir) / f"pinch_g{squeeze:.2f}_{seed:03d}.mp4", fps=round(r.hz / 2))
        return res
    finally:
        session.teardown()


def write_mp4(frames, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                          "-pix_fmt", "yuv420p", "-crf", "20", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:, :, :3]).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=int, default=2, help="最初の何試行を動画にするか")
    ap.add_argument("--out", default="out/pinch")
    ap.add_argument("--grip", type=float, nargs="+", default=[SQUEEZE], help="握りの強さ(複数で振る)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[device] cuda={wp.is_cuda_available()}", flush=True)
    results = []
    for g in args.grip:
        for k in range(args.trials):
            seed = args.seed + k
            res = trial(seed, 9700 + len(results), out if k < args.video else None, squeeze=g)
            print(json.dumps(res), flush=True)
            results.append(res)
            (out / "results.jsonl").write_text("\n".join(json.dumps(x) for x in results) + "\n")
        ok = sum(x["success"] for x in results if x["grip"] == g)
        print(f"[summary] grip {g:.2f}: success {ok}/{args.trials}", flush=True)


if __name__ == "__main__":
    main()
