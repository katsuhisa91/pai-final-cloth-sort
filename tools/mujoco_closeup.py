"""MuJoCo(flex の布)で 1 枚をつまんで運んで放す様子を、寄りのカメラで撮る。

RoboSimStudio の実験(experiments/robosimstudio/cloth_pinch.py)と見比べるための映像。
アームは本体と同じルールベースの制御(GripperArm + PickPlace)、速度も本体のまま。

    MUJOCO_GL=osmesa .venv/bin/python tools/mujoco_closeup.py --seed 3 --out runs/closeup/mujoco_pinch.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from cloth_sort.robot.executor import Phase, PickPlace
from cloth_sort.sim.mujoco_cloth_world import GripperArm, MujocoClothWorld

W, H = 1280, 720
DT = 0.05  # パイプラインと同じ制御周期 [s]
CARRY_MM = 150.0  # 運ぶ距離。RoboSimStudio の実験と同じ 15 cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="runs/closeup/mujoco_pinch.mp4")
    ap.add_argument("--still", default=None, help="最初の 1 枚だけ PNG に保存して終わる(画角の確認用)")
    args = ap.parse_args()

    world = MujocoClothWorld.random_scene(1, args.seed)
    arm = GripperArm(world)
    piece = world.pieces[0]
    pick = piece.center
    place = (pick[0] + CARRY_MM, pick[1])
    pp = PickPlace(arm, pick, place, "demo")

    # 布の位置と運び先の中間を見る、斜め上からの固定カメラ
    start = world.cloth_vertices(piece.id).mean(axis=0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = start + np.array([CARRY_MM / 2000.0, 0.0, 0.02])
    cam.distance, cam.azimuth, cam.elevation = 0.34, 120.0, -26.0
    renderer = mujoco.Renderer(world.model, height=H, width=W)

    def shot() -> bytes:
        renderer.update_scene(world.data, camera=cam)
        return renderer.render().tobytes()

    if args.still:
        from PIL import Image

        Image.frombytes("RGB", (W, H), shot()).save(args.still)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(round(1 / DT)), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for _ in range(round(0.5 / DT)):
        proc.stdin.write(shot())
    t = 0.0
    while not pp.finished and t < 60.0:
        pp.step(DT)
        t += DT
        proc.stdin.write(shot())
    for _ in range(round(1.0 / DT)):
        world.step(DT)
        proc.stdin.write(shot())
    proc.stdin.close()
    proc.wait()
    print(f"{out}: phase={pp.phase.value} retries={pp.grip_retries} lost={pp.lost_in_transit} sim={t:.1f}s")


if __name__ == "__main__":
    main()
