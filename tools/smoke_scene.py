"""シーン生成と検出の目視確認用。カメラ画像と検出結果を PNG に書く。

    python tools/smoke_scene.py runs/try                    # 2D 版
    python tools/smoke_scene.py runs/try --backend mujoco   # MuJoCo 版(3D 視点 scene_3d.png も書く)
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from cloth_sort import config
from cloth_sort.perception.detect import detect_pieces
from cloth_sort.pipeline import make_world_and_arm

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--backend", choices=["2d", "mujoco"], default="2d")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()

a.out.mkdir(parents=True, exist_ok=True)
w, arm = make_world_and_arm(a.backend, a.n, a.seed)
cam = w.render_camera()
img = Image.fromarray(cam)
d = ImageDraw.Draw(img)
dets = detect_pieces(cam, w.render_camera(include_pieces=False), region=config.SOURCE_AREA)
for det in dets:
    d.rectangle(det.bbox, outline=(255, 0, 0), width=2)
img.save(a.out / "camera_detect.png")
if hasattr(w, "render_3d"):
    w.render_3d().save(a.out / "scene_3d.png")
    w.render_3d().save(a.out / "scene.png")
else:
    w.render_scene().convert("RGB").save(a.out / "scene.png")
for i, det in enumerate(dets[:4]):
    Image.fromarray(det.crop).resize((det.crop.shape[1] * 4, det.crop.shape[0] * 4), Image.NEAREST).save(a.out / f"crop{i}.png")
print("pieces", [(p.label, [round(c) for c in p.center]) for p in w.pieces])
print("detections", len(dets), [[round(c) for c in det.center] for det in dets])
