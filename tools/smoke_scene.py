"""シーン生成と検出の目視確認用。カメラ画像と検出結果を PNG に書く。"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from cloth_sort import config
from cloth_sort.perception.detect import detect_pieces
from cloth_sort.sim.world import World

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
w = World.random_scene(12, seed=1)
cam = w.render_camera()
img = Image.fromarray(cam)
d = ImageDraw.Draw(img)
dets = detect_pieces(cam, w.render_camera(include_pieces=False), region=config.SOURCE_AREA)
for det in dets:
    d.rectangle(det.bbox, outline=(255, 0, 0), width=2)
img.save(out / "camera_detect.png")
w.render_scene().convert("RGB").save(out / "scene.png")
for i, det in enumerate(dets[:4]):
    Image.fromarray(det.crop).resize((det.crop.shape[1] * 4, det.crop.shape[0] * 4), Image.NEAREST).save(out / f"crop{i}.png")
print("pieces", [(p.label, [round(c) for c in p.center]) for p in w.pieces])
print("detections", len(dets), [[round(c) for c in det.center] for det in dets])
