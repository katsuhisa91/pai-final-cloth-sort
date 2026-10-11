"""動画用のフレームを描く。左に作業台、右に状態パネル。"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from cloth_sort import config
from cloth_sort.sim import fonts
from cloth_sort.sim.world import World

PANEL_W = 420
MARGIN = 30
FRAME_W = config.BENCH_W_MM + PANEL_W + MARGIN * 3
FRAME_H = config.BENCH_H_MM + MARGIN * 2 + 60
BG = (24, 26, 32)
FG = (235, 235, 235)
DIM = (150, 150, 160)
ACCENT = (90, 180, 250)
WARN = (250, 150, 90)
OK = (120, 210, 130)


@dataclass
class Overlay:
    """フレームに重ねる情報。パイプラインが更新する。"""
    title: str = "番号付き布片の仕分け / Cloth sorting by handwritten ID"
    status: str = "待機中"
    phase: str = ""
    current_label: str = ""
    current_conf: float = 0.0
    current_dest: str = ""
    reason: str = ""
    done: int = 0
    total: int = 0
    review: int = 0
    detections: list[tuple[tuple[int, int, int, int], str, float]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    speed_note: str = ""


def render_frame(world: World, ov: Overlay) -> Image.Image:
    frame = Image.new("RGB", (FRAME_W, FRAME_H), BG)
    d = ImageDraw.Draw(frame)
    d.text((MARGIN, 14), ov.title, fill=FG, font=fonts.ui_font(26, bold=True))
    d.text((MARGIN, 44), f"t = {world.time_s:5.1f} s   {ov.speed_note}", fill=DIM, font=fonts.ui_font(16))

    scene = world.render_scene().convert("RGB")
    ox, oy = MARGIN, MARGIN + 60
    frame.paste(scene, (ox, oy))
    d.rectangle([ox - 1, oy - 1, ox + scene.width, oy + scene.height], outline=(80, 80, 90), width=1)

    small = fonts.ui_font(15, bold=True)
    for (x0, y0, x1, y1), text, conf in ov.detections:
        color = OK if conf >= 0.6 else WARN
        d.rectangle([ox + x0, oy + y0, ox + x1, oy + y1], outline=color, width=2)
        if text:
            d.rectangle([ox + x0, oy + y0 - 18, ox + x0 + 60, oy + y0], fill=color)
            d.text((ox + x0 + 3, oy + y0 - 18), f"{text} {conf:.2f}", fill=(20, 20, 20), font=small)

    px = ox + scene.width + MARGIN
    py = oy
    d.rectangle([px, py, px + PANEL_W, py + scene.height], fill=(34, 36, 44))
    y = py + 16
    big = fonts.ui_font(22, bold=True)
    body = fonts.ui_font(17)
    d.text((px + 16, y), ov.status, fill=ACCENT, font=big)
    y += 40
    lines = [
        ("工程", ov.phase),
        ("読み取り番号", f"{ov.current_label or '-'}  (conf {ov.current_conf:.2f})" if ov.current_label or ov.current_conf else "-"),
        ("行き先", ov.current_dest or "-"),
        ("判断理由", ov.reason or "-"),
        ("完了 / 全体", f"{ov.done} / {ov.total}"),
        ("要確認へ", str(ov.review)),
    ]
    for k, v in lines:
        d.text((px + 16, y), k, fill=DIM, font=body)
        d.text((px + 150, y), v, fill=FG, font=body)
        y += 28
    y += 12
    d.line([(px + 16, y), (px + PANEL_W - 16, y)], fill=(70, 70, 80))
    y += 10
    d.text((px + 16, y), "ログ", fill=DIM, font=body)
    y += 26
    for line in ov.log[-12:]:
        d.text((px + 16, y), line, fill=FG, font=fonts.ui_font(15))
        y += 22

    d.text((MARGIN, FRAME_H - 26),
           "カメラ → 検出 → 番号読み取り(CNN) → 計画 → 仮想アーム  /  dora-rs dataflow",
           fill=DIM, font=fonts.ui_font(15))
    return frame
