"""動画用のフレームを描く。

- 2D 版(World): 左に作業台の俯瞰図、右に状態パネル(1710×720)。
- MuJoCo 版(MujocoWorld): 左に 3D の斜め視点、右上に俯瞰カメラ + 検出枠、右下に状態パネル(1920×1080)。
"""

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

# MuJoCo 版のレイアウト
MJ_W, MJ_H = 1920, 1080
MJ_VIEW = (30, 80, 1180, 660)          # x, y, w, h: 3D 視点
MJ_TOP = (1240, 80, 650, 325)          # 俯瞰カメラ(1200×600 を 0.5417 倍)
MJ_PANEL = (1240, 430, 650, 610)       # 状態パネル


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
    hand_notes: list[str] | None = None      # 手首カメラの横の説明(None ならルールベースの制御の説明)


SCRIPTED_HAND_NOTES = [
    "左指のゴムパッド(2.5 mm 長い唇)で布の縁を押して",
    "折り目を作り、右指との間でつまむ。持ち上げ・搬送は",
    "抜けないよう 3〜5 cm/s。置き場の上で放して落とす。",
    "",
    "つかみ損ねたら撮り直して最大 2 回まで再試行。",
    "それでも駄目なら要確認へ(人が拾う)。",
]
POLICY_HAND_NOTES = [
    "アームの動きは模倣学習した方策(LeRobot の ACT)の出力。",
    "入力は俯瞰カメラ + 手首カメラ + 手先位置 + 目標、",
    "出力は手先の目標位置と指の指令(20 Hz)。",
    "",
    "教師は手続き的な把持。IK と位置サーボだけが共通で、",
    "つまむ・運ぶ・放すの手順は方策が決める。",
]


def _draw_detections(d: ImageDraw.ImageDraw, ov: Overlay, ox: int, oy: int, scale: float = 1.0) -> None:
    small = fonts.ui_font(max(11, int(15 * scale)), bold=True)
    for (x0, y0, x1, y1), text, conf in ov.detections:
        color = OK if conf >= 0.6 else WARN
        X0, Y0, X1, Y1 = ox + x0 * scale, oy + y0 * scale, ox + x1 * scale, oy + y1 * scale
        d.rectangle([X0, Y0, X1, Y1], outline=color, width=2)
        if text:
            tag_h = max(14, int(18 * scale))
            d.rectangle([X0, Y0 - tag_h, X0 + 60 * scale, Y0], fill=color)
            d.text((X0 + 3, Y0 - tag_h), f"{text} {conf:.2f}", fill=(20, 20, 20), font=small)


def _draw_panel(d: ImageDraw.ImageDraw, ov: Overlay, px: int, py: int, pw: int, ph: int, log_lines: int = 12) -> None:
    d.rectangle([px, py, px + pw, py + ph], fill=(34, 36, 44))
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
    d.line([(px + 16, y), (px + pw - 16, y)], fill=(70, 70, 80))
    y += 10
    d.text((px + 16, y), "ログ", fill=DIM, font=body)
    y += 26
    for line in ov.log[-log_lines:]:
        d.text((px + 16, y), line, fill=FG, font=fonts.ui_font(15))
        y += 22


def render_frame(world, ov: Overlay) -> Image.Image:
    if hasattr(world, "render_3d"):
        return _render_frame_mujoco(world, ov)
    return _render_frame_2d(world, ov)


def _render_frame_2d(world: World, ov: Overlay) -> Image.Image:
    frame = Image.new("RGB", (FRAME_W, FRAME_H), BG)
    d = ImageDraw.Draw(frame)
    d.text((MARGIN, 14), ov.title, fill=FG, font=fonts.ui_font(26, bold=True))
    d.text((MARGIN, 44), f"t = {world.time_s:5.1f} s   {ov.speed_note}", fill=DIM, font=fonts.ui_font(16))

    scene = world.render_scene().convert("RGB")
    ox, oy = MARGIN, MARGIN + 60
    frame.paste(scene, (ox, oy))
    d.rectangle([ox - 1, oy - 1, ox + scene.width, oy + scene.height], outline=(80, 80, 90), width=1)
    _draw_detections(d, ov, ox, oy)
    _draw_panel(d, ov, ox + scene.width + MARGIN, oy, PANEL_W, scene.height)
    d.text((MARGIN, FRAME_H - 26),
           "カメラ → 検出 → 番号読み取り(CNN) → 計画 → 仮想アーム  /  dora-rs dataflow",
           fill=DIM, font=fonts.ui_font(15))
    return frame


def _render_frame_mujoco(world, ov: Overlay) -> Image.Image:
    frame = Image.new("RGB", (MJ_W, MJ_H), BG)
    d = ImageDraw.Draw(frame)
    d.text((MARGIN, 18), ov.title, fill=FG, font=fonts.ui_font(30, bold=True))
    d.text((MARGIN, 54), f"t = {world.time_s:5.1f} s   {ov.speed_note}", fill=DIM, font=fonts.ui_font(17))

    vx, vy, vw, vh = MJ_VIEW
    frame.paste(world.render_3d(vw, vh), (vx, vy))
    d.rectangle([vx - 1, vy - 1, vx + vw, vy + vh], outline=(80, 80, 90), width=1)
    if hasattr(world, "render_hand"):
        d.text((vx, vy + vh + 8), "MuJoCo: Franka Panda + 2 本指ハンド(ゴムパッド)、布は変形体(flex)。物理シミュレーション",
               fill=DIM, font=fonts.ui_font(16))
        hand = world.render_hand(560, 280)
        hx, hy = vx, vy + vh + 36
        frame.paste(hand, (hx, hy))
        d.rectangle([hx - 1, hy - 1, hx + hand.width, hy + hand.height], outline=(80, 80, 90), width=1)
        notes = ov.hand_notes if ov.hand_notes is not None else SCRIPTED_HAND_NOTES
        head = ("手首カメラ(ハンドに固定。方策の入力の 1 つ)" if ov.hand_notes is not None
                else "手首カメラ(ハンドに固定。可視化用で、AI は右上の俯瞰カメラだけを使う)")
        d.text((hx + hand.width + 16, hy), head, fill=DIM, font=fonts.ui_font(16))
        for i, line in enumerate(notes):
            d.text((hx + hand.width + 16, hy + 26 + 22 * i), line, fill=DIM, font=fonts.ui_font(15))
    else:
        d.text((vx, vy + vh + 8), "MuJoCo: Franka Panda + 吸着ツール(物理シミュレーション、等速)", fill=DIM,
               font=fonts.ui_font(16))

    tx, ty, tw, th = MJ_TOP
    top = Image.fromarray(world.render_camera(noise=False)).resize((tw, th), Image.LANCZOS)
    frame.paste(top, (tx, ty))
    d.rectangle([tx - 1, ty - 1, tx + tw, ty + th], outline=(80, 80, 90), width=1)
    _draw_detections(d, ov, tx, ty, scale=tw / config.BENCH_W_MM)
    d.text((tx, ty + th + 6), "俯瞰カメラ(AI 側が見ている画像)と検出枠", fill=DIM, font=fonts.ui_font(15))

    px, py, pw, ph = MJ_PANEL
    _draw_panel(d, ov, px, py + 20, pw, ph - 20, log_lines=14)
    d.text((MARGIN, MJ_H - 30),
           "カメラ → 検出 → 番号読み取り(CNN) → 計画 → アーム制御(IK) → 把持(物理)  /  dora-rs dataflow(sim / brain / recorder)",
           fill=DIM, font=fonts.ui_font(15))
    return frame
