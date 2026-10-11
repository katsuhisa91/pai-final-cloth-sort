"""作業台の 2D シミュレーション。

布片の位置・番号と、アームの把持状態を持ち、俯瞰カメラ画像を描く。
物理(摩擦・変形)は扱わない。本課題の焦点は「読み取り → 計画 → 実行」の運用ループなので、
把持は「グリッパ位置に布片の中心が来れば成功」とみなす。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from cloth_sort import config
from cloth_sort.sim import fonts, synth

BENCH_RGB = (196, 164, 118)   # 木目の机
GRAIN_RGB = (178, 146, 102)


@dataclass
class Piece:
    id: int
    label: str
    center: tuple[float, float]   # mm
    angle_deg: float
    color: tuple[int, int, int]
    held: bool = False
    placed_at: str | None = None  # 置いた場所の名前(スロット番号 or "review")


@dataclass
class World:
    rng: random.Random
    pieces: list[Piece] = field(default_factory=list)
    gripper_xy: tuple[float, float] = (config.ARM_BASE[0], config.ARM_BASE[1] - 40)
    gripper_z: float = 1.0           # 0 = 机面、1 = 上空(正規化)
    gripper_closed: bool = False
    held_piece: int | None = None
    time_s: float = 0.0
    label_fonts: list[tuple[str, int]] = field(
        default_factory=lambda: fonts.available(fonts.TRAIN_FONTS) + fonts.available(fonts.HELDOUT_FONTS))
    _grain: list[tuple[int, int, int, int]] = field(default_factory=list)
    _label_cache: dict[int, Image.Image] = field(default_factory=dict)

    # ---------- 生成 ----------
    @classmethod
    def random_scene(cls, n_pieces: int, seed: int, labels: list[str] | None = None) -> "World":
        rng = random.Random(seed)
        w = cls(rng=rng)
        pool = labels if labels is not None else list(config.SLOT_LABELS)
        chosen = rng.sample(pool, k=min(n_pieces, len(pool)))
        x0, y0, x1, y1 = config.SOURCE_AREA
        half = config.PIECE_MM / 2 + 6
        placed: list[tuple[float, float]] = []
        for i, label in enumerate(chosen):
            for _ in range(500):
                c = (rng.uniform(x0 + half, x1 - half), rng.uniform(y0 + half, y1 - half))
                if all(max(abs(c[0] - p[0]), abs(c[1] - p[1])) > config.PIECE_MM + config.PIECE_MIN_GAP_MM
                       for p in placed):
                    placed.append(c)
                    break
            else:
                raise RuntimeError("布片を重ならずに置けませんでした。n_pieces を減らしてください")
            w.pieces.append(Piece(i, label, c, rng.uniform(-10, 10), _cloth_color(rng)))
        w._grain = [(rng.randint(0, config.BENCH_W_MM), rng.randint(0, config.BENCH_H_MM),
                     rng.randint(60, 400), rng.randint(1, 2)) for _ in range(90)]
        return w

    # ---------- 操作 ----------
    def move_gripper(self, xy: tuple[float, float], z: float) -> None:
        self.gripper_xy = xy
        self.gripper_z = z
        if self.held_piece is not None:
            self.pieces[self.held_piece].center = xy

    def close_gripper(self) -> int | None:
        """机面で閉じたとき、グリッパ直下に布片があれば把持する。把持した id を返す。"""
        self.gripper_closed = True
        if self.gripper_z > 0.05:
            return None
        gx, gy = self.gripper_xy
        for p in self.pieces:
            if p.held:
                continue
            if abs(p.center[0] - gx) <= config.PIECE_MM * 0.35 and abs(p.center[1] - gy) <= config.PIECE_MM * 0.35:
                p.held = True
                self.held_piece = p.id
                return p.id
        return None

    def open_gripper(self, placed_at: str | None = None) -> None:
        self.gripper_closed = False
        if self.held_piece is not None:
            p = self.pieces[self.held_piece]
            p.held = False
            p.center = self.gripper_xy
            p.placed_at = placed_at
            self.held_piece = None

    def piece_at(self, xy: tuple[float, float], tol_mm: float = 30) -> Piece | None:
        for p in self.pieces:
            if abs(p.center[0] - xy[0]) <= tol_mm and abs(p.center[1] - xy[1]) <= tol_mm:
                return p
        return None

    # ---------- 描画 ----------
    def render_camera(self, noise: bool = True, include_pieces: bool = True) -> np.ndarray:
        """俯瞰カメラ画像(H, W, 3) uint8。アームは写り込まない前提(撮影時は退避)。

        include_pieces=False は段取り時の「何も置いていない机」の撮影(背景差分の基準)。
        """
        img = self._render_bench()
        self._draw_board(img)
        if include_pieces:
            for p in self.pieces:
                if not p.held:
                    self._paste_piece(img, p)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        if noise:
            g = np.random.default_rng(int(self.time_s * 1000) + 7)
            arr = arr * g.uniform(0.93, 1.07) + g.normal(0, 2.5, arr.shape)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def render_scene(self) -> Image.Image:
        """ビューア用。布片(把持中も)とアームを描く RGBA 画像。"""
        img = self._render_bench()
        self._draw_board(img)
        for p in self.pieces:
            if not p.held:
                self._paste_piece(img, p)
        if self.held_piece is not None:
            self._paste_piece(img, self.pieces[self.held_piece], lifted=self.gripper_z > 0.05)
        self._draw_arm(img)
        return img

    def _render_bench(self) -> Image.Image:
        img = Image.new("RGBA", (config.BENCH_W_MM, config.BENCH_H_MM), BENCH_RGB + (255,))
        d = ImageDraw.Draw(img)
        for x, y, length, wdt in self._grain:
            d.line([(x, y), (x + length, y + self.rng.randint(-3, 3) * 0)], fill=GRAIN_RGB + (255,), width=wdt)
        return img

    def _draw_board(self, img: Image.Image) -> None:
        d = ImageDraw.Draw(img)
        half = config.SLOT_PITCH_MM / 2 - 6
        f = fonts.ui_font(22, bold=True)
        line = (150, 118, 86, 255)   # ペンのインクより明るい色にして、読み取り時にインクと混同しない
        for s in config.board_slots():
            cx, cy = s.center
            d.rectangle([cx - half, cy - half, cx + half, cy + half], outline=line, width=2)
            d.text((cx - half + 6, cy - half + 4), s.label, fill=line, font=f)
        x0, y0, x1, y1 = config.REVIEW_TRAY
        d.rectangle([x0, y0, x1, y1], outline=(170, 60, 60, 255), width=2)
        d.text((x0 + 8, y0 + 4), "要確認 / review", fill=(170, 60, 60, 255), font=fonts.ui_font(18))

    def _piece_image(self, p: Piece) -> Image.Image:
        if p.id in self._label_cache:
            return self._label_cache[p.id]
        s = config.PIECE_MM
        rng = random.Random(p.id * 7919 + 13)
        base = Image.new("RGBA", (s, s), p.color + (255,))
        # 織り目: 薄い格子
        d = ImageDraw.Draw(base)
        weave = tuple(max(0, c - 10) for c in p.color) + (255,)
        for i in range(0, s, 4):
            d.line([(i, 0), (i, s)], fill=weave, width=1)
        synth.draw_label(base, p.label, (s - 4, s - 2), size_px=rng.randint(13, 17), rng=rng,
                         font_pool=self.label_fonts)
        rot = base.rotate(p.angle_deg, resample=Image.BICUBIC, expand=True)
        self._label_cache[p.id] = rot
        return rot

    def _paste_piece(self, img: Image.Image, p: Piece, lifted: bool = False) -> None:
        tile = self._piece_image(p)
        if lifted:
            shadow = Image.new("RGBA", tile.size, (0, 0, 0, 70))
            shadow.putalpha(tile.getchannel("A").point(lambda a: a * 70 // 255))
            img.alpha_composite(shadow, (int(p.center[0] - tile.width / 2 + 8), int(p.center[1] - tile.height / 2 + 10)))
        img.alpha_composite(tile, (int(p.center[0] - tile.width / 2), int(p.center[1] - tile.height / 2)))

    def _draw_arm(self, img: Image.Image) -> None:
        d = ImageDraw.Draw(img)
        bx, by = config.ARM_BASE
        gx, gy = self.gripper_xy
        # 2 リンクの腕を土台からグリッパまで描く(肘は中点を少し外側へ)
        mx, my = (bx + gx) / 2 + (gy - by) * 0.25, (by + gy) / 2 - (gx - bx) * 0.25
        d.line([(bx, by), (mx, my)], fill=(60, 60, 70, 255), width=14)
        d.line([(mx, my), (gx, gy)], fill=(80, 80, 95, 255), width=10)
        d.ellipse([bx - 18, by - 18, bx + 18, by + 18], fill=(50, 50, 60, 255))
        r = 16 if self.gripper_z > 0.5 else 11
        color = (230, 90, 60, 255) if self.gripper_closed else (60, 150, 230, 255)
        d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=color, width=4)
        d.line([(gx - r - 6, gy), (gx + r + 6, gy)], fill=color, width=3)


def _cloth_color(rng: random.Random) -> tuple[int, int, int]:
    palette = [
        (235, 235, 232), (240, 236, 222), (236, 200, 210), (150, 150, 158),
        (190, 160, 140), (215, 215, 225), (225, 210, 190), (170, 180, 170),
    ]
    base = rng.choice(palette)
    return tuple(int(min(255, max(0, c + rng.randint(-8, 8)))) for c in base)
