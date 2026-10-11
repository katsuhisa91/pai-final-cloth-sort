"""手書き風の番号を描く・学習用の文字画像を合成する。

シミュレータ(布片に番号を描く)と学習データ生成の両方がここを使うので、
描画の癖(フォント、傾き、太さ、にじみ)が世界と学習データで一致する。
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cloth_sort.sim import fonts

CHAR_CLASSES = list("0123456789-")  # 読み取りの語彙。番号は「12」「1-2」のように文字列で扱う
CHAR_SIZE = 28                      # CNN への入力サイズ
INK_RGB = (35, 32, 40)              # 油性ペンの色


def _render_text(text: str, font_path: str, font_index: int, size_px: int, angle_deg: float,
                 thickness: int) -> Image.Image:
    """透明背景の RGBA 画像に文字を描いて返す。"""
    font = fonts.load(font_path, font_index, size_px)
    pad = size_px
    tmp = Image.new("L", (size_px * (len(text) + 1) + pad * 2, size_px * 2 + pad * 2), 0)
    d = ImageDraw.Draw(tmp)
    d.text((pad, pad), text, fill=255, font=font)
    if thickness > 0:
        tmp = tmp.filter(ImageFilter.MaxFilter(2 * thickness + 1))
    tmp = tmp.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    bbox = tmp.getbbox()
    if bbox is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    mask = tmp.crop(bbox)
    rgba = Image.new("RGBA", mask.size, INK_RGB + (0,))
    rgba.putalpha(mask)
    return rgba


def draw_label(canvas: Image.Image, text: str, anchor_xy: tuple[int, int], size_px: int,
               rng: random.Random, font_pool: list[tuple[str, int]]) -> tuple[int, int, int, int]:
    """canvas の anchor_xy(右下の基準点)に番号を描き、描いた範囲 (x0, y0, x1, y1) を返す。"""
    path, idx = rng.choice(font_pool)
    angle = rng.uniform(-12, 12)
    thickness = rng.choice([0, 0, 1])
    glyph = _render_text(text, path, idx, size_px, angle, thickness)
    x1, y1 = anchor_xy
    x0, y0 = x1 - glyph.width, y1 - glyph.height
    canvas.alpha_composite(glyph, (x0, y0))
    return (x0, y0, x1, y1)


def char_sample(ch: str, rng: random.Random, font_pool: list[tuple[str, int]]) -> np.ndarray:
    """1 文字の学習サンプル(CHAR_SIZE 角、0〜1 の float32、インク=1)を合成する。

    読み取り側の前処理(インクの連結成分を正方形に詰めてリサイズ)と同じ形にそろえる。
    """
    path, idx = rng.choice(font_pool)
    size = rng.randint(22, 44)
    angle = rng.uniform(-15, 15)
    thickness = rng.choice([0, 0, 1, 1, 2])
    glyph = _render_text(ch, path, idx, size, angle, thickness)
    mask = np.asarray(glyph.getchannel("A"), dtype=np.float32) / 255.0
    return normalize_char(mask, rng)


def normalize_char(mask: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
    """インクのマスク(任意サイズ)を CHAR_SIZE 角に正規化する。アスペクト比は保つ。

    学習時は rng を渡してわずかな位置ずれ・ぼかし・ノイズを加える。
    """
    h, w = mask.shape
    side = max(h, w)
    margin = int(side * 0.18) + 1
    canvas = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.float32)
    oy = margin + (side - h) // 2
    ox = margin + (side - w) // 2
    if rng is not None:
        oy += rng.randint(-margin // 2, margin // 2)
        ox += rng.randint(-margin // 2, margin // 2)
    canvas[oy:oy + h, ox:ox + w] = mask
    img = Image.fromarray((canvas * 255).astype(np.uint8)).resize((CHAR_SIZE, CHAR_SIZE), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32) / 255.0
    if rng is not None:
        if rng.random() < 0.5:
            out = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)),
                             dtype=np.float32) / 255.0
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.05, out.shape).astype(np.float32)
        out = np.clip(out + noise, 0, 1)
    return out


def build_dataset(n_per_class: int, font_pool: list[tuple[str, int]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(N, 1, 28, 28) の画像と (N,) のクラス番号を返す。"""
    rng = random.Random(seed)
    xs, ys = [], []
    for ci, ch in enumerate(CHAR_CLASSES):
        for _ in range(n_per_class):
            xs.append(char_sample(ch, rng, font_pool))
            ys.append(ci)
    x = np.stack(xs)[:, None, :, :].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(len(y))
    return x[perm], y[perm]
