"""手書き風フォントの管理。

学習用と評価用でフォントを分け、「学習で見ていない筆跡」に対する汎化を測る
(sim2real のドメインランダマイゼーションを文字に適用した形)。
"""

from __future__ import annotations

import os
from functools import lru_cache

from PIL import ImageFont

_SUPPLEMENTAL = "/System/Library/Fonts/Supplemental"
_SYSTEM = "/System/Library/Fonts"

# (path, ttc index)
TRAIN_FONTS: list[tuple[str, int]] = [
    (f"{_SUPPLEMENTAL}/Chalkboard.ttc", 0),
    (f"{_SUPPLEMENTAL}/ChalkboardSE.ttc", 0),
    (f"{_SUPPLEMENTAL}/Comic Sans MS.ttf", 0),
    (f"{_SUPPLEMENTAL}/Comic Sans MS Bold.ttf", 0),
    (f"{_SUPPLEMENTAL}/Bradley Hand Bold.ttf", 0),
    (f"{_SUPPLEMENTAL}/Chalkduster.ttf", 0),
    (f"{_SUPPLEMENTAL}/Skia.ttf", 0),
    (f"{_SUPPLEMENTAL}/Trebuchet MS.ttf", 0),
]

# v2 で足した学習用フォント(筆記体・手書き風)。装飾が強いもの(Party LET、Zapfino)と数字のないもの(Kokonor)は除いた
EXTRA_TRAIN_FONTS: list[tuple[str, int]] = [
    (f"{_SUPPLEMENTAL}/Apple Chancery.ttf", 0),
    (f"{_SUPPLEMENTAL}/Brush Script.ttf", 0),
    (f"{_SUPPLEMENTAL}/Herculanum.ttf", 0),
    (f"{_SUPPLEMENTAL}/Luminari.ttf", 0),
    (f"{_SUPPLEMENTAL}/Papyrus.ttc", 0),
    (f"{_SUPPLEMENTAL}/Savoye LET.ttc", 0),
    (f"{_SUPPLEMENTAL}/SnellRoundhand.ttc", 0),
    (f"{_SUPPLEMENTAL}/Trattatello.ttf", 0),
]

# 学習に使わない筆跡。シミュレータ内の布片と評価に使う
HELDOUT_FONTS: list[tuple[str, int]] = [
    (f"{_SYSTEM}/MarkerFelt.ttc", 0),
    (f"{_SYSTEM}/Noteworthy.ttc", 0),
    (f"{_SUPPLEMENTAL}/SignPainter.ttc", 0),
]

UI_FONT_JP = f"{_SYSTEM}/ヒラギノ角ゴシック W4.ttc"
UI_FONT_JP_BOLD = f"{_SYSTEM}/ヒラギノ角ゴシック W6.ttc"


def available(fonts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(p, i) for p, i in fonts if os.path.exists(p)]


@lru_cache(maxsize=256)
def load(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


def ui_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = UI_FONT_JP_BOLD if bold else UI_FONT_JP
    if os.path.exists(path):
        return load(path, 0, size)
    return ImageFont.load_default()
