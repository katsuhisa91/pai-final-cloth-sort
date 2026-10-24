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

# Linux(CI の Ubuntu など)には手書き風フォントが無いので、標準的なフォントで代用する。
# 報告している精度の数値は macOS のフォント構成でのもの
_LINUX = "/usr/share/fonts/truetype"


def _pick(mac: list[tuple[str, int]], linux: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return mac if any(os.path.exists(p) for p, _ in mac) else linux


# (path, ttc index)
TRAIN_FONTS: list[tuple[str, int]] = _pick([
    (f"{_SUPPLEMENTAL}/Chalkboard.ttc", 0),
    (f"{_SUPPLEMENTAL}/ChalkboardSE.ttc", 0),
    (f"{_SUPPLEMENTAL}/Comic Sans MS.ttf", 0),
    (f"{_SUPPLEMENTAL}/Comic Sans MS Bold.ttf", 0),
    (f"{_SUPPLEMENTAL}/Bradley Hand Bold.ttf", 0),
    (f"{_SUPPLEMENTAL}/Chalkduster.ttf", 0),
    (f"{_SUPPLEMENTAL}/Skia.ttf", 0),
    (f"{_SUPPLEMENTAL}/Trebuchet MS.ttf", 0),
], [
    (f"{_LINUX}/dejavu/DejaVuSans.ttf", 0),
    (f"{_LINUX}/dejavu/DejaVuSans-Bold.ttf", 0),
    (f"{_LINUX}/liberation/LiberationSans-Regular.ttf", 0),
    (f"{_LINUX}/liberation/LiberationSans-Bold.ttf", 0),
])

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
HELDOUT_FONTS: list[tuple[str, int]] = _pick([
    (f"{_SYSTEM}/MarkerFelt.ttc", 0),
    (f"{_SYSTEM}/Noteworthy.ttc", 0),
    (f"{_SUPPLEMENTAL}/SignPainter.ttc", 0),
], [
    (f"{_LINUX}/dejavu/DejaVuSerif.ttf", 0),
    (f"{_LINUX}/liberation/LiberationSerif-Regular.ttf", 0),
])

UI_FONT_JP = f"{_SYSTEM}/ヒラギノ角ゴシック W4.ttc"
UI_FONT_JP_BOLD = f"{_SYSTEM}/ヒラギノ角ゴシック W6.ttc"
# ヒラギノが無い環境(Linux)では Noto Sans CJK JP(fonts-noto-cjk)を使う。ttc の 0 番が JP
UI_FONT_JP_FALLBACK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
UI_FONT_JP_BOLD_FALLBACK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def available(fonts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(p, i) for p, i in fonts if os.path.exists(p)]


@lru_cache(maxsize=256)
def load(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


def ui_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in ((UI_FONT_JP_BOLD, UI_FONT_JP_BOLD_FALLBACK) if bold else (UI_FONT_JP, UI_FONT_JP_FALLBACK)):
        if os.path.exists(path):
            return load(path, 0, size)
    return ImageFont.load_default()
