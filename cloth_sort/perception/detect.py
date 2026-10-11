"""布片の検出(古典的な画像処理)。

段取り時に撮った「何も置いていない机」の画像との背景差分で布片を分離し、
連結成分ごとに外接矩形を返す。机の色に近い布(ベージュ等)も差分なら拾える。
深層学習を使うまでもない部分は使わない、という判断。読み取りは別モジュール。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloth_sort import config


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]      # x0, y0, x1, y1 (px)
    center: tuple[float, float]          # mm(1 px = 1 mm)
    crop: np.ndarray                     # RGB uint8


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4 近傍の連結成分ラベリング(scipy 非依存の簡易実装)。"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current += 1
                labels[y, x] = current
                stack.append((y, x))
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def estimate_background(image: np.ndarray) -> np.ndarray:
    """机の色を画像の中央値で推定する(布片は面積の少数派という前提)。"""
    return np.median(image.reshape(-1, 3), axis=0)


def foreground_mask(image: np.ndarray, background: np.ndarray | None, downscale: int) -> np.ndarray:
    """背景との色差が大きい画素を True にする。background があれば差分、なければ中央値色との距離。"""
    small = image[::downscale, ::downscale].astype(np.float32)
    if background is None:
        bg = estimate_background(image)
        dist = np.sqrt(((small - bg) ** 2).sum(axis=-1))
        return dist > 38.0
    ref = background[::downscale, ::downscale].astype(np.float32)
    gain = float(np.median(ref)) / max(1.0, float(np.median(small)))   # 照明の変動を打ち消す
    dist = np.sqrt(((small * gain - ref) ** 2).sum(axis=-1))
    return dist > 16.0


def detect_pieces(image: np.ndarray, background: np.ndarray | None = None, downscale: int = 4,
                  min_area_mm2: float = 1200, max_area_mm2: float = 6000,
                  region: tuple[int, int, int, int] | None = None) -> list[Detection]:
    """俯瞰画像から布片を検出する。region を渡すとその範囲(mm)だけを見る。"""
    mask = foreground_mask(image, background, downscale)
    if region is not None:
        x0, y0, x1, y1 = (v // downscale for v in region)
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = True
        mask &= keep
    labels, n = _label_components(mask)
    out: list[Detection] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        area = len(ys) * downscale * downscale
        if not (min_area_mm2 <= area <= max_area_mm2):
            continue
        x0, x1 = xs.min() * downscale, (xs.max() + 1) * downscale
        y0, y1 = ys.min() * downscale, (ys.max() + 1) * downscale
        pad = 3
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(image.shape[1], x1 + pad), min(image.shape[0], y1 + pad)
        out.append(Detection((x0, y0, x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2), image[y0:y1, x0:x1].copy()))
    out.sort(key=lambda d: (d.center[1] // config.PIECE_MM, d.center[0]))
    return out
