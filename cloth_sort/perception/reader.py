"""布片の切り出し画像から番号を読む。

手順: インク画素のマスク → 連結成分を左から右に並べる → 1 文字ずつ CNN で分類 → 文字列に結合。
確信度は各文字の softmax 最小値。しきい値未満は「読めない」として扱い、下流で要確認トレイへ回す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cloth_sort.perception.model import CharNet
from cloth_sort.sim.synth import CHAR_CLASSES, normalize_char

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "charnet.pt"


@dataclass(frozen=True)
class Reading:
    text: str
    confidence: float
    char_boxes: tuple[tuple[int, int, int, int], ...]

    @property
    def ok(self) -> bool:
        return bool(self.text)


def ink_mask(crop: np.ndarray) -> np.ndarray:
    """暗い画素(ペンのインク)を True にする。布の色は明るい〜中間なので輝度で分けられる。"""
    v = crop.astype(np.float32).mean(axis=-1)
    sat = crop.max(axis=-1).astype(np.float32) - crop.min(axis=-1).astype(np.float32)
    return (v < 95) & (sat < 60)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """連結成分の外接矩形(8 近傍)。小さなゴミは落とす。"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                stack = [(y, x)]
                seen[y, x] = True
                ys, xs = [], []
                while stack:
                    cy, cx = stack.pop()
                    ys.append(cy)
                    xs.append(cx)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                if len(ys) >= 6:
                    boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return boxes


def _merge_vertical(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """x 範囲が重なる成分は同じ文字(「i」の点のような分離)として縦に結合する。"""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[2], b[2]) - max(m[0], b[0])
            if overlap > 0.5 * min(m[2] - m[0], b[2] - b[0]):
                m[0], m[1], m[2], m[3] = min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])
                continue
        merged.append(list(b))
    return [tuple(m) for m in merged]  # type: ignore[misc]


class LabelReader:
    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, min_confidence: float = 0.6) -> None:
        self.model = CharNet()
        state = torch.load(str(weights), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.min_confidence = min_confidence

    @torch.no_grad()
    def read(self, crop: np.ndarray) -> Reading:
        mask = ink_mask(crop)
        boxes = _merge_vertical(_components(mask))
        if not boxes:
            return Reading("", 0.0, ())
        chars = np.stack([normalize_char(mask[y0:y1, x0:x1].astype(np.float32)) for x0, y0, x1, y1 in boxes])
        logits = self.model(torch.from_numpy(chars)[:, None])
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        text = "".join(CHAR_CLASSES[i] for i in idx.tolist())
        confidence = float(conf.min())
        if text.startswith("-") or text.endswith("-") or "--" in text:
            confidence = min(confidence, 0.3)
        return Reading(text, confidence, tuple(boxes))

    def accept(self, r: Reading) -> bool:
        return r.ok and r.confidence >= self.min_confidence
