"""要確認トレイに回った布を人が確認した結果を、再学習のデータにする(運用のループ)。

1. パイプラインは、確信度がしきい値未満の布の切り出し画像を review_queue/ に残す(画像 + JSON)
2. 人が tools/review_label.py で正しい番号を付ける(JSON の "label" が埋まる)
3. ml/train.py --feedback <review_queue ...> で、番号付きの切り出しを 1 文字ずつに分けて学習データに足す

切り出しは実際に読んだ布片(ここではシミュレータ)の筆跡そのものなので、合成フォントでは足りない分を補える。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from cloth_sort.perception.reader import _components, _drop_edge_artifacts, _merge_vertical, ink_mask
from cloth_sort.sim.synth import CHAR_CLASSES, handwriting_jitter, normalize_char


def save_queue_item(queue_dir: Path, index: int, crop: np.ndarray, read: str, conf: float,
                    pick_xy: tuple[float, float], source: str) -> Path:
    """要確認に回した布の切り出しを残す。label は人が後で埋める。"""
    queue_dir.mkdir(parents=True, exist_ok=True)
    stem = queue_dir / f"{index:04d}"
    Image.fromarray(crop).save(stem.with_suffix(".png"))
    stem.with_suffix(".json").write_text(json.dumps(
        {"read": read, "conf": round(float(conf), 3), "pick_xy": [float(pick_xy[0]), float(pick_xy[1])],
         "source": source, "label": None}, ensure_ascii=False, indent=2))
    return stem.with_suffix(".png")


def load_labeled(queue_dirs: list[Path]) -> list[tuple[np.ndarray, str]]:
    """ラベルが付いた切り出し (画像, 正しい番号) を返す。ラベルが空・"skip" のものは除く。"""
    items = []
    for d in queue_dirs:
        for j in sorted(d.glob("*.json")):
            meta = json.loads(j.read_text())
            label = meta.get("label")
            if not label or label == "skip":
                continue
            png = j.with_suffix(".png")
            if png.exists():
                items.append((np.asarray(Image.open(png).convert("RGB")), str(label)))
    return items


def split_chars(crop: np.ndarray, label: str) -> list[tuple[np.ndarray, str]] | None:
    """切り出しを読み取りと同じ方法で文字に分け、ラベルの文字と対応づける。数が合わなければ使わない。"""
    if any(c not in CHAR_CLASSES for c in label):
        return None
    mask = ink_mask(crop)
    boxes = _merge_vertical(_drop_edge_artifacts(_components(mask), mask.shape))
    if len(boxes) != len(label):
        return None
    return [(mask[y0:y1, x0:x1].astype(np.float32), ch) for (x0, y0, x1, y1), ch in zip(boxes, label)]


def feedback_dataset(queue_dirs: list[Path], copies: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """番号付きの切り出しから、1 文字ごとに copies 個の揺らしたサンプルを作る。"""
    rng = random.Random(seed)
    xs, ys = [], []
    stats = {"items": 0, "used": 0, "chars": 0}
    for crop, label in load_labeled(queue_dirs):
        stats["items"] += 1
        chars = split_chars(crop, label)
        if chars is None:
            continue
        stats["used"] += 1
        for mask, ch in chars:
            stats["chars"] += 1
            ci = CHAR_CLASSES.index(ch)
            xs.append(normalize_char(mask))           # そのまま 1 つ
            ys.append(ci)
            for _ in range(copies - 1):
                xs.append(normalize_char(handwriting_jitter(mask, rng), rng))
                ys.append(ci)
    if not xs:
        return np.zeros((0, 1, 28, 28), np.float32), np.zeros((0,), np.int64), stats
    return np.stack(xs)[:, None].astype(np.float32), np.asarray(ys, dtype=np.int64), stats
