"""要確認トレイ → 人のラベル → 再学習データ、のループ。"""

import json
import random

import numpy as np
from PIL import Image

from cloth_sort.ml.feedback import feedback_dataset, load_labeled, save_queue_item, split_chars
from cloth_sort.sim import fonts, synth


def _crop(label: str, seed: int = 0) -> np.ndarray:
    canvas = Image.new("RGBA", (60, 60), (220, 215, 205, 255))
    synth.draw_label(canvas, label, (54, 52), size_px=26, rng=random.Random(seed),
                     font_pool=fonts.available(fonts.TRAIN_FONTS)[:1])
    return np.asarray(canvas.convert("RGB"))


def test_split_chars_matches_label_length():
    chars = split_chars(_crop("16"), "16")
    assert chars is not None and [c for _, c in chars] == ["1", "6"]
    assert split_chars(_crop("16"), "1") is None          # 数が合わなければ使わない
    assert split_chars(_crop("16"), "1x") is None         # 語彙外の文字


def test_queue_roundtrip_and_dataset(tmp_path):
    q = tmp_path / "review_queue"
    save_queue_item(q, 0, _crop("16"), "15", 0.37, (10.0, 20.0), "test")
    save_queue_item(q, 1, _crop("8"), "8", 0.33, (30.0, 40.0), "test")
    assert load_labeled([q]) == []                        # ラベル前は使わない
    for i, label in ((0, "16"), (1, "skip")):
        j = q / f"{i:04d}.json"
        d = json.loads(j.read_text())
        d["label"] = label
        j.write_text(json.dumps(d))
    items = load_labeled([q])
    assert len(items) == 1 and items[0][1] == "16"        # skip は除く
    x, y, stats = feedback_dataset([q], copies=5, seed=0)
    assert x.shape == (10, 1, 28, 28) and stats == {"items": 1, "used": 1, "chars": 2}
    assert sorted(set(y.tolist())) == [1, 6]
