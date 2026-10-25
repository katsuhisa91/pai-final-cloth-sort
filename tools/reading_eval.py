"""番号読み取りをシミュレータの場面の布で評価し、要確認トレイからのフィードバックを模擬する。

    # 1. MuJoCo の場面から布の切り出しを集める(描画を使う。ウィンドウサーバに繋げる環境で)
    python tools/reading_eval.py collect --seeds 70000 70040 --out data/reading/pool.npz
    python tools/reading_eval.py collect --seeds 71000 71040 --out data/reading/test.npz
    # 2. モデルの成績(布 1 枚単位: 置き場へ正しく / 置き場へ誤って / 要確認)
    python tools/reading_eval.py score --weights models/charnet.pt --data data/reading/test.npz
    # 3. pool のうち、このモデルで要確認に回る布に「人が」番号を付けた review_queue を作る(シミュレーションでは正解を使う)
    python tools/reading_eval.py feedback --weights runs/reader/v2/charnet.pt --data data/reading/pool.npz --out data/reading/review_queue

pool と test は別のシードの場面(布の番号・向き・色・筆跡の組み合わせが違う)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def collect(seeds: range, out: Path, n_pieces: int) -> None:
    from cloth_sort import config
    from cloth_sort.perception.detect import detect_pieces
    from cloth_sort.sim.mujoco_cloth_world import MujocoClothWorld
    crops, labels, meta = [], [], []
    for seed in seeds:
        w = MujocoClothWorld.random_scene(n_pieces, seed)
        w.step(0.4)
        bg = w.render_camera(include_pieces=False)
        img = w.render_camera()
        for d in detect_pieces(img, bg, region=config.SOURCE_AREA):
            p = w.piece_at(d.center)
            if p is None:
                continue
            crops.append(d.crop)
            labels.append(p.label)
            meta.append({"seed": seed, "pick_xy": [float(d.center[0]), float(d.center[1])]})
        print(f"seed {seed}: total {len(crops)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, crops=np.array(crops, dtype=object), labels=np.array(labels), meta=json.dumps(meta))


def _load(path: Path):
    z = np.load(path, allow_pickle=True)
    return list(z["crops"]), list(z["labels"]), json.loads(str(z["meta"]))


def score(weights: Path, data: Path, min_conf: float = 0.6, verbose: bool = False) -> dict:
    from cloth_sort.perception.reader import LabelReader
    reader = LabelReader(weights, min_confidence=min_conf)
    crops, labels, _ = _load(data)
    s = Counter()
    wrong = Counter()
    for crop, truth in zip(crops, labels):
        r = reader.read(crop)
        if reader.accept(r):
            if r.text == truth:
                s["slot_correct"] += 1
            else:
                s["slot_wrong"] += 1          # 確信度が高いのに読み違い = 誤配置になる
                wrong[f"{truth}->{r.text}"] += 1
        else:
            s["review"] += 1
            if r.text == truth:
                s["review_but_correct"] += 1  # 読めていたが自信がなかった
    n = len(labels)
    out = {"pieces": n, **{k: s[k] for k in ("slot_correct", "slot_wrong", "review", "review_but_correct")},
           "slot_correct_rate": round(s["slot_correct"] / n, 3), "slot_wrong_rate": round(s["slot_wrong"] / n, 3),
           "review_rate": round(s["review"] / n, 3), "wrong_examples": dict(wrong.most_common(8)), "weights": str(weights)}
    return out


def feedback(weights: Path, data: Path, out: Path, min_conf: float = 0.6) -> dict:
    """このモデルで要確認に回る布を review_queue に書き出し、人の代わりに正解の番号を付ける(simulated_human)。"""
    from cloth_sort.ml.feedback import save_queue_item
    from cloth_sort.perception.reader import LabelReader
    reader = LabelReader(weights, min_confidence=min_conf)
    crops, labels, meta = _load(data)
    n = 0
    for i, (crop, truth, m) in enumerate(zip(crops, labels, meta)):
        r = reader.read(crop)
        if reader.accept(r):
            continue
        png = save_queue_item(out, i, crop, r.text, r.confidence, tuple(m["pick_xy"]), f"seed {m['seed']}")
        j = png.with_suffix(".json")
        d = json.loads(j.read_text())
        d["label"] = truth
        d["labeled_by"] = "simulated_human"      # シミュレーションなので正解を使った。実運用では tools/review_label.py で人が付ける
        j.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        n += 1
    return {"queued": n, "pool": len(labels), "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--seeds", type=int, nargs=2, required=True, help="開始と終了(終了は含まない)")
    c.add_argument("--n", type=int, default=12)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--weights", type=Path, required=True)
    s.add_argument("--data", type=Path, required=True)
    f = sub.add_parser("feedback")
    f.add_argument("--weights", type=Path, required=True)
    f.add_argument("--data", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(range(*a.seeds), a.out, a.n)
    elif a.cmd == "score":
        print(json.dumps(score(a.weights, a.data), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(feedback(a.weights, a.data, a.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
