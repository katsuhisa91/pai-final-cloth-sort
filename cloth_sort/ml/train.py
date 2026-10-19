"""番号読み取り CNN の学習・評価(MLOps の最小ループ)。

データ生成 → 学習 → 「学習で見ていない筆跡(held-out フォント)」で評価 → 重みと指標を保存。
使い方:  python -m cloth_sort.ml.train --out models
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cloth_sort.perception.model import CharNet
from cloth_sort.sim import fonts
from cloth_sort.sim.synth import CHAR_CLASSES, build_dataset


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        pred = logits.argmax(dim=-1).numpy()
    acc = float((pred == y).mean())
    cm = np.zeros((len(CHAR_CLASSES), len(CHAR_CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per_class = {CHAR_CLASSES[i]: float(cm[i, i] / max(1, cm[i].sum())) for i in range(len(CHAR_CLASSES))}
    return {"accuracy": acc, "per_class": per_class, "confusion": cm.tolist()}


def train(out_dir: Path, n_per_class: int, epochs: int, seed: int, recipe: str = "v1",
          feedback: list[Path] | None = None, feedback_copies: int = 40) -> dict:
    """recipe="v1": 元の学習(フォント 8 種)。"v2": フォント 16 種 + 手書き風の変形 + 紛らわしい数字を 2 倍。
    feedback: 要確認トレイで人が番号を付けた切り出し(ml/feedback.py)を足す。"""
    torch.manual_seed(seed)
    v2 = recipe == "v2"
    train_fonts = fonts.available(fonts.TRAIN_FONTS + (fonts.EXTRA_TRAIN_FONTS if v2 else []))
    heldout_fonts = fonts.available(fonts.HELDOUT_FONTS)
    if not train_fonts or not heldout_fonts:
        raise RuntimeError("フォントが見つかりません。cloth_sort/sim/fonts.py のパスを確認してください")

    t0 = time.time()
    x_tr, y_tr = build_dataset(n_per_class, train_fonts, seed, handwriting=v2, confusable_factor=2.0 if v2 else 1.0)
    fb_stats = None
    if feedback:
        from cloth_sort.ml.feedback import feedback_dataset
        x_fb, y_fb, fb_stats = feedback_dataset(feedback, feedback_copies, seed + 3)
        print(f"feedback: {fb_stats} -> {len(y_fb)} samples", flush=True)
        if len(y_fb):
            x_tr, y_tr = np.concatenate([x_tr, x_fb]), np.concatenate([y_tr, y_fb])
    x_val, y_val = build_dataset(max(40, n_per_class // 8), train_fonts, seed + 1)
    x_ho, y_ho = build_dataset(max(40, n_per_class // 8), heldout_fonts, seed + 2)
    gen_s = time.time() - t0

    model = CharNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    bs = 128
    history = []
    t1 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        total = 0.0
        for i in range(0, len(yt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        val = evaluate(model, x_val, y_val)["accuracy"]
        ho = evaluate(model, x_ho, y_ho)["accuracy"]
        history.append({"epoch": ep + 1, "loss": total / len(yt), "val_acc": val, "heldout_acc": ho})
        print(f"epoch {ep + 1}/{epochs} loss={total / len(yt):.4f} val={val:.3f} heldout={ho:.3f}", flush=True)
    train_s = time.time() - t1

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "charnet.pt")
    metrics = {
        "classes": CHAR_CLASSES,
        "n_train": int(len(y_tr)),
        "train_fonts": [p.rsplit("/", 1)[-1] for p, _ in train_fonts],
        "heldout_fonts": [p.rsplit("/", 1)[-1] for p, _ in heldout_fonts],
        "val": evaluate(model, x_val, y_val),
        "heldout": evaluate(model, x_ho, y_ho),
        "history": history,
        "data_gen_seconds": gen_s,
        "train_seconds": train_s,
        "params": sum(p.numel() for p in model.parameters()),
        "recipe": recipe,
        "feedback": {"dirs": [str(d) for d in (feedback or [])], "stats": fb_stats, "copies": feedback_copies},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--n-per-class", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recipe", choices=["v1", "v2"], default="v1",
                    help="v2: フォント 16 種 + 手書き風の変形 + 紛らわしい数字(1,2,3,5,6,7)を 2 倍")
    ap.add_argument("--feedback", type=Path, nargs="*", default=None,
                    help="要確認トレイで人が番号を付けた review_queue ディレクトリ(複数可)")
    ap.add_argument("--feedback-copies", type=int, default=40, help="切り出し 1 文字あたりの揺らしたサンプル数")
    a = ap.parse_args()
    m = train(a.out, a.n_per_class, a.epochs, a.seed, a.recipe, a.feedback, a.feedback_copies)
    print(f"val acc={m['val']['accuracy']:.3f}  heldout acc={m['heldout']['accuracy']:.3f}  params={m['params']}")


if __name__ == "__main__":
    main()
