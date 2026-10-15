"""LeRobot の ACT を既定の設定で学習する(lerobot-train の薄い包み。設定をコードに残すため)。

    python -m cloth_sort.learning.train --dataset data/cloth_pickplace --out runs/train/act --steps 20000

学習済みモデルは runs/train/act/checkpoints/<step>/pretrained_model に出る(config.json + model.safetensors +
前処理の定義)。評価・実行はそのディレクトリを --policy に渡す。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def train_args(dataset: Path, out: Path, steps: int, batch_size: int, device: str, chunk: int, lr: float,
               save_freq: int, num_workers: int, seed: int) -> list[str]:
    return [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--dataset.repo_id=local/cloth_pickplace",
        f"--dataset.root={dataset}",
        "--policy.type=act",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={chunk}",
        f"--policy.n_action_steps={chunk}",
        f"--policy.optimizer_lr={lr}",
        f"--policy.optimizer_lr_backbone={lr}",
        f"--output_dir={out}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq}",
        "--log_freq=100",
        f"--num_workers={num_workers}",
        f"--seed={seed}",
        "--wandb.enable=false",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/cloth_pickplace"))
    ap.add_argument("--out", type=Path, default=Path("runs/train/act"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    cmd = train_args(a.dataset.resolve(), a.out.resolve(), a.steps, a.batch_size, a.device, a.chunk, a.lr,
                     a.save_freq, a.num_workers, a.seed)
    print(" ".join(cmd), flush=True)
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")        # ローカルのデータセットなので Hub に問い合わせない
    raise SystemExit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
