"""手続き的な把持を教師にデモを集め、LeRobot のデータセット(v3)にする。

1. 収集(並列。各ワーカーが別プロセスで物理を回し、エピソードごとに npz を書く)
     python -m cloth_sort.learning.collect --episodes 240 --workers 8 --staging data/staging
2. 詰め替え(成功したエピソードだけを LeRobotDataset にする)
     python -m cloth_sort.learning.collect --pack data/staging --out data/cloth_pickplace

成功 = 布を目標の ±50 mm に置けたもの(つかみ直して成功したものも含む)。失敗(つかめず・搬送中に落下)は学習に使わない。
MuJoCo の描画(CGL)を使うので、ウィンドウサーバに繋げる環境で実行する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cloth_sort.learning.env import (ACTION_NAMES, FPS, IMG_H, IMG_W, STATE_NAMES, STATE_NAMES_RELATIVE, TASK,
                                     PickPlaceEnv, Trajectory, run_expert, with_relative_goal)

TRAIN_SEED_BASE = 10_000


def collect_shard(staging: Path, episodes: int, shard: int, n_shards: int, noise_m: float = 0.0,
                  seed_base: int = TRAIN_SEED_BASE) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv()
    for i in range(shard, episodes, n_shards):
        seed = seed_base + i
        path = staging / f"ep_{seed:06d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        env.reset(seed)
        tr = run_expert(env, noise_m=noise_m, noise_seed=seed)
        save_trajectory(path, tr)
        print(f"[shard {shard}] seed {seed} pieces={tr.spec.n_pieces} place={tr.spec.place_name} "
              f"steps={len(tr)} success={tr.success} retries={tr.grip_retries} final={tr.final_phase} "
              f"wall={time.time() - t0:.0f}s", flush=True)


def save_trajectory(path: Path, tr: Trajectory) -> None:
    meta = {"seed": tr.spec.seed, "n_pieces": tr.spec.n_pieces, "place_name": tr.spec.place_name,
            "pick_xy": list(tr.spec.pick_xy), "place_xy": list(tr.spec.place_xy), "success": tr.success,
            "grip_retries": tr.grip_retries, "final_phase": tr.final_phase, "sim_seconds": tr.sim_seconds}
    np.savez_compressed(path, top=np.stack(tr.top), wrist=np.stack(tr.wrist), state=np.stack(tr.state),
                        action=np.stack(tr.action), phases=np.array(tr.phases), meta=json.dumps(meta))


def pack(staging: Path, out: Path, repo_id: str, state_mode: str = "basic") -> dict:
    """成功したエピソードを LeRobotDataset に詰める(lerobot が必要)。state_mode="relative" で目標との差を足す。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names = STATE_NAMES_RELATIVE if state_mode == "relative" else STATE_NAMES
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.images.top": {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    if out.exists():
        raise SystemExit(f"{out} は既にあります。消すか別の --out を指定してください")
    ds = LeRobotDataset.create(repo_id, fps=FPS, features=features, root=out, robot_type="panda_sim_2finger",
                               use_videos=True, streaming_encoding=True, encoder_queue_maxsize=2000)   # PNG を経由せず動画に直接書く。待ち行列は 1 エピソードぶん以上(溢れるとフレームが落ちる)
    stats = {"files": 0, "success": 0, "frames": 0, "with_retry": 0, "failed": 0}
    for path in sorted(staging.glob("ep_*.npz")):
        if time.time() - path.stat().st_mtime < 120:
            stats["in_progress_skipped"] = stats.get("in_progress_skipped", 0) + 1   # 収集中のワーカーが書いている途中
            continue
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        stats["files"] += 1
        if not meta["success"]:
            stats["failed"] += 1
            continue
        if meta["grip_retries"] > 0:
            stats["with_retry"] += 1          # つかみ直して成功したものも教師にする(立て直し方の例になる)
        top, wrist, state, action = z["top"], z["wrist"], z["state"], z["action"]   # npz は添字ごとに解凍し直すので先に取り出す
        if state_mode == "relative":
            state = with_relative_goal(state)
        n = len(action)
        for t in range(n):
            ds.add_frame({"observation.state": state[t], "observation.images.top": top[t],
                          "observation.images.wrist": wrist[t], "action": action[t], "task": TASK})
        ds.save_episode(parallel_encoding=False)
        stats["success"] += 1
        stats["frames"] += n
    ds.finalize()
    (out / "collect_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, default=Path("data/staging"))
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=None, help="内部用: ワーカーの番号")
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--pack", type=Path, default=None, help="staging を LeRobotDataset に詰める")
    ap.add_argument("--out", type=Path, default=Path("data/cloth_pickplace"))
    ap.add_argument("--repo-id", default="local/cloth_pickplace")
    ap.add_argument("--state", choices=["basic", "relative"], default="basic", help="relative: 目標と手先の差を状態に足す")
    ap.add_argument("--noise", type=float, default=0.0, help="DART 方式の外乱(手先目標に足す標準偏差 [m]、例 0.01)")
    ap.add_argument("--seed-base", type=int, default=TRAIN_SEED_BASE, help="収集シードの起点(外乱ありの収集は別の起点にする)")
    a = ap.parse_args()
    if a.pack is not None:
        print(json.dumps(pack(a.pack, a.out, a.repo_id, a.state), indent=2))
        return
    if a.shard is not None:
        collect_shard(a.staging, a.episodes, a.shard, a.n_shards, a.noise, a.seed_base)
        return
    procs = [subprocess.Popen([sys.executable, "-m", "cloth_sort.learning.collect", "--staging", str(a.staging),
                               "--episodes", str(a.episodes), "--shard", str(k), "--n-shards", str(a.workers),
                               "--noise", str(a.noise), "--seed-base", str(a.seed_base)])
             for k in range(a.workers)]
    codes = [p.wait() for p in procs]
    files = sorted(a.staging.glob("ep_*.npz"))
    ok = sum(json.loads(str(np.load(f)["meta"]))["success"] for f in files)
    print(f"workers exit={codes}  episodes={len(files)}  success={ok}")


if __name__ == "__main__":
    main()
