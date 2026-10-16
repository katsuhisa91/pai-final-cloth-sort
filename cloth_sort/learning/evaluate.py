"""学習した方策と手続き的な把持を、学習に使っていないシードで比べる。

    python -m cloth_sort.learning.evaluate --policy runs/train/act/checkpoints/020000/pretrained_model \\
        --episodes 20 --out runs/eval/act --video

1 エピソード = 布 1〜3 枚の場面で 1 枚を目標へ運ぶ(収集時と同じ分布、シードは別)。
成功 = 布の重心が目標の ±50 mm に載って机の上にあること。結果は JSON と(--video なら)動画に出す。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cloth_sort.learning.env import DT, MAX_EPISODE_S, PickPlaceEnv, run_expert
from cloth_sort.learning.policy_arm import ACTController, LearnedArm
from cloth_sort.robot.executor import Phase
from cloth_sort.pipeline import make_video
from cloth_sort.viewer import POLICY_HAND_NOTES, Overlay, render_frame

EVAL_SEED_BASE = 20_000
EVAL_REVIEW_SHARE = 0.15      # 評価の場面は最初の分布(要確認トレイ行き 15%)で固定し、学習データの配分を変えても比較できるようにする


class _Frames:
    def __init__(self, out: Path | None, every: int = 4) -> None:
        self.dir = out
        self.every = every
        self.n = 0
        self.tick = 0
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)

    def maybe(self, world, ov: Overlay) -> None:
        self.tick += 1
        if self.dir is None or self.tick % self.every:
            return
        render_frame(world, ov).save(self.dir / f"{self.n:05d}.png")
        self.n += 1


def run_policy_episode(env: PickPlaceEnv, ctrl: ACTController, frames: _Frames, ov: Overlay, retries: int = 0) -> dict:
    """パイプラインと同じ実行器(PolicyPickPlace)で 1 枚運ぶ。retries > 0 なら、つかみ損ねの撮り直し再試行も含めて評価する。"""
    t0 = time.time()
    world = env.world
    if getattr(world, "background", None) is None:
        world.background = world.render_camera(include_pieces=False)     # 再試行の撮り直し(背景差分)に使う
    arm = LearnedArm(world, ctrl)
    spec = env.spec
    pp = arm.pick_place(spec.pick_xy, spec.place_xy, spec.place_name)
    pp.max_grip_retries = retries
    steps = 0
    limit = MAX_EPISODE_S * (retries + 1) + 10.0
    while not pp.finished and world.time_s <= limit:
        pp.step(DT)
        world.time_s += DT
        steps += 1
        frames.maybe(world, ov)
    env.time_s = world.time_s
    x, y, z = env.cloth_centroid_mm()
    return {"success": pp.phase is Phase.DONE and bool(env.placed()), "steps": steps, "sim_seconds": round(world.time_s, 1),
            "closed": pp.closed_once, "timed_out": pp.timed_out or world.time_s > limit, "grip_retries": pp.grip_retries,
            "cloth_xy": [round(x), round(y)], "wall_seconds": round(time.time() - t0)}


def run_expert_episode(env: PickPlaceEnv, frames: _Frames, ov: Overlay) -> dict:
    t0 = time.time()
    tr = run_expert(env, record=False, on_step=lambda: frames.maybe(env.world, ov))
    x, y, _ = env.cloth_centroid_mm()
    return {"success": tr.success, "steps": len(tr.phases), "sim_seconds": round(tr.sim_seconds, 1),
            "grip_retries": tr.grip_retries, "final_phase": tr.final_phase, "cloth_xy": [round(x), round(y)],
            "wall_seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("runs/eval/act"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--skip-expert", action="store_true")
    ap.add_argument("--skip-policy", action="store_true", help="教師だけを回す(--policy は読み込まない)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ensemble", type=float, default=0.01,
                    help="時間方向の平均の係数(ACT の既定 0.01)。0 以下なら平均せず 100 ステップぶんをそのまま実行する")
    ap.add_argument("--start", type=int, default=0, help="評価シードの開始番号(並列に分けて回すとき用)")
    ap.add_argument("--retries", type=int, default=0, help="方策のつかみ損ねを撮り直して再試行する回数(パイプラインは 2)")
    ap.add_argument("--summarize", action="store_true", help="--out にある results_*.jsonl をまとめる")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        print(json.dumps(summarize(a.out, str(a.policy)), ensure_ascii=False, indent=2))
        return
    ctrl = None if a.skip_policy else ACTController(a.policy, device=a.device, ensemble_coeff=a.ensemble if a.ensemble > 0 else None)
    env = PickPlaceEnv()
    results = []
    whos = [w for w, skip in (("policy", a.skip_policy), ("expert", a.skip_expert)) if not skip]
    for i in range(a.start, a.start + a.episodes):
        seed = EVAL_SEED_BASE + i
        row = {"seed": seed}
        for who in whos:
            spec = env.reset(seed, review_share=EVAL_REVIEW_SHARE)
            env.world.time_s = 0.0
            ov = Overlay(total=1, speed_note=f"評価 seed {seed}  " + ("学習方策(ACT)" if who == "policy" else "手続き的な把持"),
                         hand_notes=POLICY_HAND_NOTES if who == "policy" else None)
            ov.status = f"{who}: 布 {spec.n_pieces} 枚、目標 {spec.place_name}"
            ov.current_dest = f"置き場 {spec.place_name}" if spec.place_name != "review" else "要確認トレイ"
            ov.phase = "学習方策(ACT)で実行中" if who == "policy" else "手続き(状態機械)"
            frames = _Frames(a.out / "frames" / f"{seed}_{who}" if a.video else None)
            row[who] = run_policy_episode(env, ctrl, frames, ov, a.retries) if who == "policy" else run_expert_episode(env, frames, ov)
            row["n_pieces"] = spec.n_pieces
            row["place"] = spec.place_name
            if a.video and frames.n:
                make_video(frames.dir, a.out / f"{seed}_{who}.mp4", 1.0 / (DT * frames.every))
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (a.out / f"results_{a.start:04d}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    print(json.dumps(summarize(a.out, str(a.policy)), ensure_ascii=False, indent=2))


def summarize(out: Path, policy: str) -> dict:
    rows = []
    for f in sorted(out.glob("results_*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    with_expert = [r for r in rows if "expert" in r]
    with_policy = [r for r in rows if "policy" in r]
    summary = {
        "episodes": len(rows),
        "policy_episodes": len(with_policy),
        "policy_success": sum(r["policy"]["success"] for r in with_policy),
        "policy_timed_out": sum(r["policy"]["timed_out"] for r in with_policy),
        "policy_retries": sum(r["policy"].get("grip_retries", 0) for r in with_policy),
        "policy_mean_sim_s": round(float(np.mean([r["policy"]["sim_seconds"] for r in with_policy])), 1) if with_policy else None,
        "expert_episodes": len(with_expert),
        "expert_success": sum(r["expert"]["success"] for r in with_expert),
        "both_success": sum(r["policy"]["success"] and r["expert"]["success"] for r in with_expert if "policy" in r),
        "policy": policy,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
