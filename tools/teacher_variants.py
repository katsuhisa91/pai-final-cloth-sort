"""教師(手続き的な把持)の上限を上げるための比較実験。描画なし、布 1〜3 枚の場面で 1 枚運ぶ。

    python tools/teacher_variants.py --variant base --seeds 30000 30001 ... --out runs/teacher/base.jsonl

variant:
  base      いまの教師
  contacts  持ち上げ直後の接触数のしきい値を 6 → 10(浅い把持を早めに握り直す)
  slow      搬送 5 → 4 cm/s、加速度 0.05 → 0.03 m/s²(揺らさない)
  both      contacts + slow
  shake     持ち上げ後に 0.5 秒止まって接触を再確認(抜けかけを握り直す)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cloth_sort.learning.env import PickPlaceEnv, run_expert
from cloth_sort.sim import mujoco_cloth_world as mcw


def apply_variant(name: str) -> None:
    if name in ("contacts", "both"):
        mcw.GripperArm.MIN_HOLD_CONTACTS = 10
    if name in ("slow", "both"):
        def _speeds(self):
            if self.holding_id is not None:
                return 0.04, 0.03, 0.03
            return 0.6, 0.5, 3.0
        mcw.GripperArm._speeds = _speeds
    if name == "shake":
        orig = mcw.GripperArm.holding

        def holding(self, strict=False):
            if strict:
                for _ in range(10):                   # 0.5 秒待ってから判定(抜けかけは落ちる)
                    self._servo(0.05)
            return orig(self, strict)
        mcw.GripperArm.holding = holding


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    apply_variant(a.variant)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv()
    with a.out.open("a") as f:
        for seed in a.seeds:
            t0 = time.time()
            spec = env.reset(seed)
            tr = run_expert(env, record=False)
            row = {"variant": a.variant, "seed": seed, "n_pieces": spec.n_pieces, "place": spec.place_name,
                   "success": tr.success, "retries": tr.grip_retries, "final": tr.final_phase,
                   "sim_s": round(tr.sim_seconds, 1), "wall_s": round(time.time() - t0)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
