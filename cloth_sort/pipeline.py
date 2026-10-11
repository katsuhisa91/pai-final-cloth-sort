"""単一プロセスで「撮影 → 検出 → 読み取り → 計画 → 実行 → 検証」を回し、動画フレームとログを出す。

dora 版(cloth_sort/nodes)と同じ部品を使う。こちらは決定的に動くので、テストと動画作成に使う。
使い方:  python -m cloth_sort.pipeline --seed 1 --n 12 --out runs/seed1 --video
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from cloth_sort import config
from cloth_sort.perception.detect import detect_pieces
from cloth_sort.perception.reader import LabelReader
from cloth_sort.planning.planner import Destination, Observation, Planner, Task
from cloth_sort.robot.executor import Phase, PickPlace, VirtualArm
from cloth_sort.sim.world import World
from cloth_sort.viewer import Overlay, render_frame

PHASE_JA = {
    Phase.APPROACH: "布片の上へ移動", Phase.DESCEND: "降下", Phase.GRIP: "把持", Phase.LIFT: "持ち上げ",
    Phase.CARRY: "置き場へ搬送", Phase.LOWER: "降下", Phase.RELEASE: "リリース", Phase.RETREAT: "退避",
    Phase.DONE: "完了", Phase.FAILED: "失敗",
}
HOME = (config.ARM_BASE[0], config.ARM_BASE[1] - 40)


class FrameWriter:
    def __init__(self, out_dir: Path | None, every: int) -> None:
        self.dir = out_dir
        self.every = every
        self.count = 0
        self.tick = 0
        if out_dir is not None:
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

    def maybe(self, world: World, ov: Overlay, force: bool = False) -> None:
        self.tick += 1
        if self.dir is None or (not force and self.tick % self.every != 0):
            return
        render_frame(world, ov).save(self.dir / f"{self.count:05d}.png")
        self.count += 1

    def hold(self, world: World, ov: Overlay, seconds: float, fps: float) -> None:
        for _ in range(int(seconds * fps)):
            self.maybe(world, ov, force=True)


def run(seed: int, n_pieces: int, out: Path | None, weights: Path | None = None, dt: float = 0.05,
        frame_every: int = 2, min_confidence: float = 0.6) -> dict:
    world = World.random_scene(n_pieces, seed)
    reader = LabelReader(weights) if weights else LabelReader()
    reader.min_confidence = min_confidence
    planner = Planner(min_confidence)
    arm = VirtualArm(world)
    ov = Overlay(total=n_pieces, speed_note="実時間 / real-time")
    fps = 1.0 / (dt * frame_every)
    frames = FrameWriter(out / "frames" if out else None, frame_every)
    records: list[dict] = []
    truth = {p.id: p.label for p in world.pieces}

    ov.status = "開始"
    ov.log.append(f"布片 {n_pieces} 枚、置き場 1〜{len(config.SLOT_LABELS)}")
    frames.hold(world, ov, 1.5, fps)

    # 0. 段取り: 何も置いていない机を撮っておく(背景差分の基準)
    background = world.render_camera(include_pieces=False)

    # 1. 撮影 → 検出 → 読み取り
    ov.status = "撮影・番号読み取り"
    t0 = time.time()
    image = world.render_camera()
    dets = detect_pieces(image, background, region=config.SOURCE_AREA)
    observations: list[Observation] = []
    for d in dets:
        r = reader.read(d.crop)
        observations.append(Observation(d.center, r.text, r.confidence))
        ov.detections.append((d.bbox, r.text or "?", r.confidence))
        gt = world.piece_at(d.center)
        records.append({"event": "read", "pick_xy": d.center, "read": r.text, "conf": round(r.confidence, 3),
                        "truth": gt.label if gt else None})
    ov.log.append(f"検出 {len(dets)} 枚、読み取り {time.time() - t0:.2f} s")
    frames.hold(world, ov, 2.0, fps)

    # 2. 計画
    tasks = planner.plan(observations, world.gripper_xy)
    n_review = sum(t.destination is Destination.REVIEW for t in tasks)
    ov.log.append(f"計画: 置き場へ {len(tasks) - n_review} 枚、要確認へ {n_review} 枚")
    frames.hold(world, ov, 1.0, fps)

    # 3. 実行(1 枚ずつ)→ 4. 置いた直後にカメラで検証
    for i, task in enumerate(tasks, 1):
        ov.current_label = task.label or "?"
        ov.current_conf = next((o.confidence for o in observations if o.pick_xy == task.pick_xy), 0.0)
        ov.current_dest = f"置き場 {task.label}" if task.destination is Destination.SLOT else "要確認トレイ"
        ov.reason = task.reason
        ov.status = f"{i}/{len(tasks)} 枚目を搬送中"
        place_name = task.label if task.destination is Destination.SLOT else "review"
        pp = PickPlace(arm, task.pick_xy, task.place_xy, place_name)
        while not pp.finished:
            phase = pp.step(dt)
            world.time_s += dt
            ov.phase = PHASE_JA[phase]
            frames.maybe(world, ov)
        outcome = _verify(world, reader, task, truth)
        records.append({"event": "place", "label": task.label, "dest": task.destination.value,
                        "reason": task.reason, "phase": pp.phase.value, **outcome})
        planner.mark_done(task)
        ov.detections = [dv for dv in ov.detections if _far(dv[0], task.pick_xy)]
        if task.destination is Destination.REVIEW:
            ov.review += 1
            ov.log.append(f"#{i} → 要確認 ({task.reason})")
        else:
            ov.done += 1
            mark = "OK" if outcome["verified"] else "NG"
            ov.log.append(f"#{i} 番号 {task.label} → 置き場 {task.label}  検証 {mark}")

    ov.status = "完了"
    ov.phase = "-"
    ov.current_label, ov.current_conf, ov.current_dest, ov.reason = "", 0.0, "", ""
    correct = sum(1 for r in records if r["event"] == "place" and r.get("verified"))
    ov.log.append(f"置き場へ {ov.done} 枚(検証 OK {correct})、要確認 {ov.review} 枚、所要 {world.time_s:.0f} s")
    # 退避
    while not arm.move_to(HOME, 1.0, dt):
        world.time_s += dt
        frames.maybe(world, ov)
    frames.hold(world, ov, 3.0, fps)

    summary = _summary(records, world, tasks)
    if out:
        (out / "log.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _far(bbox: tuple[int, int, int, int], xy: tuple[float, float]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return abs(cx - xy[0]) > 5 or abs(cy - xy[1]) > 5


def _verify(world: World, reader: LabelReader, task: Task, truth: dict[int, str]) -> dict:
    """置いた位置を撮り直して番号を読み、計画どおりかを確かめる(自己検証)。"""
    piece = world.piece_at(task.place_xy)
    if piece is None:
        return {"verified": False, "truth": None, "reread": ""}
    image = world.render_camera()
    half = config.PIECE_MM
    x0, y0 = int(task.place_xy[0] - half), int(task.place_xy[1] - half)
    crop = image[max(0, y0):y0 + 2 * half, max(0, x0):x0 + 2 * half]
    r = reader.read(crop)
    if task.destination is Destination.SLOT:
        verified = r.text == task.label and truth[piece.id] == task.label
    else:
        verified = True
    return {"verified": bool(verified), "truth": truth[piece.id], "reread": r.text, "reread_conf": round(r.confidence, 3)}


def _summary(records: list[dict], world: World, tasks: list[Task]) -> dict:
    reads = [r for r in records if r["event"] == "read"]
    places = [r for r in records if r["event"] == "place"]
    return {
        "pieces": len(world.pieces),
        "detected": len(reads),
        "read_correct": sum(r["read"] == r["truth"] for r in reads),
        "placed_slot": sum(p["dest"] == "slot" for p in places),
        "placed_slot_verified": sum(p["dest"] == "slot" and p["verified"] for p in places),
        "review": sum(p["dest"] == "review" for p in places),
        "wrong_slot": sum(p["dest"] == "slot" and p["truth"] != p["label"] for p in places),
        "sim_seconds": round(world.time_s, 1),
    }


def make_video(frames_dir: Path, out_mp4: Path, fps: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_mp4),
    ], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("runs/seed1"))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--frame-every", type=int, default=2)
    a = ap.parse_args()
    s = run(a.seed, a.n, a.out, a.weights, a.dt, a.frame_every)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if a.video:
        make_video(a.out / "frames", a.out / "cloth_sort.mp4", 1.0 / (a.dt * a.frame_every))
        print("video:", a.out / "cloth_sort.mp4")


if __name__ == "__main__":
    main()
