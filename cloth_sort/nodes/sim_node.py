"""ロボット側ノード: 作業台のシミュレーションと仮想アーム。

入力  capture_request : 撮影要求(段取り時の背景撮影も含む)
入力  task            : 1 枚ぶんの pick & place 指示(JSON)
入力  tick            : 制御周期(dora のタイマー)
出力  image           : 俯瞰カメラ画像
出力  background      : 何も置いていない机の画像
出力  result          : タスクの完了報告(JSON)
出力  frame           : 動画用フレーム(JSON の状態 + 画像)

実機に置き換えるときは、このノードだけを差し替える(上位は変わらない)。
"""

from __future__ import annotations

import os
from pathlib import Path

from dora import Node

from cloth_sort import config
from cloth_sort.nodes.common import image_to_arrow, json_to_arrow, arrow_to_json
from cloth_sort.pipeline import HOME, PHASE_JA
from cloth_sort.robot.executor import PickPlace, VirtualArm
from cloth_sort.sim.world import World
from cloth_sort.viewer import Overlay, render_frame

SEED = int(os.environ.get("SEED", "1"))
N_PIECES = int(os.environ.get("N_PIECES", "12"))
DT = float(os.environ.get("DT", "0.05"))
FRAME_DIR = Path(os.environ.get("FRAME_DIR", "runs/dora/frames"))
FRAME_EVERY = int(os.environ.get("FRAME_EVERY", "2"))


def main() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    world = World.random_scene(N_PIECES, SEED)
    arm = VirtualArm(world)
    ov = Overlay(total=N_PIECES, speed_note="dora-rs 分散実行")
    node = Node()
    current: PickPlace | None = None
    current_task: dict | None = None
    frame_idx = 0
    tick = 0
    ticks_after_finish = -1          # finish を受けたら 3 秒ぶん描いてから終了する
    ov.status = "待機中(撮影要求待ち)"

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "capture_request":
                req = arrow_to_json(event["value"])
                if req.get("background"):
                    arr, meta = image_to_arrow(world.render_camera(include_pieces=False))
                    node.send_output("background", arr, metadata=meta)
                else:
                    arr, meta = image_to_arrow(world.render_camera())
                    node.send_output("image", arr, metadata=meta)
                ov.status = "撮影"
            elif event["id"] == "task":
                current_task = arrow_to_json(event["value"])
                ov.detections = [tuple(d) for d in current_task.get("detections", [])]
                ov.current_label = current_task["label"] or "?"
                ov.current_conf = current_task["confidence"]
                ov.current_dest = f"置き場 {current_task['label']}" if current_task["dest"] == "slot" else "要確認トレイ"
                ov.reason = current_task["reason"]
                ov.status = f"{current_task['index']}/{current_task['total']} 枚目を搬送中"
                place_name = current_task["label"] if current_task["dest"] == "slot" else "review"
                current = PickPlace(arm, tuple(current_task["pick_xy"]), tuple(current_task["place_xy"]), place_name)
            elif event["id"] == "tick":
                tick += 1
                if current is not None:
                    phase = current.step(DT)
                    world.time_s += DT
                    ov.phase = PHASE_JA[phase]
                    if current.finished:
                        piece = world.piece_at(current.place_xy)
                        node.send_output("result", json_to_arrow({
                            **current_task, "phase": phase.value,
                            "truth": piece.label if piece else None,
                        }))
                        if current_task["dest"] == "slot":
                            ov.done += 1
                        else:
                            ov.review += 1
                        ov.log.append(f"#{current_task['index']} 番号 {current_task['label'] or '?'} → {ov.current_dest}")
                        current = None
                elif ticks_after_finish >= 0:
                    arm.move_to(HOME, 1.0, DT)
                    world.time_s += DT
                    ticks_after_finish -= 1
                if tick % FRAME_EVERY == 0:
                    render_frame(world, ov).save(FRAME_DIR / f"{frame_idx:05d}.png")
                    frame_idx += 1
                if ticks_after_finish == 0:
                    break
            elif event["id"] == "finish":
                ov.status = "完了"
                ov.phase = "-"
                ov.current_label, ov.current_conf, ov.current_dest, ov.reason = "", 0.0, "", ""
                ov.log.append(arrow_to_json(event["value"])["message"])
                ticks_after_finish = int(3.0 / DT)
        elif event["type"] == "STOP":
            break
    print(f"wrote {frame_idx} frames to {FRAME_DIR}", flush=True)


if __name__ == "__main__":
    main()
