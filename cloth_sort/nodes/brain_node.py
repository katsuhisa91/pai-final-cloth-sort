"""AI 側ノード: 検出・番号読み取り・計画をまとめて担当する。

起動時に背景撮影と本撮影を要求し、読み取り結果からタスク列を作って 1 枚ずつ sim ノードへ送る。
sim ノードから result が返るたびに次のタスクを送る(逐次実行)。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dora import Node

from cloth_sort import config
from cloth_sort.nodes.common import arrow_to_image, arrow_to_json, json_to_arrow
from cloth_sort.ml.feedback import save_queue_item
from cloth_sort.perception.detect import detect_pieces
from cloth_sort.perception.reader import LabelReader
from cloth_sort.planning.planner import Destination, Observation, Planner

MIN_CONF = float(os.environ.get("MIN_CONFIDENCE", "0.6"))
SETTLE_S = float(os.environ.get("SETTLE_SECONDS", "1.5"))
REVIEW_QUEUE = Path(os.environ.get("REVIEW_QUEUE", "runs/dora/review_queue"))


def main() -> None:
    reader = LabelReader(min_confidence=MIN_CONF)
    planner = Planner(MIN_CONF)
    node = Node()
    background = None
    tasks: list = []
    detections_for_overlay: list = []
    observations: list[Observation] = []
    index = 0
    started = time.time()
    requested = False

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "tick":
                # sim ノードの準備を待ってから撮影要求を出す
                if not requested and time.time() - started > SETTLE_S:
                    node.send_output("capture_request", json_to_arrow({"background": True}))
                    requested = True
            elif event["id"] == "background":
                background = arrow_to_image(event["value"], event["metadata"])
                node.send_output("capture_request", json_to_arrow({"background": False}))
            elif event["id"] == "image":
                image = arrow_to_image(event["value"], event["metadata"])
                dets = detect_pieces(image, background, region=config.SOURCE_AREA)
                observations = []
                for k, d in enumerate(dets):
                    r = reader.read(d.crop)
                    observations.append(Observation(d.center, r.text, r.confidence))
                    if not reader.accept(r):
                        # 要確認トレイへ回る布の切り出しを残す(人が番号を付けて再学習に使う)
                        save_queue_item(REVIEW_QUEUE, k, d.crop, r.text, r.confidence, d.center, "dora")
                    detections_for_overlay.append([list(d.bbox), r.text or "?", r.confidence])
                    node.send_output("reading", json_to_arrow({"pick_xy": d.center, "read": r.text, "conf": r.confidence}))
                tasks = planner.plan(observations, (config.ARM_BASE[0], config.ARM_BASE[1] - 40))
                index = 0
                _send_next(node, tasks, index, observations, detections_for_overlay)
            elif event["id"] == "result":
                res = arrow_to_json(event["value"])
                planner.mark_done(tasks[index])
                node.send_output("record", json_to_arrow(res))
                detections_for_overlay = [d for d in detections_for_overlay
                                          if abs((d[0][0] + d[0][2]) / 2 - res["pick_xy"][0]) > 5]
                index += 1
                if index < len(tasks):
                    _send_next(node, tasks, index, observations, detections_for_overlay)
                else:
                    n_review = sum(t.destination is Destination.REVIEW for t in tasks)
                    node.send_output("finish", json_to_arrow({
                        "message": f"置き場へ {len(tasks) - n_review} 枚、要確認 {n_review} 枚"}))
                    break
        elif event["type"] == "STOP":
            break


def _send_next(node: Node, tasks: list, index: int, observations: list[Observation], overlay: list) -> None:
    t = tasks[index]
    conf = next((o.confidence for o in observations if o.pick_xy == t.pick_xy), 0.0)
    node.send_output("task", json_to_arrow({
        "index": index + 1, "total": len(tasks), "pick_xy": list(t.pick_xy), "place_xy": list(t.place_xy),
        "label": t.label, "confidence": conf, "dest": t.destination.value, "reason": t.reason,
        "detections": overlay,
    }))


if __name__ == "__main__":
    main()
