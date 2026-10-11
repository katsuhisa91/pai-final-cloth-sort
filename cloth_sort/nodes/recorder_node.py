"""記録ノード: 読み取り結果とタスク結果を JSONL に残す(データセット化・振り返り用)。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dora import Node

from cloth_sort.nodes.common import arrow_to_json

OUT = Path(os.environ.get("RECORD_PATH", "runs/dora/log.jsonl"))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    node = Node()
    closed: set[str] = set()
    with OUT.open("w") as f:
        for event in node:
            if event["type"] == "INPUT":
                rec = {"event": event["id"], **arrow_to_json(event["value"])}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            elif event["type"] == "INPUT_CLOSED":
                closed.add(event["id"])
                if closed == {"reading", "record"}:
                    break
            elif event["type"] == "STOP":
                break
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
