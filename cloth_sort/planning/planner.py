"""読み取り結果から「どの布片を、どこへ、どの順で」運ぶかを決める。

方針:
- 番号が読めて置き場が空いていれば、その番号の置き場へ
- 読めない・該当する置き場がない・置き場が埋まっている布片は要確認トレイへ(止めずに逃がす)
- 作業順は「現在のグリッパ位置から近い順」。移動距離の最小化は総当たりせず貪欲法
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cloth_sort import config


class Destination(str, Enum):
    SLOT = "slot"
    REVIEW = "review"


@dataclass(frozen=True)
class Observation:
    pick_xy: tuple[float, float]
    label: str            # 読み取った文字列。読めないときは ""
    confidence: float


@dataclass(frozen=True)
class Task:
    pick_xy: tuple[float, float]
    place_xy: tuple[float, float]
    label: str
    destination: Destination
    reason: str


class Planner:
    def __init__(self, min_confidence: float = 0.6) -> None:
        self.slots = {s.label: s.center for s in config.board_slots()}
        self.filled: set[str] = set()
        self.review_count = 0
        self.min_confidence = min_confidence

    def plan(self, observations: list[Observation], gripper_xy: tuple[float, float]) -> list[Task]:
        """観測をタスク列に変換する。呼ぶたびに置き場の埋まり状況を更新する。"""
        tasks: list[Task] = []
        reserved = set(self.filled)
        for o in observations:
            if not o.label:
                tasks.append(self._review(o, "番号を読み取れない"))
            elif o.confidence < self.min_confidence:
                tasks.append(self._review(o, f"確信度が低い({o.confidence:.2f})"))
            elif o.label not in self.slots:
                tasks.append(self._review(o, f"置き場に無い番号({o.label})"))
            elif o.label in reserved:
                tasks.append(self._review(o, f"番号 {o.label} の置き場は使用済み(重複)"))
            else:
                reserved.add(o.label)
                tasks.append(Task(o.pick_xy, self.slots[o.label], o.label, Destination.SLOT, "番号どおりの置き場へ"))
        return _nearest_first(tasks, gripper_xy)

    def _review(self, o: Observation, reason: str) -> Task:
        xy = config.review_tray_position(self.review_count)
        self.review_count += 1
        return Task(o.pick_xy, xy, o.label, Destination.REVIEW, reason)

    def mark_done(self, task: Task) -> None:
        if task.destination is Destination.SLOT:
            self.filled.add(task.label)


def _nearest_first(tasks: list[Task], start: tuple[float, float]) -> list[Task]:
    remaining = list(tasks)
    ordered: list[Task] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda t: (t.pick_xy[0] - cur[0]) ** 2 + (t.pick_xy[1] - cur[1]) ** 2)
        remaining.remove(nxt)
        ordered.append(nxt)
        cur = nxt.place_xy
    return ordered
