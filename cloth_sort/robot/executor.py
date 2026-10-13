"""アームの抽象インターフェースと、仮想アーム(2D 運動学)の実装。

実機(SO-101 / OpenArm など)に載せ替えるときは RobotArm を実装するクラスを 1 つ足す。
上位(計画・読み取り)は RobotArm しか知らない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from cloth_sort.sim.world import World


class RobotArm(Protocol):
    """上位から見たアーム。座標は作業台の mm、z は 0(机面)〜1(上空)。"""

    def move_to(self, xy: tuple[float, float], z: float, dt: float) -> bool:
        """目標へ dt 秒ぶん進める。到達したら True。"""

    def grip(self) -> bool:
        """閉じる。何かを掴めたら True。"""

    def release(self, placed_at: str | None) -> None:
        """開く。"""

    @property
    def pose(self) -> tuple[float, float, float]:
        ...


class Phase(str, Enum):
    APPROACH = "approach"     # 上空を布片の真上へ
    DESCEND = "descend"       # 降りる
    GRIP = "grip"             # 掴む
    LIFT = "lift"             # 持ち上げる
    CARRY = "carry"           # 置き場の真上へ
    LOWER = "lower"           # 降ろす
    RELEASE = "release"       # 放す
    RETREAT = "retreat"       # 上空へ戻る
    PARK = "park"             # カメラの視野外へ退避(退避位置を持つアームのみ)
    DONE = "done"
    FAILED = "failed"


@dataclass
class ArmLimits:
    xy_speed_mm_s: float = 350.0
    z_speed_per_s: float = 2.5
    dwell_s: float = 0.25     # 掴む・放す動作にかかる時間


class VirtualArm:
    """World を直接動かす仮想アーム。速度制限つきの直線補間で動く。"""

    def __init__(self, world: World, limits: ArmLimits | None = None) -> None:
        self.world = world
        self.limits = limits or ArmLimits()

    @property
    def pose(self) -> tuple[float, float, float]:
        return (*self.world.gripper_xy, self.world.gripper_z)

    def move_to(self, xy: tuple[float, float], z: float, dt: float) -> bool:
        gx, gy = self.world.gripper_xy
        gz = self.world.gripper_z
        dx, dy = xy[0] - gx, xy[1] - gy
        dist = math.hypot(dx, dy)
        step = self.limits.xy_speed_mm_s * dt
        if dist <= step:
            nx, ny = xy
        else:
            nx, ny = gx + dx / dist * step, gy + dy / dist * step
        dz = z - gz
        zstep = self.limits.z_speed_per_s * dt
        nz = z if abs(dz) <= zstep else gz + math.copysign(zstep, dz)
        self.world.move_gripper((nx, ny), nz)
        return (nx, ny) == tuple(xy) and nz == z

    def grip(self) -> bool:
        return self.world.close_gripper() is not None

    def release(self, placed_at: str | None) -> None:
        self.world.open_gripper(placed_at)


@dataclass
class PickPlace:
    """1 枚ぶんの pick & place を状態機械として進める。step() を dt ごとに呼ぶ。"""

    arm: RobotArm
    pick_xy: tuple[float, float]
    place_xy: tuple[float, float]
    place_name: str
    phase: Phase = Phase.APPROACH
    _dwell: float = 0.0
    grip_retries: int = 0
    max_grip_retries: int = 2

    def step(self, dt: float) -> Phase:
        a = self.arm
        if self.phase is Phase.APPROACH:
            if a.move_to(self.pick_xy, 1.0, dt):
                self.phase = Phase.DESCEND
        elif self.phase is Phase.DESCEND:
            if a.move_to(self.pick_xy, 0.0, dt):
                self.phase = Phase.GRIP
                self._dwell = 0.0
        elif self.phase is Phase.GRIP:
            self._dwell += dt
            if self._dwell >= self.arm_dwell():
                if a.grip():
                    self.phase = Phase.LIFT
                elif self.grip_retries < self.max_grip_retries:
                    self.grip_retries += 1
                    a.release(None)
                    self.phase = Phase.APPROACH
                else:
                    self.phase = Phase.FAILED
        elif self.phase is Phase.LIFT:
            if a.move_to(self.pick_xy, 1.0, dt):
                self.phase = Phase.CARRY
        elif self.phase is Phase.CARRY:
            if a.move_to(self.place_xy, 1.0, dt):
                self.phase = Phase.LOWER
        elif self.phase is Phase.LOWER:
            if a.move_to(self.place_xy, 0.0, dt):
                self.phase = Phase.RELEASE
                self._dwell = 0.0
        elif self.phase is Phase.RELEASE:
            self._dwell += dt
            if self._dwell >= self.arm_dwell():
                a.release(self.place_name)
                self.phase = Phase.RETREAT
        elif self.phase is Phase.RETREAT:
            if a.move_to(self.place_xy, 1.0, dt):
                # 退避位置を持つアーム(MuJoCo 版)はカメラの視野外へ下がってから完了にする。
                # 置いた布片をカメラで検証するとき、アームが上に被らないようにするため
                self.phase = Phase.PARK if hasattr(a, "park") else Phase.DONE
        elif self.phase is Phase.PARK:
            if a.park(dt):   # type: ignore[attr-defined]
                self.phase = Phase.DONE
        return self.phase

    def arm_dwell(self) -> float:
        limits = getattr(self.arm, "limits", None)
        return limits.dwell_s if limits else 0.25

    @property
    def finished(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED)
