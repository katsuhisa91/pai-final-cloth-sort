"""作業台のレイアウトと共通定数。単位は mm。画像は 1 px = 1 mm で描く。"""

from __future__ import annotations

from dataclasses import dataclass

# 作業台(実験台 1 台分を想定した抽象レイアウト)
BENCH_W_MM = 1200
BENCH_H_MM = 600
PX_PER_MM = 1.0

# 布片
PIECE_MM = 50            # 5 cm 角
PIECE_MIN_GAP_MM = 18    # 配置時に確保する最小すき間(重なりは本課題の対象外)

SOURCE_AREA = (60, 60, 540, 540)  # x0, y0, x1, y1

# 番号つきの置き場(ボード)。5 列 × 4 行、番号 1〜20
BOARD_ORIGIN = (640, 80)
SLOT_PITCH_MM = 100
BOARD_COLS = 5
BOARD_ROWS = 4
SLOT_LABELS = [str(i) for i in range(1, BOARD_COLS * BOARD_ROWS + 1)]

# 読み取れなかった布片を逃がす「要確認トレイ」
REVIEW_TRAY = (640, 500, 1140, 580)
REVIEW_TRAY_PITCH_MM = 60

# アームの土台(作業台の下辺中央)
ARM_BASE = (600, 600)


@dataclass(frozen=True)
class Slot:
    label: str
    center: tuple[float, float]


def board_slots() -> list[Slot]:
    """番号順に並んだ置き場の中心座標を返す。左上が 1、右へ進み、次の行へ。"""
    slots: list[Slot] = []
    x0, y0 = BOARD_ORIGIN
    half = SLOT_PITCH_MM / 2
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            label = str(row * BOARD_COLS + col + 1)
            slots.append(Slot(label, (x0 + col * SLOT_PITCH_MM + half, y0 + row * SLOT_PITCH_MM + half)))
    return slots


def review_tray_position(index: int) -> tuple[float, float]:
    """要確認トレイの index 番目の置き位置。"""
    x0, y0, x1, y1 = REVIEW_TRAY
    cap = int((x1 - x0) // REVIEW_TRAY_PITCH_MM)
    i = index % cap
    return (x0 + REVIEW_TRAY_PITCH_MM * i + REVIEW_TRAY_PITCH_MM / 2, (y0 + y1) / 2)
