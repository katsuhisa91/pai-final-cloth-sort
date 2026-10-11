"""番号読み取りを 1 枚ぶん手で試す。布片を合成して読ませ、文字分割と確信度を表示する。

例:  python tools/try_reader.py 12          # 学習フォントと未見フォントから無作為
     python tools/try_reader.py 1-2 --font Noteworthy --out /tmp/piece.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cloth_sort import config
from cloth_sort.perception.reader import LabelReader
from cloth_sort.sim import fonts
from cloth_sort.sim.synth import draw_label
from cloth_sort.sim.world import _cloth_color


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--font", help="フォント名の一部(例: Noteworthy, Chalkboard)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="合成した布片と文字枠を PNG に書く")
    a = ap.parse_args()

    pool = fonts.available(fonts.TRAIN_FONTS + fonts.HELDOUT_FONTS)
    if a.font:
        pool = [f for f in pool if a.font.lower() in f[0].lower()]
        if not pool:
            raise SystemExit(f"フォントが見つかりません: {a.font}")
    rng = random.Random(a.seed)
    s = config.PIECE_MM
    piece = Image.new("RGBA", (s, s), _cloth_color(rng) + (255,))
    draw_label(piece, a.text, (s - 4, s - 2), size_px=rng.randint(13, 17), rng=rng, font_pool=pool)
    crop = np.asarray(piece.convert("RGB"))

    reader = LabelReader()
    r = reader.read(crop)
    heldout = {p for p, _ in fonts.HELDOUT_FONTS}
    print(f"正解: {a.text}  読み取り: {r.text or '(読めない)'}  確信度: {r.confidence:.2f}  "
          f"判定: {'採用' if reader.accept(r) else '要確認へ'}")
    print(f"文字の枠: {r.char_boxes}")
    print("使えるフォント:", ", ".join(Path(p).stem + ("(未見)" if p in heldout else "") for p, _ in pool))

    if a.out:
        big = piece.convert("RGB").resize((s * 6, s * 6), Image.NEAREST)
        d = ImageDraw.Draw(big)
        for x0, y0, x1, y1 in r.char_boxes:
            d.rectangle([x0 * 6, y0 * 6, x1 * 6, y1 * 6], outline=(255, 0, 0), width=2)
        big.save(a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
