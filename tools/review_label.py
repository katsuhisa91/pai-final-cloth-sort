"""要確認トレイに回った布に、人が正しい番号を付ける(再学習のデータになる)。

    python tools/review_label.py runs/cloth8/review_queue           # 1 枚ずつ画像を開いて番号を入力
    python tools/review_label.py runs/cloth8/review_queue --list    # 状態だけ見る

入力: 番号(例 16)/ 空 Enter = あとで / s = 使わない(汚れ・折れで判別不能など)/ q = 終了
付けた番号は各 .json の "label" に入り、ml/train.py --feedback <dir> で学習に足せる。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="画像を開かない(パスだけ表示)")
    a = ap.parse_args()
    items = sorted(a.queue.glob("*.json"))
    if a.list:
        for j in items:
            d = json.loads(j.read_text())
            print(f"{j.stem}: read={d['read']!r} conf={d['conf']} label={d.get('label')!r}")
        return
    for j in items:
        d = json.loads(j.read_text())
        if d.get("label"):
            continue
        png = j.with_suffix(".png")
        if not a.no_open and sys.platform == "darwin":
            subprocess.run(["open", str(png)], check=False)
        ans = input(f"{png}  読み取り={d['read']!r}(確信度 {d['conf']})  正しい番号: ").strip()
        if ans == "q":
            break
        if not ans:
            continue
        d["label"] = "skip" if ans == "s" else ans
        d["labeled_by"] = "human"
        j.write_text(json.dumps(d, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
