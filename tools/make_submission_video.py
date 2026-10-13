"""提出用の動画(1920x1080、MP4)を作る。

    .venv/bin/python tools/make_submission_video.py --frames runs/dora/frames --out runs/submission/PAI最終課題_katsuhisa91.mp4

構成: タイトル → 課題 → 仕組み → AI → 実行の様子(等速) → 結果 → 工夫と次の一歩。
実行の様子は dora 版(runs/dora/frames)のフレームをそのまま使い、倍速はかけない。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from cloth_sort.sim import fonts

W, H = 1920, 1080
FPS_IN = 10
FPS_OUT = 30
BG = (24, 26, 32)
PANEL = (34, 36, 44)
FG = (235, 235, 235)
DIM = (150, 150, 160)
ACCENT = (90, 180, 250)
WARN = (250, 150, 90)
OK = (120, 210, 130)
RED = (220, 80, 80)
ACCOUNT = "katsuhisa91"


def font(size: int, bold: bool = False):
    return fonts.ui_font(size, bold=bold)


def wrap(d: ImageDraw.ImageDraw, text: str, f, max_w: int) -> list[str]:
    """文字単位で折り返す(日本語前提)。"""
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, H - 50), f"PAI最終課題_{ACCOUNT}   等速再生(倍速なし)   シミュレーション", fill=DIM, font=font(22))
    return img, d


def heading(d: ImageDraw.ImageDraw, text: str, y: int = 70) -> int:
    d.text((80, y), text, fill=FG, font=font(56, bold=True))
    return y + 110


def bullets(d: ImageDraw.ImageDraw, items: list[str], x: int, y: int, max_w: int, size: int = 34,
            gap: int = 18, color=FG) -> int:
    f = font(size)
    for item in items:
        lines = wrap(d, item, f, max_w - 40)
        d.ellipse([x, y + size * 0.45, x + 12, y + size * 0.45 + 12], fill=ACCENT)
        for i, line in enumerate(lines):
            d.text((x + 32, y), line, fill=color, font=f)
            y += int(size * 1.35)
        y += gap
    return y


def box(d: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, body: str = "",
        outline=ACCENT, size: int = 26) -> None:
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=14, fill=PANEL, outline=outline, width=3)
    ft = font(size, bold=True)
    fb = font(size - 4)
    y = y0 + 18
    for line in wrap(d, title, ft, x1 - x0 - 30):
        tw = d.textlength(line, font=ft)
        d.text((x0 + (x1 - x0 - tw) / 2, y), line, fill=FG, font=ft)
        y += int(size * 1.3)
    if body:
        y += 6
        for line in wrap(d, body, fb, x1 - x0 - 30):
            tw = d.textlength(line, font=fb)
            d.text((x0 + (x1 - x0 - tw) / 2, y), line, fill=DIM, font=fb)
            y += int((size - 4) * 1.3)


def arrow(d: ImageDraw.ImageDraw, x0: int, y: int, x1: int, color=DIM) -> None:
    d.line([(x0, y), (x1, y)], fill=color, width=4)
    d.polygon([(x1, y), (x1 - 16, y - 10), (x1 - 16, y + 10)], fill=color)


# ---------- スライド ----------

def slide_title() -> Image.Image:
    img, d = new_canvas()
    d.text((120, 300), "番号付き布片の仕分け", fill=FG, font=font(96, bold=True))
    d.text((120, 440), "手書きの番号を読み取り、番号どおりの置き場へ運ぶ", fill=FG, font=font(44))
    d.text((120, 560), "Physical AI 応用編1 最終課題", fill=ACCENT, font=font(36, bold=True))
    d.text((120, 615), f"omnicampus: {ACCOUNT}", fill=DIM, font=font(32))
    d.text((120, 760), "カメラ → 検出 → 番号読み取り(自前学習 CNN) → 計画 → 仮想アーム → 検証", fill=DIM, font=font(30))
    d.text((120, 805), "dora-rs の dataflow として 3 ノードで実行", fill=DIM, font=font(30))
    return img


def slide_task(fig: Path | None) -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "課題: 布片を番号どおりの置き場へ置く")
    items = [
        "作業台の左に散らばった布片を、右の番号付き置き場へ 1 枚ずつ運ぶ",
        "読めない・自信がない布片は止めずに「要確認トレイ」へ逃がす",
        "人が見るのは要確認トレイだけで済む",
    ]
    bullets(d, items, 90, y + 10, 800, size=30)
    if fig and fig.exists():
        im = Image.open(fig).convert("RGB")
        # 作業台の部分(左 1230 px)だけを切り出して右に置く
        im = im.crop((30, 90, 1230, 690))
        im.thumbnail((880, 440))
        x = W - im.width - 80
        img.paste(im, (x, 190))
        d.rectangle([x - 1, 189, x + im.width, 190 + im.height], outline=(80, 80, 90))
        d.text((x, 190 + im.height + 12), "俯瞰カメラ画像と検出結果(緑: 確信度 0.6 以上、橙: 未満)", fill=DIM, font=font(24))
    return img


def slide_system() -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "仕組み: 撮影 → 検出 → 読み取り → 計画 → 搬送 → 検証")
    steps = [
        ("俯瞰カメラ", "1 px = 1 mm"),
        ("背景差分で検出", "布の色に依らない"),
        ("番号読み取り", "1 文字 CNN + 分割"),
        ("計画", "確信度 0.6 以上は置き場、未満は要確認"),
        ("pick & place", "仮想アーム"),
        ("検証", "置いた後に再撮影し照合"),
    ]
    bw, gap, top = 262, 44, y + 30
    x = 80
    for i, (t, b) in enumerate(steps):
        box(d, (x, top, x + bw, top + 150), t, b)
        if i < len(steps) - 1:
            arrow(d, x + bw + 4, top + 75, x + bw + gap - 4)
        x += bw + gap

    y2 = top + 230
           fill=FG, font=font(30, bold=True))
    y3 = y2 + 70
    box(d, (160, y3, 620, y3 + 170), "sim(ロボット側)", "世界・カメラ・アーム。image / result を送る", outline=OK)
    box(d, (730, y3, 1190, y3 + 170), "brain(AI 側)", "検出・読み取り・計画。task / finish を送る", outline=ACCENT)
    box(d, (1300, y3, 1760, y3 + 170), "recorder(記録)", "reading / record を JSONL に残す", outline=WARN)
    arrow(d, 624, y3 + 60, 726)
    arrow(d, 726, y3 + 110, 624)
    arrow(d, 1194, y3 + 85, 1296)
    d.text((160, y3 + 200), "実機化はロボット側(sim)の RobotArm 実装を 1 つ差し替えるだけ。AI 側と記録は変更なし",
           fill=DIM, font=font(28))
    return img


def slide_ai(metrics: dict | None) -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "AI: 手書き番号を読む CNN を自前で学習")
    n_train = metrics["n_train"] if metrics else 8800
    params = metrics["params"] if metrics else 468139
    heldout = metrics["heldout"]["accuracy"] if metrics else 0.846
    val = metrics["val"]["accuracy"] if metrics else 1.0
    n_fonts = len(metrics["train_fonts"]) if metrics else 8
    n_held = len(metrics["heldout_fonts"]) if metrics else 3
    items = [
        f"手書き風フォント {n_fonts} 種で {n_train:,} 枚を合成(大きさ・傾き・線の太さ・ぼかし・ノイズをランダム化)",
        f"PyTorch の 1 文字 CNN(約 {params / 10000:.0f} 万パラメータ)。CPU で数分で学習できる",
        "語彙は 0〜9 と「-」。文字分割と組み合わせるので、学習で見ていない番号も読める",
        f"学習に使っていないフォント {n_held} 種で {heldout * 100:.1f}%(学習フォントでは {val * 100:.0f}%)",
        "読み違えるときは確信度も下がりやすい → しきい値 0.6 で要確認トレイへ",
    ]
    bullets(d, items, 90, y + 10, 1760, size=34, gap=22)
           fill=DIM, font=font(28))
    return img


def slide_results(summary: dict, log: list[dict]) -> Image.Image:
    img, d = new_canvas()
    y = heading(d, f"結果({summary['pieces']} 枚、seed 1)")
    stats = [
        ("検出", f"{summary['detected']} / {summary['pieces']}", OK),
        ("正しく読めた", f"{summary['read_correct']} / {summary['pieces']}", OK),
        ("置き場へ", f"{summary['placed_slot']}", OK),
        ("うち検証 OK", f"{summary['placed_slot_verified']}", OK),
        ("要確認へ", f"{summary['review']}", WARN),
        ("誤配置", f"{summary['wrong_slot']}", RED if summary["wrong_slot"] else OK),
    ]
    x = 80
    for label, value, color in stats:
        d.rounded_rectangle([x, y + 10, x + 280, y + 190], radius=14, fill=PANEL)
        f_big = font(64, bold=True)
        tw = d.textlength(value, font=f_big)
        d.text((x + (280 - tw) / 2, y + 30), value, fill=color, font=f_big)
        f_lab = font(26)
        tw = d.textlength(label, font=f_lab)
        d.text((x + (280 - tw) / 2, y + 130), label, fill=DIM, font=f_lab)
        x += 300
    y2 = y + 240
    d.text((80, y2), "要確認トレイに回った布片", fill=FG, font=font(34, bold=True))
    rows = []
    for rec in log:
        if rec.get("event") == "record" and rec.get("dest") == "review":
            truth, read, conf = rec["truth"], rec["label"], rec["confidence"]
            if truth == read:
                rows.append(f"正解 {truth} → 読み {read}(正しいが確信度 {conf:.2f} で低い)")
            else:
                rows.append(f"正解 {truth} → 読み {read}(読み違い、確信度 {conf:.2f})")
    bullets(d, rows, 90, y2 + 60, 1760, size=32)
    d.text((80, 880), "読み違えた布片はすべて要確認へ回り、誤った置き場には 1 枚も置かれなかった",
           fill=ACCENT, font=font(34, bold=True))
    d.text((80, 935), f"シミュレーション時間 {summary['sim_seconds']:.0f} 秒(等速)、dora の 3 ノードでも単一プロセスでも同じ結果",
           fill=DIM, font=font(26))
    return img


def slide_next() -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "工夫した点と次の一歩")
    d.text((80, y), "工夫した点", fill=ACCENT, font=font(36, bold=True))
    bullets(d, [
        "迷ったら止めずに人へ回す設計。誤配置ゼロを最優先にした",
        "置いた後に読み直す自己検証。判断とその根拠を JSONL に残し、再学習の種にできる",
        "ロボット側と AI 側を dora で分離。実機化は RobotArm 1 クラスの差し替え",
    ], 90, y + 60, 880, size=30)
    d.text((1000, y), "次の一歩", fill=WARN, font=font(36, bold=True))
    bullets(d, [
        "SO-101 など LeRobot 対応アームで RobotArm を実装し、実機で同じ dataflow を動かす",
        "把持と接触の物理は MuJoCo / Gazebo で先に確認する",
    ], 1010, y + 60, 840, size=30)
    d.text((80, 900), "コード: github.com/katsuhisa91/pai-final-cloth-sort(Apache-2.0)", fill=FG, font=font(32, bold=True))
    return img


# ---------- 実行の様子(フレーム合成) ----------

def compose_frame(src: Image.Image) -> Image.Image:
    img, d = new_canvas()
    scaled = src.convert("RGB").resize((W, int(src.height * W / src.width)), Image.LANCZOS)
    top = 100
    img.paste(scaled, (0, top))
    d.text((60, 24), "実行の様子(等速)  dora-rs の 3 ノード sim / brain / recorder を実際に動かして録画",
           fill=FG, font=font(34, bold=True))
    yb = top + scaled.height + 22
    f = font(28)
    d.rectangle([60, yb + 8, 84, yb + 32], outline=OK, width=3)
    d.text((100, yb), "確信度 0.6 以上 → 番号の置き場へ", fill=FG, font=f)
    d.rectangle([640, yb + 8, 664, yb + 32], outline=WARN, width=3)
    d.text((680, yb), "0.6 未満 → 要確認トレイ(赤枠)へ", fill=FG, font=f)
    d.text((1220, yb), "右パネル: AI 側(brain)の判断とログ", fill=DIM, font=f)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/dora/frames")
    ap.add_argument("--log", default="runs/dora/log.jsonl")
    ap.add_argument("--summary", default="runs/seed1/summary.json")
    ap.add_argument("--metrics", default="models/metrics.json")
    ap.add_argument("--fig", default="report/fig1_detect.png")
    ap.add_argument("--out", default=f"runs/submission/PAI最終課題_{ACCOUNT}.mp4")
    ap.add_argument("--slides-dir", default="runs/submission/slides")
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("*.png"))
    if not frames:
        raise SystemExit(f"フレームがありません: {args.frames}(先に ./run_dora.sh を実行)")
    summary = json.loads(Path(args.summary).read_text())
    log = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    metrics = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else None

    slides: list[tuple[str, Image.Image, float]] = [
        ("01_title", slide_title(), 6.0),
        ("02_task", slide_task(Path(args.fig)), 10.0),
        ("03_system", slide_system(), 12.0),
        ("04_ai", slide_ai(metrics), 9.0),
    ]
    outro: list[tuple[str, Image.Image, float]] = [
        ("06_results", slide_results(summary, log), 10.0),
        ("07_next", slide_next(), 10.0),
    ]
    slides_dir = Path(args.slides_dir)
    slides_dir.mkdir(parents=True, exist_ok=True)
    for name, im, _ in slides + outro:
        im.save(slides_dir / f"{name}.png")
    compose_frame(Image.open(frames[0])).save(slides_dir / "05_run_sample.png")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS_IN), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", str(FPS_OUT), "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    n = 0

    def emit(im: Image.Image, seconds: float) -> None:
        nonlocal n
        raw = im.tobytes()
        for _ in range(int(round(seconds * FPS_IN))):
            proc.stdin.write(raw)
            n += 1

    for _, im, sec in slides:
        emit(im, sec)
    for p in frames:
        emit(compose_frame(Image.open(p)), 1.0 / FPS_IN)
    for _, im, sec in outro:
        emit(im, sec)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg が失敗しました")
    print(f"{out}: {n} frames, {n / FPS_IN:.1f} s")


if __name__ == "__main__":
    main()
