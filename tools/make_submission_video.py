"""提出用の動画(1920x1080、MP4)を作る。

    MUJOCO_GL=osmesa .venv/bin/python tools/mujoco_closeup.py --seed 3 --out runs/closeup/mujoco_pinch.mp4
    .venv/bin/python tools/make_submission_video.py

構成(合計 3 分以内):
  1. 冒頭: ロボットが布を運ぶ映像にタイトルを重ねる
  2. ロボットがやること(図と 3 行)
  3. AI は 2 か所(番号を読む CNN、腕を動かす模倣学習)
  4. 実行の様子(学習した方策でアームを動かした仕分けの録画、2 倍速、場面ごとの字幕つき)
  5. 取り組みの途中で公開された RoboSimStudio でも試した(MuJoCo との左右比較、布の柔らかさ違い)
  6. 結果と学び
氏名・所属・メールアドレスは入れない(omnicampus のアカウント名のみ)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cloth_sort.sim import fonts

W, H = 1920, 1080
FPS_IN = 10  # 合成するフレームの周期。出力は 30 fps に引き伸ばす
FPS_OUT = 30
SPEED = 2  # 実行部分の倍速
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
    """文字単位で折り返す(日本語前提)。句読点・閉じ括弧は行頭に置かない。"""
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > max_w and cur and ch not in "、。)」・":
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def canvas(footer: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, H - 50), f"PAI最終課題_{ACCOUNT}   {footer}", fill=DIM, font=font(22))
    return img, d


def heading(d: ImageDraw.ImageDraw, text: str, y: int = 60) -> int:
    d.text((80, y), text, fill=FG, font=font(60, bold=True))
    return y + 120


def bullets(d: ImageDraw.ImageDraw, items: list[str], x: int, y: int, max_w: int, size: int = 38,
            gap: int = 22, color=FG) -> int:
    f = font(size)
    for item in items:
        d.ellipse([x, y + size * 0.42, x + 14, y + size * 0.42 + 14], fill=ACCENT)
        for line in wrap(d, item, f, max_w - 40):
            d.text((x + 34, y), line, fill=color, font=f)
            y += int(size * 1.35)
        y += gap
    return y


def badge(d: ImageDraw.ImageDraw, label: str = f"{SPEED} 倍速") -> None:
    f = font(36, bold=True)
    tw = d.textlength(label, font=f)
    d.rectangle([W - tw - 70, 16, W - 20, 70], fill=(200, 60, 40))
    d.text((W - tw - 45, 22), label, fill=(255, 255, 255), font=f)


def arrow(d: ImageDraw.ImageDraw, x0: int, y: int, x1: int, color=DIM) -> None:
    d.line([(x0, y), (x1, y)], fill=color, width=5)
    d.polygon([(x1, y), (x1 - 18, y - 12), (x1 - 18, y + 12)], fill=color)


def frames_of(path: Path, size: tuple[int, int], step: int):
    """MP4 を FPS_IN のフレームとして読み、step 枚おきに返す(2 なら 2 倍速)。"""
    w, h = size
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={FPS_IN},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    i = 0
    while True:
        raw = proc.stdout.read(w * h * 3)
        if len(raw) < w * h * 3:
            break
        if i % step == 0:
            yield Image.frombytes("RGB", (w, h), raw)
        i += 1
    proc.wait()


# ---------- 1. 冒頭 ----------

def hook_frames(run_video: Path, start_s: float, seconds: float, summary: dict):
    """実行の録画から布を運んでいる場面を切り出し、タイトルを重ねる。"""
    skip = round(start_s * FPS_IN)
    for i, fr in enumerate(frames_of(run_video, (W, H), SPEED)):
        if i < skip // SPEED:
            continue
        if i >= skip // SPEED + round(seconds * FPS_IN):
            break
        img = compose_run(fr, None, summary)
        shade = Image.new("RGB", (W, 300), (8, 9, 12))
        img.paste(Image.blend(img.crop((0, 0, W, 300)), shade, 0.93), (0, 0))
        d = ImageDraw.Draw(img)
        d.text((80, 40), "番号付き布片の仕分け", fill=FG, font=font(84, bold=True))
        d.text((84, 150), "手書きの番号を読み、番号どおりの置き場へ運ぶ。腕は模倣学習した AI が動かす",
               fill=FG, font=font(38))
        d.text((84, 220), f"Physical AI 応用編1 最終課題   omnicampus: {ACCOUNT}", fill=ACCENT, font=font(30, bold=True))
        badge(d)
        yield img


# ---------- 2. ロボットがやること ----------

def slide_overview(fig: Path) -> Image.Image:
    img, d = canvas("MuJoCo シミュレーション")
    y = heading(d, "ロボットがやること")
    im = Image.open(fig).convert("RGB")
    im.thumbnail((860, 430))
    img.paste(im, (80, y + 10))
    d.text((80, y + 20 + im.height), "AI が見ている俯瞰カメラの画像(緑: 読めた / 橙: 自信が低い)", fill=DIM, font=font(24))
    bullets(d, [
        "5 cm 角の布に手書きの番号。番号どおりの置き場へ 1 枚ずつ運ぶ",
        "読めない・自信がない布は止めずに「要確認トレイ」へ。人が見るのはそこだけ",
        "MuJoCo 上の Franka Panda。布は変形する。指の摩擦だけでつまむ",
    ], 1000, y + 10, 840, size=36)
    steps = [("カメラで布を探す", DIM), ("番号を読む AI ①", ACCENT), ("腕を動かす AI ②", OK), ("置いたら確認", DIM)]
    bw, gap, top = 380, 80, 800
    x = 80
    for i, (t, color) in enumerate(steps):
        d.rounded_rectangle([x, top, x + bw, top + 110], radius=16, fill=PANEL, outline=color, width=4)
        f = font(36, bold=True)
        tw = d.textlength(t, font=f)
        d.text((x + (bw - tw) / 2, top + 32), t, fill=FG, font=f)
        if i < len(steps) - 1:
            arrow(d, x + bw + 8, top + 55, x + bw + gap - 8)
        x += bw + gap
    return img


# ---------- 3. AI は 2 か所 ----------

def slide_ai(metrics: dict | None) -> Image.Image:
    img, d = canvas("MuJoCo シミュレーション")
    y = heading(d, "AI は 2 か所で使った")
    heldout = metrics["heldout"]["accuracy"] if metrics else 0.846
    cw = 840
    for k, (title, color) in enumerate((("AI ① 番号を読む", ACCENT), ("AI ② 腕を動かす", OK))):
        x0 = 80 + k * (cw + 80)
        d.rounded_rectangle([x0, y, x0 + cw, 960], radius=18, fill=PANEL)
        d.text((x0 + 40, y + 30), title, fill=color, font=font(46, bold=True))
    x0 = 80
    bullets(d, ["自前で学習した文字認識 CNN(PyTorch)", "手書き風に合成した文字で学習"], x0 + 40, y + 120, cw - 60, size=34)
    d.text((x0 + 40, y + 330), f"{heldout * 100:.1f}%", fill=ACCENT, font=font(120, bold=True))
    d.text((x0 + 44, y + 480), "学習に使っていない筆跡での正解率", fill=DIM, font=font(30))
    bullets(d, ["自信が低い読みは、置き場ではなく要確認トレイへ"], x0 + 40, y + 560, cw - 60, size=34)
    x0 = 80 + cw + 80
    bullets(d, ["LeRobot の ACT で模倣学習。手本は手で作った動き", "カメラ画像を見て、つまむ・運ぶ・放す"],
            x0 + 40, y + 120, cw - 60, size=34)
    d.text((x0 + 40, y + 330), "学習に使っていない 20 場面での成功", fill=FG, font=font(30, bold=True))
    by = y + 390
    for label, n, color in (("手本(手で作った動き)", 12, DIM), ("AI(そのまま)", 14, ACCENT), ("AI + 実行の工夫", 19, OK)):
        d.text((x0 + 40, by), label, fill=FG, font=font(28))
        bx, bw = x0 + 380, 300
        d.rectangle([bx, by + 6, bx + bw, by + 36], fill=(50, 52, 62))
        d.rectangle([bx, by + 6, bx + int(bw * n / 20), by + 36], fill=color)
        d.text((bx + bw + 16, by - 4), f"{n}/20", fill=color, font=font(36, bold=True))
        by += 70
    return img


# ---------- 4. 実行の様子 ----------

def run_captions(log: list[dict]) -> list[str]:
    """録画の状態欄の切り替わりごとの字幕。0, 1 は開始と読み取り、2 から 1 枚ずつ、最後が完了。"""
    caps = ["俯瞰カメラで布を見つけ、手書きの番号を AI ① が読む", "俯瞰カメラで布を見つけ、手書きの番号を AI ① が読む"]
    k = 0
    for rec in log:
        if rec.get("event") != "place":
            continue
        k += 1
        label = rec.get("label")
        if rec.get("phase") == "failed":
            caps.append(f"{k} 枚目: 番号 {label} を運ぶ途中で落とした → つかみ直しても置けず、人に回す")
        elif rec.get("dest") == "review":
            conf = rec.get("reason", "").split("(")[-1].rstrip(")")
            truth = rec.get("truth")
            what = f"「{truth}」を「{label}」と読み違え、" if truth and truth != label else f"「{label}」と読めたが"
            caps.append(f"{k} 枚目: {what}自信が低い({conf}) → 止めずに要確認トレイへ")
        else:
            extra = "腕は AI ② が動かす" if k == 2 else "置いた後にカメラで確かめる" if k == 4 else ""
            caps.append(f"{k} 枚目: 番号 {label} → 置き場 {label}" + (f"。{extra}" if extra else ""))
    caps.append("完了: 置き場へ 5 枚、要確認トレイへ 3 枚。誤った置き場には 1 枚も置かなかった")
    return caps


class StatusWatcher:
    """録画の右パネルの状態欄(「3/8 枚目を搬送中」)が変わった回数を数える。"""

    def __init__(self) -> None:
        self.prev = None
        self.state = 0
        self.since = 99

    def update(self, img: Image.Image) -> int:
        a = np.asarray(img.crop((1252, 464, 1460, 494)).convert("L"), dtype=np.float32)
        self.since += 1
        if self.prev is not None and float(np.abs(a - self.prev).mean()) > 1.0 and self.since > 8:
            self.state += 1
            self.since = 0
        self.prev = a
        return self.state


def compose_run(src: Image.Image, caption: str | None, summary: dict) -> Image.Image:
    img = src.convert("RGB")
    d = ImageDraw.Draw(img)
    # 録画のフッタは手続き版の文言のままなので、方策版の流れに書き換える
    d.rectangle([0, 1046, 1230, H], fill=BG)
    d.text((30, 1052), "カメラ → 検出 → 番号読み取り(CNN) → 計画 → 学習方策(ACT)が手先目標を出力 → IK → 把持(物理) → 検証",
           fill=DIM, font=font(17))
    if img.crop((1250, 966, 1880, 979)).convert("L").getextrema()[1] > 150:
        # 録画の最終行は「検証 OK」の数え方に誤りがあった(要確認トレイの分も数えていた。pipeline.py は修正済み)
        s = summary
        d.rectangle([1250, 961, 1885, 983], fill=PANEL)
        d.text((1256, 960), f"置き場へ {s['placed_slot']} 枚(検証 OK {s['placed_slot_verified']})、"
               f"要確認 {s['review'] + s.get('grasp_failed', 0)} 枚、所要 {s['sim_seconds']:.0f} s", fill=FG, font=font(16))
    # 左上の「実時間 / real-time」表示を上書きして矛盾をなくす
    d.rectangle([126, 50, 900, 80], fill=BG)
    d.text((130, 54), f"{SPEED} 倍速再生(録画は等速)  MuJoCo(変形体の布 + 2 本指)  アームは学習方策(ACT)", fill=DIM, font=font(17))
    badge(d)
    if caption:
        # 字幕(3D 視点の下の方に帯で出す)。長い字幕は文字を小さくして 1 行に収める
        size = 36
        while d.textlength(caption, font=font(size, bold=True)) > 1140 and size > 24:
            size -= 1
        band = Image.new("RGB", (1180, 70), (0, 0, 0))
        img.paste(Image.blend(img.crop((30, 662, 1210, 732)), band, 0.72), (30, 662))
        d.text((50, 697 - size * 0.6), caption, fill=(255, 255, 255), font=font(size, bold=True))
    return img


# ---------- 5. RoboSimStudio ----------

def side_by_side(left: Path, right: Path, left_label: str, right_label: str, title: str, lines: list[str],
                 left_note: str, right_note: str, left_size: tuple[int, int], right_size: tuple[int, int]):
    cw, ch = 880, 495
    base, d = canvas(f"{SPEED} 倍速再生   左右とも物理シミュレーション")
    d.text((80, 40), title, fill=FG, font=font(52, bold=True))
    for k, (label, note) in enumerate(((left_label, left_note), (right_label, right_note))):
        x0 = 60 + k * (cw + 40)
        d.text((x0, 140), label, fill=ACCENT if k == 0 else OK, font=font(36, bold=True))
        d.text((x0, 200 + ch + 16), note, fill=FG, font=font(30))
    y = 200 + ch + 80
    for line in lines:
        d.text((80, y), line, fill=DIM, font=font(30))
        y += 46
    badge(d)
    lf = list(frames_of(left, left_size, SPEED))
    rf = list(frames_of(right, right_size, SPEED))
    for i in range(max(len(lf), len(rf))):
        img = base.copy()
        img.paste(lf[min(i, len(lf) - 1)].resize((cw, ch)), (60, 200))
        img.paste(rf[min(i, len(rf) - 1)].resize((cw, ch)), (60 + cw + 40, 200))
        yield img


def rss_screens(mujoco_clip: Path, soft: Path, mid: Path, results: list[dict]):
    def stat(bend: float) -> str:
        rows = [r for r in results if r.get("bend") == bend]
        return f"{sum(r['success'] for r in rows)}/{len(rows)} 回成功"

    yield from side_by_side(
        mujoco_clip, soft, "本体: MuJoCo(flex の布)", "RoboSimStudio: NVIDIA Newton(VBD の布)",
        "取り組みの途中で公開された RoboSimStudio でも試した",
        ["同じ 5 cm 角の布・Panda・速さで、つまむ → 運ぶ → 放す。布は指に固定せず摩擦だけで持つ",
         "RoboSimStudio は UC Berkeley と NVIDIA が 2026 年 10 月に公開(Apache-2.0)。Colab の GPU で実行"],
        "布は板に近く、ほぼ平らに垂れる", "布が指から垂れ、折れて着地する",
        (1280, 720), (1280, 720))
    yield from side_by_side(
        mid, soft, "やや柔らかい布", "柔らかい布", "RoboSimStudio: 布の柔らかさを変える",
        ["動かすまでに詰まった点: GPU が無いと摘めない、厚みのある布は指をすり抜ける(→ 膜で作る)、",
         "腕が 3〜4 mm 手前で止まる(→ 指先を机まで追い込む)、空気抵抗が無く布が揺れ続ける(→ 減衰を足す)"],
        f"{stat(0.5)}。置くと平らなまま", f"{stat(0.05)}。置くと折れて着地",
        (1280, 720), (1280, 720))


# ---------- 6. 結果と学び ----------

def slide_closing(summary: dict) -> Image.Image:
    img, d = canvas("MuJoCo シミュレーション")
    y = heading(d, "結果と学び")
    n = summary["pieces"]
    stats = [("検出", f"{summary['detected']}/{n}", OK), ("正しく読めた", f"{summary['read_correct']}/{n}", OK),
             ("置き場へ", f"{summary['placed_slot']}", OK),
             ("要確認(人へ)", f"{summary['review'] + summary.get('grasp_failed', 0)}", WARN),
             ("誤った置き場", f"{summary['wrong_slot']}", RED if summary["wrong_slot"] else OK)]
    x = 80
    for label, value, color in stats:
        d.rounded_rectangle([x, y, x + 330, y + 170], radius=16, fill=PANEL)
        fb = font(72, bold=True)
        tw = d.textlength(value, font=fb)
        d.text((x + (330 - tw) / 2, y + 14), value, fill=color, font=fb)
        fl = font(28)
        tw = d.textlength(label, font=fl)
        d.text((x + (330 - tw) / 2, y + 116), label, fill=DIM, font=fl)
        x += 352
    y += 210
    d.text((80, y), "工夫した点", fill=ACCENT, font=font(36, bold=True))
    bullets(d, ["迷ったら止めずに人へ回す。誤った置き場はゼロ",
                "AI の失敗を 1 件ずつ分解し、学習のやり直しでなく実行の工夫で 14 → 19/20"], 86, y + 60, 860, size=32)
    d.text((1000, y), "講座の学びの使い方", fill=OK, font=font(36, bold=True))
    bullets(d, ["MLOps: 決めた場面での成功率で評価。データとモデルは Hugging Face、テストは CI",
                "OSS: dora-rs・MuJoCo・LeRobot で組み、Apache-2.0 で公開"], 1006, y + 60, 840, size=32)
    d.text((80, 940), "コード: github.com/katsuhisa91/pai-final-cloth-sort", fill=FG, font=font(34, bold=True))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-video", default="runs/cloth8_act2/cloth_sort.mp4")
    ap.add_argument("--log", default="runs/cloth8_act2/log.jsonl")
    ap.add_argument("--summary", default="runs/cloth8_act2/summary.json")
    ap.add_argument("--metrics", default="models/metrics.json")
    ap.add_argument("--fig", default="runs/cloth8/top_detect.png")
    ap.add_argument("--mujoco-clip", default="runs/closeup/mujoco_pinch.mp4")
    ap.add_argument("--rss-dir", default="experiments/robosimstudio/results")
    ap.add_argument("--out", default=f"runs/submission/PAI最終課題_{ACCOUNT}.mp4")
    ap.add_argument("--slides-dir", default="runs/submission/slides")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    log = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    metrics = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else None
    rss_dir = Path(args.rss_dir)
    rss_results = [json.loads(line) for line in (rss_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    slides_dir = Path(args.slides_dir)
    slides_dir.mkdir(parents=True, exist_ok=True)
    for old in slides_dir.glob("*.png"):
        old.unlink()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS_IN), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", "-r", str(FPS_OUT), "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    n = 0

    def emit(im: Image.Image, seconds: float = 1.0 / FPS_IN) -> None:
        nonlocal n
        raw = im.convert("RGB").tobytes()
        for _ in range(max(1, int(round(seconds * FPS_IN)))):
            proc.stdin.write(raw)
            n += 1

    def emit_seq(name: str, frames) -> None:
        for i, im in enumerate(frames):
            if i == 25:
                im.save(slides_dir / f"{name}.png")
            emit(im)

    emit_seq("01_hook", hook_frames(Path(args.run_video), start_s=58.0, seconds=5.0, summary=summary))
    for name, im, sec in (("02_overview", slide_overview(Path(args.fig)), 8.0), ("03_ai", slide_ai(metrics), 10.0)):
        im.save(slides_dir / f"{name}.png")
        emit(im, sec)

    caps = run_captions(log)
    watch = StatusWatcher()
    for i, fr in enumerate(frames_of(Path(args.run_video), (W, H), SPEED)):
        state = watch.update(fr)
        im = compose_run(fr, caps[min(state, len(caps) - 1)], summary)
        if i == 300:
            im.save(slides_dir / "04_run_sample.png")
        emit(im)

    screens = rss_screens(Path(args.mujoco_clip), rss_dir / "pinch_soft.mp4", rss_dir / "pinch_mid.mp4", rss_results)
    emit_seq("05_robosimstudio", screens)
    im = slide_closing(summary)
    im.save(slides_dir / "06_closing.png")
    emit(im, 12.0)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg が失敗しました")
    print(f"{out}: {n} frames, {n / FPS_IN:.1f} s")


if __name__ == "__main__":
    main()
