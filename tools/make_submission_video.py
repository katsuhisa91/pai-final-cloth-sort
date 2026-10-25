"""提出用の動画(1920x1080、MP4)を作る。

    # 提出版: 学習した方策(ACT)でアームを動かした仕分けの録画を 2 倍速で使う
    .venv/bin/python tools/make_submission_video.py --run-video runs/cloth8_act2/cloth_sort.mp4 \
        --log runs/cloth8_act2/log.jsonl --summary runs/cloth8_act2/summary.json \
        --fig runs/cloth8/top_detect.png --run-speed 2 --arm policy

構成: タイトル → 課題 → 仕組み → AI ①番号読み取り → AI ②アームの動き(模倣学習) → 実行の様子 → 結果 → 工夫・講座との対応・次の一歩。
実行の様子は MuJoCo で描画した 1920x1080 のフレーム(--frames の PNG か、--run-video の MP4)をそのまま使う。
氏名・所属・メールアドレスは入れない(omnicampus のアカウント名のみ)。
"""

from __future__ import annotations

import argparse
import re
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
SEED = 4
SPEED = 1
ARM = "procedural"  # 実行部分のアーム: procedural(状態機械 + IK)/ policy(模倣学習した ACT)
SUMMARY: dict = {}


def speed_label() -> str:
    return f"{SPEED} 倍速再生" if SPEED > 1 else "等速再生(倍速なし)"


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


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, H - 50), f"PAI最終課題_{ACCOUNT}   実行部分は{speed_label()}   MuJoCo シミュレーション", fill=DIM, font=font(22))
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
    arm = "搬送(模倣学習した方策 ACT)" if ARM == "policy" else "アーム制御(IK)"
    d.text((120, 760), f"カメラ → 検出 → 番号読み取り(自前学習 CNN) → 計画 → {arm} → 検証", fill=DIM, font=font(30))
    d.text((120, 805), "MuJoCo 上の Franka Panda が、変形体の布を 2 本指でつまんで運ぶ(把持は物理任せ)", fill=DIM, font=font(30))
    return img


def slide_task(fig: Path | None) -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "布片を番号どおりの置き場へ置く")
    items = [
        "作業台の左に散らばった布片を、右の番号付き置き場へ 1 枚ずつ運ぶ",
        "読めない・自信がない布片は止めずに「要確認トレイ」へ逃がす",
        "人が見るのは要確認トレイだけで済む",
    ]
    bullets(d, items, 90, y + 10, 800, size=30)
    if fig and fig.exists():
        im = Image.open(fig).convert("RGB")
        if im.size != (1200, 600):
            # 2D 版のフレーム(1710x720)なら作業台の部分だけを切り出す
            im = im.crop((30, 90, 1230, 690))
        im.thumbnail((880, 440))
        x = W - im.width - 80
        img.paste(im, (x, 190))
        d.rectangle([x - 1, 189, x + im.width, 190 + im.height], outline=(80, 80, 90))
        d.text((x, 190 + im.height + 12), "MuJoCo の俯瞰カメラ画像と検出結果(緑: 確信度 0.6 以上、橙: 未満)", fill=DIM, font=font(24))
    return img


def slide_system() -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "仕組み 撮影 → 検出 → 読み取り → 計画 → 搬送 → 検証")
    steps = [
        ("俯瞰カメラ", "1 px = 1 mm"),
        ("背景差分で検出", "布の色に依らない"),
        ("番号読み取り", "1 文字 CNN + 分割"),
        ("計画", "確信度 0.6 以上は置き場、未満は要確認"),
        ("pick & place", "学習方策(ACT)が手先を動かし 2 本指でつまむ" if ARM == "policy" else "2 本指で布をつまむ(物理)"),
        ("検証", "退避後に置き場を再撮影"),
    ]
    bw, gap, top = 262, 44, y + 30
    x = 80
    for i, (t, b) in enumerate(steps):
        box(d, (x, top, x + bw, top + 150), t, b)
        if i < len(steps) - 1:
            arrow(d, x + bw + 4, top + 75, x + bw + gap - 4)
        x += bw + gap

    y2 = top + 230
    d.text((80, y2), "dora-rs の dataflow として 3 プロセスで実行(yaml でノードをつなぐ + Python スクリプト + Arrow)",
           fill=FG, font=font(30, bold=True))
    y3 = y2 + 70
    box(d, (160, y3, 620, y3 + 170), "sim(ロボット側)", "MuJoCo の世界・カメラ・Panda。画像と結果を送る", outline=OK)
    box(d, (730, y3, 1190, y3 + 170), "brain(AI 側)", "検出・読み取り・計画。task / finish を送る", outline=ACCENT)
    box(d, (1300, y3, 1760, y3 + 170), "recorder(記録)", "reading / record を JSONL に残す", outline=WARN)
    arrow(d, 624, y3 + 60, 726)
    arrow(d, 726, y3 + 110, 624)
    arrow(d, 1194, y3 + 85, 1296)
    d.text((160, y3 + 200), "物理: MuJoCo 3(Apache-2.0)。布は変形体(flex、厚さ 3 mm、フェルト程度)。アーム: menagerie の Franka Panda + ゴムパッド付き 2 本指",
           fill=DIM, font=font(26))
    d.text((160, y3 + 240), "実機化はロボット側(sim)の RobotArm 実装を 1 つ差し替えるだけ。AI 側と記録は変更なし",
           fill=DIM, font=font(26))
    d.text((160, y3 + 280), "動画の実行部分は同じ部品を 1 プロセスにまとめた版(pipeline.py)の録画",
           fill=DIM, font=font(26))
    return img


def slide_ai(metrics: dict | None) -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "AI ① 手書き番号を読む CNN を自前で学習")
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


def slide_policy() -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "AI ② アームの動きを模倣学習(LeRobot の ACT)")
    bullets(d, [
        "教師: 手で作った把持の手順(状態機械 + IK)。これでデモを 154 本集め、成功 123 本・5.1 万フレームを LeRobot 形式で保存",
        "入力: 俯瞰カメラ + 手首カメラの画像、手先位置、目標(つまむ点・置く点)。出力: 手先の目標位置と指の開閉(20 Hz、5 秒先まで)",
        "ACT(52M パラメータ)を 20,000 ステップ学習(Mac の GPU で約 1 時間 45 分)",
        "どの布をどこへ運ぶかは上位(読み取り・計画)が決め、方策は「つまんで運んで放す」を担当",
    ], 90, y + 10, 1000, size=30, gap=16)

    # 評価: 学習に使っていない 20 場面(固定シード)での成功数
    x0, y0 = 1160, y + 10
    d.rounded_rectangle([x0, y0, W - 80, y0 + 560], radius=14, fill=PANEL)
    d.text((x0 + 30, y0 + 24), "学習に使っていない 20 場面(固定シード)", fill=FG, font=font(28, bold=True))
    rows = [
        ("教師(手続き的な把持)", 12, DIM),
        ("学習した方策(そのまま)", 14, ACCENT),
        ("+ 実行器の改善", 19, OK),
    ]
    by = y0 + 90
    for label, n, color in rows:
        d.text((x0 + 30, by), label, fill=FG, font=font(26))
        bar_x, bar_w = x0 + 30, 520
        d.rectangle([bar_x, by + 42, bar_x + bar_w, by + 74], fill=(50, 52, 62))
        d.rectangle([bar_x, by + 42, bar_x + int(bar_w * n / 20), by + 74], fill=color)
        d.text((bar_x + bar_w + 20, by + 36), f"{n} / 20", fill=color, font=font(34, bold=True))
        by += 120
    note = "改善: つかみ直し・布の振り落とし・5 秒ごとの開ループ実行・置く位置の系統誤差(22 mm)の補正。データを 4 倍にしても差は出なかった"
    for line in wrap(d, note, font(23), W - 80 - x0 - 60):
        d.text((x0 + 30, by + 4), line, fill=DIM, font=font(23))
        by += 34

    d.text((90, 900), "評価は loss ではなく、固定シードのシミュレータでの成功率で測った。モデルとデータは Hugging Face Hub に置いた",
           fill=DIM, font=font(26))
    return img


def slide_results(summary: dict, log: list[dict]) -> Image.Image:
    img, d = new_canvas()
    y = heading(d, f"結果({summary['pieces']} 枚、seed {SEED})")
    stats = [
        ("検出", f"{summary['detected']} / {summary['pieces']}", OK),
        ("正しく読めた", f"{summary['read_correct']} / {summary['pieces']}", OK),
        ("置き場へ", f"{summary['placed_slot']}", OK),
        ("うち検証 OK", f"{summary['placed_slot_verified']}", OK),
        ("要確認へ", f"{summary['review'] + summary.get('grasp_failed', 0)}", WARN),
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
    d.text((80, y2), "要確認トレイに回った布片と、つかみ損ねの再試行", fill=FG, font=font(34, bold=True))
    rows = []
    for rec in log:
        # dora 版のログは event=record、単一プロセス版は event=place。確信度は理由の文字列からも拾う
        if rec.get("event") not in ("record", "place"):
            continue
        truth, read = rec.get("truth") or rec.get("label"), rec.get("label")
        conf = rec.get("confidence")
        if conf is None:
            m = re.search(r"\(([0-9.]+)\)", rec.get("reason", ""))
            conf = float(m.group(1)) if m else 0.0
        if rec.get("phase") == "failed":
            retries = rec.get("grip_retries", 0)
            why = (f"{retries} 回つかみ直したが搬送中に落下" if retries else "搬送中に落下") if rec.get("lost_in_transit") \
                else f"つかめず(再試行 {retries} 回)"
            rows.append(f"番号 {truth} → {why}。要確認として人へ")
        elif rec.get("dest") == "review":
            if truth == read:
                rows.append(f"正解 {truth} → 読み {read}(正しいが確信度 {conf:.2f} で低い)")
            else:
                rows.append(f"正解 {truth} → 読み {read}(読み違い、確信度 {conf:.2f})")
        if rec.get("grip_retries", 0) and rec.get("phase") == "done":
            rows.append(f"正解 {truth}: つかみ損ねて撮り直し、{rec['grip_retries']} 回目の再試行で回収")
    if not rows:
        rows.append("なし")
    bullets(d, rows, 90, y2 + 60, 1760, size=30, gap=10)
    d.text((80, 885), "読み違えた布片はすべて要確認へ回り、誤った置き場には 1 枚も置かれなかった",
           fill=ACCENT, font=font(32, bold=True))
    arm = "アームは学習した方策(ACT)" if ARM == "policy" else "アームは手続き的な把持(状態機械 + IK)"
    d.text((80, 935), f"{arm}。シミュレーション時間 {summary['sim_seconds']:.0f} 秒(MuJoCo、変形体の布)。"
           f"再試行 {summary.get('grasp_retries', 0)} 回。検証は置き場に布があるかをカメラで確認",
           fill=DIM, font=font(24))
    if ARM == "policy":
        d.text((80, 970), "同じ seed で手続き的な把持を使うと、置き場 5 + 要確認 3、失敗 0。把持の成否は物理の微小な差で揺れる",
               fill=DIM, font=font(24))
    return img


def slide_next() -> Image.Image:
    img, d = new_canvas()
    y = heading(d, "工夫した点・講座の学びの使い方・次の一歩")
    col_w = 560
    cols = [
        ("工夫した点", ACCENT, [
            "迷ったら止めずに人へ回す。誤配置ゼロを最優先にした",
            "平らな布を平行グリッパでつまむため、片方の指に 2.5 mm の唇を付け、押さえ指を止めて閉じる",
            "方策の失敗を 1 件ずつ再現して分解し、学習のやり直しでなく実行器の修正で 14 → 19 / 20",
        ]),
        ("講座の学びの使い方", OK, [
            "AI 画像認識 + ルールのアーム制御から始め、タスクの前提条件を先に決めた",
            "MLOps: 固定シードで成功率を評価、データとモデルは HF Hub、GitHub Actions でテスト",
            "ミドルウェアと OSS: dora-rs の dataflow、MuJoCo、LeRobot。Apache-2.0 で公開",
        ]),
        ("次の一歩", WARN, [
            "布は折れたまま落ちる。敷く置き方を教師に足したが、方策は 8 / 20 止まり。ここを伸ばす",
            "SO-101 など LeRobot 対応アームで RobotArm を実装し、実機のデモで同じ学習を回す",
        ]),
    ]
    x = 80
    for title, color, items in cols:
        d.text((x, y), title, fill=color, font=font(34, bold=True))
        bullets(d, items, x + 6, y + 64, col_w, size=28, gap=18)
        x += col_w + 40
    d.text((80, 910), "コード: github.com/katsuhisa91/pai-final-cloth-sort(Apache-2.0)", fill=FG, font=font(32, bold=True))
    return img


# ---------- 実行の様子(フレーム合成) ----------

def compose_frame(src: Image.Image) -> Image.Image:
    if src.size == (W, H):
        # MuJoCo 版のフレームは 1920x1080 で、タイトル・3D 視点・俯瞰カメラ・パネルを含む。そのまま使う
        img = src.convert("RGB")
        d = ImageDraw.Draw(img)
        arm = "  アームは学習方策(ACT)" if ARM == "policy" else ""
        if ARM == "policy":
            # 録画のフッタは手続き版の文言(アーム制御(IK))のままなので、方策版の流れに書き換える
            d.rectangle([0, 1046, 1230, H], fill=BG)
            d.text((30, 1052), "カメラ → 検出 → 番号読み取り(CNN) → 計画 → 学習方策(ACT)が手先目標を出力 → IK → 把持(物理) → 検証",
                   fill=DIM, font=font(17))
        if SUMMARY and img.crop((1250, 966, 1880, 979)).convert("L").getextrema()[1] > 150:
            # 録画の最終行は「検証 OK」の数え方に誤りがあった(要確認トレイの分も数えていた。pipeline.py は修正済み)
            s = SUMMARY
            d.rectangle([1250, 961, 1885, 983], fill=PANEL)
            d.text((1256, 960), f"置き場へ {s['placed_slot']} 枚(検証 OK {s['placed_slot_verified']})、"
                   f"要確認 {s['review'] + s.get('grasp_failed', 0)} 枚、所要 {s['sim_seconds']:.0f} s",
                   fill=FG, font=font(16))
        if SPEED > 1:
            # フレーム左上の「実時間 / real-time」表示を上書きして矛盾をなくす
            d.rectangle([126, 50, 900, 80], fill=BG)
            d.text((130, 54), f"{SPEED} 倍速再生(録画は等速)  MuJoCo(変形体の布 + 2 本指){arm}", fill=DIM, font=font(17))
            label = f"{SPEED} 倍速"
            f = font(36, bold=True)
            tw = d.textlength(label, font=f)
            d.rectangle([W - tw - 70, 16, W - 20, 70], fill=(200, 60, 40))
            d.text((W - tw - 45, 22), label, fill=(255, 255, 255), font=f)
        return img
    img, d = new_canvas()
    scaled = src.convert("RGB").resize((W, int(src.height * W / src.width)), Image.LANCZOS)
    top = 100
    img.paste(scaled, (0, top))
    d.text((60, 24), "実行の様子  同じ部品を dora-rs の 3 ノード(sim / brain / recorder)でも実行できる",
           fill=FG, font=font(34, bold=True))
    yb = top + scaled.height + 22
    f = font(28)
    d.rectangle([60, yb + 8, 84, yb + 32], outline=OK, width=3)
    d.text((100, yb), "確信度 0.6 以上 → 番号の置き場へ", fill=FG, font=f)
    d.rectangle([640, yb + 8, 664, yb + 32], outline=WARN, width=3)
    d.text((680, yb), "0.6 未満 → 要確認トレイ(赤枠)へ", fill=FG, font=f)
    d.text((1220, yb), "右パネル: AI 側(brain)の判断とログ", fill=DIM, font=f)
    return img


def video_frames(path: Path, step: int):
    """MP4 を 10 fps のフレームとして読み、step 枚おきに返す(2 なら 2 倍速)。"""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={FPS_IN},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    i = 0
    while True:
        raw = proc.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            break
        if i % step == 0:
            yield Image.frombytes("RGB", (W, H), raw)
        i += 1
    proc.wait()


def rss_frames(path: Path, results: list[dict], speed: int):
    """RoboSimStudio(Newton の VBD 布)で布をつまんで運んだ録画を、説明つきの 1920x1080 にして返す。"""
    cw, ch = 1280, 720
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={FPS_IN},scale={cw}:{ch}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    base, d = new_canvas()
    d.text((60, 40), "取り組みの途中で公開された RoboSimStudio でも試した", fill=FG, font=font(46, bold=True))
    d.text((60, 110), "UC Berkeley と NVIDIA が 2026 年 10 月に公開(Apache-2.0)。物理は NVIDIA の Newton、布は VBD で計算",
           fill=DIM, font=font(26))
    x0 = 60 + cw + 40
    lines = ["本体と同じ 5 cm 角の布・Panda・速度で、つまむ → 運ぶ → 放す",
             "布を指に固定しない。指の摩擦だけで持つ",
             "Colab の T4(GPU)で実行。GPU が無いと布をつまめない"]
    for bend, label in ((0.5, "やや柔らかい布"), (0.05, "柔らかい布")):
        rows = [r for r in results if r.get("bend") == bend]
        if rows:
            ok = sum(r["success"] for r in rows)
            lines.append(f"{label}: {ok} / {len(rows)} 回成功")
    lines.append("柔らかいほど垂れて本物らしいが、指から抜けやすい")
    bullets(d, lines, x0, 190, W - x0 - 40, size=26, gap=14)
    d.text((60, 190 + ch + 20), "スクリプト: experiments/robosimstudio/cloth_pinch.py", fill=DIM, font=font(24))
    d.rectangle([0, H - 60, W, H], fill=BG)  # 下の帯は MuJoCo の文言なので、この場面用に書き換える
    d.text((60, H - 50), f"PAI最終課題_{ACCOUNT}   {speed} 倍速再生   RoboSimStudio(Newton)シミュレーション", fill=DIM, font=font(22))
    label = f"{speed} 倍速"
    f = font(36, bold=True)
    tw = d.textlength(label, font=f)
    d.rectangle([W - tw - 70, 16, W - 20, 70], fill=(200, 60, 40))
    d.text((W - tw - 45, 22), label, fill=(255, 255, 255), font=f)
    i = 0
    while True:
        raw = proc.stdout.read(cw * ch * 3)
        if len(raw) < cw * ch * 3:
            break
        if i % speed == 0:
            im = base.copy()
            im.paste(Image.frombytes("RGB", (cw, ch), raw), (60, 170))
            yield im
        i += 1
    proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/dora/frames")
    ap.add_argument("--run-video", default=None, help="フレームの代わりに実行の録画(MP4、1920x1080)を使う")
    ap.add_argument("--arm", choices=["procedural", "policy"], default="procedural", help="録画のアームの種類(スライドの文言が変わる)")
    ap.add_argument("--rss-video", default=None, help="RoboSimStudio で布をつまんで運んだ録画(1280x720)")
    ap.add_argument("--rss-results", default=None, help="その評価結果(cloth_pinch.py の results.jsonl)")
    ap.add_argument("--log", default="runs/dora/log.jsonl")
    ap.add_argument("--summary", default="runs/mj4/summary.json")
    ap.add_argument("--metrics", default="models/metrics.json")
    ap.add_argument("--fig", default="runs/mj4/top_detect.png")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--run-speed", type=int, default=1, help="実行部分の倍速(2 なら 2 倍速。スライドと画面に明記する)")
    ap.add_argument("--out", default=f"runs/submission/PAI最終課題_{ACCOUNT}.mp4")
    ap.add_argument("--slides-dir", default="runs/submission/slides")
    args = ap.parse_args()

    global SEED, SPEED, ARM, SUMMARY
    SEED = args.seed
    SPEED = args.run_speed
    ARM = args.arm
    # 1920x1080 のフレームは 1 枚 6 MB あるので、全部をメモリに載せずに 1 枚ずつ流す
    if args.run_video:
        frames = video_frames(Path(args.run_video), max(1, args.run_speed))
    else:
        paths = sorted(Path(args.frames).glob("*.png"))[::max(1, args.run_speed)]
        if not paths:
            raise SystemExit(f"フレームがありません: {args.frames}(先に ./run_dora.sh を実行)")
        frames = (Image.open(p) for p in paths)
    summary = json.loads(Path(args.summary).read_text())
    SUMMARY = summary
    log = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    metrics = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else None

    slides: list[tuple[str, Image.Image, float]] = [
        ("01_title", slide_title(), 4.0),
        ("02_task", slide_task(Path(args.fig)), 5.0),
        ("03_system", slide_system(), 8.0),
        ("04_ai", slide_ai(metrics), 6.0),
    ]
    if ARM == "policy":
        slides.append(("05_policy", slide_policy(), 10.0))
    outro: list[tuple[str, Image.Image, float]] = [
        ("07_results", slide_results(summary, log), 7.0),
        ("09_next", slide_next(), 9.0),
    ]
    slides_dir = Path(args.slides_dir)
    slides_dir.mkdir(parents=True, exist_ok=True)
    for name, im, _ in slides + outro:
        im.save(slides_dir / f"{name}.png")

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
    for i, fr in enumerate(frames):
        im = compose_frame(fr)
        if i == 300:
            im.save(slides_dir / "06_run_sample.png")
        emit(im, 1.0 / FPS_IN)
    for name, im, sec in outro:
        if name == "09_next" and args.rss_video:
            rss_results = [json.loads(line) for line in Path(args.rss_results).read_text().splitlines() if line.strip()]
            for i, fr in enumerate(rss_frames(Path(args.rss_video), rss_results, SPEED or 2)):
                if i == 30:
                    fr.save(slides_dir / "08_robosimstudio.png")
                emit(fr, 1.0 / FPS_IN)
        emit(im, sec)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg が失敗しました")
    print(f"{out}: {n} frames, {n / FPS_IN:.1f} s")


if __name__ == "__main__":
    main()
