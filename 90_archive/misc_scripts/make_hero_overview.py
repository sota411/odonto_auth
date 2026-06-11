# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "proposal" / "figures"


def font(path: str, size: int):
    return ImageFont.truetype(path, size)


FONT_TITLE = font(r"C:\Windows\Fonts\meiryob.ttc", 34)
FONT_HEAD = font(r"C:\Windows\Fonts\meiryob.ttc", 26)
FONT_BODY = font(r"C:\Windows\Fonts\meiryo.ttc", 19)
FONT_SMALL = font(r"C:\Windows\Fonts\meiryo.ttc", 16)
FONT_TINY = font(r"C:\Windows\Fonts\meiryo.ttc", 14)


INPUT_IMG = Image.open(FIG_DIR / "v6_example_input.png").convert("RGB")
PRED_IMG = Image.open(FIG_DIR / "v6_example_pred.jpg").convert("RGB")


TOOTH_BOX = (200, 276, 264, 361)  # R1 around the center


COLORS = {
    "bg": "#F4F7FB",
    "panel": "#FFFFFF",
    "panel_edge": "#C9D4E0",
    "blue_badge": "#DDEEFF",
    "blue_text": "#1E3F66",
    "orange_badge": "#FBE4D6",
    "orange_text": "#7A4620",
    "red_badge": "#FBE1DE",
    "red_text": "#8C3028",
    "ink": "#203243",
    "sub": "#53677A",
    "accent": "#3E6E96",
    "line": "#BE5147",
    "soft_red": "#FFF4F2",
    "soft_blue": "#EEF5FC",
    "soft_gray": "#F7F9FC",
    "score_green": "#36A269",
    "r1": "#2D57FF",
}


def rounded_panel(draw: ImageDraw.ImageDraw, box, title, badge, badge_fill):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=COLORS["panel"], outline=COLORS["panel_edge"], width=3)
    draw.rounded_rectangle((x0 + 16, y0 + 16, x0 + 150, y0 + 54), radius=14, fill=badge_fill)
    draw.text((x0 + 32, y0 + 24), badge, font=FONT_SMALL, fill=COLORS["sub"])
    draw.text((x0 + 20, y0 + 72), title, font=FONT_HEAD, fill=COLORS["ink"])


def arrow(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, color=None, width=7, head=16):
    color = color or COLORS["accent"]
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    draw.polygon(
        [(x2, y2), (x2 - head, y2 - head // 2), (x2 - head, y2 + head // 2)],
        fill=color,
    )


def multiline(draw: ImageDraw.ImageDraw, xy, text, font_obj, fill, spacing=6):
    draw.multiline_text(xy, text, font=font_obj, fill=fill, spacing=spacing)


def draw_phone(canvas: Image.Image, draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    phone = (x0 + 56, y0 + 118, x1 - 56, y1 - 88)
    draw.rounded_rectangle(phone, radius=30, fill="#1F2C39", outline="#1F2C39")
    draw.rounded_rectangle((phone[0] + 16, phone[1] + 16, phone[2] - 16, phone[3] - 16), radius=20, fill="#EFF2F6")
    screen = ImageOps.fit(INPUT_IMG, (phone[2] - phone[0] - 32, phone[3] - phone[1] - 76))
    canvas.paste(screen, (phone[0] + 16, phone[1] + 46))
    draw.rounded_rectangle((phone[0] + 90, phone[1] + 20, phone[2] - 90, phone[1] + 31), radius=5, fill="#97A3B0")
    multiline(
        draw,
        (x0 + 24, y1 - 74),
        "重要操作のときだけ使う.\n口元をガイドに合わせて撮影.",
        FONT_SMALL,
        COLORS["sub"],
    )


def draw_detection(canvas: Image.Image, draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    img = ImageOps.contain(PRED_IMG, (x1 - x0 - 34, 252))
    canvas.paste(img, (x0 + (x1 - x0 - img.width) // 2, y0 + 124))
    multiline(
        draw,
        (x0 + 18, y1 - 72),
        "R1, R2, R3, L1, L2, L3 を認識し,\n歯種ごとに領域を切り出す.",
        FONT_SMALL,
        COLORS["sub"],
    )


def make_feature_concept() -> Image.Image:
    crop = INPUT_IMG.crop((TOOTH_BOX[0] - 10, TOOTH_BOX[1] - 10, TOOTH_BOX[2] + 10, TOOTH_BOX[3] + 10))
    canvas = Image.new("RGB", (640, 300), COLORS["soft_blue"])
    draw = ImageDraw.Draw(canvas)

    left = (16, 28, 186, 272)
    mid = (212, 28, 428, 272)
    right = (458, 50, 620, 250)

    draw.rounded_rectangle(left, radius=18, fill="white", outline=COLORS["panel_edge"], width=2)
    fit_left = ImageOps.contain(crop, (left[2] - left[0] - 24, left[3] - left[1] - 56))
    canvas.paste(fit_left, (left[0] + (left[2] - left[0] - fit_left.width) // 2, left[1] + 42))
    draw.text((left[0] + 16, left[1] + 10), "切り出した歯牙", font=FONT_SMALL, fill=COLORS["ink"])

    draw.rounded_rectangle(mid, radius=18, fill="white", outline=COLORS["panel_edge"], width=2)
    fit_mid = ImageOps.contain(crop, (mid[2] - mid[0] - 26, mid[3] - mid[1] - 52))
    paste_x = mid[0] + (mid[2] - mid[0] - fit_mid.width) // 2
    paste_y = mid[1] + 36
    canvas.paste(fit_mid, (paste_x, paste_y))
    draw.text((mid[0] + 16, mid[1] + 10), "見る形状", font=FONT_SMALL, fill=COLORS["ink"])

    fx0 = paste_x
    fy0 = paste_y
    fx1 = paste_x + fit_mid.width
    fy1 = paste_y + fit_mid.height
    # width arrow
    draw.line((fx0, fy1 + 12, fx1, fy1 + 12), fill=COLORS["line"], width=3)
    draw.polygon([(fx0, fy1 + 12), (fx0 + 8, fy1 + 7), (fx0 + 8, fy1 + 17)], fill=COLORS["line"])
    draw.polygon([(fx1, fy1 + 12), (fx1 - 8, fy1 + 7), (fx1 - 8, fy1 + 17)], fill=COLORS["line"])
    draw.text((fx0 + 32, fy1 + 16), "幅", font=FONT_TINY, fill=COLORS["line"])
    # height arrow
    draw.line((fx1 + 12, fy0, fx1 + 12, fy1), fill=COLORS["line"], width=3)
    draw.polygon([(fx1 + 12, fy0), (fx1 + 7, fy0 + 8), (fx1 + 17, fy0 + 8)], fill=COLORS["line"])
    draw.polygon([(fx1 + 12, fy1), (fx1 + 7, fy1 - 8), (fx1 + 17, fy1 - 8)], fill=COLORS["line"])
    draw.text((fx1 + 18, fy0 + 28), "高さ", font=FONT_TINY, fill=COLORS["line"])
    # contour callout
    contour_pt = (fx0 + fit_mid.width // 2, fy0 + 18)
    draw.line((contour_pt[0], contour_pt[1], contour_pt[0] - 36, contour_pt[1] - 28), fill=COLORS["line"], width=2)
    draw.ellipse((contour_pt[0] - 4, contour_pt[1] - 4, contour_pt[0] + 4, contour_pt[1] + 4), fill=COLORS["line"])
    draw.text((contour_pt[0] - 100, contour_pt[1] - 56), "輪郭", font=FONT_TINY, fill=COLORS["line"])
    edge_pt = (fx0 + fit_mid.width // 2 - 8, fy1 - 14)
    draw.line((edge_pt[0], edge_pt[1], edge_pt[0] - 52, edge_pt[1] + 28), fill=COLORS["line"], width=2)
    draw.ellipse((edge_pt[0] - 4, edge_pt[1] - 4, edge_pt[0] + 4, edge_pt[1] + 4), fill=COLORS["line"])
    draw.text((edge_pt[0] - 128, edge_pt[1] + 26), "切縁の形", font=FONT_TINY, fill=COLORS["line"])

    draw.rounded_rectangle(right, radius=18, fill="white", outline=COLORS["panel_edge"], width=2)
    draw.text((right[0] + 18, right[1] + 12), "局所形状特徴", font=FONT_SMALL, fill=COLORS["ink"])
    vec_lines = ["幅 = 0.82", "高さ = 0.91", "輪郭曲率 = 0.34", "切縁形状 = 0.57"]
    for idx, line in enumerate(vec_lines):
        draw.rounded_rectangle(
            (right[0] + 18, right[1] + 46 + idx * 34, right[0] + 140, right[1] + 72 + idx * 34),
            radius=10,
            fill=COLORS["soft_red"],
            outline=None,
        )
        draw.text((right[0] + 30, right[1] + 52 + idx * 34), line, font=FONT_TINY, fill=COLORS["red_text"])

    arrow(draw, left[2] + 10, 150, mid[0] - 10, 150, color=COLORS["accent"], width=5, head=14)
    arrow(draw, mid[2] + 10, 150, right[0] - 10, 150, color=COLORS["accent"], width=5, head=14)
    return canvas


def save_feature_concept():
    img = make_feature_concept()
    out = FIG_DIR / "local_shape_feature_concept.png"
    img.save(out)
    return out


def draw_feature_stage(canvas: Image.Image, draw: ImageDraw.ImageDraw, box, concept: Image.Image):
    x0, y0, x1, y1 = box
    img = ImageOps.contain(concept, (x1 - x0 - 22, 248))
    canvas.paste(img, (x0 + (x1 - x0 - img.width) // 2, y0 + 128))
    multiline(
        draw,
        (x0 + 18, y1 - 72),
        "歯ごとの輪郭, 幅, 高さなどを\n照合用の特徴へ変換する.",
        FONT_SMALL,
        COLORS["sub"],
    )


def draw_compare_stage(draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    left = (x0 + 28, y0 + 146, x0 + 122, y0 + 276)
    right = (x0 + 148, y0 + 146, x1 - 28, y0 + 276)
    for rect, title in ((left, "今回"), (right, "登録済み")):
        draw.rounded_rectangle(rect, radius=18, fill=COLORS["soft_gray"], outline=COLORS["panel_edge"], width=2)
        draw.text((rect[0] + 14, rect[1] + 10), title, font=FONT_SMALL, fill=COLORS["ink"])
    bars_left = [0.76, 0.58, 0.83, 0.41]
    bars_right = [0.72, 0.60, 0.80, 0.39]
    for i, (lv, rv) in enumerate(zip(bars_left, bars_right)):
        y = left[1] + 42 + i * 20
        draw.rounded_rectangle((left[0] + 16, y, left[0] + 16 + int(52 * lv), y + 12), radius=6, fill="#7AA7D1")
        draw.rounded_rectangle((right[0] + 16, y, right[0] + 16 + int(52 * rv), y + 12), radius=6, fill="#7AA7D1")
        draw.text((x0 + 122, y - 4), "≒", font=FONT_SMALL, fill=COLORS["accent"])
    arrow(draw, x0 + 126, y0 + 210, x0 + 144, y0 + 210, color=COLORS["accent"], width=4, head=12)
    draw.rounded_rectangle((x0 + 54, y0 + 302, x1 - 54, y0 + 366), radius=16, fill=COLORS["soft_blue"], outline=COLORS["panel_edge"], width=2)
    draw.text((x0 + 86, y0 + 322), "特徴の近さを比較", font=FONT_SMALL, fill=COLORS["blue_text"])
    multiline(
        draw,
        (x0 + 18, y1 - 72),
        "今回の特徴と登録済み特徴を比べ,\n本人らしさを計算する.",
        FONT_SMALL,
        COLORS["sub"],
    )


def draw_score_stage(draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    gauge = (x0 + 34, y0 + 148, x1 - 34, y0 + 248)
    draw.rounded_rectangle(gauge, radius=20, fill=COLORS["soft_gray"], outline=COLORS["panel_edge"], width=2)
    draw.text((gauge[0] + 18, gauge[1] + 14), "一致スコア", font=FONT_SMALL, fill=COLORS["ink"])
    bar = (gauge[0] + 18, gauge[1] + 46, gauge[1] + 18 + 152, gauge[1] + 72)
    draw.rounded_rectangle((gauge[0] + 18, gauge[1] + 46, gauge[0] + 170, gauge[1] + 72), radius=12, fill="#DDE8D8")
    draw.rounded_rectangle((gauge[0] + 18, gauge[1] + 46, gauge[0] + 18 + 122, gauge[1] + 72), radius=12, fill=COLORS["score_green"])
    draw.text((gauge[0] + 26, gauge[1] + 80), "0.93 / 一致", font=FONT_HEAD, fill=COLORS["score_green"])
    draw.rounded_rectangle((x0 + 58, y0 + 290, x1 - 58, y0 + 360), radius=18, fill=COLORS["soft_red"], outline="#E4B2AC", width=2)
    draw.text((x0 + 76, y0 + 316), "追加認証 OK", font=FONT_SMALL, fill=COLORS["red_text"])
    multiline(
        draw,
        (x0 + 18, y1 - 72),
        "一致スコアを返し,\n重要操作の追加認証に使う.",
        FONT_SMALL,
        COLORS["sub"],
    )


def make_hero_overview():
    concept_path = save_feature_concept()
    concept = Image.open(concept_path).convert("RGB")
    canvas = Image.new("RGB", (1960, 980), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    draw.text((46, 28), "スマートフォン撮影から一致スコア出力までの流れ", font=FONT_TITLE, fill=COLORS["ink"])
    multiline(
        draw,
        (48, 72),
        "現時点では 2 まで確認済み. 未踏では 3 から 5 を実写条件で詰める.",
        FONT_BODY,
        COLORS["sub"],
    )

    panels = [
        ((40, 130, 380, 830), "1. スマートフォンで口腔内を撮影", "入力", COLORS["orange_badge"]),
        ((420, 130, 760, 830), "2. 撮影画像から歯牙を抽出", "現時点", COLORS["blue_badge"]),
        ((800, 130, 1140, 830), "3. 各歯牙から局所形状特徴を抽出", "未踏", COLORS["red_badge"]),
        ((1180, 130, 1520, 830), "4. 登録済みの本人データと照合", "未踏", COLORS["red_badge"]),
        ((1560, 130, 1900, 830), "5. 一致スコアを出力", "未踏", COLORS["red_badge"]),
    ]

    for box, title, badge, fill in panels:
        rounded_panel(draw, box, title, badge, fill)

    draw_phone(canvas, draw, panels[0][0])
    draw_detection(canvas, draw, panels[1][0])
    draw_feature_stage(canvas, draw, panels[2][0], concept)
    draw_compare_stage(draw, panels[3][0])
    draw_score_stage(draw, panels[4][0])

    for idx in range(4):
        left = panels[idx][0]
        right = panels[idx + 1][0]
        arrow(draw, left[2] + 12, 476, right[0] - 12, 476)

    draw.rounded_rectangle((40, 866, 1900, 944), radius=20, fill="#E8F0F8", outline="#B8CADB", width=2)
    multiline(
        draw,
        (70, 890),
        "現時点: 擬似口腔内画像で前歯6クラスの認識と, 各歯牙の個別切り出しまで確認済み.\n未踏: 実写条件で局所形状特徴, 照合, 一致スコア出力, FAR / FRR / EER 評価までつなぐ.",
        FONT_BODY,
        COLORS["ink"],
        spacing=8,
    )

    out = FIG_DIR / "hero_overview.png"
    canvas.save(out)
    return out


if __name__ == "__main__":
    print(save_feature_concept())
    print(make_hero_overview())
