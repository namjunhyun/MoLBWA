#!/usr/bin/env python3
"""시선 씬 픽셀 캘리브용 표적 시트 — A4 네 귀퉁이 태그 + 가운데 응시점 (2026-10-30).

왜 태그 한 장의 중심이 아니라 따로 찍은 점을 보나
--------------------------------------------------
70mm 태그는 0.5m 에서 시야각 약 8° — "중심"을 어디로 보느냐가 1° 이상 흔들린다.
목표 정확도가 1.5° 라 그대로 오차가 된다. 그래서 작은 과녁(지름 10mm, 1.2m 에서 0.5°)을
보게 하고, 그 픽셀 위치는 귀퉁이 태그로 계산한다. 태그와 과녁의 상대 위치는 인쇄 치수로
알고 있으므로 네 장 중 한 장만 보여도 된다.

규격: tag36h11, 60mm (씬 카메라 fx ≈ 461px -> 1.2m 에서 약 23px, 검출 하한 약 20px 위),
      ID 20~23 (클립보드 10~13, 책상 14~17, 옛 앵커 0~4 와 겹치지 않게).

    python3 tools/make_gaze_target_sheet.py
결과: calibration/gaze_target_sheet_60mm.pdf + _layout.yaml (과녁 기준 태그 중심 좌표)
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from make_apriltag_sheet import A4_H_MM, A4_W_MM, TAG_CELLS, mm2px, render_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mm", type=float, default=60.0, help="검은 사각형 한 변(mm)")
    ap.add_argument("--ids", type=int, nargs=4, default=[20, 21, 22, 23],
                    help="좌상, 우상, 좌하, 우하")
    ap.add_argument("--dx-mm", type=float, default=55.0, help="과녁 -> 태그 중심 가로 거리")
    ap.add_argument("--dy-mm", type=float, default=75.0, help="과녁 -> 태그 중심 세로 거리")
    ap.add_argument("--dot-mm", type=float, default=10.0, help="과녁 지름")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.abspath(args.out or os.path.join(
        here, "..", "calibration", f"gaze_target_sheet_{int(args.size_mm)}mm.pdf"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    dpi = args.dpi
    page = Image.new("L", (mm2px(A4_W_MM, dpi), mm2px(A4_H_MM, dpi)), 255)
    draw = ImageDraw.Draw(page)
    quiet_mm = args.size_mm / TAG_CELLS          # 흰 여백 1셀
    half = args.size_mm / 2 + quiet_mm
    cx, cy = A4_W_MM / 2, A4_H_MM / 2 - 5.0      # 아래쪽에 자선 자리를 남긴다
    offsets = [(-args.dx_mm, -args.dy_mm), (args.dx_mm, -args.dy_mm),
               (-args.dx_mm, args.dy_mm), (args.dx_mm, args.dy_mm)]
    for (ox, oy) in offsets:
        if not (8 <= cx + ox - half and cx + ox + half <= A4_W_MM - 8
                and 18 <= cy + oy - half and cy + oy + half <= A4_H_MM - 34):
            raise SystemExit("A4 에 안 들어간다 — --dx-mm/--dy-mm/--size-mm 를 줄일 것")

    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                  mm2px(3.5, dpi))
    except OSError:
        font = ImageFont.load_default()

    for tid, (ox, oy) in zip(args.ids, offsets):
        tag_img, _ = render_tag(tid, mm2px(args.size_mm, dpi))
        x0, y0 = cx + ox - args.size_mm / 2, cy + oy - args.size_mm / 2
        page.paste(Image.fromarray(tag_img), (mm2px(x0, dpi), mm2px(y0, dpi)))
        # 라벨은 흰 여백 바깥에
        ly = y0 - quiet_mm - 5.0 if oy < 0 else y0 + args.size_mm + quiet_mm + 1.0
        draw.text((mm2px(x0, dpi), mm2px(ly, dpi)), f"id={tid}  {args.size_mm:.0f}mm",
                  fill=0, font=font)

    # 과녁: 검은 원 + 가운데 흰 점 (눈을 한 점에 고정시키기 쉽게)
    r, r_in = args.dot_mm / 2, args.dot_mm * 0.15
    draw.ellipse([mm2px(cx - r, dpi), mm2px(cy - r, dpi), mm2px(cx + r, dpi), mm2px(cy + r, dpi)],
                 fill=0)
    draw.ellipse([mm2px(cx - r_in, dpi), mm2px(cy - r_in, dpi),
                  mm2px(cx + r_in, dpi), mm2px(cy + r_in, dpi)], fill=255)

    draw.text((mm2px(10, dpi), mm2px(6, dpi)),
              f"MoLBWA 시선 캘리브 표적 — tag36h11 {args.size_mm:.0f}mm, id {args.ids}, "
              "가운데 점을 응시", fill=0, font=font)

    # 인쇄 배율 검증용 100mm 자선
    ry, x_start = A4_H_MM - 26.0, 20.0
    draw.line([mm2px(x_start, dpi), mm2px(ry, dpi), mm2px(x_start + 100, dpi), mm2px(ry, dpi)],
              fill=0, width=max(2, dpi // 150))
    for k in range(11):
        h = 4.0 if k % 5 == 0 else 2.5
        x = x_start + k * 10.0
        draw.line([mm2px(x, dpi), mm2px(ry - h, dpi), mm2px(x, dpi), mm2px(ry, dpi)],
                  fill=0, width=max(2, dpi // 200))
    draw.text((mm2px(x_start, dpi), mm2px(ry + 2, dpi)),
              "이 선이 정확히 100mm 여야 한다 (인쇄 설정: 실제 크기 / 100%). "
              f"검은 사각형 한 변 {args.size_mm:.0f}mm 도 확인.", fill=0, font=font)

    page.save(out, "PDF", resolution=float(dpi))
    layout = os.path.splitext(out)[0] + "_layout.yaml"
    with open(layout, "w") as f:
        f.write("# 시선 표적 시트: 과녁(가운데 점) 원점, x = 오른쪽, y = 아래, 단위 m. 태그 중심.\n")
        f.write(f"tag_family: tag36h11\ntag_size_m: {args.size_mm / 1000:.4f}\ntags:\n")
        for tid, (ox, oy) in zip(args.ids, offsets):
            f.write(f"  {tid}: [{ox / 1000:.4f}, {oy / 1000:.4f}]\n")
    print(f"[생성] {out}\n[생성] {layout}")


if __name__ == "__main__":
    main()
