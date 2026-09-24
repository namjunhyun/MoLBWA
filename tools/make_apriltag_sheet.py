#!/usr/bin/env python3
"""AprilTag(tag36h11) 인쇄 시트 생성 — 물리 치수가 정확한 PDF.

왜 직접 만드는가
----------------
태그의 물리적 한 변 길이가 자세 추정의 절대 스케일을 결정한다. 6cm 태그를
6.3cm 로 잘못 알면 거리가 5% 틀리고, 그대로 로봇팔 좌표가 된다. 그래서
  - PDF 에 DPI 를 박아 프린터가 원본 크기로 찍게 하고,
  - 인쇄물에 검증용 100mm 자선을 같이 찍는다.
프린터는 "맞춤(fit to page)"으로 조용히 축소하는 일이 흔하다. 반드시 자로 확인할 것.

크기 규약 (중요)
---------------
AprilTag 의 tag_size 는 **검은 테두리의 바깥 변 길이**다. 흰 여백은 포함하지 않는다.
cv2.aruco 가 생성하는 이미지가 정확히 그 검은 사각형(tag36h11 = 8x8 셀)이므로,
그 이미지를 tag_size 만큼 인쇄하면 된다. 검출기(pupil_apriltags)에 넘기는
tag_size 와 이 값이 같아야 한다.

여백은 검출 신뢰도를 위해 최소 1셀(=tag_size/8) 이상 둔다. 여기서는 2셀을 준다.

사용:
    python3 make_apriltag_sheet.py                 # 60mm 태그 4개, A4
    python3 make_apriltag_sheet.py --size-mm 80    # 더 크게 (멀리서 볼 때)
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A4_W_MM, A4_H_MM = 210.0, 297.0
QUIET_CELLS = 2          # 검은 사각형 바깥 흰 여백(셀 단위). tag36h11 은 8셀이 검은 사각형.
TAG_CELLS = 8


def mm2px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def render_tag(tag_id, side_px):
    """검은 사각형만. 여백 없음 — 여백은 호출자가 배치하면서 준다."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # side_px 는 8셀이 정확히 나눠떨어져야 셀 경계가 흐려지지 않는다.
    side_px = (side_px // TAG_CELLS) * TAG_CELLS
    # OpenCV 4.7 에서 drawMarker -> generateImageMarker 로 이름이 바뀌었다.
    gen = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
    img = gen(d, tag_id, side_px)
    return img, side_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mm", type=float, default=60.0,
                    help="검은 사각형 한 변(mm). 검출 시 tag_size 로 넘길 값과 같아야 한다.")
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "..", "calibration",
                                   f"apriltag_36h11_{int(args.size_mm)}mm.pdf")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    dpi = args.dpi
    W, H = mm2px(A4_W_MM, dpi), mm2px(A4_H_MM, dpi)
    page = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(page)

    cell_mm = args.size_mm / TAG_CELLS
    quiet_mm = cell_mm * QUIET_CELLS
    block_mm = args.size_mm + 2 * quiet_mm      # 태그 + 여백 한 덩어리

    n = len(args.ids)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    if block_mm * cols > A4_W_MM - 20 or block_mm * rows > A4_H_MM - 60:
        raise SystemExit(f"A4 에 안 들어간다: 덩어리 {block_mm:.1f}mm x {cols}x{rows}. "
                         f"--size-mm 를 줄이거나 --ids 개수를 줄일 것.")

    # 배치: 위쪽에 제목, 아래쪽에 검증용 자선을 남긴다.
    top_mm = 22.0
    gap_mm = (A4_W_MM - block_mm * cols) / (cols + 1)
    centers_mm = []
    for i, tid in enumerate(args.ids):
        r, c = divmod(i, cols)
        x0 = gap_mm + c * (block_mm + gap_mm)
        y0 = top_mm + r * (block_mm + 8.0)
        tag_img, side_px = render_tag(tid, mm2px(args.size_mm, dpi))
        tx = mm2px(x0 + quiet_mm, dpi)
        ty = mm2px(y0 + quiet_mm, dpi)
        page.paste(Image.fromarray(tag_img), (tx, ty))
        # 태그 바깥 여백 경계(잘라 붙일 때 기준). 아주 얇은 회색 선 — 검출에 영향 없음.
        draw.rectangle([mm2px(x0, dpi), mm2px(y0, dpi),
                        mm2px(x0 + block_mm, dpi), mm2px(y0 + block_mm, dpi)],
                       outline=200, width=max(1, dpi // 300))
        # 라벨은 여백 바깥에 — 흰 여백 안에 글자가 들어가면 검출이 흔들린다.
        label_y = mm2px(y0 + block_mm + 1.5, dpi)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", mm2px(4.0, dpi))
        except OSError:
            font = ImageFont.load_default()
        draw.text((mm2px(x0, dpi), label_y),
                  f"tag36h11  id={tid}  {args.size_mm:.0f}mm", fill=0, font=font)
        # 태그 중심의 페이지 좌표(좌상단 원점, mm). 번들 배치 파일에 쓴다.
        centers_mm.append((tid, x0 + block_mm / 2, y0 + block_mm / 2))

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", mm2px(5.0, dpi))
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", mm2px(3.5, dpi))
    except OSError:
        title_font = small_font = ImageFont.load_default()

    draw.text((mm2px(10, dpi), mm2px(8, dpi)),
              f"MoLBWA  AprilTag tag36h11  tag_size = {args.size_mm:.0f} mm",
              fill=0, font=title_font)

    # --- 인쇄 배율 검증용 자선 ---
    # 프린터의 "맞춤" 축소를 잡아내는 유일한 방법. 100mm 로 안 나오면 전부 무효다.
    ruler_y = A4_H_MM - 28.0
    x_start = 20.0
    draw.line([mm2px(x_start, dpi), mm2px(ruler_y, dpi),
               mm2px(x_start + 100.0, dpi), mm2px(ruler_y, dpi)],
              fill=0, width=max(2, dpi // 150))
    for k in range(11):
        x = x_start + k * 10.0
        h = 4.0 if k % 5 == 0 else 2.5
        draw.line([mm2px(x, dpi), mm2px(ruler_y - h, dpi),
                   mm2px(x, dpi), mm2px(ruler_y, dpi)],
                  fill=0, width=max(2, dpi // 200))
    draw.text((mm2px(x_start, dpi), mm2px(ruler_y + 2.0, dpi)),
              "이 선이 정확히 100mm 여야 한다. 아니면 인쇄 배율이 틀린 것 "
              "(인쇄 설정에서 '실제 크기 / 100% / no scaling' 선택).",
              fill=0, font=small_font)
    draw.text((mm2px(x_start, dpi), mm2px(ruler_y + 7.0, dpi)),
              f"검은 사각형 한 변도 자로 확인: {args.size_mm:.0f}mm (흰 여백 제외)",
              fill=0, font=small_font)

    page.save(out, "PDF", resolution=float(dpi))
    print(f"[생성] {out}")
    print(f"  태그 {len(args.ids)}개 id={args.ids}, 한 변 {args.size_mm:.0f}mm, "
          f"흰 여백 {quiet_mm:.1f}mm, {dpi}dpi")

    # ★ 이 시트는 "잘라서 계단형으로 붙이기" 위한 것이다. 통째로 붙이면 안 된다.
    #
    # 태그가 전부 한 평면에 있으면 PnP 에 거울상 해가 생긴다. 2026-08-19 기록:
    # 재투영 오차 0.000px 인데 팔 베이스가 120mm(= 판 깊이의 2배) 틀린 해가 나왔다.
    # 재투영 게이트로는 절대 못 거른다 — 기하학적으로 완벽한 해이기 때문이다.
    # 유일한 방어는 물리적으로 깊이 차를 주는 것이다.
    layout = os.path.splitext(out)[0] + "_bundle_template.yaml"
    with open(layout, "w") as f:
        f.write(
            "# arm/config.yaml 의 anchor.bundle 에 붙여넣을 템플릿.\n"
            "#\n"
            "# ★ 이 값들은 자리표시자다. 실제로 붙인 위치를 자로 재서 바꿔야 한다.\n"
            "#\n"
            "# 좌표계: armbase (x=팔 정면, y=왼쪽, z=위, 원점=베이스 바닥 pan 축).\n"
            "# pos = 태그 중심. 태그 평면 법선은 +x(팔 정면, 즉 사용자 쪽)를 향한다.\n"
            "#\n"
            "# ★★ 반드시 비평면으로 붙일 것. 아래는 뒤판 2장(x=-0.060) + 앞단 2장(x=-0.010)\n"
            "#    의 계단 배치다. 네 장을 한 평면에 붙이면 재투영 오차 0.000px 짜리\n"
            "#    거울상 해가 나와서 팔 베이스가 120mm 틀린다 (실제로 겪음).\n"
            "#    두 단의 깊이 차는 최소 30mm 이상 줄 것.\n"
            "anchor:\n"
            f"  tag_family: tag36h11\n"
            f"  tag_size_m: {args.size_mm/1000:.4f}   # MEASURE: 인쇄 후 검은 테두리 바깥 한 변\n"
            "  bundle:\n")
        # config.yaml 의 기존 계단 배치를 그대로 따른다 (id 순서도 맞춘다).
        staircase = [(0, [-0.060,  0.055, 0.105]),
                     (1, [-0.060, -0.055, 0.105]),
                     (2, [-0.010,  0.070, 0.035]),
                     (3, [-0.010, -0.070, 0.035])]
        for tid, pos in staircase:
            if tid not in args.ids:
                continue
            step = "뒤판" if pos[0] < -0.03 else "앞단"
            f.write(f"    - {{id: {tid}, pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]}}"
                    f"   # {step}, MEASURE\n")
        f.write("  tag_axes:\n    right: [0, 1, 0]\n    up:    [0, 0, 1]\n")
    print(f"[생성] {layout}")
    print("  ★ 시트를 통째로 붙이지 말 것 — 4장을 잘라 두 단(깊이차 30mm+)으로 붙여야 한다.")
    print("    평면 배치는 재투영 오차 0px 짜리 거울상 해를 만든다 (팔 베이스 120mm 오차).")


if __name__ == "__main__":
    main()
