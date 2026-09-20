#!/usr/bin/env python3
"""
통합 GUI 화면 구성.

`control_panel_node` 에서 **그리기만** 떼어냈습니다. ROS 의존이 없어서
하드웨어나 rclpy 없이도 화면을 그려볼 수 있습니다:

    python -m gaze_hri.panel_render preview.png     # 미리보기 이미지 생성

레이아웃

    ┌─────────────────────────────┬──────────────┐
    │ 상태 배지 / 안내      개수 fps│              │
    ├─────────────────────────────┤  파이프라인   │
    │                             │  선택        │
    │      카메라 영상 + 오버레이   │  로봇        │
    │                             │  준비 상태    │
    ├─────────────────────────────┴──────────────┤
    │ 키 안내 또는 알림                            │
    └────────────────────────────────────────────┘
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from gaze_hri import hud as H
from gaze_hri.hud import Hud

HEADER_H = 46
FOOTER_H = 30
BG = (18, 17, 16)
PANEL_BG = (26, 24, 22)

STAGES = [("시선", "Gaze"), ("응시", "Dwell"), ("목표", "Target"), ("로봇", "Robot")]


@dataclass
class PanelState:
    """화면에 그릴 것 전부. 노드가 이걸 채워서 넘긴다."""

    task_state: str = "IDLE"
    source: str = "mouse"
    objects: list = field(default_factory=list)   # [((u,v), (x,y), contour), ...]
    pointer_px: tuple = None          # 화면 위 시선/커서 픽셀
    pointer_holding: bool = False
    hold_frac: float = 0.0            # 마우스 클릭 고정 진행률
    dwell_progress: float = 0.0
    pick_target: tuple = None         # 로봇 (x, y)
    place_target: tuple = None
    target_info: dict = None
    tcp: tuple = None                 # 팔 끝 로봇 (x, y, z)
    arm_phase: str = ""
    has_fixation: bool = False
    detect_on: bool = True
    calibrated: bool = False
    reproj_error_mm: float = None
    fps: float = 0.0
    notice: str = ""
    sidebar_w: int = 320


def _is_same(a, b, tol=0.02):
    return a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) < tol


def render(frame, st, hud, homography):
    """한 프레임을 그려서 캔버스를 돌려준다."""
    view = frame.copy()
    _draw_objects(view, st, hud)
    _draw_targets(view, st, hud, homography)
    _draw_arm_tip(view, st, hud, homography)
    _draw_pointer(view, st, hud, homography)

    vh, vw = view.shape[:2]
    canvas = np.full((HEADER_H + vh + FOOTER_H, vw + st.sidebar_w, 3), BG, np.uint8)
    canvas[HEADER_H:HEADER_H + vh, 0:vw] = view

    _draw_header(canvas, st, hud)
    _draw_sidebar(canvas, vw, HEADER_H, vh, st, hud)
    _draw_footer(canvas, HEADER_H + vh, st, hud)
    return canvas


# ----------------------------------------------------------------------
# 영상 위 오버레이
# ----------------------------------------------------------------------
def _object_radius(contour, default=20):
    """컵의 화면상 반지름. 마커와 번호 배지를 컵 밖에 놓기 위해 쓴다."""
    if contour is None:
        return default
    (_, _), r = cv2.minEnclosingCircle(contour)
    return max(default, float(r))


def _draw_objects(view, st, hud):
    """검출된 컵. 선택된 것만 강조하고 나머지는 은은하게 둔다."""
    for i, ((u, v), (x, y), contour) in enumerate(st.objects, 1):
        if _is_same(st.pick_target, (x, y)):
            continue                       # 선택된 컵은 _draw_targets 에서 그린다
        r = _object_radius(contour)
        if contour is not None:
            cv2.drawContours(view, [contour], -1, H.OBJ, 1, cv2.LINE_AA)
        hud.marker(view, (u, v), H.OBJ, radius=int(r + 6), thickness=1, dashed=True)
        off = (r + 16) * 0.707             # 대각선으로 컵 바깥에
        hud.badge(view, (u + off, v - off), str(i), H.OBJ, radius=10)


def _draw_targets(view, st, hud, homo):
    """선택 결과를 영상 위에 되돌려 그린다 (로봇 좌표 -> 픽셀)."""
    if st.pick_target:
        px = homo.to_pixel(*st.pick_target)
        if px:
            # 선택된 컵의 실제 크기에 맞춰 링을 두른다
            r = 22.0
            for ((ou, ov), (ox, oy), contour) in st.objects:
                if _is_same(st.pick_target, (ox, oy)):
                    r = _object_radius(contour)
                    break
            hud.marker(view, px, H.PICK, radius=int(r + 8), thickness=3)
            hud.marker(view, px, H.PICK, radius=int(r + 2), thickness=1)
            cv2.circle(view, (int(px[0]), int(px[1])), 3, H.PICK, -1, cv2.LINE_AA)
            hud.chip(view, (px[0], px[1] - r - 14), hud.label("집을 것", "PICK"),
                     H.PICK, 13, bold=True, anchor="ct")
    if st.place_target:
        px = homo.to_pixel(*st.place_target)
        if px:
            hud.marker(view, px, H.PLACE, radius=28, thickness=2, dashed=True)
            hud.crosshair(view, px, H.PLACE, size=16)
            hud.chip(view, (px[0], px[1] - 32), hud.label("놓을 곳", "PLACE"),
                     H.PLACE, 13, bold=True, anchor="ct")


def _draw_arm_tip(view, st, hud, homo):
    if st.tcp is None:
        return
    px = homo.to_pixel(st.tcp[0], st.tcp[1])
    if not px:
        return
    u, v = int(px[0]), int(px[1])
    cv2.drawMarker(view, (u, v), (30, 28, 26), cv2.MARKER_TILTED_CROSS,
                   15, 4, cv2.LINE_AA)          # 어두운 테두리로 가독성 확보
    cv2.drawMarker(view, (u, v), (250, 250, 250), cv2.MARKER_TILTED_CROSS,
                   14, 2, cv2.LINE_AA)
    # 목표 칩(위쪽)과 겹치지 않게 아래에 붙인다
    hud.chip(view, (u, v + 12), hud.label("팔 끝", "TCP"),
             (250, 250, 250), 11, anchor="cb", alpha=0.6)


def _draw_pointer(view, st, hud, homo):
    px = st.pointer_px
    if px is None:
        return
    active = st.task_state != "EXECUTING"
    if not active:
        color = H.MUTED
    elif st.task_state == "IDLE":
        color = H.IDLE
    else:
        color = H.PLACE

    hud.crosshair(view, px, color)
    hud.progress_ring(view, px, st.dwell_progress if active else 0.0, color)
    if st.pointer_holding:
        hud.progress_ring(view, px, st.hold_frac, color, radius=26)

    xy = homo.to_robot(*px)
    if xy:
        hud.chip(view, (px[0] + 30, px[1] - 10),
                 f"{xy[0]:+.3f}, {xy[1]:+.3f}", color, 12, alpha=0.66)


# ----------------------------------------------------------------------
# 헤더 / 푸터
# ----------------------------------------------------------------------
def _state_style(st, hud):
    if st.task_state == "IDLE":
        return H.IDLE, hud.label("대기", "IDLE"), \
            hud.label("집을 컵을 보세요", "Look at a cup to pick")
    if st.task_state == "PICK_SELECTED":
        return H.PICK, hud.label("집을 것 선택됨", "PICKED"), \
            hud.label("이제 놓을 자리를 보세요", "Now look where to place")
    if st.task_state == "EXECUTING":
        return H.BUSY, hud.label("동작 중", "RUNNING"), \
            hud.label("로봇이 움직이는 중입니다", "Robot is moving")
    return H.MUTED, st.task_state, ""


def _draw_header(canvas, st, hud):
    color, badge, hint = _state_style(st, hud)
    Hud.panel(canvas, 0, 0, canvas.shape[1], HEADER_H, alpha=1.0, color=PANEL_BG)
    Hud.accent_bar(canvas, 0, 0, 5, HEADER_H, color)
    cv2.line(canvas, (0, HEADER_H - 1), (canvas.shape[1], HEADER_H - 1),
             (44, 42, 40), 1)

    items = [((18, 6), badge, color, 17, True),
             ((18, 27), hint, H.MUTED, 13, False)]

    src = (hud.label("시선 입력", "gaze input") if st.source == "gaze"
           else hud.label("마우스 입력", "mouse input"))
    right = f"{len(st.objects)} " + hud.label("개 검출", "objects") + "   " + src
    w, _ = hud.text_size(right, 13)
    items.append(((canvas.shape[1] - w - 18, 7), right, H.FG, 13, False))
    fps = f"{st.fps:4.1f} fps"
    w2, _ = hud.text_size(fps, 12)
    items.append(((canvas.shape[1] - w2 - 18, 27), fps, H.MUTED, 12, False))
    hud.texts(canvas, items)


def _draw_footer(canvas, y, st, hud):
    Hud.panel(canvas, 0, y, canvas.shape[1], FOOTER_H, alpha=1.0, color=PANEL_BG)
    cv2.line(canvas, (0, y), (canvas.shape[1], y), (44, 42, 40), 1)
    if st.notice:
        hud.text(canvas, (18, y + 7), st.notice, H.BUSY, 13, bold=True)
    else:
        hud.text(canvas, (18, y + 8), hud.label(
            "좌클릭 선택   c 취소   space 비상정지   d 검출 전환   q 종료",
            "click=select   c=cancel   space=E-STOP   d=detect   q=quit"),
            H.MUTED, 12)


# ----------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------
def _section(items, x, y, hud, ko, en):
    items.append(((x, y), hud.label(ko, en), H.MUTED, 12, True))
    return y + 21


def _draw_sidebar(canvas, vw, top, vh, st, hud):
    x0 = vw
    Hud.panel(canvas, x0, top, st.sidebar_w, vh, alpha=1.0, color=PANEL_BG)
    cv2.line(canvas, (x0, top), (x0, top + vh), (44, 42, 40), 1)
    x = x0 + 18
    y = top + 14
    items = []

    # 파이프라인
    y = _section(items, x, y, hud, "파이프라인", "PIPELINE")
    # 단계는 누적해서 켠다. 뒤 단계가 진행됐는데 앞 단계가 꺼져 있으면
    # (예: 동작 중이라 커서가 화면 밖) 진행 상황을 잘못 읽게 된다.
    reached = [st.pointer_px is not None, st.has_fixation,
               st.pick_target is not None or st.place_target is not None,
               st.task_state == "EXECUTING"]
    last = max([i for i, v in enumerate(reached) if v], default=-1)
    lit = [i <= last for i in range(len(reached))]
    for i, (ko, en) in enumerate(STAGES):
        on = lit[i]
        cy = y + 7
        if i < len(STAGES) - 1:
            cv2.line(canvas, (x + 5, cy + 6), (x + 5, cy + 19), (58, 56, 54), 1)
        cv2.circle(canvas, (x + 5, cy), 5, H.PLACE if on else (68, 66, 64), -1,
                   cv2.LINE_AA)
        items.append(((x + 20, y), hud.label(ko, en),
                      H.FG if on else (108, 106, 104), 13, False))
        y += 25
    y += 8

    # 선택
    y = _section(items, x, y, hud, "선택", "SELECTION")
    if st.pick_target:
        items.append(((x, y), hud.label("집을 곳", "Pick"), H.PICK, 12, False))
        items.append(((x + 78, y), f"{st.pick_target[0]:+.3f}, "
                                   f"{st.pick_target[1]:+.3f}", H.FG, 12, False))
        y += 19
    if st.place_target:
        items.append(((x, y), hud.label("놓을 곳", "Place"), H.PLACE, 12, False))
        items.append(((x + 78, y), f"{st.place_target[0]:+.3f}, "
                                   f"{st.place_target[1]:+.3f}", H.FG, 12, False))
        y += 19
    info = st.target_info
    if info:
        snapped = info.get("snapped")
        items.append(((x, y),
                      hud.label("물체 중심으로 보정됨", "snapped to object") if snapped
                      else hud.label("보정 없음 — 응시점 그대로", "no snap"),
                      H.FG if snapped else H.WARN, 11, False))
        y += 17
        items.append(((x, y), hud.label("신뢰도 ", "conf ")
                      + f"{info.get('confidence', 0.0):.2f}", H.MUTED, 11, False))
        y += 21
    if not st.pick_target and not st.place_target:
        items.append(((x, y), hud.label("아직 없음", "none yet"),
                      (108, 106, 104), 12, False))
        y += 23
    y += 6

    # 로봇
    y = _section(items, x, y, hud, "로봇", "ROBOT")
    items.append(((x, y), hud.label("단계", "phase"), H.MUTED, 12, False))
    items.append(((x + 78, y), st.arm_phase or hud.label("정지", "idle"),
                  H.BUSY if st.arm_phase else (108, 106, 104), 12, False))
    y += 19
    if st.tcp:
        items.append(((x, y), hud.label("팔 끝", "TCP"), H.MUTED, 12, False))
        items.append(((x + 78, y), f"{st.tcp[0]:+.3f}, {st.tcp[1]:+.3f}, "
                                   f"{st.tcp[2]:+.3f}", H.FG, 11, False))
        y += 19
    y += 10

    # 준비 상태
    y = _section(items, x, y, hud, "준비 상태", "READY")
    err = st.reproj_error_mm
    checks = [
        (st.calibrated,
         "캘리브레이션" + (f" ({err:.1f}mm)" if err is not None else ""),
         "calibration" + (f" ({err:.1f}mm)" if err is not None else "")),
        (st.detect_on, "물체 검출", "detection"),
        (len(st.objects) > 0, "컵이 보임", "cup visible"),
    ]
    for ok, ko, en in checks:
        cv2.circle(canvas, (x + 5, y + 7), 4, H.PLACE if ok else H.WARN, -1,
                   cv2.LINE_AA)
        items.append(((x + 20, y), hud.label(ko, en),
                      H.FG if ok else H.WARN, 12, False))
        y += 19

    hud.texts(canvas, items)


# ----------------------------------------------------------------------
# 미리보기 (하드웨어·ROS 없이 화면만 확인)
# ----------------------------------------------------------------------
def _demo_frame(w=640, h=480):
    """책상 위에 컵 세 개가 있는 가짜 탑다운 영상."""
    img = np.full((h, w, 3), (78, 88, 104), np.uint8)
    for i in range(0, h, 4):                      # 나뭇결 느낌
        cv2.line(img, (0, i), (w, i), (74, 84, 99), 1)
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), (96, 108, 126), 2)
    cups = [(180, 300), (320, 250), (470, 320)]
    radius = 34
    for (cx, cy) in cups:
        cv2.circle(img, (cx, cy), radius, (40, 40, 180), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (30, 30, 130), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 24, (60, 60, 205), -1, cv2.LINE_AA)
    # 실제 검출과 같은 형태의 contour 를 만들어 둔다(마커/배지 크기가 여기 맞춰진다)
    contours = []
    for (cx, cy) in cups:
        pts = [[[int(cx + radius * math.cos(a)), int(cy + radius * math.sin(a))]]
               for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
        contours.append(np.array(pts, dtype=np.int32))
    return img, cups, contours


class _DemoHomography:
    """미리보기용 가짜 호모그래피 (픽셀 <-> 로봇 좌표 선형 근사)."""
    ready = True
    reproj_error_mm = 4.2

    def to_robot(self, u, v):
        return 0.18 + (480 - float(v)) * 0.00045, (float(u) - 320) * 0.00075

    def to_pixel(self, x, y):
        return 320 + float(y) / 0.00075, 480 - (float(x) - 0.18) / 0.00045


def main():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "control_panel_preview.png"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "pick"

    frame, cups, contours = _demo_frame()
    hud = Hud()
    homo = _DemoHomography()
    objects = [((float(u), float(v)), homo.to_robot(u, v), c)
               for (u, v), c in zip(cups, contours)]

    st = PanelState(objects=objects, fps=29.7, calibrated=True,
                    reproj_error_mm=4.2, detect_on=True)
    if scenario == "idle":
        st.task_state = "IDLE"
        st.pointer_px = (250, 270)
        st.dwell_progress = 0.45
    elif scenario == "pick":
        st.task_state = "PICK_SELECTED"
        st.pick_target = objects[1][1]
        st.target_info = {"snapped": True, "confidence": 0.86}
        st.has_fixation = True
        st.pointer_px = (470, 380)
        st.dwell_progress = 0.7
        st.tcp = (0.22, -0.05, 0.16)
    else:                                   # executing
        st.task_state = "EXECUTING"
        st.pick_target = objects[1][1]
        st.place_target = homo.to_robot(250, 410)      # 컵이 없는 빈 자리
        st.target_info = {"snapped": True, "confidence": 0.91}
        st.has_fixation = True
        st.arm_phase = "GRASP"
        st.tcp = (0.30, 0.00, 0.06)
        st.notice = hud.label("로봇이 컵을 집는 중입니다", "Robot is grasping")

    canvas = render(frame, st, hud, homo)
    cv2.imwrite(out, canvas)
    print(f"{out} ({canvas.shape[1]}x{canvas.shape[0]}), "
          f"한글표시={'예' if hud.korean else '아니오'}")


if __name__ == "__main__":
    main()
