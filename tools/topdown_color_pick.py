#!/usr/bin/env python3
"""탑다운 카메라 컵 색 설정 — 화면에서 컵을 클릭하면 HSV 범위를 추정하고 검출을 실시간으로 보여 준다.

    python3 tools/topdown_color_pick.py --cam 0

조작:
  좌클릭  컵 위의 한 점 -> 그 주변 색으로 범위 추정 (여러 번 클릭하면 범위가 합쳐진다)
  r       범위 초기화
  w       gaze_hri.yaml 의 topdown_click / control_panel hsv_lower·hsv_upper 에 저장
  q       종료 (저장 안 함)

~/.ros/topdown_homography.yaml 이 있으면 검출된 컵마다 로봇 좌표(base_link, m)를 같이 띄운다.
컵을 자로 잰 자리에 놓고 이 숫자와 비교하면 탑다운 정확도를 바로 검증할 수 있다.

흰/회색 컵(채도 낮음)은 색상(H)이 의미가 없어서 밝기(V)/채도(S) 기준 범위로 자동 전환한다.
★ 컵 윤곽 무게중심을 테이블 호모그래피에 넣으므로, 카메라가 기울어 있으면 컵 높이만큼
  카메라 반대쪽으로 밀린 XY 가 나온다 (60cm 높이·30° 기울기·중심높이 4.5cm 면 약 2.6cm).
  자로 검증할 때 이 방향의 치우침이 보이면 그 때문이다.
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "..", "ros2_ws", "src", "gaze_hri", "config", "gaze_hri.yaml")
HOMO = os.path.expanduser("~/.ros/topdown_homography.yaml")


def range_from_patch(hsv_patch):
    h, s, v = [hsv_patch[..., i].ravel().astype(int) for i in range(3)]
    s_med = int(np.median(s))
    if s_med < 45:     # 흰/회색: 색상 무의미 -> 밝고 채도 낮은 영역
        lo = [0, 0, max(0, int(np.percentile(v, 5)) - 35)]
        hi = [180, min(255, int(np.percentile(s, 95)) + 30), 255]
    else:
        hm = int(np.median(h))
        lo = [max(0, hm - 10), max(0, int(np.percentile(s, 5)) - 40), max(0, int(np.percentile(v, 5)) - 50)]
        hi = [min(180, hm + 10), 255, 255]
    return np.array(lo), np.array(hi)


def write_params(lo, hi, min_area):
    s = open(PARAMS, encoding="utf-8").read()
    n = 0
    for sec in ("control_panel", "topdown_click"):
        m = re.search(rf"^{sec}:\n(?:[ \t].*\n|\n)*", s, flags=re.M)
        if not m:
            continue
        block = m.group(0)
        new = re.sub(r"(hsv_lower:\s*)\[[^\]]*\]", rf"\g<1>[{lo[0]}, {lo[1]}, {lo[2]}]", block)
        new = re.sub(r"(hsv_upper:\s*)\[[^\]]*\]", rf"\g<1>[{hi[0]}, {hi[1]}, {hi[2]}]", new)
        new = re.sub(r"(min_area:\s*)\d+", rf"\g<1>{min_area}", new)
        if sec == "topdown_click":
            new = re.sub(r"(detect_objects:\s*)(true|false)", r"\g<1>true", new)
        s = s.replace(block, new)
        n += 1
    open(PARAMS, "w", encoding="utf-8").write(s)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=400)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"카메라 {a.cam} 열기 실패")
    H = np.array(yaml.safe_load(open(HOMO))["H"], float) if os.path.exists(HOMO) else None
    print("호모그래피:", "있음 — 컵마다 로봇 좌표 표시" if H is not None else "없음 (캘리브 전)")

    st = {"lo": None, "hi": None, "click": None}
    win = "topdown color - click a cup (w=save r=reset q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, f, p: st.update(click=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if st["click"]:
            x, y = st["click"]
            st["click"] = None
            patch = hsv[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
            lo, hi = range_from_patch(patch)
            if st["lo"] is not None:
                lo, hi = np.minimum(lo, st["lo"]), np.maximum(hi, st["hi"])
            st["lo"], st["hi"] = lo, hi
            print(f"범위: lower {lo.tolist()} upper {hi.tolist()}")
        vis = frame.copy()
        if st["lo"] is not None:
            mask = cv2.inRange(hsv, st["lo"].astype(np.uint8), st["hi"].astype(np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            vis[mask > 0] = (0.5 * vis[mask > 0] + [0, 90, 0]).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            k = 0
            for c in cnts:
                area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                if area < a.min_area:
                    if area >= 60:      # 버려진 덩어리 (min_area 미달) — 왜 안 잡히는지 보라고 회색으로
                        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
                        cv2.putText(vis, f"{int(area)}", (bx, by - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (200, 200, 200), 1)
                    continue
                M = cv2.moments(c)
                u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
                k += 1
                cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                label = f"#{k} a={int(area)}"
                if H is not None:
                    p = H @ [u, v, 1.0]
                    label += f" ({p[0] / p[2]:.3f}, {p[1] / p[2]:+.3f})"
                cv2.putText(vis, label, (bx, max(12, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, f"cups: {k}  min_area={a.min_area}  (gray=too small)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("mask", mask)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            st["lo"] = st["hi"] = None
        if key == ord("w") and st["lo"] is not None:
            n = write_params(st["lo"].astype(int).tolist(), st["hi"].astype(int).tolist(), a.min_area)
            print(f"저장: {PARAMS} ({n}개 섹션). 이제 컵 위치를 자로 재서 화면 좌표와 비교해 볼 것.")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
