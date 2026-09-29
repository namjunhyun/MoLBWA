#!/usr/bin/env python3
"""SGBM 깊이 스케일 실측 검증 — 어느 baseline 이 맞는지 줄자로 판정한다.

왜 필요한가
-----------
지금 코드베이스에는 서로 다른 baseline 두 개가 있고, 1.63배 차이가 난다:

  ocams_calib.GEOMETRIC_BASELINE_M = 0.1716 m   # P_right 에서 계산된 기하값.
                                                # ORB-SLAM3 설정(Stereo.T_c1_c2)도 이 값.
  ocams_calib.DEPTH_BASELINE_M     = 0.105  m   # 2026-09-22 에 SGBM 이 너무 멀게
                                                # 나온다고 경험적으로 낮춘 값.

둘 중 하나는 틀렸다. 그런데 융합(fusion.gaze_point_world)은 이 둘을 섞어 쓴다 —
카메라 위치는 SLAM(=기하 baseline 스케일), 카메라에서 대상까지 거리는 SGBM
(=깊이 baseline 스케일). 스케일이 어긋나면 p_W 는 거리에 비례해 틀린다.
컵이 60cm 앞에 있는데 37cm 로 계산되면 로봇팔은 허공을 집는다.

이건 추론으로 정할 수 없다. 줄자로 재는 수밖에 없다.

쓰는 법
-------
  1. 평평하고 무늬가 있는 벽/상자를 정면으로 본다. (민무늬 벽은 SGBM 이 못 푼다.
     신문지나 박스를 붙이면 된다.)
  2. 카메라 앞면에서 그 면까지 줄자로 잰다.
  3. python3 verify_depth_scale.py --truth 0.60 --source ros
  4. 화면 중앙 십자에 그 면이 오도록 맞추고 스페이스. 여러 거리에서 반복하면 더 좋다.
  5. q 로 끝내면 두 baseline 중 어느 쪽이 실측과 맞는지 표로 나온다.

거리를 3개 이상(예: 0.4 / 0.8 / 1.5 m) 재는 게 중요하다. baseline 오차는 거리에
비례해서 커지므로 한 점만 보면 상수 오프셋과 구분이 안 된다.
"""
import argparse
import sys

import cv2
import numpy as np

import ocams_calib
from gaze_on_scene import depth_at, make_stereo_matcher, RosFrameSource, scene_stereo, open_scene

FX = ocams_calib.RECTIFIED_K[0, 0]


def measure(disparity, radius):
    """화면 중앙의 disparity 중앙값. baseline 과 무관하게 disparity 자체를 반환한다."""
    if disparity is None:
        return None
    h, w = disparity.shape
    cu, cv = w // 2, h // 2
    roi = disparity[cv - radius:cv + radius + 1, cu - radius:cu + radius + 1]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if len(valid) < 0.7 * roi.size:
        return None
    return float(np.median(valid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=float, required=True,
                    help="줄자로 잰 실제 거리(m). 샘플마다 --truth 를 바꿔 여러 번 실행해도 되고, "
                         "실행 중 t 키로 바꿔도 된다.")
    ap.add_argument("--source", choices=["local", "ros"], default="ros")
    ap.add_argument("--left-topic", default="/camera/left/compressed")
    ap.add_argument("--right-topic", default="/camera/right/compressed")
    ap.add_argument("--eye-topic", default="/eye/image_raw/compressed")
    ap.add_argument("--radius", type=int, default=15, help="중앙 ROI 반경(px)")
    args = ap.parse_args()

    ros_src = scene_cap = left_maps = right_maps = None
    if args.source == "ros":
        ros_src = RosFrameSource(args.eye_topic, args.left_topic, args.right_topic)
    else:
        scene_cap = open_scene(ocams_calib.IMAGE_WIDTH, ocams_calib.IMAGE_HEIGHT)
        left_maps, right_maps = ocams_calib.build_rectify_maps()

    matcher = make_stereo_matcher()
    truth = args.truth
    samples = []   # (실측거리, disparity)

    print(f"[검증] 중앙 십자에 대상을 맞추고 스페이스=기록 / t=실측거리 변경 / q=종료")
    win = "depth scale check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            if ros_src is not None:
                left, right = ros_src.scene_pair()
            else:
                left, right = scene_stereo(scene_cap, left_maps, right_maps)
            if left is None or right is None:
                cv2.waitKey(30)
                continue

            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            disp = measure(disparity, args.radius)

            vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            h, w = left.shape
            cv2.drawMarker(vis, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
            cv2.rectangle(vis, (w // 2 - args.radius, h // 2 - args.radius),
                          (w // 2 + args.radius, h // 2 + args.radius), (0, 255, 0), 1)
            if disp is None:
                txt = "disparity 없음 — 무늬 있는 면을 보여줄 것"
                cv2.putText(vis, "NO DISPARITY (need texture)", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                d_geo = FX * ocams_calib.GEOMETRIC_BASELINE_M / disp
                d_sgbm = FX * ocams_calib.DEPTH_BASELINE_M / disp
                cv2.putText(vis, f"truth={truth:.3f}m  disp={disp:.2f}px", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(vis, f"geometric(17.16cm) -> {d_geo:.3f}m", (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.putText(vis, f"sgbm(10.50cm)      -> {d_sgbm:.3f}m", (20, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(vis, f"samples={len(samples)}  space=record t=truth q=quit",
                        (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' ') and disp is not None:
                samples.append((truth, disp))
                print(f"  기록 {len(samples)}: truth={truth:.3f}m disparity={disp:.2f}px")
            elif key == ord('t'):
                cv2.destroyWindow(win)
                try:
                    truth = float(input("새 실측거리(m): ").strip())
                except ValueError:
                    print("  숫자가 아님 — 유지")
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    finally:
        cv2.destroyAllWindows()
        if ros_src is not None:
            ros_src.shutdown()
        if scene_cap is not None:
            scene_cap.release()

    if not samples:
        print("기록된 샘플이 없다.")
        return 1

    print(f"\n{'실측(m)':>9} {'disp(px)':>9} {'기하 17.16cm':>13} {'SGBM 10.50cm':>13}")
    for t, d in samples:
        print(f"{t:>9.3f} {d:>9.2f} "
              f"{FX*ocams_calib.GEOMETRIC_BASELINE_M/d:>13.3f} "
              f"{FX*ocams_calib.DEPTH_BASELINE_M/d:>13.3f}")

    # 최소제곱으로 실측에 가장 맞는 baseline 을 직접 구한다.
    # D = fx*B/disp 이므로 B = D*disp/fx. 샘플별 B 의 분포를 본다.
    b_est = np.array([t * d / FX for t, d in samples])
    print(f"\n실측에서 역산한 baseline: 평균 {b_est.mean()*100:.2f}cm "
          f"(표준편차 {b_est.std()*100:.2f}cm, 샘플 {len(b_est)}개)")
    for name, b in (("기하 (SLAM 과 동일)", ocams_calib.GEOMETRIC_BASELINE_M),
                    ("SGBM 현재 기본값", ocams_calib.DEPTH_BASELINE_M)):
        err = np.array([abs(FX * b / d - t) / t for t, d in samples])
        print(f"  {name:<22} {b*100:>6.2f}cm  평균 상대오차 {err.mean()*100:>5.1f}%  "
              f"최대 {err.max()*100:.1f}%")

    if b_est.std() / b_est.mean() > 0.05:
        print("\n[주의] 역산 baseline 이 샘플마다 5% 넘게 흔들린다. 단순 baseline 오차가 아니라 "
              "rectification 자체가 틀렸을 수 있다 — 거리별로 오차 방향이 다른지 위 표를 볼 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
