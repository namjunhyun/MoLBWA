#!/usr/bin/env python3
"""SGBM 깊이 스케일 vs SLAM 스케일 — 줄자 없이 판정한다.

무엇을 재는가
-------------
융합(fusion.gaze_point_world)은 스케일이 다른 두 출처를 섞는다:

  - 카메라의 세계 위치      <- ORB-SLAM3 (IMU 초기화 후 실제 미터)
  - 카메라에서 대상까지 거리 <- StereoSGBM (D = fx * baseline / disparity)

두 '미터'가 같아야 p_W 가 의미를 갖는다. 그런데 baseline 후보가 두 개고 1.63배
차이난다(ocams_calib.GEOMETRIC_BASELINE_M 0.1716 vs DEPTH_BASELINE_M 0.105).

여기서는 SLAM 을 자로 쓴다. 벽을 보며 앞뒤로 움직이면
    SLAM 이 말하는 이동거리  ==  SGBM 깊이의 변화량
이어야 한다. 어긋난 비율이 곧 baseline 의 배율 오차다.

    D = fx*B/disp  이므로  D 에 곱해지는 배율 = B 에 곱해지는 배율.
    즉 B_참값 = B_사용값 * (SLAM 이동량 / SGBM 깊이변화량)

줄자보다 나은 이유: 절대 거리가 아니라 '두 스케일이 일치하는가'를 직접 본다.
융합에서 문제가 되는 건 정확히 그것이다. 조준 정밀도도 필요 없다 — 같은 면을
계속 보고만 있으면 차이만 쓰므로 오프셋은 상쇄된다.

전제
----
  1. ORB-SLAM3 가 IMU 초기화까지 끝나 있을 것 (로그에 "start VIBA 2" 가 떠야 한다).
     초기화 전에는 SLAM 스케일 자체가 미확정이라 이 검증이 무의미하다.
  2. 무늬가 있는 평면을 정면으로 볼 것. 민무늬 벽은 SGBM 이 못 푼다.
  3. 카메라 광축 방향으로 앞뒤 이동할 것. 옆으로 움직이면 보는 지점이 바뀐다.

쓰는 법
-------
    python3 verify_scale_vs_slam.py
    - 벽/박스를 중앙 십자에 두고 스페이스 -> 기준점 기록
    - 30~50cm 앞뒤로 이동 (같은 면을 계속 보면서)
    - 다시 스페이스 -> 짝 하나 완성. 5짝 이상 모으면 결론이 안정된다.
    - q 로 종료하면 판정 결과가 나온다.
"""
import argparse
import sys

import cv2
import numpy as np

import ocams_calib
from gaze_on_scene import make_stereo_matcher, RosFrameSource

FX = ocams_calib.RECTIFIED_K[0, 0]
# 판정에 쓰는 기준 baseline. 결과는 "이 값에 몇 배를 곱해야 하는가" 로 나온다.
REF_BASELINE = ocams_calib.DEPTH_BASELINE_M

# 짝을 이루는 두 샘플 사이의 최소 이동량. 너무 작으면 잡음이 신호를 덮는다.
MIN_MOVE_M = 0.10


def center_disparity(disparity, radius):
    if disparity is None:
        return None
    h, w = disparity.shape
    cu, cv = w // 2, h // 2
    roi = disparity[cv - radius:cv + radius + 1, cu - radius:cu + radius + 1]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if len(valid) < 0.7 * roi.size:
        return None
    disp = float(np.median(valid))
    # 평면을 보고 있는지 확인. 깊이가 섞인 ROI 는 차분이 엉뚱하게 나온다.
    mad = float(np.median(np.abs(valid - disp)))
    if disp <= 0 or mad / disp > 0.08:
        return None
    return disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-topic", default="/camera/left/compressed")
    ap.add_argument("--right-topic", default="/camera/right/compressed")
    ap.add_argument("--eye-topic", default="/eye/image_raw/compressed")
    ap.add_argument("--pose-topic", default="/orbslam3/pose")
    ap.add_argument("--radius", type=int, default=15)
    args = ap.parse_args()

    src = RosFrameSource(args.eye_topic, args.left_topic, args.right_topic)
    src.enable_pose(args.pose_topic, stale_after=0.5)
    matcher = make_stereo_matcher()

    anchor = None      # (SLAM 위치 3벡터, disparity)
    pairs = []         # (SLAM 이동거리, SGBM 깊이변화)

    win = "scale check (SGBM vs SLAM)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[검증] 스페이스=기록(2번 눌러 한 짝) / r=기준점 버리기 / q=종료")
    try:
        while True:
            left, right = src.scene_pair()
            if left is None or right is None:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                continue

            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            disp = center_disparity(disparity, args.radius)
            T_WS = src.latest_pose()
            D_ref = FX * REF_BASELINE / disp if disp else None

            h, w = left.shape
            vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(vis, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
            cv2.rectangle(vis, (w // 2 - args.radius, h // 2 - args.radius),
                          (w // 2 + args.radius, h // 2 + args.radius), (0, 255, 0), 1)

            slam_txt = "SLAM 끊김 (추적 상실)" if T_WS is None else \
                       f"SLAM pos=({T_WS[0,3]:+.3f},{T_WS[1,3]:+.3f},{T_WS[2,3]:+.3f})"
            cv2.putText(vis, slam_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if T_WS is None else (0, 255, 0), 2)
            disp_txt = "disparity 없음 (무늬 필요/평면 아님)" if disp is None else \
                       f"disp={disp:.2f}px  D={D_ref:.3f}m (baseline {REF_BASELINE*100:.1f}cm 기준)"
            cv2.putText(vis, disp_txt, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if disp is None else (255, 200, 0), 2)
            anchor_txt = "기준점 없음 — 스페이스로 기록" if anchor is None else \
                         f"기준점 있음 — {MIN_MOVE_M*100:.0f}cm 이상 앞뒤 이동 후 스페이스"
            cv2.putText(vis, anchor_txt, (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)
            cv2.putText(vis, f"짝 {len(pairs)}개  space=기록 r=기준점버리기 q=종료",
                        (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                anchor = None
                print("  기준점 버림")
            elif key == ord(' '):
                if disp is None or T_WS is None:
                    print("  기록 불가 — disparity 또는 SLAM 포즈 없음")
                    continue
                pos = T_WS[:3, 3].copy()
                if anchor is None:
                    anchor = (pos, disp)
                    print(f"  기준점: D={FX*REF_BASELINE/disp:.3f}m "
                          f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")
                    continue
                pos0, disp0 = anchor
                slam_move = float(np.linalg.norm(pos - pos0))
                d0 = FX * REF_BASELINE / disp0
                d1 = FX * REF_BASELINE / disp
                depth_change = abs(d1 - d0)
                if slam_move < MIN_MOVE_M:
                    print(f"  이동이 너무 작다 ({slam_move*100:.1f}cm < {MIN_MOVE_M*100:.0f}cm) — 무시")
                    continue
                if depth_change < 1e-3:
                    print("  깊이 변화가 없다 — 광축 방향으로 움직였는지 확인")
                    continue
                pairs.append((slam_move, depth_change))
                ratio = slam_move / depth_change
                print(f"  짝 {len(pairs)}: SLAM {slam_move*100:.1f}cm vs "
                      f"SGBM {depth_change*100:.1f}cm -> 배율 {ratio:.3f}")
                anchor = None
    finally:
        cv2.destroyAllWindows()
        src.shutdown()

    if not pairs:
        print("\n짝이 하나도 없다 — 결론 없음.")
        return 1

    ratios = np.array([m / c for m, c in pairs])
    r_med = float(np.median(ratios))
    b_true = REF_BASELINE * r_med

    print(f"\n{'SLAM 이동(cm)':>14} {'SGBM 변화(cm)':>15} {'배율':>8}")
    for m, c in pairs:
        print(f"{m*100:>14.1f} {c*100:>15.1f} {m/c:>8.3f}")

    print(f"\n배율 중앙값 {r_med:.3f} (표준편차 {ratios.std():.3f}, 짝 {len(pairs)}개)")
    print(f"=> 실제 baseline 추정: {REF_BASELINE*100:.2f}cm x {r_med:.3f} = {b_true*100:.2f}cm")
    print(f"   후보와 비교: 기하 {ocams_calib.GEOMETRIC_BASELINE_M*100:.2f}cm / "
          f"SGBM 현재값 {ocams_calib.DEPTH_BASELINE_M*100:.2f}cm")

    for name, b in (("기하 (SLAM 과 동일)", ocams_calib.GEOMETRIC_BASELINE_M),
                    ("SGBM 현재 기본값", ocams_calib.DEPTH_BASELINE_M)):
        print(f"   {name:<22} 오차 {abs(b-b_true)/b_true*100:>5.1f}%")

    if ratios.std() > 0.15:
        print("\n[주의] 배율이 짝마다 0.15 넘게 흔들린다. 단순 스케일 문제가 아닐 수 있다 — "
              "SLAM 추적이 불안정했거나, 옆으로 움직여 보는 지점이 바뀌었을 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
