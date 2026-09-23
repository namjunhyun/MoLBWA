#!/usr/bin/env python3
"""gaze_on_scene.fuse_and_send() 단위 검증 — 카메라/SLAM 없이 순수 기하로.

이 함수는 시선 픽셀 + 스테레오 깊이 + SLAM 포즈를 합쳐 세계좌표를 만들고 로봇팔로
쏜다. 여기서 틀리면 팔이 엉뚱한 데로 간다. 특히 검증하는 것:

  1. 유효하지 않은 상황마다 valid=False 가 나가는가 (조용히 멈추면 로봇팔이
     마지막 좌표를 붙들고 움직인다)
  2. --scene-flip 일 때 픽셀을 되돌려서 깊이/역투영에 쓰는가
     (표시용으로만 뒤집힌 좌표를 그대로 K 에 넣으면 좌우상하가 뒤집힌 3D점이 나온다)
  3. p_W 가 실제로 T_WS 로 변환된 세계좌표인가

실행: python3 test_fuse_and_send.py
"""
import sys

import numpy as np

import ocams_calib
import fusion
from gaze_on_scene import fuse_and_send

K = ocams_calib.RECTIFIED_K
FX = K[0, 0]
BASELINE = ocams_calib.DEPTH_BASELINE_M


class FakeSender:
    """GazeUdpSender 대역. 마지막으로 보낸 것을 붙잡아 둔다."""

    def __init__(self):
        self.last = None
        self.sent = 0
        self.dropped = 0

    def send(self, p_W, valid=True, T_WS=None):
        self.last = (np.asarray(p_W, dtype=float), bool(valid), T_WS)
        self.sent += 1
        return True


class FakeRosSrc:
    def __init__(self, T_WS):
        self._T = T_WS

    def latest_pose(self):
        return self._T


def uniform_disparity(depth_m, shape=(480, 640)):
    """전 화면이 같은 거리인 disparity 맵. D = fx*B/disp 를 역으로 푼다."""
    disp = FX * BASELINE / depth_m
    return np.full(shape, disp, dtype=np.float32)


def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return cond


def main():
    ok = True
    SW, SH = 640, 480
    T_identity = np.eye(4)

    # 1) 시선 무효 -> valid=False
    s = FakeSender()
    depth, status = fuse_and_send(s, FakeRosSrc(T_identity), False, None, None,
                                  uniform_disparity(0.8), BASELINE, None, False, SW, SH)
    ok &= check("시선 무효 -> valid=False", s.last[1] is False and status == "시선없음", status)

    # 2) disparity 없음 -> valid=False
    s = FakeSender()
    depth, status = fuse_and_send(s, FakeRosSrc(T_identity), True, 320, 240,
                                  None, BASELINE, None, False, SW, SH)
    ok &= check("깊이 없음 -> valid=False", s.last[1] is False and status == "깊이없음", status)

    # 3) 깊이가 범위 밖(10m) -> valid=False
    s = FakeSender()
    depth, status = fuse_and_send(s, FakeRosSrc(T_identity), True, 320, 240,
                                  uniform_disparity(10.0), BASELINE, None, False, SW, SH)
    ok &= check("깊이 범위 밖 -> valid=False",
                s.last[1] is False and status.startswith("깊이범위밖"), status)

    # 4) SLAM 포즈 없음(추적 상실) -> valid=False
    s = FakeSender()
    depth, status = fuse_and_send(s, FakeRosSrc(None), True, 320, 240,
                                  uniform_disparity(0.8), BASELINE, None, False, SW, SH)
    ok &= check("SLAM 추적 상실 -> valid=False",
                s.last[1] is False and status == "SLAM끊김", status)

    # 5) 항등 포즈 + 주점(cx,cy) 응시 -> p_W 는 광축 위 D 미터
    #    주점을 쓰는 이유: (u-cx)/fx = 0 이라 x,y 가 정확히 0 이어야 한다.
    D_true = 0.8
    s = FakeSender()
    cu, cv = int(round(K[0, 2])), int(round(K[1, 2]))
    depth, status = fuse_and_send(s, FakeRosSrc(T_identity), True, cu, cv,
                                  uniform_disparity(D_true), BASELINE, None, False, SW, SH)
    p_W, valid, T_sent = s.last
    ok &= check("깊이 복원", valid and abs(depth - D_true) < 1e-3, f"D={depth:.4f} (기대 {D_true})")
    ok &= check("항등 포즈 주점 -> p_W=(0,0,D)",
                valid and np.allclose(p_W, [0, 0, D_true], atol=2e-3), f"p_W={np.round(p_W, 4)}")
    ok &= check("head 포즈 동봉", T_sent is not None)

    # 6) 포즈 적용: 카메라가 (1, 2, 3) 에 있고 z축 90도 회전
    T = fusion.pose_to_matrix([1.0, 2.0, 3.0],
                              fusion.rotation_matrix_to_quat(
                                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)))
    s = FakeSender()
    fuse_and_send(s, FakeRosSrc(T), True, cu, cv,
                  uniform_disparity(D_true), BASELINE, None, False, SW, SH)
    p_W, valid, _ = s.last
    # 카메라 좌표 (0,0,D) -> 회전 후 (0,0,D) -> 평행이동 더해 (1,2,3+D)
    ok &= check("포즈 적용된 p_W",
                valid and np.allclose(p_W, [1.0, 2.0, 3.0 + D_true], atol=2e-3),
                f"p_W={np.round(p_W, 4)}")

    # 7) --scene-flip: 뒤집힌 화면에서 (u,v) 를 클릭했을 때, 되돌린 픽셀로 역투영해야 한다.
    #    뒤집힌 좌표 (SW-1-cu, SH-1-cv) 는 원본 주점에 해당하므로 결과는 5)와 같아야 한다.
    s = FakeSender()
    fuse_and_send(s, FakeRosSrc(T_identity), True, SW - 1 - cu, SH - 1 - cv,
                  uniform_disparity(D_true), BASELINE, None, True, SW, SH)
    p_W_flip, valid, _ = s.last
    ok &= check("scene-flip 픽셀 되돌리기",
                valid and np.allclose(p_W_flip, [0, 0, D_true], atol=2e-3),
                f"p_W={np.round(p_W_flip, 4)}")

    # 8) 되돌리기를 안 하면 실제로 틀리는지(테스트가 무의미하지 않은지) 확인
    s = FakeSender()
    fuse_and_send(s, FakeRosSrc(T_identity), True, SW - 1 - cu, SH - 1 - cv,
                  uniform_disparity(D_true), BASELINE, None, False, SW, SH)
    p_W_wrong, _, _ = s.last
    ok &= check("되돌리기 없으면 어긋남(대조군)",
                not np.allclose(p_W_wrong, [0, 0, D_true], atol=0.01),
                f"p_W={np.round(p_W_wrong, 4)} 만큼 빗나감")

    print("\n전체:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
