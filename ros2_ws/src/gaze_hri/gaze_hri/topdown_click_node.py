#!/usr/bin/env python3
"""
[테스트 노드] topdown_click  —  마우스로 시선을 대신한다

안경 없이 로봇 쪽만 독립적으로 개발/테스트하기 위한 노드입니다.
탑다운 카메라 영상을 창에 띄우고, 마우스 커서를 '시선'처럼 흘려보냅니다.

  커서 이동  ->  /gaze/point_raw 로 계속 발행 (시선이 떠도는 것과 동일)
  클릭       ->  그 지점에 2초간 고정 -> dwell_detector가 자연스럽게 발동

이렇게 가장 앞단에서 주입하기 때문에, 아래 노드들이 전부 실제와 똑같이 돕니다.
  dwell_detector -> target_resolver -> task_manager -> arm_server

즉 이 노드로 통과한 시나리오는 나중에 안경을 붙였을 때 그대로 동작합니다.
바뀌는 건 좌표의 출처뿐입니다.

---------------------------------------------------------------------------
핵심 원리: 호모그래피

책상은 평면입니다. 위에서 내려다보는 카메라와 평면 사이에는 3x3 행렬 하나로
표현되는 관계(호모그래피)가 성립합니다. 즉 '화면 픽셀 -> 로봇 좌표 (x, y)'가
한 번의 행렬 곱으로 끝납니다. 깊이도 SLAM도 필요 없습니다.

캘리브레이션은 로봇팔 자신을 자로 씁니다:
  1. 팔을 미리 정한 자세로 보낸다 -> 순기구학으로 손끝의 로봇 좌표를 정확히 안다
  2. 그 손끝이 화면 어디에 있는지 사람이 클릭한다
  3. 네 쌍 이상 모으면 cv2.findHomography 로 행렬이 나온다

---------------------------------------------------------------------------
사용법

  # 1) 캘리브레이션 (최초 1회, 카메라나 로봇을 옮기면 다시)
  ros2 run gaze_hri topdown_click --ros-args -p mode:=calib

  # 2) 테스트 실행
  ros2 run gaze_hri topdown_click

조작:
  좌클릭    집을 물체 -> (상태가 바뀌면) 놓을 자리
  c         현재 선택 취소
  r         화면 새로고침 / 물체 재검출
  q 또는 ESC 종료
"""

import os
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, Float64MultiArray, String

from gaze_hri.kinematics import (TOPDOWN_CALIB_XY, ArmGeometry, IKError,
                                 forward_kinematics, solve_with_fallback)

# 캘리브레이션 지점은 kinematics.py 에 있습니다(ROS 없이 테스트할 수 있도록).
# 테이블면 위 (x, y) 목록이며, 각 지점의 관절각은 IK로 풉니다.
# 전부 같은 높이여야 호모그래피의 평면 가정이 성립합니다 — 자세한 이유는 그쪽 주석 참고.
CALIB_XY = TOPDOWN_CALIB_XY

WIN = "topdown (click = gaze)"


class TopdownClick(Node):

    def __init__(self):
        super().__init__("topdown_click")

        self.declare_parameter("mode", "run")            # run | calib
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("hold_time", 2.0)         # 클릭 후 고정 유지 시간
        self.declare_parameter("table_z", 0.0)           # 테이블면의 로봇 기준 높이
        self.declare_parameter("homography_file",
                               os.path.expanduser("~/.ros/topdown_homography.yaml"))
        # 이 노드가 내보내는 좌표는 호모그래피 결과, 즉 로봇 베이스 기준이다.
        # frame_id 를 "map" 으로 달면 안경 연동 후 SLAM 좌표와 뒤섞인다.
        self.declare_parameter("base_frame", "base_link")
        # 캘리브 때 그리퍼 끝을 테이블면에서 띄울 높이 [m].
        # 0이면 테이블을 긁으므로 살짝 띄우되, 작을수록 평면 가정이 정확해진다.
        self.declare_parameter("calib_height", 0.005)
        # findHomography RANSAC 임계값 [m]. 목표값이 미터이므로 임계값도 미터다.
        self.declare_parameter("ransac_threshold_m", 0.005)

        # 색 기반 물체 검출 (선택). HSV 범위. 기본값은 붉은 계열.
        self.declare_parameter("detect_objects", False)
        self.declare_parameter("hsv_lower", [0, 120, 80])
        self.declare_parameter("hsv_upper", [12, 255, 255])
        self.declare_parameter("min_area", 400)

        # 기구학 (순기구학으로 손끝 좌표를 구할 때 사용)
        self.declare_parameter("base_height", 0.0563)
        self.declare_parameter("shoulder_offset", 0.0304)
        self.declare_parameter("l1", 0.1160)
        self.declare_parameter("l2", 0.1350)
        self.declare_parameter("l3", 0.1350)

        self.mode = self.get_parameter("mode").value
        self.hold_time = float(self.get_parameter("hold_time").value)
        self.table_z = float(self.get_parameter("table_z").value)
        self.homography_path = self.get_parameter("homography_file").value
        self.base_frame = self.get_parameter("base_frame").value
        self.calib_height = float(self.get_parameter("calib_height").value)

        self.geo = ArmGeometry(
            base_height=float(self.get_parameter("base_height").value),
            shoulder_offset=float(self.get_parameter("shoulder_offset").value),
            l1=float(self.get_parameter("l1").value),
            l2=float(self.get_parameter("l2").value),
            l3=float(self.get_parameter("l3").value),
        )

        # ---- 카메라 ----
        idx = int(self.get_parameter("camera_index").value)
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter("width").value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter("height").value))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"카메라 {idx}를 열 수 없습니다. "
                "`ls /dev/video*` 로 인덱스를 확인하세요."
            )

        # ---- 퍼블리셔 ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_point = self.create_publisher(PointStamped, "/gaze/point_raw", qos)
        self.pub_valid = self.create_publisher(Bool, "/gaze/valid", qos)
        self.pub_objects = self.create_publisher(PoseArray, "/objects/poses", 10)
        self.pub_cancel = self.create_publisher(Empty, "/task/cancel", 10)
        self.pub_joints = self.create_publisher(
            Float64MultiArray, "/arm/goto_joints", 10)

        self.create_subscription(String, "/task/state", self.on_state, 10)
        self.task_state = "IDLE"

        # ---- 상태 ----
        self.cursor = None          # (u, v) 픽셀
        self.held_until = 0.0       # 이 시각까지 클릭 지점 고정
        self.held_px = None
        self.H = None               # 호모그래피 3x3
        self.calib_poses = []       # IK로 생성된 캘리브 자세
        self.calib_targets = []     # 각 자세의 테이블면 목표 (x, y)
        self.calib_target = (0.0, 0.0)
        self.calib_pixels = []
        self.calib_world = []
        self.calib_idx = 0
        self.frame = None

        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, self.on_mouse)

        if self.mode == "calib":
            self.start_calibration()
        else:
            self.load_homography()

        self.create_timer(1.0 / 30.0, self.tick)

    # ==================================================================
    # 호모그래피
    # ==================================================================
    def load_homography(self):
        if not os.path.exists(self.homography_path):
            self.get_logger().error(
                f"호모그래피 파일이 없습니다: {self.homography_path}\n"
                "  먼저 캘리브레이션을 하세요:\n"
                "    ros2 run gaze_hri topdown_click --ros-args -p mode:=calib"
            )
            return
        with open(self.homography_path) as f:
            data = yaml.safe_load(f)
        self.H = np.array(data["H"], dtype=float)
        err = data.get("reproj_error_mm", 0.0)
        self.get_logger().info(f"호모그래피 로드 완료 (재투영 오차 {err:.1f} mm)")

    def pixel_to_robot(self, u, v):
        """화면 픽셀 -> 로봇 베이스 기준 (x, y). 한 번의 행렬 곱."""
        if self.H is None:
            return None
        p = self.H @ np.array([float(u), float(v), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def _build_calib_poses(self):
        """테이블면 위 지점들을 IK로 풀어 캘리브레이션 자세를 만든다.

        전부 같은 높이(table_z + calib_height)라서 호모그래피의 평면 가정이 성립한다.
        도달 불가능한 지점은 건너뛰고 경고한다.
        """
        z = self.table_z + self.calib_height
        poses, targets = [], []
        for (x, y) in CALIB_XY:
            try:
                q, _ = solve_with_fallback([x, y, z], self.geo)
            except IKError as exc:
                self.get_logger().warn(
                    f"캘리브 지점 ({x:.2f}, {y:.2f}) 도달 불가 — 건너뜁니다. {exc}")
                continue
            poses.append(q)
            targets.append((x, y))
        return poses, targets

    def start_calibration(self):
        self.calib_poses, self.calib_targets = self._build_calib_poses()
        if len(self.calib_poses) < 4:
            self.get_logger().error(
                f"도달 가능한 캘리브 지점이 {len(self.calib_poses)}개뿐입니다(최소 4개). "
                "링크 길이를 실측했는지, CALIB_XY 가 작업 영역 안인지 확인하세요."
            )
            return
        self.get_logger().info("=" * 58)
        self.get_logger().info(" 호모그래피 캘리브레이션")
        self.get_logger().info(
            f" 캘리브 지점 {len(self.calib_poses)}개, 전부 테이블면 위 "
            f"{self.calib_height * 1000:.0f}mm 높이입니다.")
        self.get_logger().info(" 로봇팔이 자세를 잡으면, 화면에서 그리퍼 끝을 클릭하세요.")
        self.get_logger().info(" 잘못 클릭하면 'c'로 취소하고 다시 클릭합니다.")
        self.get_logger().info("=" * 58)
        self.goto_calib_pose(0)

    def goto_calib_pose(self, i):
        q = self.calib_poses[i]
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q] + [0.0, 0.8]
        self.pub_joints.publish(msg)
        self.calib_target = self.calib_targets[i]
        p = forward_kinematics(q, self.geo)
        self.get_logger().info(
            f"[{i + 1}/{len(self.calib_poses)}] 로봇 좌표 "
            f"({self.calib_target[0]:.3f}, {self.calib_target[1]:.3f}), "
            f"높이 {p[2] * 1000:.0f}mm -- 화면에서 그리퍼 끝을 클릭"
        )

    def finish_calibration(self):
        src = np.array(self.calib_pixels, dtype=np.float32)
        dst = np.array(self.calib_world, dtype=np.float32)
        # ★ 임계값 단위 주의: dst 가 미터라서 임계값도 미터여야 한다.
        #   예전 값 5.0 은 '5미터'라 어떤 오클릭도 걸러내지 못했다(사실상 최소제곱).
        thr = float(self.get_parameter("ransac_threshold_m").value)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr)
        if H is None:
            self.get_logger().error("호모그래피 계산 실패. 클릭 지점을 다시 확인하세요.")
            return
        if mask is not None:
            n_in = int(mask.sum())
            if n_in < len(src):
                self.get_logger().warn(
                    f" 클릭 {len(src)}개 중 {len(src) - n_in}개를 이상치로 제외했습니다 "
                    f"(임계 {thr * 1000:.0f}mm). 그 지점은 잘못 클릭했을 가능성이 큽니다.")
            if n_in < 4:
                self.get_logger().error(
                    " 유효 대응점이 4개 미만입니다. 캘리브레이션을 다시 하세요.")
                return

        # 재투영 오차 확인
        pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.sqrt(((pred - dst) ** 2).sum(axis=1)).mean()) * 1000.0

        os.makedirs(os.path.dirname(self.homography_path), exist_ok=True)
        with open(self.homography_path, "w") as f:
            yaml.safe_dump({
                "H": H.tolist(),
                "reproj_error_mm": err,
                "num_points": len(self.calib_pixels),
            }, f, default_flow_style=False)

        self.H = H
        self.get_logger().info("=" * 58)
        self.get_logger().info(f" 완료. 재투영 오차 {err:.1f} mm")
        if err > 15:
            self.get_logger().warn(" 15mm를 넘습니다. 클릭 정확도나 링크 길이를 확인하세요.")
        self.get_logger().warn(
            " 주의: 재투영 오차는 '캘리브 점 자신'에 대해서만 맞춘 숫자입니다. "
            "실사용 정확도를 보증하지 않습니다. 반드시 다음 단계에서 테이블 위 "
            "여러 지점을 자로 재서 검증하세요.")
        self.get_logger().info(f" 저장: {self.homography_path}")
        self.get_logger().info(" 이제 mode:=run 으로 다시 실행하세요.")
        self.get_logger().info("=" * 58)
        self.mode = "run"

    # ==================================================================
    # 마우스
    # ==================================================================
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "calib":
                self.calib_pixels.append([x, y])
                self.calib_world.append(list(self.calib_target))
                self.calib_idx += 1
                if self.calib_idx >= len(self.calib_poses):
                    self.finish_calibration()
                else:
                    self.goto_calib_pose(self.calib_idx)
            else:
                self.held_px = (x, y)
                self.held_until = time.time() + self.hold_time
                xy = self.pixel_to_robot(x, y)
                if xy:
                    self.get_logger().info(
                        f"클릭: 화면({x}, {y}) -> 로봇({xy[0]:.3f}, {xy[1]:.3f})"
                    )

    def on_state(self, msg: String):
        if msg.data != self.task_state:
            self.task_state = msg.data

    # ==================================================================
    # 물체 검출 (선택)
    # ==================================================================
    def _detect_enabled(self):
        return bool(self.get_parameter("detect_objects").value) and self.H is not None

    def detect_objects(self, frame):
        if not self._detect_enabled():
            return []
        lo = np.array(self.get_parameter("hsv_lower").value, dtype=np.uint8)
        hi = np.array(self.get_parameter("hsv_upper").value, dtype=np.uint8)
        min_area = int(self.get_parameter("min_area").value)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            xy = self.pixel_to_robot(u, v)
            if xy:
                found.append(((u, v), xy, c))
        return found

    # ==================================================================
    # 메인 루프
    # ==================================================================
    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        self.frame = frame
        vis = frame.copy()
        now = time.time()

        objects = self.detect_objects(frame)

        # 검출이 켜져 있으면 결과가 0개여도 발행한다.
        # 컵을 치웠는데 옛 목록이 남아 있으면 없는 컵으로 스냅하기 때문이다.
        if self.mode == "run" and self._detect_enabled():
            pa = PoseArray()
            pa.header.stamp = self.get_clock().now().to_msg()
            pa.header.frame_id = self.base_frame
            for (_, (x, y), _) in objects:
                p = Pose()
                p.position = Point(x=x, y=y, z=self.table_z)
                p.orientation.w = 1.0
                pa.poses.append(p)
            self.pub_objects.publish(pa)

        for (px, xy, contour) in objects:
            cv2.drawContours(vis, [contour], -1, (0, 200, 255), 2)
            cv2.circle(vis, (int(px[0]), int(px[1])), 4, (0, 200, 255), -1)

        # ---- 시선 지점 결정 ----
        if self.mode == "run":
            if now < self.held_until and self.held_px is not None:
                gaze_px = self.held_px
                holding = True
            else:
                gaze_px = self.cursor
                holding = False

            if gaze_px is not None:
                xy = self.pixel_to_robot(*gaze_px)
                if xy is not None:
                    msg = PointStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = self.base_frame
                    msg.point.x, msg.point.y = xy
                    # 집을 물체는 테이블보다 살짝 위, 놓을 자리는 테이블면
                    msg.point.z = self.table_z + (0.05 if self.task_state == "IDLE" else 0.0)
                    self.pub_point.publish(msg)
                    self.pub_valid.publish(Bool(data=True))

                self.draw_cursor(vis, gaze_px, holding, now)

        self.draw_hud(vis)
        cv2.imshow(WIN, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        elif key == ord("c"):
            self.held_until = 0.0
            self.held_px = None
            if self.mode == "calib" and self.calib_pixels:
                self.calib_pixels.pop()
                self.calib_world.pop()
                self.calib_idx -= 1
                self.goto_calib_pose(self.calib_idx)
            else:
                self.pub_cancel.publish(Empty())
                self.get_logger().info("취소")

    # ==================================================================
    # 화면 표시
    # ==================================================================
    def draw_cursor(self, vis, px, holding, now):
        u, v = int(px[0]), int(px[1])
        color = (80, 220, 120) if holding else (200, 200, 200)
        cv2.line(vis, (u - 14, v), (u + 14, v), color, 1)
        cv2.line(vis, (u, v - 14), (u, v + 14), color, 1)

        if holding:
            # 남은 고정 시간을 원호로 표시 (응시 진행률과 비슷한 느낌)
            remain = max(0.0, self.held_until - now)
            frac = 1.0 - remain / self.hold_time
            cv2.ellipse(vis, (u, v), (22, 22), -90, 0, int(360 * frac), color, 3)
        else:
            cv2.circle(vis, (u, v), 22, color, 1)

        xy = self.pixel_to_robot(u, v)
        if xy:
            cv2.putText(vis, f"({xy[0]:+.3f}, {xy[1]:+.3f})", (u + 28, v + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def draw_hud(self, vis):
        h = vis.shape[0]
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (30, 30, 30), -1)

        if self.mode == "calib":
            text = (f"CALIB {self.calib_idx + 1}/{len(self.calib_poses)}"
                    " - click gripper tip")
            color = (120, 200, 255)
        elif self.H is None:
            text = "NO CALIBRATION - run with mode:=calib first"
            color = (80, 80, 255)
        else:
            hint = {
                "IDLE": "click the OBJECT to pick",
                "PICK_SELECTED": "now click WHERE to place it",
                "EXECUTING": "robot is moving...",
            }.get(self.task_state, self.task_state)
            text = f"[{self.task_state}] {hint}"
            color = (120, 255, 160)

        cv2.putText(vis, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        cv2.putText(vis, "click=select   c=cancel   q=quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    def destroy_node(self):
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = TopdownClick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
