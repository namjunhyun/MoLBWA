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
from rclpy.executors import ExternalShutdownException
import yaml
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, PoseStamped
from gaze_hri_msgs.msg import GazeTarget
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, Float32, Float64MultiArray, String

from gaze_hri.kinematics import (TOPDOWN_CALIB_XY, ArmGeometry, IKError,
                                 forward_kinematics, solve_with_fallback)

# 캘리브레이션 지점은 kinematics.py 에 있습니다(ROS 없이 테스트할 수 있도록).
# 테이블면 위 (x, y) 목록이며, 각 지점의 관절각은 IK로 풉니다.
# 전부 같은 높이여야 호모그래피의 평면 가정이 성립합니다 — 자세한 이유는 그쪽 주석 참고.
CALIB_XY = TOPDOWN_CALIB_XY

WIN = "topdown (click = gaze)"


class _HomographyAdapter:
    """table_view.YoloDetector.detect() 가 기대하는 .ready / .to_robot(u,v) 모양."""

    def __init__(self, node):
        self.node = node

    @property
    def ready(self):
        return self.node.H is not None

    def to_robot(self, u, v):
        return self.node.pixel_to_robot(u, v)


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
        # 실측 점이 없는 호모그래피(시연 촬영용 임시값 등)는 기본적으로 거부한다 (2026-09-24).
        # ~/.ros/topdown_homography.yaml 에 2026-09-22 촬영용 가짜 H 가 있었고, reproj null 로
        # 로드 로그에서 TypeError 로 죽었다. 그대로 로드됐다면 팔이 무의미한 좌표로 갔다.
        self.declare_parameter("allow_uncalibrated_homography", False)
        self.declare_parameter("calib_height", 0.005)
        # findHomography RANSAC 임계값 [m]. 목표값이 미터이므로 임계값도 미터다.
        self.declare_parameter("ransac_threshold_m", 0.005)

        # 색 기반 물체 검출 (선택). HSV 범위. 기본값은 붉은 계열.
        self.declare_parameter("detect_objects", False)
        self.declare_parameter("hsv_lower", [0, 120, 80])
        self.declare_parameter("hsv_upper", [12, 255, 255])
        self.declare_parameter("min_area", 400)
        # 2026-09-27: 안경 시연(run_b_demo)은 이 노드를 쓰는데 YOLO 는 control_panel 에만
        # 붙어 있어서, 컵이 있어도 /objects/poses 가 늘 비어 스냅이 전부 실패했다.
        # 2026-09-28: 안경 시연에서 마우스가 창에 한 번 들어가면 커서 위치가 매 프레임
        # /gaze/point_raw 로 나가 안경 시선과 섞였다(응시 확정 0회). 안경 모드에서는 끈다.
        self.declare_parameter("mouse_gaze", True)
        self.declare_parameter("detector", "color")          # color | yolo
        self.declare_parameter("yolo_python", "~/yolo-env/bin/python")
        self.declare_parameter("yolo_model", "~/yolo-env/yolo11l.pt")
        self.declare_parameter("yolo_conf", 0.15)
        self.declare_parameter("yolo_classes", ["cup", "bottle"])
        # 이 노드의 호모그래피는 tip_z 0.13m 평면 -> 박스 중심이 맞다 (table_view 주석 참고)
        self.declare_parameter("yolo_anchor", "center")

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
        self.pub_confirm = self.create_publisher(Empty, "/gaze/confirm", 10)
        self.pub_joints = self.create_publisher(
            Float64MultiArray, "/arm/goto_joints", 10)

        self.create_subscription(String, "/task/state", self.on_state, 10)
        # 2026-09-28: 안경 모드(mouse_gaze=False)에선 커서를 마우스가 아니라 안경 시선으로 그린다.
        # gaze_bridge 가 내는 /gaze/point_raw(base_link, 테이블 위 4.5cm 평면)를 호모그래피 역변환.
        self._eye_pt = None          # (x, y, 수신시각)
        self._dwell_prog = 0.0
        if not bool(self.get_parameter("mouse_gaze").value):
            self.create_subscription(PointStamped, "/gaze/point_raw", self.on_eye_point, qos)
            self.create_subscription(Float32, "/gaze/dwell_progress", self.on_dwell_prog, qos)
            # 광선 원점(머리 = p_eye). 평면 교차점 하나만 찍으면 키 큰 물체를 볼 때 교차점이
            # 물체 뒤로 밀려 "어긋나 보인다" -> 광선 전체를 선으로 그린다.
            self._head = None
            self.create_subscription(PoseStamped, "/head/pose", self.on_head, qos)
            # 확정된 목표(집을 물체/놓을 자리)를 마커로 남긴다. 작업이 IDLE 로 돌아가면 지운다.
            self._targets = {}
            self.create_subscription(GazeTarget, "/gaze/target", self.on_target, 10)
        self.task_state = "IDLE"

        # ---- 상태 ----
        self.cursor = None          # (u, v) 픽셀
        self.held_until = 0.0       # 이 시각까지 클릭 지점 고정
        self.held_px = None
        self.H = None               # 호모그래피 3x3
        self._yolo = None           # detector:=yolo 일 때 첫 검출에서 띄운다
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
        n = int(data.get("num_points") or 0)
        err = data.get("reproj_error_mm")
        if n < 4 or err is None:
            if not bool(self.get_parameter("allow_uncalibrated_homography").value):
                self.get_logger().error(
                    f"호모그래피가 실측값이 아닙니다 (점 {n}개, 오차 {err}) — 로드하지 않습니다. "
                    "tools/topdown_calib.py 로 캘리브하세요. (촬영용으로 일부러 쓰려면 "
                    "allow_uncalibrated_homography:=true)")
                self.H = None
                return
            self.get_logger().warn(f"★ 실측이 아닌 호모그래피를 씁니다 (점 {n}개) — 팔을 보내지 마세요")
        self.H = np.array(data["H"], dtype=float)
        loo = data.get("loo_error_mm")
        self.get_logger().info(
            f"호모그래피 로드 완료 (점 {n}개, 재투영 {float(err or 0):.1f} mm"
            + (f", 처음 보는 점 {float(loo):.1f} mm)" if loo is not None else ")"))

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
    def _sticky_objects(self, objects, now, keep_s=1.5, merge_m=0.05):
        """2026-09-29: YOLO 신뢰도가 기준 근처면 프레임마다 잡혔다 놓쳤다 해서 /objects/poses 가
        비었다 -> 최근 keep_s 초 안에 본 물체는 유지한다(5cm 안이면 같은 물체로 보고 갱신)."""
        if not hasattr(self, "_seen"):
            self._seen = []          # [(px, xy, contour, t)]
        for (px, xy, c) in objects:
            for i, (_, xy0, _, _) in enumerate(self._seen):
                if np.hypot(xy[0] - xy0[0], xy[1] - xy0[1]) < merge_m:
                    self._seen[i] = (px, xy, c, now)
                    break
            else:
                self._seen.append((px, xy, c, now))
        self._seen = [o for o in self._seen if now - o[3] <= keep_s]
        return [(px, xy, c) for (px, xy, c, _) in self._seen]

    def on_eye_point(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            return
        self._eye_pt = (msg.point.x, msg.point.y, time.time())

    def on_head(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            return
        p = msg.pose.position
        self._head = (np.array([p.x, p.y, p.z]), time.time())

    def on_target(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            return
        self._targets[msg.role] = (msg.point.x, msg.point.y)

    def draw_targets(self, vis):
        for role, color, label in (("pick", (0, 220, 0), "PICK"), ("place", (255, 0, 255), "PLACE")):
            t = getattr(self, "_targets", {}).get(role)
            if t is None:
                continue
            q = self.robot_to_pixel(*t)
            if q is None:
                continue
            u, v = int(q[0]), int(q[1])
            if role == "pick":
                cv2.circle(vis, (u, v), 30, color, 3)
            else:
                cv2.rectangle(vis, (u - 22, v - 22), (u + 22, v + 22), color, 3)
            cv2.drawMarker(vis, (u, v), color, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(vis, label, (u - 24, v - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def on_dwell_prog(self, msg):
        self._dwell_prog = float(msg.data)

    def robot_to_pixel(self, x, y):
        """로봇 베이스 (x, y) -> 화면 픽셀 (pixel_to_robot 의 역)."""
        if self.H is None:
            return None
        p = np.linalg.inv(self.H) @ np.array([float(x), float(y), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def draw_eye_cursor(self, vis, now):
        # 팔이 놓을 수 있는 범위(0.16~0.29m, ±65°) 경계를 옅게 그린다
        for rr in (0.16, 0.29):
            arc = [self.robot_to_pixel(rr * np.cos(a), rr * np.sin(a))
                   for a in np.radians(np.linspace(-65, 65, 27))]
            arc = [(int(a[0]), int(a[1])) for a in arc if a is not None]
            if len(arc) > 1:
                cv2.polylines(vis, [np.array(arc, np.int32)], False, (180, 120, 180), 1, cv2.LINE_AA)
        pt = self._eye_pt
        if pt is None or now - pt[2] > 0.3:
            cv2.putText(vis, "gaze: -", (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            return
        px = self.robot_to_pixel(pt[0], pt[1])
        if px is None:
            return
        h, w = vis.shape[:2]
        u, v = int(np.clip(px[0], 0, w - 1)), int(np.clip(px[1], 0, h - 1))
        color = (0, 140, 255)                      # 주황 = 안경 시선
        # 광선을 선으로: 머리 -> 응시점 방향으로 높이 25cm ~ 0cm 구간을 그린다. 이 선이 물체를
        # 지나가면 그 물체를 보고 있는 것 (선택도 광선-물체 중심 거리로 한다).
        hd = self._head
        if hd is not None and now - hd[1] < 0.5:
            o = hd[0]
            d = np.array([pt[0], pt[1], self.table_z + 0.045]) - o
            if abs(d[2]) > 1e-6:
                seg, table_xy = [], None
                for z in (0.25, 0.0):
                    t = (self.table_z + z - o[2]) / d[2]
                    if t > 0:
                        w = o + t * d
                        if z == 0.0:
                            table_xy = w[:2]
                        q = self.robot_to_pixel(*w[:2])
                        if q is not None:
                            seg.append((int(np.clip(q[0], -2000, 4000)), int(np.clip(q[1], -2000, 4000))))
                if len(seg) == 2:
                    cv2.line(vis, seg[0], seg[1], (0, 140, 255), 1, cv2.LINE_AA)
                if table_xy is not None:
                    # 실제로 놓이는 점 = target_resolver._clamp_place 와 같은 규칙으로 자른 테이블 교점
                    r = float(np.hypot(*table_xy)); yaw = float(np.arctan2(table_xy[1], table_xy[0]))
                    r_c = min(max(r, 0.16), 0.29); yaw_c = max(-np.radians(65), min(np.radians(65), yaw))
                    q = self.robot_to_pixel(r_c * np.cos(yaw_c), r_c * np.sin(yaw_c))
                    if q is not None:
                        qp = (int(q[0]), int(q[1]))
                        cv2.circle(vis, qp, 6, (255, 0, 255), -1)
                        cv2.putText(vis, "place" + (" (clamped)" if abs(r_c - r) > 1e-3 or abs(yaw_c - yaw) > 1e-3 else ""),
                                    (qp[0] + 8, qp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (u - 16, v), (u + 16, v), color, 2)
        cv2.line(vis, (u, v - 16), (u, v + 16), color, 2)
        cv2.circle(vis, (u, v), 24, color, 1)
        if self._dwell_prog > 0:
            cv2.ellipse(vis, (u, v), (24, 24), -90, 0, int(360 * min(1.0, self._dwell_prog)), color, 4)
        cv2.putText(vis, f"gaze ({pt[0]:+.3f}, {pt[1]:+.3f})", (u + 30, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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
        if msg.data == "IDLE" and hasattr(self, "_targets"):
            self._targets = {}                 # 작업 끝/취소/타임아웃 -> 확정 마커 지움
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
        if self.get_parameter("detector").value == "yolo":
            if self._yolo is None:
                from gaze_hri.table_view import YoloDetector
                self._yolo = YoloDetector(
                    self.get_parameter("yolo_python").value,
                    self.get_parameter("yolo_model").value,
                    self.get_parameter("yolo_conf").value,
                    self.get_parameter("yolo_classes").value,
                    self.get_parameter("min_area").value,
                    logger=self.get_logger(),
                    anchor=self.get_parameter("yolo_anchor").value)
                self.get_logger().info(
                    f"물체 검출: YOLO (좌표 기준점: {self.get_parameter('yolo_anchor').value})")
            return self._yolo.detect(frame, _HomographyAdapter(self))
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

        objects = self._sticky_objects(self.detect_objects(frame), now)
        # 2026-09-28: 컵이 화면에 있는데 노란 박스가 안 뜨는 원인을 밖에서 보려고 — 2초마다
        # 최신 프레임을 저장하고 검출 개수를 남긴다 (카메라는 이 노드만 열 수 있다).
        if now - getattr(self, "_last_dump", 0.0) > 2.0:
            self._last_dump = now
            try:
                cv2.imwrite(os.path.expanduser("~/.ros/topdown_last.jpg"), frame)
            except Exception:
                pass
            if self._detect_enabled():
                self.get_logger().info(
                    f"물체 검출 {len(objects)}개: "
                    + ", ".join(f"({xy[0]:.3f},{xy[1]:.3f})" for (_, xy, _) in objects),
                    throttle_duration_sec=5.0)

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
        if self.mode == "run" and not bool(self.get_parameter("mouse_gaze").value):
            self.draw_targets(vis)
            self.draw_eye_cursor(vis, now)
        elif self.mode == "run":
            if now < self.held_until and self.held_px is not None:
                gaze_px = self.held_px
                holding = True
            else:
                gaze_px = self.cursor
                holding = False

            if gaze_px is not None:
                xy = (self.pixel_to_robot(*gaze_px)
                      if bool(self.get_parameter("mouse_gaze").value) else None)
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
        elif key == ord(" ") and not bool(self.get_parameter("mouse_gaze").value):
            # 2026-09-28: 안경 모드 — 스페이스바 = 지금 보고 있는 곳으로 확정 (dwell 대기 대신)
            self.pub_confirm.publish(Empty())
            self.get_logger().info("스페이스: 지금 시선으로 확정 요청")
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
        if self._yolo is not None and hasattr(self._yolo, "close"):
            self._yolo.close()
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():                # SIGINT/SIGTERM 이면 rclpy 가 이미 내렸다
            rclpy.shutdown()


if __name__ == "__main__":
    main()
