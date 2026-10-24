#!/usr/bin/env python3
"""
[통합 GUI] control_panel — 한 창에서 전체 파이프라인을 보고 조작한다

무엇이 보이는가
---------------
  · 탑다운 카메라 영상
  · 검출된 컵 (번호와 로봇 좌표)
  · 시선(또는 마우스) 위치와 응시 진행률
  · 어느 컵이 선택됐는지 — 집을 것(주황) / 놓을 곳(초록)
  · 로봇이 지금 무슨 동작 중인지 (APPROACH_PICK, GRASP, ...)
  · 팔 끝이 지금 어디 있는지 (관절각 -> 순기구학 -> 화면 위 점)
  · 좌표계/캘리브레이션이 준비됐는지

무엇을 할 수 있는가
-------------------
  좌클릭   지금 보는 곳을 선택 (마우스 모드)
  c        선택 취소 / 동작 중단
  space    비상 정지 (동작 중단)
  d        물체 검출 켜기·끄기
  q, ESC   종료

입력 소스 (`input_source`)
--------------------------
  mouse : 마우스 커서를 시선 대신 쓴다. 이 노드가 /gaze/point_raw 를 발행한다.
  gaze  : 안경이 만든 시선을 쓴다. gaze_bridge 가 발행한 것을 구독해 표시만 한다.

  두 모드에서 아래 노드들(dwell_detector -> target_resolver -> task_manager ->
  arm_server)은 똑같이 동작한다. 바뀌는 건 좌표의 출처뿐이다.

좌표계
------
이 노드가 내보내는 좌표는 호모그래피 결과, 즉 **로봇 베이스 기준**이다.
그래서 frame_id 에 base_frame("base_link")을 단다. 시선은 SLAM world(map)로
들어오므로, 둘을 섞지 않도록 target_resolver 가 전부 base_frame 으로 변환해
비교한다. 자세한 내용은 docs/13_gaze_to_arm.md.

화면 그리기는 `panel_render.py` 에 있다. ROS 없이도 미리 볼 수 있다:

    python -m gaze_hri.panel_render preview.png executing
"""

import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from gaze_hri_msgs.msg import Fixation, GazeTarget
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32, Int32, String

from std_msgs.msg import Float64MultiArray

from gaze_hri.hud import Hud
from gaze_hri.kinematics import (TOPDOWN_CALIB_XY, ArmGeometry,
                                 forward_kinematics)
from gaze_hri.panel_render import PanelState, render
from gaze_hri.table_view import (HomographyCalibration, ObjectDetector, YoloDetector,
                                 TableHomography, find_camera,
                                 hsv_range_from_pixel)

WIN = "MoLBWA gaze pick and place"


class ControlPanel(Node):

    def __init__(self):
        super().__init__("control_panel")

        self.declare_parameter("input_source", "mouse")   # mouse | gaze
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("ocams", False)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fourcc", "")
        # 2026-10-22: 1280x720 화면을 640 으로 줄여 넣으니 컵이 75% 프레임에서 안 잡혔다 -> 화면 크기 그대로
        self.declare_parameter("yolo_imgsz", 640)
        # 2026-10-22: 화면 녹화. 폴더를 주면 GUI 화면(탑다운 + 광선·PICK/PLACE + 사이드바)을 15fps avi 로 남긴다
        self.declare_parameter("record_dir", "")
        self._rec, self._rec_t, self._rec_path = None, 0.0, None       # 2026-10-22: 탑다운 1280x720 은 "MJPG" (YUYV 는 10fps)
        self.declare_parameter("sidebar_width", 320)
        self.declare_parameter("table_z", 0.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("hold_time", 2.0)
        self.declare_parameter("homography_file",
                               os.path.expanduser("~/.ros/topdown_homography.yaml"))
        self.declare_parameter("detect_objects", True)
        # 2026-10-12: 검출 박스 아래 변(카메라 쪽 바닥 가장자리) -> 물체 중심 보정 [m]
        self.declare_parameter("object_radius", 0.0)
        self.declare_parameter("mouse_click_only", True)
        self.declare_parameter("reach_band", [0.0])     # [r_min, r_max] m, 2개가 아니면 안 그림   # 마우스 모드: 클릭만 응시로 친다
        self.declare_parameter("hsv_lower", [0, 120, 80])
        self.declare_parameter("hsv_upper", [12, 255, 255])
        self.declare_parameter("min_area", 400)
        self.declare_parameter("detector", "color")          # color | yolo
        self.declare_parameter("yolo_python", "~/yolo-env/bin/python")
        self.declare_parameter("yolo_model", "~/yolo-env/yolo11s.pt")
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_classes", ["cup", "bottle"])
        self.declare_parameter("show_arm_tip", True)
        # 2026-10-14: 두 팔. 이 노드의 homography_file 은 첫 팔(좌표 기준) 것이고, 다른 팔은
        # 띠/팔 끝 표시만 그 팔의 호모그래피로 그린다.
        self.declare_parameter("joint_states_topic", "/joint_states")
        # 두 팔 GUI 에서는 k(캘리브)를 막는다 — 캘리브 자세 명령을 받을 팔이 없고, 끝까지 가면
        # 팔 a 호모그래피를 덮어쓴다 (2026-10-14 실수로 k 눌림)
        self.declare_parameter("allow_calib", True)
        self.declare_parameter("arm_label", "")
        self.declare_parameter("other_arm_labels", [""])
        self.declare_parameter("other_arm_homographies", [""])
        self.declare_parameter("other_arm_joint_topics", [""])
        # 캘리브레이션 (GUI 안에서 k 키로 실행)
        self.declare_parameter("calib_height", 0.005)
        self.declare_parameter("ransac_threshold_m", 0.005)
        # 기구학 (팔 끝을 화면에 그릴 때 사용)
        self.declare_parameter("base_height", 0.0563)
        self.declare_parameter("shoulder_offset", 0.0304)
        self.declare_parameter("l1", 0.1160)
        self.declare_parameter("l2", 0.1350)
        self.declare_parameter("l3", 0.1100)

        self.source = self.get_parameter("input_source").value
        self.table_z = float(self.get_parameter("table_z").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.hold_time = float(self.get_parameter("hold_time").value)
        self.sidebar_w = int(self.get_parameter("sidebar_width").value)
        self.detect_on = bool(self.get_parameter("detect_objects").value)
        self.show_tip = bool(self.get_parameter("show_arm_tip").value)

        self.geo = ArmGeometry(
            base_height=float(self.get_parameter("base_height").value),
            shoulder_offset=float(self.get_parameter("shoulder_offset").value),
            l1=float(self.get_parameter("l1").value),
            l2=float(self.get_parameter("l2").value),
            l3=float(self.get_parameter("l3").value),
        )

        # ---- 화면 / 좌표 ----
        self.hud = Hud()
        self.homography = TableHomography()
        ok, why = self.homography.load(self.get_parameter("homography_file").value)
        if not ok:
            self.get_logger().error(
                f"호모그래피를 못 읽었습니다 ({why}). 좌표를 낼 수 없습니다.\n"
                "  먼저 캘리브레이션하세요:\n"
                "    ros2 launch gaze_hri robot_only.launch.py mode:=calib "
                "backend:=feetech"
            )
        if self.get_parameter("detector").value == "yolo":
            self.detector = YoloDetector(
                self.get_parameter("yolo_python").value,
                self.get_parameter("yolo_model").value,
                self.get_parameter("yolo_conf").value,
                self.get_parameter("yolo_classes").value,
                self.get_parameter("min_area").value,
                logger=self.get_logger(),
                imgsz=int(self.get_parameter("yolo_imgsz").value),
                anchor_radius=self.get_parameter("object_radius").value)
            self.get_logger().info(f"물체 검출: YOLO (입력 {int(self.get_parameter('yolo_imgsz').value)})")
        else:
            self.detector = ObjectDetector(
                self.get_parameter("hsv_lower").value,
                self.get_parameter("hsv_upper").value,
                self.get_parameter("min_area").value)
        self.ocams = bool(self.get_parameter("ocams").value)
        if self.ocams:
            # oCamS-1MGN-U 는 YUYV raw 로 받아야 Y채널(=좌영상)을 꺼낼 수 있다.
            # 그냥 읽으면 YUYV 를 BGR 로 오해해서 화면 전체가 초록색으로 나온다.
            cam_idx = int(self.get_parameter("camera_index").value)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                         int(self.get_parameter("width").value))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                         int(self.get_parameter("height").value))
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"oCamS 카메라 {cam_idx} 를 열 수 없습니다. "
                    "`ls /dev/v4l/by-id` 로 인덱스를 확인하세요.")
            self.get_logger().info(f"카메라 {cam_idx} 사용 (oCamS 좌영상)")
        else:
            # 카메라 인덱스는 USB 를 다시 꽂으면 바뀐다. 지정한 게 안 열리면 훑는다.
            self.cap, cam_idx = find_camera(
                self.get_parameter("camera_index").value,
                self.get_parameter("width").value,
                self.get_parameter("height").value,
                fourcc=self.get_parameter("fourcc").value)
            self.get_logger().info(f"카메라 {cam_idx} 사용")
        self._check_image_size(self.homography, "A" if not self.get_parameter("arm_label").value
                               else self.get_parameter("arm_label").value)

        # 캘리브레이션 (GUI 안에서 바로 한다 — 카메라를 두 프로그램이 다투지 않도록)
        self.calib = None
        self.calib_file = self.get_parameter("homography_file").value

        # ---- 통신 ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_objects = self.create_publisher(PoseArray, "/objects/poses", 10)
        # 2026-10-14: /objects/poses 와 같은 순서의 클래스 이름 (JSON 목록). 물병->컵 이면 따르기
        self.pub_labels = self.create_publisher(String, "/objects/labels", 10)
        # 2026-10-14: 따르기 보정 — p = 보정 모드 켜기/끄기, r = 리더로 맞춘 자세 기록 (+화면 저장)
        self.pub_teleop = self.create_publisher(String, "/teleop/cmd", 10)
        self.pub_pour_calib = self.create_publisher(Bool, "/task/pour_calib", 10)
        self.pour_calib = False
        # 2026-10-22: g = 집기 보정 모드, 따르기 자리 표시 (task_manager pour_spot 과 같은 값)
        self.pub_grasp_calib = self.create_publisher(Bool, "/task/grasp_calib", 10)
        self.grasp_calib = False
        self.declare_parameter("pour_spot", [0.33, -0.205])
        self.declare_parameter("snapshot_dir", "~/MoLBWA-gaze-hri/notes/pour_calib/snap")
        self.pub_cancel = self.create_publisher(Empty, "/task/cancel", 10)
        # 캘리브레이션 때 팔을 기준 자세로 보낸다
        self.pub_joints = self.create_publisher(
            Float64MultiArray, "/arm/goto_joints", 10)
        self.pub_point = self.pub_valid = None
        if self.source == "mouse":
            self.pub_point = self.create_publisher(PointStamped, "/gaze/point_raw", qos)
            self.pub_valid = self.create_publisher(Bool, "/gaze/valid", qos)
        else:
            self.create_subscription(PointStamped, "/gaze/point_raw",
                                     self.on_gaze_point, qos)
            # 2026-10-22: 10-01 탑다운 창처럼 광선을 그린다 (머리 -> 스냅 전 응시점) + 광선이 붙은 물체
            self.create_subscription(PoseStamped, "/head/pose", self.on_head, qos)
            self.create_subscription(PointStamped, "/gaze/ray_point", self.on_ray_point, qos)
            self.create_subscription(Int32, "/gaze/snapped", self.on_snapped, qos)
        self._head = self._ray_pt = None
        self._snapped = (-1, 0.0)

        self.create_subscription(Fixation, "/gaze/fixation", self.on_fixation, 10)
        self.create_subscription(Float32, "/gaze/dwell_progress", self.on_progress, qos)
        self.create_subscription(GazeTarget, "/gaze/target", self.on_target, 10)
        self.create_subscription(String, "/task/state", self.on_state, 10)
        self.create_subscription(String, "/arm/phase", self.on_phase, 10)
        self.create_subscription(String, "/task/message", self.on_task_message, 10)
        self.create_subscription(JointState, self.get_parameter("joint_states_topic").value,
                                 self.on_joints, 10)
        self.other_arms = []          # [[이름, TableHomography, tcp]]
        self._reach_cache = {}
        for name, hf, topic in zip(self.get_parameter("other_arm_labels").value,
                                   self.get_parameter("other_arm_homographies").value,
                                   self.get_parameter("other_arm_joint_topics").value):
            if not name:
                continue
            h = TableHomography()
            ok, why = h.load(os.path.expanduser(hf))
            if not ok:
                self.get_logger().error(f"팔 {name} 호모그래피를 못 읽었습니다 ({why})")
                continue
            self._check_image_size(h, name)
            entry = [name, h, None]
            self.other_arms.append(entry)
            self.create_subscription(JointState, topic,
                                     lambda m, e=entry: self._on_other_joints(e, m), 10)

        # ---- 상태 ----
        self.mode = "run"            # run | calib
        self.frame = None
        self.objects = []
        self.cursor = None
        self.gaze_px = None
        self.held_until = 0.0
        self.held_px = None
        self.task_state = "IDLE"
        self.dwell_progress = 0.0
        self.has_fixation = False
        self.pick_target = None
        self.place_target = None
        self.target_info = None
        self.arm_phase = ""
        self.tcp = None
        self.notice = ""
        self.notice_until = 0.0
        self._fps = 0.0
        self._fps_t = time.time()

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info(
            f"control_panel 시작 (입력 {self.source}, "
            f"{'한글' if self.hud.korean else '영문'} 표시)")

    # ==================================================================
    # 수신
    # ==================================================================
    def on_gaze_point(self, msg: PointStamped):
        """시선 모드: 들어온 점을 화면 위치로 되돌려 표시한다."""
        self.gaze_px = self.homography.to_pixel(msg.point.x, msg.point.y)

    def on_head(self, msg):
        p = msg.pose.position
        self._head = (np.array([p.x, p.y, p.z]), time.time())

    def on_ray_point(self, msg):
        self._ray_pt = (np.array([msg.point.x, msg.point.y, msg.point.z]), time.time())

    def on_snapped(self, msg):
        self._snapped = (int(msg.data), time.time())

    def _ray_pixels(self, now):
        """머리 -> 응시점 광선의 책상 위 25cm ~ 0cm 구간을 화면 픽셀 두 점으로."""
        if self._head is None or self._ray_pt is None:
            return None
        if now - self._head[1] > 0.5 or now - self._ray_pt[1] > 0.5:
            return None
        o, d = self._head[0], self._ray_pt[0] - self._head[0]
        if abs(d[2]) < 1e-6:
            return None
        tz = float(self.get_parameter("table_z").value)
        seg = []
        for z in (0.25, 0.0):
            t = (tz + z - o[2]) / d[2]
            if t <= 0:
                return None
            w = o + t * d
            q = self.homography.to_pixel(float(w[0]), float(w[1]))
            if q is None:
                return None
            seg.append((float(np.clip(q[0], -4000, 6000)), float(np.clip(q[1], -4000, 6000))))
        return seg

    def on_progress(self, msg: Float32):
        self.dwell_progress = float(msg.data)

    def on_fixation(self, _msg: Fixation):
        self.has_fixation = True

    def on_target(self, msg: GazeTarget):
        self.target_info = {"snapped": msg.snapped, "confidence": msg.confidence,
                            "label": msg.label, "frame": msg.header.frame_id}
        if msg.role == "pick":
            self.pick_target = (msg.point.x, msg.point.y)
            self.place_target = None
        else:
            self.place_target = (msg.point.x, msg.point.y)

    def on_state(self, msg: String):
        if msg.data == self.task_state:
            return
        self.task_state = msg.data
        if msg.data == "IDLE":
            self.pick_target = None
            self.place_target = None
            self.target_info = None
            self.arm_phase = ""
            self.has_fixation = False

    def on_phase(self, msg: String):
        self.arm_phase = msg.data

    def on_joints(self, msg: JointState):
        if not self.show_tip or len(msg.position) < 4:
            return
        try:
            p = forward_kinematics(list(msg.position[:4]), self.geo)
            self.tcp = (float(p[0]), float(p[1]), float(p[2]))
        except Exception:                               # noqa: BLE001
            self.tcp = None

    def _reach_arc(self, label):
        """~/.ros/so101_reach_{a|b}.json (손으로 그린 띠). 파일이 바뀌면 다시 읽는다."""
        if not label:
            return None
        f = os.path.expanduser(f"~/.ros/so101_reach_{label.lower()}.json")
        try:
            mt = os.path.getmtime(f)
        except OSError:
            return None
        c = self._reach_cache.get(f)
        if c is None or c[0] != mt:
            try:
                c = (mt, json.load(open(f)))
            except ValueError:
                return None
            self._reach_cache[f] = c
        return c[1]

    def _band_at(self, label, xy, lo, hi):
        """그 팔의 띠 (손으로 그린 게 있으면 가장 가까운 pan 칸)."""
        arc = self._reach_arc(label)
        if not arc or not arc.get("pan_deg"):
            return lo, hi
        pan = math.degrees(math.atan2(xy[1], xy[0]))
        k = min(range(len(arc["pan_deg"])), key=lambda i: abs(arc["pan_deg"][i] - pan))
        if abs(arc["pan_deg"][k] - pan) > 7.5:
            return None                     # 그 방향은 손으로 그린 범위 밖
        return arc["r_min"][k], arc["r_max"][k]

    def _object_info(self):
        """검출 물체마다 각 팔 받침대까지 거리 글자와, 어느 팔이든 집기 띠 안인지."""
        band = self.get_parameter("reach_band").value
        if len(band) != 2:
            return []
        lo, hi = band
        homs = [(self.get_parameter("arm_label").value or "", self.homography)] + \
               [(n, h) for n, h, _ in self.other_arms]
        labels = list(getattr(self, "_obj_labels", []))

        def dists(u, v):
            out_ = []
            for name, h in homs:
                xy = h.to_robot(u, v)
                out_.append((name, None if xy is None else math.hypot(*xy)))
            return out_

        def in_band(name, u, v, r, label=None):
            h = dict(homs)[name]
            xy = h.to_robot(u, v)
            # 물병은 가까우면 위에서 수직으로 내려와 잡는다 (arm_server vertical_min_radius 0.27)
            b = self._band_at(name, xy, lo, hi) if xy else None
            return b is not None and r is not None and b[0] <= r <= b[1]
        # 2026-10-14: 물병을 고른 상태면 컵마다 '따르기 가능' 을 따로 본다. 컵은 따르는 팔에서
        # 컵 중심(기준점에서 팔 쪽으로 3cm) 23~40cm, 그 팔에서 병도 집기 띠(수평 집기) 안.
        bottle_r = None
        if self.task_state == "PICK_SELECTED" and self.pick_target is not None:
            for i, ((u, v), (x, y), _) in enumerate(self.objects):
                if i < len(labels) and labels[i] == "bottle" and \
                        math.hypot(x - self.pick_target[0], y - self.pick_target[1]) < 0.06:
                    bottle_r = {n: (r, in_band(n, u, v, r, "bottle")) for n, r in dists(u, v)}
        out = []
        for i, ((u, v), _, _) in enumerate(self.objects):
            ds = dists(u, v)
            txt = " ".join(f"{n}{r * 100:.0f}" for n, r in ds if r is not None) + "cm"
            if bottle_r is not None and i < len(labels) and labels[i] == "cup":
                ok = any(r is not None and 0.23 <= r <= 0.40
                         and bottle_r.get(n) is not None and bottle_r[n][1]
                         for n, r in ds)
                out.append((txt + (" 따르기 OK" if ok else " 따르기 X"), ok))
            else:
                lab = labels[i] if i < len(labels) else None
                out.append((txt, any(in_band(n, u, v, r, lab) for n, r in ds)))
        return out

    def _on_other_joints(self, entry, msg):
        if not self.show_tip or len(msg.position) < 4:
            return
        try:
            p = forward_kinematics(list(msg.position[:4]), self.geo)
            entry[2] = (float(p[0]), float(p[1]), float(p[2]))
        except Exception:                               # noqa: BLE001
            entry[2] = None

    # ==================================================================
    # 입력
    # ==================================================================
    def on_mouse(self, event, x, y, flags, param):
        view_w = self.frame.shape[1] if self.frame is not None else 640
        if x >= view_w:                                  # 사이드바는 무시
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.pick_color(x, y)
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "calib":
            self.calib_click(x, y)
        elif self.source == "mouse":
            # 2026-10-14: 물체 박스 안을 찍으면 그 물체의 기준점(박스 아래 변 가운데)을 찍은 것으로
            # 친다. 컵 몸통을 찍었는데 기준점에서 9.6cm 라 '빈 곳'으로 보고 물병을 컵 옆에 그냥
            # 옮겨 버렸다 (따르기 대신).
            for ((u, v), _, c) in self.objects:
                if c is not None and cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0:
                    x, y = int(round(u)), int(round(v))
                    break
            self.held_px = (x, y)
            self.held_until = time.time() + self.hold_time
            xy = self.homography.to_robot(x, y)
            if xy:
                self.say(f"선택: ({xy[0]:+.3f}, {xy[1]:+.3f})",
                         f"Selected: ({xy[0]:+.3f}, {xy[1]:+.3f})")

    def handle_key(self, key):
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("k"):
            if not self.get_parameter("allow_calib").value and self.mode != "calib":
                self.say("두 팔 화면에서는 캘리브를 할 수 없습니다 (팔별 topdown_calib 사용)",
                         "Calibration disabled in dual-arm GUI")
                return
            self.toggle_calibration()
        elif key == ord("u") and self.mode == "calib":
            if self.calib.undo():
                self.goto_calib_pose()
                self.say("직전 클릭을 취소했습니다", "Undid last click")
        elif key == ord("c"):
            if self.mode == "calib":
                self.toggle_calibration()          # 캘리브 중단
                return
            self.held_until, self.held_px = 0.0, None
            self.pub_cancel.publish(Empty())
            self.say("취소했습니다", "Cancelled")
        elif key == ord(" "):
            self.held_until, self.held_px = 0.0, None
            self.pub_cancel.publish(Empty())
            self.say("비상 정지 — 동작을 중단했습니다", "EMERGENCY STOP")
        elif key == ord("p"):
            self.pour_calib = not self.pour_calib
            self.say("따르기 보정 모드 켬 — 물병 -> 컵 을 고르면 리더로 맞춰 기록합니다"
                     if self.pour_calib else "따르기 보정 모드 끔",
                     "Pour calibration ON" if self.pour_calib else "Pour calibration OFF", 4.0)
        elif key in (ord("8"), ord("4"), ord("2"), ord("6")):
            self.nudge(*{ord("8"): (0, -1), ord("4"): (-1, 0), ord("2"): (0, 1),
                         ord("6"): (1, 0)}[key])
        elif key in (ord("+"), ord("="), ord("-")):
            d = -1 if key == ord("-") else 1
            self.pub_teleop.publish(String(data=f"tilt:{d}"))
            self.say("더 기울임 +5°" if d > 0 else "덜 기울임 -5°", f"tilt {d * 5:+d} deg", 1.0)
        elif key == ord("g"):
            self.grasp_calib = not self.grasp_calib
            self.say("집기 보정 모드 켬 — 집기 직전에 멈춥니다 (8/4/2/6 로 맞추고 r)"
                     if self.grasp_calib else "집기 보정 모드 끔",
                     "Grasp calibration ON" if self.grasp_calib else "Grasp calibration OFF", 4.0)
        elif key == ord("r"):
            path = self.save_snapshot()
            self.pub_teleop.publish(String(data=f"record:{path or ''}"))
            self.say("기록했습니다", "Recorded")
        elif key == ord("d"):
            self.detect_on = not self.detect_on
            self.say(f"물체 검출 {'켬' if self.detect_on else '끔'}",
                     f"Detection {'ON' if self.detect_on else 'OFF'}")

    # ==================================================================
    # 캘리브레이션 (창 하나에서 바로)
    # ==================================================================
    def toggle_calibration(self):
        if self.mode == "calib":
            self.mode = "run"
            self.calib = None
            self.say("캘리브레이션을 중단했습니다", "Calibration aborted")
            return

        self.calib = HomographyCalibration(
            self.geo, table_z=self.table_z,
            calib_height=float(self.get_parameter("calib_height").value),
            ransac_threshold_m=float(
                self.get_parameter("ransac_threshold_m").value),
            logger=self.get_logger())
        n = self.calib.build(TOPDOWN_CALIB_XY)
        if n < 4:
            self.calib = None
            self.say(f"도달 가능한 캘리브 지점이 {n}개뿐입니다 — 링크 길이 확인",
                     f"Only {n} reachable points — check link lengths", 5.0)
            return
        self.mode = "calib"
        self.pub_cancel.publish(Empty())    # 진행 중이던 작업은 정리
        self.say(f"캘리브레이션 시작 — 그리퍼 끝을 클릭하세요 (0/{n})",
                 f"Calibration started — click the gripper tip (0/{n})", 4.0)
        self.goto_calib_pose()

    def goto_calib_pose(self):
        """현재 캘리브 지점으로 팔을 보낸다. 그리퍼는 살짝 벌려 끝이 잘 보이게."""
        q = self.calib.current_pose()
        if q is None:
            return
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q] + [0.0, 0.8]
        self.pub_joints.publish(msg)

    def calib_click(self, x, y):
        if not self.calib.add_click(x, y):
            return
        if not self.calib.done:
            self.say(f"{self.calib.index}/{self.calib.total} — 다음 자세로",
                     f"{self.calib.index}/{self.calib.total} — next pose")
            self.goto_calib_pose()
            return

        result, why = self.calib.solve()
        if result is None:
            self.say(f"캘리브레이션 실패: {why}", f"Calibration failed: {why}", 6.0)
            self.mode, self.calib = "run", None
            return
        H, err, n = result
        self.homography.set(H)
        self.homography.reproj_error_mm = err
        self.homography.num_points = n
        self.calib.save(self.calib_file)
        self.mode, self.calib = "run", None
        self.say(f"캘리브레이션 완료 — 재투영 {err:.1f}mm. 자로 재서 꼭 확인하세요",
                 f"Calibration done — reproj {err:.1f}mm. Verify with a ruler", 6.0)
        self.get_logger().info(
            f"호모그래피 저장: {self.calib_file} (재투영 {err:.1f}mm, 점 {n}개)\n"
            "  주의: 재투영 오차는 캘리브 점 자신에 대해서만 맞춘 값입니다. "
            "테이블 위 여러 지점을 자로 재서 검증하세요.")

    # ==================================================================
    def pick_color(self, x, y):
        """우클릭한 자리의 색으로 컵 검출 범위를 잡는다."""
        if self.frame is None:
            return
        lower, upper, (h, s, v) = hsv_range_from_pixel(self.frame, x, y)
        self.detector.lower = np.array(lower, dtype=np.uint8)
        self.detector.upper = np.array(upper, dtype=np.uint8)
        self.detect_on = True
        self.say(f"컵 색 지정: H{h:.0f} S{s:.0f} V{v:.0f} → {lower}~{upper}",
                 f"Color set: H{h:.0f} S{s:.0f} V{v:.0f}", 4.0)
        self.get_logger().info(
            "컵 색을 잡았습니다. config 에 남기려면 아래를 복사하세요:\n"
            f"    hsv_lower: {lower}\n    hsv_upper: {upper}")

    def on_task_message(self, msg):
        """task_manager 의 거부/실패 이유 — 화면 아래에 8초. 길면 앞부분만."""
        t = msg.data if len(msg.data) <= 110 else msg.data[:108] + "…"
        self.say(t, t, 8.0)

    def say(self, ko, en, seconds=2.5):
        self.notice = self.hud.label(ko, en)
        self.notice_until = time.time() + seconds

    def _check_image_size(self, h, name):
        """호모그래피 해상도 != 카메라 해상도면 좌표가 조용히 틀린다 (2026-10-22, 720p 환산)."""
        size = getattr(h, "image_size", None)
        want = [int(self.get_parameter("width").value), int(self.get_parameter("height").value)]
        if size is None:
            self.get_logger().warn(f"팔 {name} 호모그래피에 image_size 가 없습니다 — {want} 로 캘리브한 건지 확인하세요")
        elif list(size) != want:
            raise RuntimeError(f"팔 {name} 호모그래피는 {size} 로 캘리브됐는데 카메라는 {want} 입니다. "
                               "gaze_hri.yaml control_panel width/height 를 맞추거나 "
                               "tools/topdown_to_720p.py 로 환산하세요.")

    # ==================================================================
    # 메인 루프
    # ==================================================================
    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        if self.ocams and frame.ndim == 3 and frame.shape[2] == 2:
            frame = cv2.cvtColor(np.ascontiguousarray(frame[:, :, 0]),
                                 cv2.COLOR_GRAY2BGR)
        self.frame = frame
        now = time.time()

        dt = now - self._fps_t
        self._fps_t = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst

        # 캘리브레이션 중에는 파이프라인에 좌표를 흘리지 않는다.
        # (그리퍼 끝을 클릭하는 동작이 '물체 선택'으로 해석되면 팔이 제멋대로 움직인다)
        if self.mode == "calib":
            self.objects = []
        else:
            if self.detect_on:
                dets = self.detector.detect(frame, self.homography)
                self.objects = self._persist_objects(dets, list(getattr(self.detector, "last_labels", [])))
            else:
                self.objects, self._tracks, self._obj_labels = [], [], []
            if self.detect_on and self.homography.ready:
                self.publish_objects()
            if self.source == "mouse":
                self.publish_mouse_point(now)

        view = render(frame, self.build_state(now), self.hud, self.homography)
        cv2.imshow(WIN, view)
        self._record(view, now)
        raw = cv2.waitKeyEx(1)
        if raw != -1:
            # 화살표 (GTK/Qt 코드) -> 따르기 보정 입구 조정. 일반 키는 아래 바이트로
            arrow = {65361: (-1, 0), 65362: (0, -1), 65363: (1, 0), 65364: (0, 1),
                     0x1000012: (-1, 0), 0x1000013: (0, -1), 0x1000014: (1, 0), 0x1000015: (0, 1)}
            if raw in arrow:
                self.nudge(*arrow[raw])
            else:
                self.handle_key(raw & 0xFF)

    def nudge(self, dx, dy):
        """따르기 보정 멈춤 중: 병 입구를 화면 방향으로 1cm (8 위, 2 아래, 4 왼쪽, 6 오른쪽)."""
        self.pub_teleop.publish(String(data=f"nudge:{dx}:{dy}"))
        self.say({(0, -1): "입구 ↑ 1cm", (0, 1): "입구 ↓ 1cm", (-1, 0): "입구 ← 1cm",
                  (1, 0): "입구 → 1cm"}[(dx, dy)], "nudge 1cm", 1.0)

    def save_snapshot(self):
        """보정 기록용: 지금 화면 + 검출 결과(픽셀 박스, 라벨, 좌표)."""
        if self.frame is None:
            return None
        d = os.path.expanduser(self.get_parameter("snapshot_dir").value)
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, time.strftime("snap_%m%d_%H%M%S"))
        cv2.imwrite(base + ".png", self.frame)
        labels = list(getattr(self, "_obj_labels", []))
        objs = [{"px": [float(u), float(v)], "xy": [float(x), float(y)],
                 "box": c.reshape(-1, 2).tolist() if c is not None else None,
                 "label": labels[i] if i < len(labels) else None}
                for i, ((u, v), (x, y), c) in enumerate(self.objects)]
        json.dump({"objects": objs, "homography_file": self.calib_file},
                  open(base + ".json", "w"), ensure_ascii=False)
        return base + ".png"

    def publish_objects(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.base_frame
        for (_, (x, y), _) in self.objects:
            p = Pose()
            p.position = Point(x=x, y=y, z=self.table_z)
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pub_objects.publish(pa)
        self.pub_pour_calib.publish(Bool(data=self.pour_calib))
        self.pub_grasp_calib.publish(Bool(data=self.grasp_calib))
        labels = list(getattr(self, "_obj_labels", []))
        if len(labels) != len(pa.poses):
            labels = ["object"] * len(pa.poses)
        labels = self._vote_labels([(x, y) for (_, (x, y), _) in self.objects], labels)
        self.pub_labels.publish(String(data=json.dumps(labels)))

    def _persist_objects(self, dets, labels, hold=1.5, radius=0.04):
        """2026-10-22: 검출이 한두 프레임 끊겨도(물병 ~60% 프레임만 잡힘) 물체를 hold 초 유지한다.
        2초 응시가 끝나는 순간 물병이 빠져 있으면 "물체 특정 못함"으로 선택이 거부됐다.
        같은 자리(radius 안)에서 다시 잡히면 같은 물체로 잇는다. 순서도 고정돼 시선 스냅 번호가 안 바뀐다."""
        now = time.time()
        tracks = getattr(self, "_tracks", [])
        if len(labels) != len(dets):
            labels = ["object"] * len(dets)
        used = set()
        for (uv, xy, c), lab in zip(dets, labels):
            best, bd = None, radius
            for i, t in enumerate(tracks):
                d = math.hypot(t["xy"][0] - xy[0], t["xy"][1] - xy[1])
                if i not in used and d < bd:
                    best, bd = i, d
            if best is None:
                tracks.append({"uv": uv, "xy": xy, "c": c, "lab": lab, "t": now})
                used.add(len(tracks) - 1)
            else:
                tracks[best].update(uv=uv, xy=xy, c=c, lab=lab, t=now)
                used.add(best)
        self._tracks = [t for t in tracks if now - t["t"] <= hold]
        self._obj_labels = [t["lab"] for t in self._tracks]
        return [(t["uv"], t["xy"], t["c"]) for t in self._tracks]

    def _vote_labels(self, xys, labels, window=1.0, radius=0.04):
        """2026-10-22: 라벨 흔들림(cup <-> vase <-> bottle) 대비 — 최근 1초 동안 같은 자리(4cm 안)에서
        나온 라벨 중 가장 많은 것으로. 클릭 순간 한 프레임 라벨로 따르기/거부가 갈리지 않게."""
        now = time.time()
        hist = getattr(self, "_label_hist", [])
        hist = [h for h in hist if now - h[0] <= window]
        hist += [(now, xy, lab) for xy, lab in zip(xys, labels)]
        self._label_hist = hist
        out = []
        for xy in xys:
            votes = {}
            for _, hxy, lab in hist:
                if math.hypot(hxy[0] - xy[0], hxy[1] - xy[1]) < radius:
                    votes[lab] = votes.get(lab, 0) + 1
            out.append(max(votes, key=votes.get) if votes else "object")
        return out

    def publish_mouse_point(self, now):
        holding = now < self.held_until and self.held_px is not None
        # 2026-10-12: 클릭만 선택으로 친다. 커서를 화면 위에 올려 둔 것만으로 2초 응시가
        # 확정돼 엉뚱한 집기가 두 번 실행됐다 (팔 뒤/옆 좌표 -> 거부 -> home 복귀).
        if not holding and bool(self.get_parameter("mouse_click_only").value):
            self.pub_valid.publish(Bool(data=False))
            return
        px = self.held_px if holding else self.cursor
        if px is None or not self.homography.ready:
            return
        xy = self.homography.to_robot(*px)
        if xy is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.point.x, msg.point.y = xy
        # 집을 것은 테이블보다 살짝 위, 놓을 자리는 테이블면
        msg.point.z = self.table_z + (0.05 if self.task_state == "IDLE" else 0.0)
        self.pub_point.publish(msg)
        self.pub_valid.publish(Bool(data=True))

    # ==================================================================
    def build_state(self, now):
        """지금 화면에 그릴 것을 한 덩어리로 모은다."""
        if self.source == "mouse":
            holding = now < self.held_until and self.held_px is not None
            px = self.held_px if holding else self.cursor
            frac = (1.0 - max(0.0, self.held_until - now) / self.hold_time
                    if holding else 0.0)
        else:
            px, holding, frac = self.gaze_px, False, 0.0

        return PanelState(
            mode=self.mode,
            calib_index=self.calib.index if self.calib else 0,
            calib_total=self.calib.total if self.calib else 0,
            calib_target=self.calib.current_target() if self.calib else None,
            calib_pixels=list(self.calib.pixels) if self.calib else [],
            task_state=self.task_state,
            source=self.source,
            objects=self.objects,
            pointer_px=px,
            ray_px=self._ray_pixels(now) if self.source != "mouse" else None,
            snapped_index=(self._snapped[0] if self.source != "mouse"
                           and now - self._snapped[1] < 0.5 else -1),
            pointer_holding=holding,
            hold_frac=frac,
            dwell_progress=self.dwell_progress,
            pick_target=self.pick_target,
            place_target=self.place_target,
            target_info=self.target_info,
            tcp=self.tcp if self.show_tip else None,
            reach_band=(tuple(self.get_parameter("reach_band").value)
                        if len(self.get_parameter("reach_band").value) == 2 else None),
            arm_phase=self.arm_phase,
            tip_label=self.get_parameter("arm_label").value,
            other_arms=[(h, tcp if self.show_tip else None, n) for n, h, tcp in self.other_arms],
            object_info=self._object_info(),
            pour_spot=tuple(self.get_parameter("pour_spot").value),
            modes=(["따르기 보정"] if self.pour_calib else []) + (["집기 보정"] if self.grasp_calib else []),
            reach_arcs={id(h): a for h, a in
                        [(self.homography, self._reach_arc(self.get_parameter("arm_label").value))]
                        + [(h, self._reach_arc(n)) for n, h, _ in self.other_arms] if a},
            has_fixation=self.has_fixation,
            detect_on=self.detect_on,
            calibrated=self.homography.ready,
            reproj_error_mm=self.homography.reproj_error_mm,
            fps=self._fps,
            notice=self.notice if now < self.notice_until else "",
            sidebar_w=self.sidebar_w,
        )

    # ------------------------------------------------------------------
    def _record(self, view, now):
        d = str(self.get_parameter("record_dir").value)
        if not d or now - self._rec_t < 1.0 / 15:
            return
        if self._rec is None:
            d = os.path.expanduser(d)
            os.makedirs(d, exist_ok=True)
            # MJPG avi — GUI 를 강제로 꺼도 앞부분은 재생된다 (mp4 는 끝 정리가 안 되면 통째로 깨짐)
            self._rec_path = os.path.join(d, time.strftime("topdown_%m%d_%H%M%S.avi"))
            self._rec = cv2.VideoWriter(self._rec_path, cv2.VideoWriter_fourcc(*"MJPG"), 15.0,
                                        (view.shape[1], view.shape[0]))
            self.get_logger().info(f"화면 녹화 시작: {self._rec_path}")
        self._rec.write(view)
        self._rec_t = now

    def destroy_node(self):
        if self._rec is not None:
            self._rec.release()
            self.get_logger().info(f"화면 녹화 저장: {self._rec_path}")
        if hasattr(self.detector, "close"):
            self.detector.close()
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception:                               # noqa: BLE001
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ControlPanel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
