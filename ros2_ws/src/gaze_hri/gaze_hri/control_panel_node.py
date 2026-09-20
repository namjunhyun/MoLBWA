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

import os
import time

import cv2
import rclpy
from gaze_hri_msgs.msg import Fixation, GazeTarget
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32, String

from gaze_hri.hud import Hud
from gaze_hri.kinematics import ArmGeometry, forward_kinematics
from gaze_hri.panel_render import PanelState, render
from gaze_hri.table_view import ObjectDetector, TableHomography, open_camera

WIN = "MoLBWA gaze pick and place"


class ControlPanel(Node):

    def __init__(self):
        super().__init__("control_panel")

        self.declare_parameter("input_source", "mouse")   # mouse | gaze
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("sidebar_width", 320)
        self.declare_parameter("table_z", 0.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("hold_time", 2.0)
        self.declare_parameter("homography_file",
                               os.path.expanduser("~/.ros/topdown_homography.yaml"))
        self.declare_parameter("detect_objects", True)
        self.declare_parameter("hsv_lower", [0, 120, 80])
        self.declare_parameter("hsv_upper", [12, 255, 255])
        self.declare_parameter("min_area", 400)
        self.declare_parameter("show_arm_tip", True)
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
        self.detector = ObjectDetector(
            self.get_parameter("hsv_lower").value,
            self.get_parameter("hsv_upper").value,
            self.get_parameter("min_area").value)
        self.cap = open_camera(self.get_parameter("camera_index").value,
                               self.get_parameter("width").value,
                               self.get_parameter("height").value)

        # ---- 통신 ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_objects = self.create_publisher(PoseArray, "/objects/poses", 10)
        self.pub_cancel = self.create_publisher(Empty, "/task/cancel", 10)
        self.pub_point = self.pub_valid = None
        if self.source == "mouse":
            self.pub_point = self.create_publisher(PointStamped, "/gaze/point_raw", qos)
            self.pub_valid = self.create_publisher(Bool, "/gaze/valid", qos)
        else:
            self.create_subscription(PointStamped, "/gaze/point_raw",
                                     self.on_gaze_point, qos)

        self.create_subscription(Fixation, "/gaze/fixation", self.on_fixation, 10)
        self.create_subscription(Float32, "/gaze/dwell_progress", self.on_progress, qos)
        self.create_subscription(GazeTarget, "/gaze/target", self.on_target, 10)
        self.create_subscription(String, "/task/state", self.on_state, 10)
        self.create_subscription(String, "/arm/phase", self.on_phase, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)

        # ---- 상태 ----
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

    # ==================================================================
    # 입력
    # ==================================================================
    def on_mouse(self, event, x, y, flags, param):
        view_w = self.frame.shape[1] if self.frame is not None else 640
        if x >= view_w:                                  # 사이드바는 무시
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and self.source == "mouse":
            self.held_px = (x, y)
            self.held_until = time.time() + self.hold_time
            xy = self.homography.to_robot(x, y)
            if xy:
                self.say(f"선택: ({xy[0]:+.3f}, {xy[1]:+.3f})",
                         f"Selected: ({xy[0]:+.3f}, {xy[1]:+.3f})")

    def handle_key(self, key):
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("c"):
            self.held_until, self.held_px = 0.0, None
            self.pub_cancel.publish(Empty())
            self.say("취소했습니다", "Cancelled")
        elif key == ord(" "):
            self.held_until, self.held_px = 0.0, None
            self.pub_cancel.publish(Empty())
            self.say("비상 정지 — 동작을 중단했습니다", "EMERGENCY STOP")
        elif key == ord("d"):
            self.detect_on = not self.detect_on
            self.say(f"물체 검출 {'켬' if self.detect_on else '끔'}",
                     f"Detection {'ON' if self.detect_on else 'OFF'}")

    def say(self, ko, en, seconds=2.5):
        self.notice = self.hud.label(ko, en)
        self.notice_until = time.time() + seconds

    # ==================================================================
    # 메인 루프
    # ==================================================================
    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        self.frame = frame
        now = time.time()

        dt = now - self._fps_t
        self._fps_t = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst

        self.objects = (self.detector.detect(frame, self.homography)
                        if self.detect_on else [])
        if self.detect_on and self.homography.ready:
            self.publish_objects()
        if self.source == "mouse":
            self.publish_mouse_point(now)

        cv2.imshow(WIN, render(frame, self.build_state(now), self.hud,
                               self.homography))
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            self.handle_key(key)

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

    def publish_mouse_point(self, now):
        holding = now < self.held_until and self.held_px is not None
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
            task_state=self.task_state,
            source=self.source,
            objects=self.objects,
            pointer_px=px,
            pointer_holding=holding,
            hold_frac=frac,
            dwell_progress=self.dwell_progress,
            pick_target=self.pick_target,
            place_target=self.place_target,
            target_info=self.target_info,
            tcp=self.tcp if self.show_tip else None,
            arm_phase=self.arm_phase,
            has_fixation=self.has_fixation,
            detect_on=self.detect_on,
            calibrated=self.homography.ready,
            reproj_error_mm=self.homography.reproj_error_mm,
            fps=self._fps,
            notice=self.notice if now < self.notice_until else "",
            sidebar_w=self.sidebar_w,
        )

    # ------------------------------------------------------------------
    def destroy_node(self):
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
