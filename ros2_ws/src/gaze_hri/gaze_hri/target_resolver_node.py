#!/usr/bin/env python3
"""
[노드 3] target_resolver  —  "무엇을 보고 있는가 / 어디에 놓으려는가"

/gaze/fixation 은 그냥 3D 점입니다. 그대로 로봇에 넘기면 문제가 생깁니다.

  - 컵을 볼 때 사람은 보통 컵의 "표면"을 봅니다. 그 점을 그대로 파지점으로 쓰면
    그리퍼가 컵 옆구리를 스치거나 테두리를 칩니다. -> 물체 중심으로 스냅 필요.
  - "저기에 놓아줘" 할 때 시선 깊이 오차가 몇 cm 있으면 로봇이 허공이나
    테이블 속으로 내려갑니다. -> 테이블 평면으로 투영 필요.

그래서 이 노드는 응시점을 "의미 있는 목표점"으로 바꿔줍니다.

────────────────────────────────────────────────────────────────────────
좌표계 — 이 노드에서 가장 중요한 부분

입력 두 갈래는 **서로 다른 좌표계로 들어옵니다.**

  /gaze/fixation   통합 시에는 SLAM world(map) 기준         <- 안경이 만든 점
  /objects/poses   탑다운 카메라는 로봇 base_link 기준       <- 호모그래피 결과

이 둘을 변환 없이 거리 비교하면 같은 컵인데도 수십 cm 떨어진 것으로 보입니다.
로봇 단독 모드에서는 map=base_link 항등이라 이 결함이 드러나지 않다가,
안경을 붙이는 순간 나타납니다.

그래서 이 노드는 **들어온 모든 점을 base_frame 으로 변환한 뒤에** 비교하고,
목표도 base_frame 으로 내보냅니다(GazeTarget.header.frame_id 에 명시).
table_plane 도 base_frame 기준으로 해석합니다.

구독:
  /gaze/fixation       gaze_hri_msgs/Fixation
  /task/expected_role  std_msgs/String       "pick" | "place" | "idle"
  /objects/poses       geometry_msgs/PoseArray  탑다운/YOLO 노드가 주는 물체 중심

발행:
  /gaze/target         gaze_hri_msgs/GazeTarget   (frame_id = base_frame)
  /gaze/target_markers visualization_msgs/MarkerArray

테이블 평면은 config/gaze_hri.yaml 의 table_plane [a,b,c,d]에 넣습니다.
아직 모르면 `table_calibration: true` 로 두고 테이블 여기저기를 5~10번
응시하세요. RANSAC으로 평면을 피팅해서 로그에 찍어줍니다.
"""

import numpy as np
import rclpy
import tf2_ros
from gaze_hri_msgs.msg import Fixation, GazeTarget
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from gaze_hri.kinematics import (fit_plane_ransac, height_above_plane,
                                 intersect_ray_plane, project_onto_plane,
                                 quat_to_matrix, ray_point_distance)


class TargetResolver(Node):

    def __init__(self):
        super().__init__("target_resolver")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("table_plane", [0.0, 0.0, 1.0, 0.0])   # ax+by+cz+d=0
        self.declare_parameter("table_calibration", False)
        self.declare_parameter("snap_radius", 0.08)     # 물체 중심 스냅 허용 반경 [m]
        self.declare_parameter("object_min_height", 0.02)   # 이보다 높으면 "물체"
        self.declare_parameter("grasp_height", 0.045)   # 테이블에서 파지점 높이 [m]
        self.declare_parameter("place_clearance", 0.01)  # 놓을 때 테이블 위 여유 [m]
        # 1등과 2등 물체가 이만큼도 차이 안 나면 "어느 컵인지 모르겠다"로 보고 거부.
        # 틀린 컵을 집는 것보다 안 집는 게 낫다.
        self.declare_parameter("ambiguity_margin", 0.05)
        # 물체 목록이 이 시간보다 오래되면 신뢰하지 않는다.
        # (검출 노드가 죽었는데 옛 목록으로 계속 집으려 드는 것을 막는다)
        self.declare_parameter("object_timeout", 2.0)
        # pick 인데 물체 스냅에 실패했을 때 목표를 낼지 여부.
        #   False(기본): 응시점을 그대로 파지점으로 쓴다.
        #     로봇 단독 테스트는 마우스 클릭이 곧 호모그래피를 거친 정확한 좌표라
        #     스냅이 없어도 정확하다. 그래서 기본값은 False 다.
        #   True: 거부한다. 안경을 붙인 뒤에는 이쪽을 쓴다 — 스냅 실패는 곧
        #     '정확한 물체 위치를 모른다'는 뜻이고, 그대로 진행하면 시선 오차가
        #     그대로 파지 오차가 된다. gaze_hri.launch.py 가 True 로 켠다.
        self.declare_parameter("require_snap_for_pick", False)

        self.base_frame = self.get_parameter("base_frame").value
        self.plane = np.array(self.get_parameter("table_plane").value, dtype=float)
        self.calibrating = bool(self.get_parameter("table_calibration").value)
        self.snap_radius = float(self.get_parameter("snap_radius").value)
        self.object_min_height = float(self.get_parameter("object_min_height").value)
        self.grasp_height = float(self.get_parameter("grasp_height").value)
        self.place_clearance = float(self.get_parameter("place_clearance").value)
        self.ambiguity_margin = float(self.get_parameter("ambiguity_margin").value)
        self.object_timeout = float(self.get_parameter("object_timeout").value)
        self.require_snap = bool(self.get_parameter("require_snap_for_pick").value)

        self.expected_role = "pick"
        self.objects = []          # [np.array([x,y,z]), ...]  (base_frame 기준)
        self.objects_stamp = None  # 마지막으로 물체 목록을 받은 시각 [s]
        self.calib_points = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Fixation, "/gaze/fixation", self.on_fixation, 10)
        self.create_subscription(String, "/task/expected_role", self.on_role, 10)
        self.create_subscription(PoseArray, "/objects/poses", self.on_objects, 10)

        self.pub_target = self.create_publisher(GazeTarget, "/gaze/target", 10)
        self.pub_marker = self.create_publisher(MarkerArray, "/gaze/target_markers", 10)

        if self.calibrating:
            self.get_logger().warn(
                "테이블 평면 캘리브레이션 모드입니다. "
                "테이블 위 서로 다른 지점을 5~10회 응시하세요."
            )
        self.get_logger().info(f"target_resolver 시작 (기준 좌표계 {self.base_frame})")

    # ------------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def to_base(self, xyz, frame_id):
        """어떤 프레임의 점이든 base_frame 좌표로 변환한다.

        변환을 못 하면 None을 돌려준다. 이때 '그냥 쓰는' 선택지는 두지 않는다.
        좌표계가 다른 값을 그대로 쓰면 팔이 엉뚱한 곳으로 가기 때문이다.
        """
        p = np.asarray(xyz, dtype=float)
        src = frame_id or self.base_frame
        if src == self.base_frame:
            return p
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, src, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"TF {src}->{self.base_frame} 없음 ({exc}). 이 샘플은 버립니다.",
                throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        R = quat_to_matrix(r.x, r.y, r.z, r.w)
        return R @ p + np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------
    def on_role(self, msg: String):
        self.expected_role = msg.data

    def on_objects(self, msg: PoseArray):
        pts = []
        for p in msg.poses:
            q = self.to_base([p.position.x, p.position.y, p.position.z],
                             msg.header.frame_id)
            if q is not None:
                pts.append(q)
        # 빈 목록도 그대로 반영한다. 컵을 치웠는데 옛 목록이 남아 있으면
        # 없는 컵으로 스냅해 버린다.
        self.objects = pts
        self.objects_stamp = self.now()

    def objects_fresh(self):
        if self.objects_stamp is None:
            return False
        return (self.now() - self.objects_stamp) <= self.object_timeout

    # ------------------------------------------------------------------
    def on_fixation(self, msg: Fixation):
        p = self.to_base([msg.point.x, msg.point.y, msg.point.z],
                         msg.header.frame_id)
        if p is None:
            return

        if self.calibrating:
            self.calib_points.append(p)
            n = len(self.calib_points)
            self.get_logger().info(f"평면 캘리브레이션 샘플 {n}개 수집")
            if n >= 5:
                plane = fit_plane_ransac(np.array(self.calib_points), threshold=0.02)
                self.get_logger().info(
                    "테이블 평면 추정 결과 -> config에 복사하세요:\n"
                    f"    table_plane: [{plane[0]:.6f}, {plane[1]:.6f}, "
                    f"{plane[2]:.6f}, {plane[3]:.6f}]"
                )
            return

        if self.expected_role == "idle":
            return

        # 머리 위치(광선 원점). msg.point 와 같은 좌표계로 들어오므로 반드시
        # 같은 변환을 태워야 한다. TF가 없으면 None 이 되고, 그러면 예전처럼
        # 직교 투영으로 물러선다.
        origin_base = None
        if msg.has_origin:
            origin_base = self.to_base([msg.origin.x, msg.origin.y, msg.origin.z],
                                       msg.header.frame_id)

        h = height_above_plane(p, self.plane)
        n = np.asarray(self.plane[:3], dtype=float)

        target = GazeTarget()
        target.header.stamp = msg.header.stamp
        target.header.frame_id = self.base_frame
        target.role = self.expected_role
        target.height_above_table = float(h)
        target.snapped = False
        target.label = "unknown"
        target.confidence = float(msg.confidence)

        if self.expected_role == "pick":
            snapped = self._snap_to_object(p, origin_base)
            if snapped is not None:
                base = snapped
                target.snapped = True
                target.label = "object"
            elif self.require_snap:
                # 스냅에 실패했다는 건 '정확한 물체 위치를 모른다'는 뜻이다.
                # 그대로 진행하면 시선 오차(수 cm)가 곧바로 파지 오차가 된다.
                self.get_logger().warn(
                    "집을 물체를 특정하지 못했습니다. 목표를 내지 않습니다. "
                    "(물체 검출이 켜져 있는지, snap_radius 안에 있는지 확인하세요)")
                return
            else:
                base = p
                if self.objects:
                    # 목록은 있는데 못 붙였다 = 시선 오차가 그대로 파지 오차가 된다.
                    self.get_logger().warn(
                        "물체 스냅 실패 — 응시점을 그대로 파지점으로 씁니다. "
                        "안경 연동 후에는 require_snap_for_pick 을 켜세요.",
                        throttle_duration_sec=5.0)
                if h < self.object_min_height:
                    self.get_logger().warn(
                        f"응시점이 테이블에 너무 붙어 있습니다 (높이 {h * 100:.1f}cm). "
                        "물체가 아니라 테이블을 본 것 같습니다. 무시합니다."
                    )
                    return

            # 파지 높이 보정: 물체 표면이 아니라 "테이블에서 grasp_height 만큼 위"를 잡는다.
            # 스냅된 점은 탑다운 카메라가 준 정확한 물체 중심이라 광선과 무관하다.
            # 그대로 수직 투영한다. 스냅에 실패해 응시점을 쓰는 경우에만 광선을 쓴다.
            if target.snapped:
                on_plane = project_onto_plane(base, self.plane)
            else:
                on_plane = self._gaze_to_table(p, origin_base)
            final = on_plane + n * self.grasp_height

        else:  # place
            # 놓을 자리는 무조건 테이블 평면 위로 눌러준다
            on_plane = self._gaze_to_table(p, origin_base)
            final = on_plane + n * (self.grasp_height + self.place_clearance)
            target.label = "table"

        target.point.x, target.point.y, target.point.z = map(float, final)
        self.pub_target.publish(target)
        self._publish_marker(final, self.expected_role)

        self.get_logger().info(
            f"[{self.expected_role}] 목표 확정: "
            f"({final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f}) "
            f"{'(물체 스냅됨)' if target.snapped else ''}"
        )

    # ------------------------------------------------------------------
    def _gaze_to_table(self, p, origin_base):
        """응시점을 테이블 평면 위의 점으로 내린다 (둘 다 base_frame 기준).

        머리 위치를 알면 origin -> p 광선을 평면과 교차시킨다. 이러면 추정
        응시점이 광선 위 어디에 있든(= 깊이 오차가 있어도) 같은 자리를 뚫으므로,
        깊이 오차가 테이블 위 가로 오차로 새지 않는다. 직교 투영은 머리
        0.45m / 거리 0.35m 기준으로 깊이 오차의 약 0.62배를 가로로 흘린다.

        머리 위치가 없거나(has_origin=false, TF 실패) 교차가 실패하면
        (평행하거나 머리 뒤쪽) 예전 그대로 직교 투영으로 물러선다.
        """
        if origin_base is not None:
            hit = intersect_ray_plane(origin_base, p - origin_base, self.plane)
            if hit is not None:
                self.get_logger().info(
                    "테이블 점: 광선-평면 교차 사용 (깊이 오차에 둔감)",
                    throttle_duration_sec=5.0)
                return hit
            self.get_logger().warn(
                "광선-평면 교차 실패(평면과 평행이거나 교점이 머리 뒤쪽). "
                "직교 투영으로 물러섭니다.", throttle_duration_sec=5.0)
        else:
            self.get_logger().info(
                "테이블 점: 직교 투영 사용 (머리 위치 없음 — /head/pose 확인)",
                throttle_duration_sec=5.0)
        return project_onto_plane(p, self.plane)

    # ------------------------------------------------------------------
    def _snap_to_object(self, p, origin_base=None):
        """가장 가까운 물체 중심으로 끌어당긴다.

        1등과 2등이 엇비슷하면 어느 컵인지 특정할 수 없으므로 거부한다.
        시선 오차가 3cm일 때 컵이 6cm 이내로 붙어 있으면 여기에 걸린다.

        머리 위치(origin_base)를 알면 "시선 광선이 각 물체의 파지 높이 중심을
        얼마나 가깝게 지나가는가"로 잰다. 물체 좌표는 테이블면(z=table)으로 들어오고
        시선은 컵 몸통을 보므로, 응시점과 점-점으로 비교하면 응시점이 광선 위 어느
        높이에 찍혔는지에 따라 5~8cm 가 앞뒤로 흔들린다 (머리 0.35m 위, 0.6m 거리,
        컵 중간 4.5cm 기준). 광선 거리는 그 선택과 무관하다.
        """
        if not self.objects:
            return None
        if not self.objects_fresh():
            self.get_logger().warn(
                f"물체 목록이 {self.object_timeout:.1f}초 넘게 갱신되지 않았습니다. "
                "검출 노드가 살아 있는지 확인하세요.", throttle_duration_sec=5.0)
            return None

        if origin_base is not None and float(np.linalg.norm(p - origin_base)) > 1e-6:
            n = np.asarray(self.plane[:3], dtype=float)
            centers = [project_onto_plane(o, self.plane) + n * self.grasp_height
                       for o in self.objects]
            dists = np.array([ray_point_distance(origin_base, p - origin_base, c)
                              for c in centers])
        else:
            dists = np.array([float(np.linalg.norm(o - p)) for o in self.objects])
        order = np.argsort(dists)
        i = int(order[0])
        if dists[i] > self.snap_radius:
            return None

        if len(order) > 1:
            gap = float(dists[order[1]] - dists[i])
            if gap < self.ambiguity_margin:
                self.get_logger().warn(
                    f"어느 물체인지 모호합니다 (1등 {dists[i] * 100:.1f}cm / "
                    f"2등 {dists[order[1]] * 100:.1f}cm, 차이 {gap * 100:.1f}cm "
                    f"< 기준 {self.ambiguity_margin * 100:.1f}cm). 선택하지 않습니다."
                )
                return None
        return self.objects[i]

    def _publish_marker(self, point, role):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f"target_{role}"
        m.id = 0 if role == "pick" else 1
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, point)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.06
        m.scale.z = 0.005
        if role == "pick":
            m.color.r, m.color.g, m.color.b = 1.0, 0.35, 0.1
        else:
            m.color.r, m.color.g, m.color.b = 0.1, 0.9, 0.4
        m.color.a = 0.85
        arr.markers.append(m)
        self.pub_marker.publish(arr)


def main():
    rclpy.init()
    node = TargetResolver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
