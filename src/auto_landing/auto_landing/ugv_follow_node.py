#!/usr/bin/env python3
"""
UGV Follow Node - CO-Paint
============================
TAKEOFF~PAINT 중 UGV 가 드론의 XY 위치(+yaw)를 일정 거리 안으로 따라간다.

좌표계:
  - 드론 위치는 PX4 NED 로 들어옴 (/fmu/out/vehicle_odometry).
  - UGV 는 ENU 로 동작. 내부에서 NED → ENU 변환.
  - 변환: X_enu = Y_ned, Y_enu = X_ned, Z_enu = -Z_ned, yaw_enu = π/2 - yaw_ned

가정:
  - INIT 시점에 드론과 UGV 가 같은 원점 (0,0,0) 공유.
  - UGV 는 메카넘 휠 (linear.x, linear.y, angular.z 모두 사용).

인터페이스:
  [Sub] /ugv/follow_enable          (Bool)     True = follow 시작
  [Sub] /fmu/out/vehicle_odometry   (VehicleOdometry, PX4 NED)
  [Sub] /odom                       (Odometry, UGV ENU) - 토픽명 파라미터
  [Pub] /cmd_vel                    (Twist)

설계:
  - 활성화 전: cmd_vel 미발행, 내부 상태만 갱신.
  - 활성화 후: 드론과 UGV 의 XY 거리가 데드밴드 밖이면 UGV 가 드론 쪽으로 이동.
  - 드론 yaw 와 UGV yaw 차이가 yaw 데드밴드 밖이면 회전.
  - 드론 고도가 min_drone_altitude 미만이면 안전상 UGV 정지.
  - 비활성화 시 마지막에 정지 명령 1회.

PID 가 아닌 P 제어 (단순함, 충분). 필요 시 PID 로 확장 가능.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry

from px4_msgs.msg import VehicleOdometry


# ── PX4 NED → ENU 변환 ─────────────────────────────────
def ned_to_enu_xyz(x_ned: float, y_ned: float, z_ned: float):
    return y_ned, x_ned, -z_ned


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """quaternion → yaw (라디안)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def ned_yaw_to_enu_yaw(yaw_ned: float) -> float:
    """NED yaw → ENU yaw.

    NED: yaw=0 → North 향함 (+X_ned)
    ENU: yaw=0 → East  향함 (+X_enu).  North = +Y_enu = yaw π/2.
    변환: yaw_enu = π/2 - yaw_ned
    """
    return wrap_to_pi(math.pi / 2.0 - yaw_ned)


def wrap_to_pi(angle: float) -> float:
    """각도를 [-π, π] 로 정규화."""
    return math.atan2(math.sin(angle), math.cos(angle))


class UgvFollowNode(Node):
    def __init__(self):
        super().__init__('ugv_follow_node')

        # ---- 파라미터 ----
        self.declare_parameter('enable_topic',        '/ugv/follow_enable')
        self.declare_parameter('drone_odom_topic',    '/fmu/out/vehicle_odometry')
        self.declare_parameter('ugv_odom_topic',      '/odom')
        self.declare_parameter('cmd_vel_topic',       '/cmd_vel')
        self.declare_parameter('distance_topic',      '/ugv/tether_distance')

        # XY 추종
        self.declare_parameter('xy_deadband',         0.3)    # m
        self.declare_parameter('kp_xy',               0.5)
        self.declare_parameter('xy_speed_limit',      0.2)    # m/s

        # Yaw 추종
        self.declare_parameter('yaw_deadband',        0.087)  # rad (≈5°)
        self.declare_parameter('kp_yaw',              0.8)
        self.declare_parameter('yaw_speed_limit',     0.3)    # rad/s

        # 안전
        self.declare_parameter('min_drone_altitude',  0.0)    # m, ENU Z (양수=위)
                                                              # 0이면 안전 체크 비활성
        # 루프
        self.declare_parameter('control_rate',        20.0)   # Hz

        # 로드
        self.xy_deadband      = float(self.get_parameter('xy_deadband').value)
        self.kp_xy            = float(self.get_parameter('kp_xy').value)
        self.xy_speed_limit   = float(self.get_parameter('xy_speed_limit').value)
        self.yaw_deadband     = float(self.get_parameter('yaw_deadband').value)
        self.kp_yaw           = float(self.get_parameter('kp_yaw').value)
        self.yaw_speed_limit  = float(self.get_parameter('yaw_speed_limit').value)
        self.min_alt          = float(self.get_parameter('min_drone_altitude').value)
        control_rate          = float(self.get_parameter('control_rate').value)

        # ---- QoS ----
        # PX4 토픽: BEST_EFFORT
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # 마스터 명령: RELIABLE + TRANSIENT_LOCAL
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # UGV odom: 일반적인 Reliable 가정
        ugv_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.dist_pub = self.create_publisher(
            Float32, self.get_parameter('distance_topic').value, 10)

        # ---- Subscriber ----
        self.create_subscription(
            Bool, self.get_parameter('enable_topic').value,
            self._on_enable, cmd_qos)
        self.create_subscription(
            VehicleOdometry, self.get_parameter('drone_odom_topic').value,
            self._on_drone_odom, px4_qos)
        self.create_subscription(
            Odometry, self.get_parameter('ugv_odom_topic').value,
            self._on_ugv_odom, ugv_qos)

        # ---- 상태 ----
        self.active = False
        # ENU 좌표로 캐싱
        self.drone_x   = 0.0
        self.drone_y   = 0.0
        self.drone_z   = 0.0   # ENU Z (양수 = 위)
        self.drone_yaw = 0.0
        self.drone_odom_ok = False

        self.ugv_x   = 0.0
        self.ugv_y   = 0.0
        self.ugv_z   = 0.0
        self.ugv_yaw = 0.0
        self.ugv_odom_ok = False

        # ---- 제어 루프 ----
        self.create_timer(1.0 / control_rate, self._control_loop)

        self.get_logger().info(
            f'UGV Follow Node started.\n'
            f'  active=False\n'
            f'  xy_deadband={self.xy_deadband}m  kp_xy={self.kp_xy}\n'
            f'  yaw_deadband={math.degrees(self.yaw_deadband):.1f}°  kp_yaw={self.kp_yaw}\n'
            f'  min_drone_altitude={self.min_alt}m (0=안전체크 OFF)'
        )

    # ================= 콜백 =================
    def _on_enable(self, msg: Bool):
        if msg.data and not self.active:
            self.active = True
            self.get_logger().info('✅ FOLLOW 활성화')
        elif not msg.data and self.active:
            self.active = False
            self._send_stop()
            self.get_logger().info('FOLLOW 비활성화')

    def _on_drone_odom(self, msg: VehicleOdometry):
        # NED → ENU
        x_enu, y_enu, z_enu = ned_to_enu_xyz(
            float(msg.position[0]),
            float(msg.position[1]),
            float(msg.position[2]),
        )
        # quaternion (PX4 순서: w, x, y, z)
        yaw_ned = quat_to_yaw(msg.q[1], msg.q[2], msg.q[3], msg.q[0])
        yaw_enu = ned_yaw_to_enu_yaw(yaw_ned)

        self.drone_x   = x_enu
        self.drone_y   = y_enu
        self.drone_z   = z_enu
        self.drone_yaw = yaw_enu
        self.drone_odom_ok = True

    def _on_ugv_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.ugv_x = float(p.x)
        self.ugv_y = float(p.y)
        self.ugv_z = float(p.z)
        self.ugv_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.ugv_odom_ok = True

    # ================= 제어 루프 =================
    def _control_loop(self):
        if not (self.drone_odom_ok and self.ugv_odom_ok):
            if self.active:
                self.get_logger().warn(
                    'odometry 대기 중 (drone={}, ugv={})'.format(
                        self.drone_odom_ok, self.ugv_odom_ok),
                    throttle_duration_sec=2.0)
            return

        # ── 3D 거리 계산 + 발행 (follow 비활성이어도 항상) ──
        dx = self.drone_x - self.ugv_x
        dy = self.drone_y - self.ugv_y
        dz = self.drone_z - self.ugv_z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.dist_pub.publish(Float32(data=float(dist_3d)))

        if not self.active:
            return

        # 안전: 드론 고도가 너무 낮으면 정지
        if self.min_alt > 0.0 and self.drone_z < self.min_alt:
            self._send_stop()
            self.get_logger().warn(
                f'드론 고도 ({self.drone_z:.2f}m) < 임계값 ({self.min_alt:.2f}m). UGV 정지',
                throttle_duration_sec=2.0)
            return

        # ---- 오차 계산 (월드 ENU 기준) ----
        err_x_world = self.drone_x - self.ugv_x
        err_y_world = self.drone_y - self.ugv_y
        dist = math.hypot(err_x_world, err_y_world)

        err_yaw = wrap_to_pi(self.drone_yaw - self.ugv_yaw)

        cmd = Twist()

        # ---- XY 제어 (월드 → UGV body frame 변환) ----
        # /cmd_vel 의 linear 는 UGV body frame 기준이라
        # 월드 오차를 UGV yaw 로 회전시켜야 함.
        if dist >= self.xy_deadband:
            cos_y = math.cos(self.ugv_yaw)
            sin_y = math.sin(self.ugv_yaw)
            err_x_body =  cos_y * err_x_world + sin_y * err_y_world
            err_y_body = -sin_y * err_x_world + cos_y * err_y_world

            vx = self.kp_xy * err_x_body
            vy = self.kp_xy * err_y_body

            # 속도 벡터 크기 제한
            speed = math.hypot(vx, vy)
            if speed > self.xy_speed_limit:
                scale = self.xy_speed_limit / speed
                vx *= scale
                vy *= scale

            cmd.linear.x = float(vx)
            cmd.linear.y = float(vy)

        # ---- Yaw 제어 ----
        if abs(err_yaw) >= self.yaw_deadband:
            w = self.kp_yaw * err_yaw
            w = max(-self.yaw_speed_limit, min(self.yaw_speed_limit, w))
            cmd.angular.z = float(w)

        self.cmd_pub.publish(cmd)

    def _send_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = UgvFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C → 종료')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
