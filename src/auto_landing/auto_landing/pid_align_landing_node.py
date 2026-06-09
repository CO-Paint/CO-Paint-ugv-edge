#!/usr/bin/env python3
"""
PID Landing Controller Node - CO-Paint
========================================
UGV 상단 카메라로 드론 하단 ArUco 마커를 추적해
UGV 바퀴(/cmd_vel)를 PID 제어하여 드론 정중앙으로 정렬 + 방향 일치.

마스터 노드 인터페이스:
  [Sub] /landing/start_auto_land       (Bool)   True 받으면 추적 활성화
  [Sub] /flight_control/status         (String) LANDED_CONFIRM 받으면 추적 종료
  [Pub] /aruco_landing/marker_detected (Bool)   마커 감지 시 True
  [Pub] /cmd_vel                       (Twist)  UGV 바퀴 명령 (linear.x, linear.y, angular.z)
  [Pub] /landing_status                (String) 디버그 상태

설계:
- 카메라는 노드 시작 시 점유 (계속 열어둠).
- 활성화 전: PID 미수행, cmd_vel 미발행. 마커 감지만 publish.
- 활성화 후: 마커 감지 시 XY + yaw PID 동시에 발행 (메카넘 평행이동 + 회전).
- LANDED_CONFIRM 수신 시 비활성화 + 마지막으로 정지 명령.
- debug_gui 파라미터로 cv2.imshow 토글.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import cv2
import cv2.aruco as aruco
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool


class PidLandingControllerNode(Node):
    def __init__(self):
        super().__init__('pid_landing_controller_node')

        # ---- 파라미터: 카메라/마커 ----
        self.declare_parameter('camera_index', 1)
        self.declare_parameter('camera_fallback_index', 0)
        self.declare_parameter('debug_gui', True)
        self.declare_parameter('marker_length', 0.055)   # m

        # ---- 파라미터: 카메라 → UGV 중심 오프셋 (m) ----
        # 카메라가 UGV 중심에서 얼마나 떨어져 있는지.
        # 양수 = UGV 전방, 음수 = UGV 후방
        self.declare_parameter('camera_offset_x', 0.047)  # 전방 4.7cm
        self.declare_parameter('camera_offset_y', 0.0)    # 좌우 오프셋 없음

        # ---- 파라미터: XY PID ----
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.01)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('error_deadband', 0.03)   # m
        self.declare_parameter('output_limit', 0.3)      # m/s (0.15에서 상향)

        # ---- 파라미터: Yaw PID ----
        self.declare_parameter('kp_yaw', 0.8)
        self.declare_parameter('ki_yaw', 0.0)
        self.declare_parameter('kd_yaw', 0.1)
        self.declare_parameter('yaw_deadband', 0.087)    # rad (~5도)
        self.declare_parameter('angular_limit', 0.3)     # rad/s

        # 로드
        self.debug_gui      = bool(self.get_parameter('debug_gui').value)
        self.marker_length  = float(self.get_parameter('marker_length').value)

        self.cam_offset_x   = float(self.get_parameter('camera_offset_x').value)
        self.cam_offset_y   = float(self.get_parameter('camera_offset_y').value)

        self.kp             = float(self.get_parameter('kp').value)
        self.ki             = float(self.get_parameter('ki').value)
        self.kd             = float(self.get_parameter('kd').value)
        self.error_deadband = float(self.get_parameter('error_deadband').value)
        self.output_limit   = float(self.get_parameter('output_limit').value)

        self.kp_yaw         = float(self.get_parameter('kp_yaw').value)
        self.ki_yaw         = float(self.get_parameter('ki_yaw').value)
        self.kd_yaw         = float(self.get_parameter('kd_yaw').value)
        self.yaw_deadband   = float(self.get_parameter('yaw_deadband').value)
        self.angular_limit  = float(self.get_parameter('angular_limit').value)

        # ---- QoS ----
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Publisher ----
        self.cmd_pub      = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.status_pub   = self.create_publisher(String, '/landing_status', 10)
        self.detected_pub = self.create_publisher(
            Bool, '/aruco_landing/marker_detected', cmd_qos)

        # ---- Subscriber ----
        self.create_subscription(
            Bool, '/landing/start_auto_land',
            self._on_start_auto_land, cmd_qos)
        self.create_subscription(
            String, '/flight_control/status',
            self._on_flight_status, cmd_qos)

        # ---- 카메라 ----
        cam_idx     = int(self.get_parameter('camera_index').value)
        cam_idx_alt = int(self.get_parameter('camera_fallback_index').value)
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().warn(
                f'카메라 {cam_idx} 열기 실패 → {cam_idx_alt} 재시도')
            self.cap = cv2.VideoCapture(cam_idx_alt, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                self.get_logger().error('카메라 열기 최종 실패')

        # ---- ArUco ----
        try:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        except AttributeError:
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        try:
            self.aruco_params = aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_params = aruco.DetectorParameters()

        # 카메라 캘리브레이션 (임시 보정값)
        self.camera_matrix = np.array([
            [657.8,   0.0, 320.0],
            [  0.0, 657.8, 240.0],
            [  0.0,   0.0,   1.0],
        ], dtype=float)
        self.dist_coeffs = np.zeros((4, 1))

        # ---- PID 상태 ----
        self.integral_x   = 0.0
        self.integral_y   = 0.0
        self.integral_yaw = 0.0
        self.prev_error_x   = 0.0
        self.prev_error_y   = 0.0
        self.prev_error_yaw = 0.0

        # ---- 활성화 플래그 ----
        self.active = False

        # ---- 30Hz 메인 루프 ----
        self.timer_period = 1.0 / 30.0
        self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info(
            f'PID Landing Node started. '
            f'active=False, debug_gui={self.debug_gui}')

    # ================= 콜백 =================
    def _on_start_auto_land(self, msg: Bool):
        if msg.data and not self.active:
            self.active = True
            self._reset_pid()
            self.get_logger().info('✅ START_AUTO_LAND 수신 → 추적 활성화')
        elif not msg.data and self.active:
            self.active = False
            self._send_stop()
            self.get_logger().info('추적 비활성화 (False 수신)')

    def _on_flight_status(self, msg: String):
        if msg.data.strip().upper() == 'LANDED_CONFIRM' and self.active:
            self.active = False
            self._send_stop()
            self.get_logger().info('✅ LANDED_CONFIRM 수신 → 추적 종료')

    def _reset_pid(self):
        self.integral_x   = 0.0
        self.integral_y   = 0.0
        self.integral_yaw = 0.0
        self.prev_error_x   = 0.0
        self.prev_error_y   = 0.0
        self.prev_error_yaw = 0.0

    def _send_stop(self):
        self.cmd_pub.publish(Twist())

    # ================= 30Hz 루프 =================
    def control_loop(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('카메라 프레임 수신 실패', throttle_duration_sec=2.0)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        marker_detected = ids is not None and len(ids) > 0
        self.detected_pub.publish(Bool(data=marker_detected))

        debug_info = []
        color = (255, 255, 255)

        if marker_detected:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length,
                self.camera_matrix, self.dist_coeffs)

            cam_x = tvecs[0][0][0]
            cam_y = tvecs[0][0][1]
            cam_z = tvecs[0][0][2]

            # 카메라 → UGV base 좌표 매핑 (실기에서 부호 확인됨)
            # 카메라가 UGV 중심에서 전방으로 offset만큼 떨어져 있으므로 보정.
            # 드론이 UGV 정중앙 위에 있을 때 error=0이 되도록 함.
            error_x   =  cam_y + self.cam_offset_x   # 전후 오차 (m)
            error_y   = -cam_x + self.cam_offset_y   # 좌우 오차 (m)
            drone_alt =  cam_z                        # 고도 (m)

            # rvec → yaw 오차 (rad). 부호는 실기 테스트에서 확정.
            error_yaw = self._extract_yaw_error(rvecs[0])

            if self.active:
                cmd, state_str, color = self._pid_step(
                    error_x, error_y, error_yaw)
                self.cmd_pub.publish(cmd)
                self.status_pub.publish(String(
                    data=f'{state_str} | alt: {drone_alt:.2f}m'))
            else:
                state_str = 'STATE: [ DETECTED, IDLE ]'
                color = (200, 200, 200)
                self.status_pub.publish(String(data=state_str))

            debug_info.append(state_str)
            debug_info.append(f'Drone Alt (Z) : {drone_alt:.3f} m')
            debug_info.append(
                f'Error X (Fwd) : {error_x*100:+6.1f} cm')
            debug_info.append(
                f'Error Y (Lat) : {error_y*100:+6.1f} cm')
            debug_info.append(
                f'Error Yaw     : {math.degrees(error_yaw):+6.1f} deg')

            if self.debug_gui:
                aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(
                    frame, self.camera_matrix, self.dist_coeffs,
                    rvecs[0], tvecs[0], 0.1)
        else:
            if self.active:
                self._send_stop()
                self._reset_pid()
                state_str = 'STATE: [ SEARCHING ]'
                color = (0, 0, 255)
            else:
                state_str = 'STATE: [ IDLE ]'
                color = (128, 128, 128)
            self.status_pub.publish(String(data=state_str))
            debug_info.append(state_str)

        if self.debug_gui:
            self._draw_overlay(frame, debug_info, color)
            cv2.imshow('ROS 2 Auto Landing Debug GUI', frame)
            cv2.waitKey(1)

    # ================= 자세 → yaw 오차 =================
    def _extract_yaw_error(self, rvec) -> float:
        """
        ArUco rvec → yaw 오차 (rad).

        카메라가 위(천장)를 보는 자세에서, 마커 평면의 회전이 곧 yaw 차이.
        rvec (Rodrigues) → 회전행렬 R → atan2(R[1,0], R[0,0]) 으로 yaw 추출.

        실기 보정:
          - 정면일 때 ±180° 오프셋이 나오므로 ±π 만큼 빼서 0 근처로 정규화.
          - 시계방향 회전 시 음수가 되도록 부호 반전 (UGV 시계방향 = angular.z < 0).
        """
        R, _ = cv2.Rodrigues(rvec)
        yaw = math.atan2(R[1, 0], R[0, 0])

        # 1) 180° 오프셋 제거: 마커 정면 ↔ 0 으로 정규화
        if yaw > 0:
            yaw -= math.pi
        else:
            yaw += math.pi

        # 2) 부호 반전: 마커 시계방향 회전 → error 음수 → angular.z 음수 (= UGV 시계방향)
        return yaw

    # ================= PID 한 스텝 =================
    def _pid_step(self, error_x: float, error_y: float, error_yaw: float):
        """XY + yaw PID. (Twist, state_str, color) 반환."""
        cmd = Twist()

        in_xy_band  = (abs(error_x) < self.error_deadband
                       and abs(error_y) < self.error_deadband)
        in_yaw_band = abs(error_yaw) < self.yaw_deadband

        if in_xy_band and in_yaw_band:
            return cmd, 'STATE: [ LOCKED ON ] - Ready to Land', (0, 255, 255)

        dt = self.timer_period

        # XY (데드밴드 안이면 출력 0)
        if not in_xy_band:
            self.integral_x += error_x * dt
            dx = (error_x - self.prev_error_x) / dt
            out_x = (self.kp * error_x
                     + self.ki * self.integral_x
                     + self.kd * dx)
            self.prev_error_x = error_x

            self.integral_y += error_y * dt
            dy = (error_y - self.prev_error_y) / dt
            out_y = (self.kp * error_y
                     + self.ki * self.integral_y
                     + self.kd * dy)
            self.prev_error_y = error_y

            cmd.linear.x = float(np.clip(out_x, -self.output_limit, self.output_limit))
            cmd.linear.y = float(np.clip(out_y, -self.output_limit, self.output_limit))

        # Yaw (데드밴드 안이면 출력 0)
        if not in_yaw_band:
            self.integral_yaw += error_yaw * dt
            dyaw = (error_yaw - self.prev_error_yaw) / dt
            out_yaw = (self.kp_yaw * error_yaw
                       + self.ki_yaw * self.integral_yaw
                       + self.kd_yaw * dyaw)
            self.prev_error_yaw = error_yaw

            cmd.angular.z = float(np.clip(
                out_yaw, -self.angular_limit, self.angular_limit))

        # 상태 표시
        if in_xy_band:
            state_str = 'STATE: [ XY OK, ALIGNING YAW ]'
        elif in_yaw_band:
            state_str = 'STATE: [ YAW OK, ALIGNING XY ]'
        else:
            state_str = 'STATE: [ TRACKING ]'

        return cmd, state_str, (0, 255, 0)

    def _draw_overlay(self, frame, debug_info, color):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (635, 155), (0, 0, 0), -1)
        frame_blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        frame[:] = frame_blended
        y_offset = 30
        for text in debug_info:
            txt_color = color if 'STATE' in text else (255, 255, 255)
            cv2.putText(frame, text, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)
            y_offset += 25

    # ================= 종료 =================
    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.debug_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PidLandingControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C 입력 감지 → 종료')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
