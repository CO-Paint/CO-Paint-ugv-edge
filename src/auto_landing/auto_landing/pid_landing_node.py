#!/usr/bin/env python3
"""
PID Landing Controller Node - CO-Paint
========================================
UGV 상단 카메라로 드론 하단 ArUco 마커를 추적해
UGV 바퀴(/cmd_vel)를 PID 제어하여 드론 정중앙으로 정렬.

마스터 노드 인터페이스:
  [Sub] /landing/start_auto_land       (Bool)   True 받으면 추적 활성화
  [Sub] /flight_control/status         (String) LANDED_CONFIRM 받으면 추적 종료
  [Pub] /aruco_landing/marker_detected (Bool)   마커 감지 시 True
  [Pub] /cmd_vel                       (Twist)  UGV 바퀴 명령
  [Pub] /landing_status                (String) 디버그 상태

설계:
- 카메라는 노드 시작 시 점유 (계속 열어둠).
- 활성화 전: PID 미수행, cmd_vel 미발행. 마커 감지만 publish (정보용).
- 활성화 후: 마커 감지 시 PID로 cmd_vel 발행.
- LANDED_CONFIRM 수신 시 비활성화 + 마지막으로 정지 명령.
- debug_gui 파라미터로 cv2.imshow 토글 (헤드리스 운용 시 False).
"""

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

        # ---- 파라미터 ----
        self.declare_parameter('camera_index', 1)
        self.declare_parameter('camera_fallback_index', 0)
        self.declare_parameter('debug_gui', True)
        self.declare_parameter('marker_length', 0.055)   # m (5.5cm)
        self.declare_parameter('kp', 0.6)
        self.declare_parameter('ki', 0.01)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('error_deadband', 0.03)   # m, 이 안이면 정지
        self.declare_parameter('output_limit', 0.15)      # m/s, 최대 속도

        self.debug_gui      = bool(self.get_parameter('debug_gui').value)
        self.marker_length  = float(self.get_parameter('marker_length').value)
        self.kp             = float(self.get_parameter('kp').value)
        self.ki             = float(self.get_parameter('ki').value)
        self.kd             = float(self.get_parameter('kd').value)
        self.error_deadband = float(self.get_parameter('error_deadband').value)
        self.output_limit   = float(self.get_parameter('output_limit').value)

        # ---- QoS ----
        # 마스터 명령: RELIABLE + TRANSIENT_LOCAL
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
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

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
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

    def _send_stop(self):
        cmd = Twist()  # 모든 필드 0
        self.cmd_pub.publish(cmd)

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

        # 감지 결과는 활성/비활성 무관하게 항상 발행
        # (마스터가 phase 2에서 이걸 보고 _force_auto_land 호출)
        self.detected_pub.publish(Bool(data=marker_detected))

        debug_info = []
        color = (255, 255, 255)

        if marker_detected:
            # 3D 자세 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length,
                self.camera_matrix, self.dist_coeffs)

            cam_x = tvecs[0][0][0]
            cam_y = tvecs[0][0][1]
            cam_z = tvecs[0][0][2]

            # 카메라 → UGV base 좌표 매핑
            error_x   =  cam_y    # 전후 오차
            error_y   = -cam_x    # 좌우 오차
            drone_alt =  cam_z    # 고도

            if self.active:
                cmd, state_str, color = self._pid_step(error_x, error_y)
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
                f'Error X (Fwd) : {error_x*100:+.1f} cm')
            debug_info.append(
                f'Error Y (Lat) : {error_y*100:+.1f} cm')

            if self.debug_gui:
                aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(
                    frame, self.camera_matrix, self.dist_coeffs,
                    rvecs[0], tvecs[0], 0.1)
        else:
            if self.active:
                # 마커 놓침 → 정지 + 적분 리셋
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
            cv2.waitKey(1)   # 단순히 GUI 이벤트 처리만 (종료 로직 제거)

    def _pid_step(self, error_x: float, error_y: float):
        """PID 한 스텝. (Twist, state_str, color) 반환."""
        cmd = Twist()

        if abs(error_x) < self.error_deadband and abs(error_y) < self.error_deadband:
            # 데드밴드 안 → 정지
            return cmd, 'STATE: [ LOCKED ON ] - Ready to Land', (0, 255, 255)

        self.integral_x += error_x * self.timer_period
        deriv_x = (error_x - self.prev_error_x) / self.timer_period
        out_x = self.kp * error_x + self.ki * self.integral_x + self.kd * deriv_x
        self.prev_error_x = error_x

        self.integral_y += error_y * self.timer_period
        deriv_y = (error_y - self.prev_error_y) / self.timer_period
        out_y = self.kp * error_y + self.ki * self.integral_y + self.kd * deriv_y
        self.prev_error_y = error_y

        cmd.linear.x = float(np.clip(out_x, -self.output_limit, self.output_limit))
        cmd.linear.y = float(np.clip(out_y, -self.output_limit, self.output_limit))
        return cmd, 'STATE: [ TRACKING ]', (0, 255, 0)

    def _draw_overlay(self, frame, debug_info, color):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (635, 130), (0, 0, 0), -1)
        frame_blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        # in-place 대체
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