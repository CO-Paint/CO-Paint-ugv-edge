#!/usr/bin/env python3

import time
import math
import threading
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

try:
    import serial
except ImportError as exc:
    raise RuntimeError(
        'pyserial is not installed. Install it with: sudo apt install python3-serial'
    ) from exc


class ArduinoMecanumBridge(Node):
    """
    ROS2 <-> Arduino USB serial bridge.

    Subscribes:
      /wheel_vel : std_msgs/Float32MultiArray, [LF, RF, LR, RR] in rad/s
      /cmd_vel   : geometry_msgs/Twist, mecanum command in m/s and rad/s

    Publishes:
      /wheel_state : std_msgs/Float32MultiArray, [LF, RF, LR, RR] in rad/s

    Serial protocol:
      ROS2 -> Arduino: W LF RF LR RR\n
      Arduino -> ROS2: S LF RF LR RR\n
    """

    def __init__(self):
        super().__init__('arduino_mecanum_bridge')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('wheel_radius', 0.034)  # onboard_MAH01 WHEEL_DIAMETER 0.068 / 2
        self.declare_parameter('base_width', 0.26)
        self.declare_parameter('base_length', 0.26)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('wheel_vel_topic', '/wheel_vel')
        self.declare_parameter('wheel_state_topic', '/wheel_state')
        self.declare_parameter('command_timeout_sec', 0.5)

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.base_width = float(self.get_parameter('base_width').value)
        self.base_length = float(self.get_parameter('base_length').value)
        self.command_timeout_sec = float(self.get_parameter('command_timeout_sec').value)

        self.serial_lock = threading.Lock()
        self.running = True

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.05,
            write_timeout=0.1,
        )

        
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.get_logger().info(f'Opened {self.port} @ {self.baudrate}')

        self.wheel_state_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('wheel_state_topic').value,
            10,
        )

        self.wheel_vel_sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter('wheel_vel_topic').value,
            self.wheel_vel_callback,
            10,
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            self.cmd_vel_callback,
            10,
        )

        # self.watchdog_timer = self.create_timer(
        #     self.command_timeout_sec,
        #     self.send_stop,
        # )

        self.rx_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.rx_thread.start()

        self.get_logger().info(
            'Ready. Send /cmd_vel or /wheel_vel. '
            'Publishing measured wheels on /wheel_state.'
        )

    def write_line(self, line: str):
        data = (line + '\n').encode('ascii')
        with self.serial_lock:
            self.ser.write(data)

    def send_wheel_command(self, lf: float, rf: float, lr: float, rr: float):
        lf = 2 * lf;
        rf = 2 * rf;
        lr = 2 * lr;
        rr = 2 * rr;
        line = f'W {lf:.4f} {rf:.4f} {lr:.4f} {rr:.4f}'
        self.write_line(line)
        self.get_logger().info(f'TX: {line}')

    def send_stop(self):
        # Keep sending STOP periodically if no command is active.
        # The Arduino also has its own timeout, so this is extra safety.
        self.write_line('STOP')

    def wheel_vel_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn('/wheel_vel needs 4 values: [LF, RF, LR, RR] rad/s')
            return

        lf, rf, lr, rr = [float(v) for v in msg.data[:4]]
        self.send_wheel_command(lf, rf, lr, rr)

    def cmd_vel_callback(self, msg: Twist):
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        wz = float(msg.angular.z)

        r = self.wheel_radius
        k = (self.base_length + self.base_width) / 2.0

        if r <= 0.0:
            self.get_logger().error('wheel_radius must be greater than 0')
            return

        # Mecanum inverse kinematics.
        # Wheel order: LF, RF, LR, RR.
        lf = (vx - vy - k * wz) / r
        rf = (vx + vy + k * wz) / r
        lr = (vx + vy - k * wz) / r
        rr = (vx - vy + k * wz) / r

        self.send_wheel_command(lf, rf, lr, rr)

    def read_loop(self):
        while self.running and rclpy.ok():
            try:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode('ascii', errors='ignore').strip()
                if not line:
                    continue

                if line.startswith('S '):
                    self.handle_state_line(line)
                elif line.startswith('BOOT') or line.startswith('PONG') or line.startswith('OK'):
                    self.get_logger().info(f'Arduino: {line}')
                elif line.startswith('ERR'):
                    self.get_logger().warn(f'Arduino: {line}')
                else:
                    self.get_logger().debug(f'Arduino: {line}')

            except Exception as exc:
                self.get_logger().warn(f'Serial read error: {exc}')

    def handle_state_line(self, line: str):
        parts = line.split()
        if len(parts) != 5:
            return

        try:
            values = [float(parts[i]) for i in range(1, 5)]
        except ValueError:
            return

        msg = Float32MultiArray()
        msg.data = values
        self.wheel_state_pub.publish(msg)

    def destroy_node(self):
        self.running = False
        try:
            self.write_line('STOP')
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = ArduinoMecanumBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
