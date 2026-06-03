from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='arduino_mecanum_bridge',
            executable='serial_bridge',
            name='arduino_mecanum_bridge',
            output='screen',
            parameters=[{
                'port': '/dev/ttyUSB0',
                'baudrate': 115200,

                # Match these to your real robot.
                # Current onboard_MAH01.hpp uses WHEEL_DIAMETER = 0.068 m.
                'wheel_radius': 0.034,
                'base_width': 0.26,
                'base_length': 0.26,

                'cmd_vel_topic': '/cmd_vel',
                'wheel_vel_topic': '/wheel_vel',
                'wheel_state_topic': '/wheel_state',
                'command_timeout_sec': 0.5,
            }],
        )
    ])
