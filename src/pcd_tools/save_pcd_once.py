#!/usr/bin/env python3

import sys
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class SavePCDOnce(Node):
    def __init__(self, topic_name, output_path):
        super().__init__('save_pcd_once')
        self.topic_name = topic_name
        self.output_path = output_path
        self.saved = False

        self.sub = self.create_subscription(
            PointCloud2,
            topic_name,
            self.callback,
            10
        )

        self.get_logger().info(f'Waiting for PointCloud2 topic: {topic_name}')
        self.get_logger().info(f'Output PCD: {output_path}')

    def callback(self, msg):
        if self.saved:
            return

        field_names = [field.name for field in msg.fields]
        self.get_logger().info(f'PointCloud2 fields: {field_names}')

        required = ['x', 'y', 'z']
        for name in required:
            if name not in field_names:
                self.get_logger().error(f'Missing required field: {name}')
                return

        use_intensity = 'intensity' in field_names

        if use_intensity:
            read_fields = ['x', 'y', 'z', 'intensity']
        else:
            read_fields = ['x', 'y', 'z']

        points = []

        for p in point_cloud2.read_points(msg, field_names=read_fields, skip_nans=True):
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue

            if use_intensity:
                intensity = float(p[3])
                points.append((x, y, z, intensity))
            else:
                points.append((x, y, z))

        if len(points) == 0:
            self.get_logger().error('No valid points to save.')
            return

        with open(self.output_path, 'w') as f:
            f.write('# .PCD v0.7 - Point Cloud Data file format\n')
            f.write('VERSION 0.7\n')

            if use_intensity:
                f.write('FIELDS x y z intensity\n')
                f.write('SIZE 4 4 4 4\n')
                f.write('TYPE F F F F\n')
                f.write('COUNT 1 1 1 1\n')
            else:
                f.write('FIELDS x y z\n')
                f.write('SIZE 4 4 4\n')
                f.write('TYPE F F F\n')
                f.write('COUNT 1 1 1\n')

            f.write(f'WIDTH {len(points)}\n')
            f.write('HEIGHT 1\n')
            f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
            f.write(f'POINTS {len(points)}\n')
            f.write('DATA ascii\n')

            for p in points:
                if use_intensity:
                    f.write(f'{p[0]} {p[1]} {p[2]} {p[3]}\n')
                else:
                    f.write(f'{p[0]} {p[1]} {p[2]}\n')

        self.saved = True
        self.get_logger().info(f'Saved {len(points)} points to {self.output_path}')
        rclpy.shutdown()


def main():
    if len(sys.argv) != 3:
        print('Usage:')
        print('  ros2 run or python3 save_pcd_once.py <topic_name> <output.pcd>')
        print('')
        print('Example:')
        print('  python3 save_pcd_once.py /kiss/local_map ~/pcd_maps/map.pcd')
        return

    topic_name = sys.argv[1]
    output_path = sys.argv[2].replace('~', '/home/ugv')

    rclpy.init()
    node = SavePCDOnce(topic_name, output_path)
    rclpy.spin(node)


if __name__ == '__main__':
    main()