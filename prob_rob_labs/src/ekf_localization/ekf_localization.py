#!/usr/bin/env python3
import os
import yaml
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped


@dataclass
class Landmark:
    color: str
    x: float
    y: float
    radius: float
    height: float


class LandmarkEKFNode(Node):

    def __init__(self):
        super().__init__('landmark_ekf')

        # --- use_sim_time ---
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        if use_sim_time:
            self.get_logger().info('Using simulation time')
        else:
            self.get_logger().info('Using wall time')

        # --- map_file ---
        if not self.has_parameter('map_file'):
            self.declare_parameter('map_file', '')

        map_file_param = (
            self.get_parameter('map_file')
            .get_parameter_value()
            .string_value
        )

        if not map_file_param:
            try:
                share_dir = get_package_share_directory('prob_rob_labs')
                map_file_param = os.path.join(
                    share_dir, 'maps', 'landmarks_lab6.yaml'
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to get default map path from package share: {e}"
                )
                raise

        self.map_file = map_file_param
        self.get_logger().info(f"Loading landmark map from: {self.map_file}")

        # --- load landmark map ---
        self.landmarks_by_color = self._load_landmark_map(self.map_file)
        colors = list(self.landmarks_by_color.keys())
        self.get_logger().info(
            f"Loaded {len(colors)} landmarks: {colors}"
        )

        # ---- Assignment 2: subscribe to vision measurements for each landmark color ----
        self._measurement_subs = []
        self.last_measurements = {}

        for color, lm in self.landmarks_by_color.items():
            topic = f"/vision_{color}/measurement"

            sub = self.create_subscription(
                PointStamped,
                topic,
                lambda msg, color=color, lm=lm: self._vision_measurement_callback(
                    msg, color, lm
                ),
                10,
            )
            self._measurement_subs.append(sub)

            self.get_logger().info(
                f"Subscribed to {topic} for landmark color={color} "
                f"at world position x={lm.x:.2f}, y={lm.y:.2f}"
            )



    def _load_landmark_map(self, path: str):
        if not os.path.exists(path):
            self.get_logger().error(f"Map file does not exist: {path}")
            raise FileNotFoundError(path)

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict) and 'landmarks' in data:
            entries = data['landmarks']
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                "Unexpected YAML structure in landmark map. "
                "Expected a 'landmarks' list or top-level list."
            )

        landmarks_by_color = {}
        for lm in entries:
            color = str(lm['color']).strip()
            x = float(lm['x'])
            y = float(lm['y'])
            radius = float(lm.get('radius', 0.1))
            height = float(lm.get('height', 0.5))

            landmarks_by_color[color] = Landmark(
                color=color,
                x=x,
                y=y,
                radius=radius,
                height=height,
            )

        return landmarks_by_color



    def _vision_measurement_callback(self, msg: PointStamped, color: str, lm: Landmark):
        """
        - msg.point.x: distance
        - msg.point.y: bearing
        - color: of landmark
        - lm: (lm.x, lm.y) of landmark
        """

        d = float(msg.point.x)
        theta = float(msg.point.y)

        self.last_measurements[color] = {
            'stamp': msg.header.stamp,
            'd': d,
            'theta': theta,
            'landmark': lm,
        }

        self.get_logger().info(
            f"[measurement] color={color} d={d:.3f} m, theta={theta:.3f} rad "
            f"(landmark world: x={lm.x:.2f}, y={lm.y:.2f})"
        )



    def get_landmark_for_color(self, color: str):
        """ for correspondence mapping """
        
        return self.landmarks_by_color.get(color)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkEKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
