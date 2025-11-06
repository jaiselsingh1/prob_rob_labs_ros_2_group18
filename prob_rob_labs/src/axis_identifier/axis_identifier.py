#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from prob_rob_msgs.msg import Point2DArrayStamped

HEARTBEAT_PERIOD = 0.1


class LandmarkAxisIdentifier(Node):
    """
    reads landmark corners and logs estimated height and vertical symmetry axis
    """

    def __init__(self):
        super().__init__('landmark_axis_identifier')
        self.log = self.get_logger()

        self.declare_parameter('landmark_color', 'cyan')
        self.landmark_color = (
            self.get_parameter('landmark_color')
            .get_parameter_value().string_value
        )

        self.sub = self.create_subscription(
            Point2DArrayStamped,
            f'/vision_{self.landmark_color}/corners',
            self.landmark_positioning_callback,
            10,
        )

        self.timer = self.create_timer(HEARTBEAT_PERIOD, self.heartbeat)

    def heartbeat(self):
        self.log.info('heartbeat')

    def landmark_positioning_callback(self, msg: Point2DArrayStamped):
        """
        for fewer than 8 points, treat as rectangular and use (min_x + max_x)/2
        for 8 or more points, treat as cylindrical and use mean x
        in both cases, height = max_y - min_y
        """
        num_points = len(msg.points)
        if num_points == 0:
            return

        xs = [p.x for p in msg.points]
        ys = [p.y for p in msg.points]

        height = max(ys) - min(ys)
        if height <= 0.0:
            return

        if num_points < 8:
            min_x = min(xs)
            max_x = max(xs)
            x_sym = 0.5 * (min_x + max_x)
            shape = 'rectangular'
        else:
            x_sym = sum(xs) / float(num_points)
            shape = 'cylindrical'

        self.log.info(
            f'{shape} landmark: height={height:.2f}, vertical_axis_position={x_sym:.2f}'
        )


def main():
    rclpy.init()
    node = LandmarkAxisIdentifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
