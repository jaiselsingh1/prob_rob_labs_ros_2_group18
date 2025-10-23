#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.msg import LinkStates


class GroundTruthFromLinkStates(Node):

    def __init__(self):
        super().__init__('ground_truth_from_link_states')
        self.log = self.get_logger()
        # store last warning times for throttling
        self._last_warn_times = {}
        
        self.declare_parameter('link_name', 'base_footprint')
        self.declare_parameter('suffix_match', True)
        # Target frame to express outputs in (e.g., "map", "odom", or "world")
        self.declare_parameter('ref_frame', 'odom')

        self.link_name_param = self.get_parameter('link_name').get_parameter_value().string_value
        self.suffix_match = self.get_parameter('suffix_match').get_parameter_value().bool_value
        self.ref_frame = self.get_parameter('ref_frame').get_parameter_value().string_value

        # TF buffer/listener (used to transform from Gazebo "world" frame -> ref_frame)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pub_pose = self.create_publisher(PoseStamped, '/tb3/ground_truth/pose', 1)
        self.pub_twist = self.create_publisher(TwistStamped, '/tb3/ground_truth/twist', 1)

        # Subscriber
        self.sub = self.create_subscription(LinkStates, '/gazebo/link_states', self._on_link_states, 1)

        self.get_logger().info(
            f'Listening to /gazebo/link_states for link="{self.link_name_param}" '
            f'(suffix_match={self.suffix_match}); publishing in frame "{self.ref_frame}".'
        )

    # Helper to find the link index
    def _find_link_index(self, names):
        # exact match first
        if self.link_name_param in names:
            return names.index(self.link_name_param)

        if self.suffix_match:
            # allow names like "turtlebot3_burger::base_footprint"
            for i, n in enumerate(names):
                if n.endswith(self.link_name_param) or n.endswith('::' + self.link_name_param):
                    return i
        return None

    def _on_link_states(self, msg: LinkStates):
        idx = self._find_link_index(msg.name)
        if idx is None:
            # Throttle warn to avoid spam
            self._warn_throttle('link_not_found', 2000, f'Link "{self.link_name_param}" not found in /gazebo/link_states.')
            return

        now = self.get_clock().now().to_msg()

        ps_ref = PoseStamped()
        ps_ref.header.stamp = now
        ps_ref.header.frame_id = self.ref_frame
        ps_ref.pose = msg.pose[idx]

        ts_ref = TwistStamped()
        ts_ref.header.stamp = now
        ts_ref.header.frame_id = self.ref_frame
        ts_ref.twist = msg.twist[idx]

        # Publish the (possibly-fallback) transformed messages
        self.pub_pose.publish(ps_ref)
        self.pub_twist.publish(ts_ref)

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    ground_truth_from_link_states = GroundTruthFromLinkStates()
    ground_truth_from_link_states.spin()
    ground_truth_from_link_states.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
