import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.msg import LinkStates
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from rclpy.time import Time


class GroundTruthFromLinkStates(Node):
    """Subscribe to /gazebo/link_states, extract base_footprint pose/twist,
    transform into `ref_frame` and publish as PoseStamped and TwistStamped.

    The reference frame is a ROS parameter `ref_frame` (settable from the
    launch file or command line). If the requested frame is not available the
    node will publish the original 'world' stamped messages as a fallback.
    """

    def __init__(self):
        super().__init__('ground_truth_from_link_states')
        self._last_warn_times = {}

        # parameters
        self.declare_parameter('link_name', 'base_footprint')
        self.declare_parameter('suffix_match', True)
        self.declare_parameter('ref_frame', 'world')

        self.link_name_param = self.get_parameter('link_name').get_parameter_value().string_value
        self.suffix_match = self.get_parameter('suffix_match').get_parameter_value().bool_value
        self.ref_frame = self.get_parameter('ref_frame').get_parameter_value().string_value

        # TF buffer/listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # publishers
        qos = QoSProfile(depth=10)
        self.pub_pose = self.create_publisher(PoseStamped, '/tb3/ground_truth/pose', qos)
        self.pub_twist = self.create_publisher(TwistStamped, '/tb3/ground_truth/twist', qos)

        # subscriber
        self.sub = self.create_subscription(LinkStates, '/gazebo/link_states', self._on_link_states, qos)

        self.get_logger().info(
            f'Listening to /gazebo/link_states for link="{self.link_name_param}" '
            f'(suffix_match={self.suffix_match}); publishing in frame "{self.ref_frame}".'
        )

    def _find_link_index(self, names):
        # exact match first
        if self.link_name_param in names:
            return names.index(self.link_name_param)

        if self.suffix_match:
            for i, n in enumerate(names):
                if n.endswith(self.link_name_param) or n.endswith('::' + self.link_name_param):
                    return i
        return None

    def _on_link_states(self, msg: LinkStates):
        idx = self._find_link_index(msg.name)
        if idx is None:
            self._warn_throttle('link_not_found', 2000, f'Link "{self.link_name_param}" not found in /gazebo/link_states.')
            return

        now = self.get_clock().now().to_msg()

        # package the world-frame messages (Gazebo uses 'world')
        ps_world = PoseStamped()
        ps_world.header.stamp = now
        ps_world.header.frame_id = 'world'
        ps_world.pose = msg.pose[idx]

        ts_world = TwistStamped()
        ts_world.header.stamp = now
        ts_world.header.frame_id = 'world'
        ts_world.twist = msg.twist[idx]

        # if the requested ref frame is world, skip transforms
        if self.ref_frame == 'world':
            self.pub_pose.publish(ps_world)
            self.pub_twist.publish(ts_world)
            return

        # otherwise attempt to transform to the requested ref_frame; use latest
        # available transform (Time() == latest) but wait briefly if necessary.
        try:
            # wait up to 0.2s for a transform to appear
            if self.tf_buffer.can_transform(self.ref_frame, ps_world.header.frame_id, Time(), timeout=Duration(seconds=0.2)):
                ps_out = self.tf_buffer.transform(ps_world, self.ref_frame, timeout=Duration(seconds=0.05))
            else:
                raise RuntimeError('transform not available')
        except Exception as e:
            self._warn_throttle('pose_transform_failed', 2000, f'Pose transform world -> {self.ref_frame} failed; publishing in world. ({e})')
            ps_out = ps_world

        try:
            if self.tf_buffer.can_transform(self.ref_frame, ts_world.header.frame_id, Time(), timeout=Duration(seconds=0.2)):
                ts_out = self.tf_buffer.transform(ts_world, self.ref_frame, timeout=Duration(seconds=0.05))
            else:
                raise RuntimeError('transform not available')
        except Exception as e:
            # tf2 may not support TwistStamped transform in some environments; fallback
            self._warn_throttle('twist_transform_failed', 2000, f'Twist transform world -> {self.ref_frame} failed; publishing in world. ({e})')
            ts_out = ts_world

        self.pub_pose.publish(ps_out)
        self.pub_twist.publish(ts_out)

    def _warn_throttle(self, key: str, period_ms: int, msg: str):
        import time
        now = time.time() * 1000.0
        last = self._last_warn_times.get(key, 0.0)
        if (now - last) >= float(period_ms):
            try:
                self.get_logger().warning(msg)
            except Exception:
                print('WARNING:', msg)
            self._last_warn_times[key] = now

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = GroundTruthFromLinkStates()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.msg import LinkStates
from rclpy.qos import QoSProfile
import tf2_geometry_msgs
import tf2_ros

class GroundTruthFromLinkStates(Node):

    def __init__(self):
        super().__init__('ground_truth_from_link_states')
        self.log = self.get_logger()
        # store last warning times for throttling
        self._last_warn_times = {}
        
        self.declare_parameter('link_name', 'base_footprint')
        self.declare_parameter('suffix_match', True)
        # Target frame to express outputs in (e.g., "map", "odom", or "world")
        self.declare_parameter('ref_frame', 'world')

        self.link_name_param = self.get_parameter('link_name').get_parameter_value().string_value
        self.suffix_match = self.get_parameter('suffix_match').get_parameter_value().bool_value
        self.ref_frame = self.get_parameter('ref_frame').get_parameter_value().string_value

        # TF buffer/listener (used to transform from Gazebo "world" frame -> ref_frame)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        qos = QoSProfile(depth=10)
        self.pub_pose = self.create_publisher(PoseStamped, '/tb3/ground_truth/pose', qos)
        self.pub_twist = self.create_publisher(TwistStamped, '/tb3/ground_truth/twist', qos)

        # Subscriber
        self.sub = self.create_subscription(LinkStates, '/gazebo/link_states', self._on_link_states, qos)

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

        # Wrap pose & twist in stamped messages with Gazebo's "world" as the source frame
        ps_world = PoseStamped()
        ps_world.header.stamp = now
        ps_world.header.frame_id = 'world'
        ps_world.pose = msg.pose[idx]

        ts_world = TwistStamped()
        ts_world.header.stamp = now
        ts_world.header.frame_id = 'world'
        ts_world.twist = msg.twist[idx]

        # If target frame is "world", just pass through quickly
        if self.ref_frame == 'world':
            self.pub_pose.publish(ps_world)
            self.pub_twist.publish(ts_world)
            return

        # Otherwise try to transform to the requested reference frame
        try:
            ps_out = self.tf_buffer.transform(ps_world, self.ref_frame, timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception as e:
            # Throttled warning with fallback
            self._warn_throttle('pose_transform_failed', 2000, f'Pose transform world -> {self.ref_frame} failed; publishing in world. ({e})')
            ps_out = ps_world

        try:
            # tf2 supports TwistStamped transforms (rotates linear/angular parts)
            ts_out = self.tf_buffer.transform(ts_world, self.ref_frame, timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception as e:
            self._warn_throttle('twist_transform_failed', 2000, f'Twist transform world -> {self.ref_frame} failed; publishing in world. ({e})')
            ts_out = ts_world

        # Publish the (possibly-fallback) transformed messages
        self.pub_pose.publish(ps_out)
        self.pub_twist.publish(ts_out)

    def _warn_throttle(self, key: str, period_ms: int, msg: str):
        """Emit a warning message at most once every period_ms milliseconds per key.

        This is a small compatibility helper because the RcutilsLogger returned
        by rclpy.get_logger() does not implement warn_throttle in some rclpy
        versions. key should uniquely identify the warning location/message.
        """
        import time
        now = time.time() * 1000.0
        last = self._last_warn_times.get(key, 0.0)
        if (now - last) >= float(period_ms):
            # use the node's logger to emit the warning
            try:
                self.get_logger().warning(msg)
            except Exception:
                # fallback to print if logging fails for some reason
                print('WARNING:', msg)
            self._last_warn_times[key] = now

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
