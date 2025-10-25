#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import math


def yaw_from_quaternion(q):
    """Convert quaternion to yaw (radians)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomErrorPublisher(Node):
    """
    Subscribes to ground truth and estimated odometry,
    computes Euclidean (pos) and yaw errors, and publishes them.
    """

    def __init__(self):
        super().__init__('odom_error_publisher')

        # ---- Parameters ----
        self.declare_parameter('ground_truth_topic', '/tb3/ground_truth/pose')
        self.declare_parameter('estimated_odom_topic', '/ekf_odom')
        self.declare_parameter('error_topic', '/odom_error')

        gt_topic = self.get_parameter('ground_truth_topic').get_parameter_value().string_value
        est_topic = self.get_parameter('estimated_odom_topic').get_parameter_value().string_value
        error_topic = self.get_parameter('error_topic').get_parameter_value().string_value

        # ---- Subscribers ----
        self.gt_pose = None
        self.est_pose = None
        # ground-truth is published as PoseStamped by the simulator node
        self.gt_sub = self.create_subscription(PoseStamped, gt_topic, self._gt_callback, 10)
        # estimated odometry is published as Odometry
        self.est_sub = self.create_subscription(Odometry, est_topic, self._est_callback, 10)

        # ---- Publisher ----
        self.error_pub = self.create_publisher(Float32MultiArray, error_topic, 10)

        self.get_logger().info(
            f"Listening to ground_truth='{gt_topic}', estimated='{est_topic}', "
            f"publishing errors to '{error_topic}'"
        )

    # --- Callbacks ---
    def _gt_callback(self, msg: PoseStamped):
        # msg.pose is a geometry_msgs/Pose
        self.gt_pose = msg.pose
        self._compute_and_publish()

    def _est_callback(self, msg):
        self.est_pose = msg.pose.pose
        self._compute_and_publish()

    # --- Compute and publish error ---
    def _compute_and_publish(self):
        if self.gt_pose is None or self.est_pose is None:
            return

        # Position error (Euclidean distance)
        dx = self.est_pose.position.x - self.gt_pose.position.x
        dy = self.est_pose.position.y - self.gt_pose.position.y
        pos_error = math.sqrt(dx ** 2 + dy ** 2)

        # Orientation error (yaw)
        yaw_gt = yaw_from_quaternion(self.gt_pose.orientation)
        yaw_est = yaw_from_quaternion(self.est_pose.orientation)
        yaw_err = yaw_est - yaw_gt
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))  # wrap to [-pi, pi]

        # Publish [pos_error, yaw_error_deg]
        msg = Float32MultiArray()
        msg.data = [float(pos_error), math.degrees(yaw_err)]
        self.error_pub.publish(msg)

        # Optional log (debug)
        self.get_logger().debug(f"Δpos={pos_error:.3f} m | Δyaw={math.degrees(yaw_err):.2f}°")


def main(args=None):
    rclpy.init(args=args)
    node = OdomErrorPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
