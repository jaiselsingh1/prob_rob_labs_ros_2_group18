#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Float32MultiArray

class EKFAccuracyNode(Node):
    def __init__(self):
        super().__init__('ekf_accuracy')
        
        # Topic names matched to your system configuration
        GT_TOPIC = '/tb3/ground_truth/pose' 
        EKF_TOPIC = '/ekf_pose'

        self.gt_sub = self.create_subscription(PoseStamped, GT_TOPIC, self.gt_callback, 10)
        self.ekf_sub = self.create_subscription(PoseWithCovarianceStamped, EKF_TOPIC, self.ekf_callback, 10)
        
        self.err_pub = self.create_publisher(Float32MultiArray, '/ekf_error', 10)
        
        self.last_gt = None
        self.last_ekf = None
        
        self.has_received_gt = False
        self.has_received_ekf = False

        self.get_logger().info(f"Accuracy Node Started. Waiting for GT on: {GT_TOPIC}, EKF on: {EKF_TOPIC}")

        self.timer = self.create_timer(2.0, self.check_status)

    def check_status(self):
        if not self.has_received_gt and not self.has_received_ekf:
            self.get_logger().warn("Waiting for BOTH Ground Truth and EKF...", throttle_duration_sec=2.0)
        elif not self.has_received_gt:
            self.get_logger().warn("Got EKF, but WAITING FOR GROUND TRUTH...", throttle_duration_sec=2.0)
        elif not self.has_received_ekf:
            self.get_logger().warn("Got Ground Truth, but WAITING FOR EKF...", throttle_duration_sec=2.0)

    def gt_callback(self, msg):
        self.last_gt = msg
        if not self.has_received_gt:
            self.has_received_gt = True
            self.get_logger().info("Received first Ground Truth message!")

    def ekf_callback(self, msg):
        self.last_ekf = msg
        if not self.has_received_ekf:
            self.has_received_ekf = True
            self.get_logger().info("Received first EKF message!")
        
        self._compute_error()

    def _compute_error(self):
        if self.last_gt is None or self.last_ekf is None:
            return

        gt_x = self.last_gt.pose.position.x
        gt_y = self.last_gt.pose.position.y
        ekf_x = self.last_ekf.pose.pose.position.x
        ekf_y = self.last_ekf.pose.pose.position.y

        pos_err = math.sqrt((gt_x - ekf_x)**2 + (gt_y - ekf_y)**2)

        gt_yaw = self._quat_to_yaw(self.last_gt.pose.orientation)
        ekf_yaw = self._quat_to_yaw(self.last_ekf.pose.pose.orientation)
        yaw_err = abs(self._angle_diff(gt_yaw, ekf_yaw))

        err_msg = Float32MultiArray()
        err_msg.data = [pos_err, yaw_err]
        self.err_pub.publish(err_msg)

        self.get_logger().info(f"Error: Pos={pos_err:.4f}m, Yaw={yaw_err:.4f}rad", throttle_duration_sec=1.0)


    @staticmethod
    def _quat_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = EKFAccuracyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()