#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

class MapOdomPublisher(Node):
    def __init__(self):
        super().__init__('map_odom_publisher')

        # --- Parameters ---
        # The frame names usually used in ROS
        self.map_frame = 'map'
        self.odom_frame = 'odom'
        self.base_frame = 'base_footprint' # Or 'base_link', check your tf tree

        # --- TF Buffer & Listener (to get odom -> base) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- TF Broadcaster (to send map -> odom) ---
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Subscriber (to get map -> base from EKF) ---
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/ekf_pose',
            self._ekf_callback,
            10
        )

        # --- State ---
        # Initial transform is identity (x=0, y=0, yaw=0)
        self.latest_transform = TransformStamped()
        self.latest_transform.header.frame_id = self.map_frame
        self.latest_transform.child_frame_id = self.odom_frame
        self.latest_transform.transform.rotation.w = 1.0

        # --- Timer (Publish at 30Hz) --- 
        self.create_timer(1.0 / 30.0, self._publish_transform)

        self.get_logger().info("Map->Odom Publisher Started. Broadcasting at 30Hz.")

    def _ekf_callback(self, msg: PoseWithCovarianceStamped):
        """
        Triggered when EKF publishes a new pose estimate (map -> base).
        We calculate: T_map_odom = T_map_base * (T_odom_base)^-1
        """
        # 1. Get T_map_base from EKF message
        x_ekf = msg.pose.pose.position.x
        y_ekf = msg.pose.pose.position.y
        q_ekf = msg.pose.pose.orientation
        yaw_ekf = self._quat_to_yaw(q_ekf)

        # 2. Get T_odom_base from TF tree
        try:
            # Look up the latest available transform
            # We want: odom -> base_link
            trans = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                rclpy.time.Time()) # Get latest
        except Exception as e:
            # If odom is not ready yet, skip update
            # self.get_logger().warn(f"Could not get odom->base transform: {e}", throttle_duration_sec=2.0)
            return

        x_odom = trans.transform.translation.x
        y_odom = trans.transform.translation.y
        q_odom = trans.transform.rotation
        yaw_odom = self._quat_to_yaw(q_odom)

        # 3. Calculate T_map_odom
        # Using 2D Homogeneous Matrix Math
        
        # Matrix T_map_base
        T_mb = self._get_matrix(x_ekf, y_ekf, yaw_ekf)
        
        # Matrix T_odom_base
        T_ob = self._get_matrix(x_odom, y_odom, yaw_odom)
        
        # Matrix T_map_odom = T_mb * (T_ob)^-1
        T_ob_inv = np.linalg.inv(T_ob)
        T_mo = np.dot(T_mb, T_ob_inv)

        # Extract x, y, yaw from T_mo
        dx = T_mo[0, 2]
        dy = T_mo[1, 2]
        dyaw = math.atan2(T_mo[1, 0], T_mo[0, 0])

        # 4. Update the transform to be published
        self.latest_transform.transform.translation.x = dx
        self.latest_transform.transform.translation.y = dy
        self.latest_transform.transform.translation.z = 0.0
        
        q = self._yaw_to_quat(dyaw)
        self.latest_transform.transform.rotation.x = q[0]
        self.latest_transform.transform.rotation.y = q[1]
        self.latest_transform.transform.rotation.z = q[2]
        self.latest_transform.transform.rotation.w = q[3]

    def _publish_transform(self):
        """
        Timer callback to publish the latest calculated transform at 30Hz.
        """
        # Update timestamp to current time (required by TF)
        self.latest_transform.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.latest_transform)

    # --- Helpers ---
    def _get_matrix(self, x, y, theta):
        return np.array([
            [math.cos(theta), -math.sin(theta), x],
            [math.sin(theta),  math.cos(theta), y],
            [0.0,              0.0,             1.0]
        ])

    def _quat_to_yaw(self, q):
        # Handle both msg object and raw list/array
        if hasattr(q, 'w'):
            w, x, y, z = q.w, q.x, q.y, q.z
        else:
            x, y, z, w = q
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _yaw_to_quat(self, yaw):
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

def main(args=None):
    rclpy.init(args=args)
    node = MapOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
