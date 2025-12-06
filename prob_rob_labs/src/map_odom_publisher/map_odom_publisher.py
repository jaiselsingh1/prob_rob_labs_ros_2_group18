#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

class MapOdomPublisher(Node):
    def __init__(self):
        super().__init__('map_odom_publisher')

        # --- Parameters ---
        self.map_frame = 'map'
        self.odom_frame = 'odom'
        self.base_frame = 'base_footprint' 

        # --- TF Buffer ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Multi-threading Setup
        self.cb_group = ReentrantCallbackGroup()

        # --- Shared State (The "Global Variable") ---
        # Initialize with Identity transform to ensure we can publish immediately
        self.latest_transform = TransformStamped()
        self.latest_transform.header.frame_id = self.map_frame
        self.latest_transform.child_frame_id = self.odom_frame
        self.latest_transform.transform.rotation.w = 1.0
        
        # This lock is optional in Python due to GIL but good practice
        self._data_lock = False 

        # --- Subscriber (Heavy Calculation Logic) ---
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/ekf_pose',
            self._ekf_callback,  # This callback does the heavy lifting
            10,
            callback_group=self.cb_group
        )

        # --- Timer (Lightweight Publishing Logic) --- 
        # This purely broadcasts the *last known good* transform
        self.create_timer(
            1.0 / 30.0, 
            self._publish_transform, 
            callback_group=self.cb_group
        )

        self.get_logger().info("Map->Odom Publisher Started. Broadcasting at 30Hz (Decoupled Mode).")

    def _ekf_callback(self, msg: PoseWithCovarianceStamped):
        """
        Heavy Calculation: Updates the internal state, but DOES NOT publish.
        Run frequency: Depends on EKF frequency (variable).
        """
        # 1. Get map -> base
        x_ekf = msg.pose.pose.position.x
        y_ekf = msg.pose.pose.position.y
        q_ekf = msg.pose.pose.orientation
        yaw_ekf = self._quat_to_yaw(q_ekf)

        # 2. Get odom -> base (Heavy TF lookup)
        try:
            trans = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                rclpy.time.Time()) 
        except Exception:
            # If TF fails, we just don't update the transform this time.
            # The timer will keep publishing the OLD transform, satisfying the 30Hz rule.
            return

        x_odom = trans.transform.translation.x
        y_odom = trans.transform.translation.y
        q_odom = trans.transform.rotation
        yaw_odom = self._quat_to_yaw(q_odom)

        # 3. Calculate map -> odom
        T_mb = self._get_matrix(x_ekf, y_ekf, yaw_ekf)
        T_ob = self._get_matrix(x_odom, y_odom, yaw_odom)
        T_ob_inv = np.linalg.inv(T_ob)
        T_mo = np.dot(T_mb, T_ob_inv)

        dx = T_mo[0, 2]
        dy = T_mo[1, 2]
        dyaw = math.atan2(T_mo[1, 0], T_mo[0, 0])
        q = self._yaw_to_quat(dyaw)

        # 4. Update the shared state (Atomic assignment is safe enough here)
        # We only update the values, we don't send anything yet.
        self.latest_transform.transform.translation.x = dx
        self.latest_transform.transform.translation.y = dy
        self.latest_transform.transform.translation.z = 0.0
        self.latest_transform.transform.rotation.x = q[0]
        self.latest_transform.transform.rotation.y = q[1]
        self.latest_transform.transform.rotation.z = q[2]
        self.latest_transform.transform.rotation.w = q[3]

    def _publish_transform(self):
        """
        Lightweight Broadcast: Just stamps and sends.
        Run frequency: Strictly 30Hz.
        """
        # Update timestamp to NOW (crucial for TF tree continuity)
        self.latest_transform.header.stamp = self.get_clock().now().to_msg()
        
        # Send the latest calculated transform (even if it's old, it's valid)
        self.tf_broadcaster.sendTransform(self.latest_transform)

    # --- Helpers ---
    def _get_matrix(self, x, y, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    def _quat_to_yaw(self, q):
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
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
