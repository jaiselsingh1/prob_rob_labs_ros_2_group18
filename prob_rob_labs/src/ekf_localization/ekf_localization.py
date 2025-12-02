#!/usr/bin/env python3
import os
import math
import yaml
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry


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

        # --- EKF landmark color ---
        if not self.has_parameter('ekf_landmark_color'):
            self.declare_parameter('ekf_landmark_color', 'cyan')

        self.ekf_landmark_color = (
            self.get_parameter('ekf_landmark_color')
            .get_parameter_value()
            .string_value
        )
        self.get_logger().info(
            f"EKF will use landmark color='{self.ekf_landmark_color}' for updates"
        )

        # --- EKF state and covariance ---
        self.mu = np.zeros(3, dtype=float)
        self.Sigma = np.eye(3, dtype=float) * 1e-3

        # Motion noise Q
        pos_noise = 0.01
        yaw_noise = (5.0 * math.pi / 180.0) ** 2
        self.Q = np.diag([pos_noise, pos_noise, yaw_noise])

        # Measurement noise R (start with constant noise from Lab 5 variance_model.py)
        base_var_d = 1.761867e-02
        base_var_theta = 1.573656e-04
        self.R_const = np.diag([base_var_d, base_var_theta])

        # Odometry bookkeeping
        self.have_odom = False
        self.last_odom_pose = None  # (x, y, yaw)

        # ---  Assignment 3: subscribe to odometry for EKF prediction ---
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            50,
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


    @staticmethod
    def _angle_normalize(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        a = math.fmod(angle + math.pi, 2.0 * math.pi)
        if a < 0.0:
            a += 2.0 * math.pi
        return a - math.pi

    def _angle_diff(self, a: float, b: float) -> float:
        """Compute a - b and wrap to [-pi, pi]."""
        return self._angle_normalize(a - b)

    @staticmethod
    def _yaw_from_quat(q) -> float:
        """Extract yaw from a geometry_msgs/Quaternion."""
        x = q.x
        y = q.y
        z = q.z
        w = q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw


    # Odometry callback: EKF prediction step
    def _odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        yaw = self._yaw_from_quat(msg.pose.pose.orientation)

        if not self.have_odom:
            # First odom message: initialize EKF state
            self.mu[:] = [px, py, yaw]
            self.last_odom_pose = (px, py, yaw)
            self.have_odom = True
            self.get_logger().info(
                f"Initialized EKF state from odom: "
                f"x={px:.2f}, y={py:.2f}, yaw={yaw:.2f} rad"
            )
            return

        prev_x, prev_y, prev_yaw = self.last_odom_pose

        dx = px - prev_x
        dy = py - prev_y
        dyaw = self._angle_diff(yaw, prev_yaw)

        u = np.array([dx, dy, dyaw], dtype=float)

        # EKF prediction
        self.mu = self.mu + u
        self.mu[2] = self._angle_normalize(self.mu[2])


        self.Sigma = self.Sigma + self.Q

        self.last_odom_pose = (px, py, yaw)


    # Vision measurement callback: store measurement and trigger EKF update
    def _vision_measurement_callback(self, msg: PointStamped, color: str, lm: Landmark):
        """
        - msg.point.x: distance (d)
        - msg.point.y: bearing (theta)
        - color: landmark color
        - lm: landmark in world frame (lm.x, lm.y)
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

        # If EKF state has not been initialized from odom, skip update
        if not self.have_odom:
            return

        # In Assignment 3, only use one landmark color for EKF update
        if color != self.ekf_landmark_color:
            return

        self._ekf_update_single_landmark(color, d, theta, lm)


    # EKF measurement update for a single landmark
    def _ekf_update_single_landmark(
        self,
        color: str,
        d_meas: float,
        theta_meas: float,
        lm: Landmark,
    ):
        """
        EKF measurement update using a single landmark.

        State: mu = [x, y, theta]^T
        Landmark: (lm.x, lm.y) in world frame
        Measurement:
          z = [d_meas, theta_meas]^T
        Measurement model:
          d_hat     = sqrt((x_l - x)^2 + (y_l - y)^2)
          theta_hat = atan2(y_l - y, x_l - x) - theta
        """

        x, y, yaw = self.mu

        dx = lm.x - x
        dy = lm.y - y
        q = dx * dx + dy * dy

        if q < 1e-6:
            # Too close, avoid division by zero
            self.get_logger().warn(
                f"Landmark {color} is extremely close to the robot; skipping update."
            )
            return

        d_hat = math.sqrt(q)
        theta_hat = self._angle_normalize(math.atan2(dy, dx) - yaw)

        # Measurement vector and prediction
        z = np.array([d_meas, theta_meas], dtype=float)
        z_hat = np.array([d_hat, theta_hat], dtype=float)

        # Residual (wrap bearing residual)
        y_res = z - z_hat
        y_res[1] = self._angle_normalize(y_res[1])

        # Measurement Jacobian H
        d = d_hat
        H = np.array([
            [-dx / d,      -dy / d,        0.0],
            [ dy / q,      -dx / q,       -1.0],
        ])

        # EKF update
        S = H @ self.Sigma @ H.T + self.R_const
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ y_res
        self.mu[2] = self._angle_normalize(self.mu[2])

        I = np.eye(3)
        self.Sigma = (I - K @ H) @ self.Sigma

        self.get_logger().info(
            f"[EKF update] color={color} -> "
            f"mu = [x={self.mu[0]:.2f}, y={self.mu[1]:.2f}, yaw={self.mu[2]:.2f} rad]"
        )


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
