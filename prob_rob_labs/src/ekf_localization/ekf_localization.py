#!/usr/bin/env python3
import os
import math
import yaml
import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

try:
    from landmark_positioner.variance_model import make_variance_model
except ImportError:
    try:
        from prob_rob_labs.landmark_positioner.variance_model import make_variance_model
    except ImportError:
        def make_variance_model(var_d, var_th):
            return (lambda d: var_d, lambda th: var_th, lambda d, th: np.diag([var_d, var_th]))

@dataclass
class Landmark:
    color: str
    x: float
    y: float
    radius: float
    height: float

class LandmarkEKFNode(Node):
    def __init__(self):
        super().__init__('ekf_localization')

        self._declare_params()
        
        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.get_logger().info(f"Using {'Simulation' if use_sim_time else 'Wall'} Time.")

        map_path = self._get_map_path()
        self.landmarks = self._load_landmark_map(map_path)
        self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks for multi-fusion.")

        self.mu = np.zeros(3, dtype=float) 
        self.Sigma = np.eye(3, dtype=float) * 1e-3

        self.alpha_pos = 0.01
        self.alpha_yaw = (5.0 * math.pi / 180.0) ** 2
        self.Q = np.diag([self.alpha_pos, self.alpha_pos, self.alpha_yaw])

        base_var_d = 1.761867e-02
        base_var_theta = 1.573656e-04
        self.sigma_d2, self.sigma_theta2, self.R_func = make_variance_model(base_var_d, base_var_theta)

        self.state_time = None
        self.initialized = False
        self.last_v = 0.0
        self.last_omega = 0.0

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_callback, 50)
        
        self.vision_subs = []
        for color, lm in self.landmarks.items():
            topic = f"/vision_{color}/measurement"
            sub = self.create_subscription(
                PointStamped, topic,
                lambda msg, c=color, l=lm: self._vision_callback(msg, c, l),
                10
            )
            self.vision_subs.append(sub)
            self.get_logger().info(f"Subscribed to {topic}")

    def _declare_params(self):
        params = {'use_sim_time': True, 'map_file': ''}
        for name, default in params.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

    def _get_map_path(self):
        path = self.get_parameter('map_file').get_parameter_value().string_value
        if not path:
            try:
                share_dir = get_package_share_directory('prob_rob_labs')
                path = os.path.join(share_dir, 'maps', 'landmarks_lab6.yaml')
            except Exception:
                pass
        return path

    def _load_landmark_map(self, path):
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            lms = {}
            entries = data.get('landmarks', []) if isinstance(data, dict) else data
            for item in entries:
                c = item['color']
                lms[c] = Landmark(c, float(item['x']), float(item['y']), 
                                  float(item.get('radius', 0.1)), float(item.get('height', 0.5)))
            return lms
        except Exception:
            return {}

    @staticmethod
    def _stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _norm_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _publish_pose(self, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.mu[0]
        msg.pose.pose.position.y = self.mu[1]
        
        th = self.mu[2]
        msg.pose.pose.orientation.z = math.sin(th * 0.5)
        msg.pose.pose.orientation.w = math.cos(th * 0.5)

        P = np.zeros((6, 6))
        P[0,0] = self.Sigma[0,0]; P[0,1] = self.Sigma[0,1]; P[0,5] = self.Sigma[0,2]
        P[1,0] = self.Sigma[1,0]; P[1,1] = self.Sigma[1,1]; P[1,5] = self.Sigma[1,2]
        P[5,0] = self.Sigma[2,0]; P[5,1] = self.Sigma[2,1]; P[5,5] = self.Sigma[2,2]
        msg.pose.covariance = P.flatten().tolist()

        self.pose_pub.publish(msg)

    def _predict(self, dt, v, omega):
        if dt <= 0.0: return
        x, y, th = self.mu
        
        if abs(omega) < 1e-5:
            x_pred = x + v * dt * math.cos(th)
            y_pred = y + v * dt * math.sin(th)
            th_pred = th
        else:
            ratio = v / omega
            x_pred = x + ratio * (math.sin(th + omega*dt) - math.sin(th))
            y_pred = y + ratio * (math.cos(th) - math.cos(th + omega*dt))
            th_pred = th + omega * dt

        self.mu = np.array([x_pred, y_pred, self._norm_angle(th_pred)])

        F = np.eye(3)
        F[0, 2] = -v * math.sin(th) * dt
        F[1, 2] =  v * math.cos(th) * dt
        Q_scaled = self.Q * dt
        self.Sigma = F @ self.Sigma @ F.T + Q_scaled

    def _update(self, d_meas, th_meas, lm, stamp):
        x, y, th = self.mu
        dx = lm.x - x
        dy = lm.y - y
        q = dx**2 + dy**2
        
        # Avoid Singularity
        if q < 1e-4: return

        d_hat = math.sqrt(q)
        th_hat = self._norm_angle(math.atan2(dy, dx) - th)

        y_res = np.array([d_meas - d_hat, self._norm_angle(th_meas - th_hat)])
        H = np.array([[-dx/d_hat, -dy/d_hat, 0.0], [ dy/q, -dx/q, -1.0]])

        try:
            R = self.R_func(d_meas, th_meas)
            S = H @ self.Sigma @ H.T + R
            K = self.Sigma @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.mu = self.mu + K @ y_res
        self.mu[2] = self._norm_angle(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

        self.get_logger().info(f"Update {lm.color}: mu=[{self.mu[0]:.2f}, {self.mu[1]:.2f}, {self.mu[2]:.2f}]")
        self._publish_pose(stamp)

    def _odom_callback(self, msg):
        t_now = self._stamp_to_sec(msg.header.stamp)
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.last_v = v
        self.last_omega = w

        if not self.initialized:
            return

        dt = t_now - self.state_time
        if dt > 0.0:
            self._predict(dt, v, w)
            self.state_time = t_now
            self._publish_pose(msg.header.stamp)
            self.get_logger().info(f"Predicting (v={v:.2f}, w={w:.2f})...", throttle_duration_sec=2.0)

    def _vision_callback(self, msg, color, lm):
        t_now = self._stamp_to_sec(msg.header.stamp)

        if not self.initialized:
            self.state_time = t_now
            self.initialized = True
            self.get_logger().info(f"System initialized at time {t_now:.2f}")
            return

        dt = t_now - self.state_time
        
        # Time tolerance for late packets
        if dt > 0.0:
            self._predict(dt, self.last_v, self.last_omega)
            self.state_time = t_now
        elif dt < 0.0:
            if dt > -3.0: 
                pass 
            else:
                return

        # Update with any landmark
        self._update(msg.point.x, msg.point.y, lm, msg.header.stamp)

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