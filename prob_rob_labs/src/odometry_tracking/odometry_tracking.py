import threading
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np


def yaw_to_quaternion(yaw:float):
    # Convert a yaw angle (in radians) into a quaternion representation
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)

class OdometryTracking(Node):

    def __init__(self):
        super().__init__('odometry_tracking')
        
        #----------------parameters-----------------
        #topics
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/ekf_odom')
        # sync parameters
        self.declare_parameter('sync_queue', 20)
        self.declare_parameter('sync_slop', 0.05)
        #TF frame
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        #initial state/covariance
        self.declare_parameter('init_theta', 0.0)
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_v', 0.0)
        self.declare_parameter('init_w', 0.0)
        self.declare_parameter('init_P_diag', [1e-4, 1e-4, 1e-4, 1e-3, 1e-3]) #initial covariance diagonal
        #process noise Q (diagonal)
        self.declare_parameter('Q_diag', [0.5, 0.5, 0.5, 0.5, 0.5]) #[theta, x, y, v, yaw_rate]
        #measurement noise R (diagonal)
        self.declare_parameter('R_diag', [0.1, 0.1, 0.001]) #[wheel_left, wheel_right, gyro_z]
        #model parameters
        self.declare_parameter('model_tau_v',1.8)
        self.declare_parameter('model_tau_w',0.45)   
        self.declare_parameter('model_G_v',1.0)  
        self.declare_parameter('model_G_w',1.0)   
        # robot parameters
        self.declare_parameter('wheel_radius', 0.033)  # m
        self.declare_parameter('wheel_separation', 0.1435)  # m
        #debug flag
        self.declare_parameter('debug', False)
        
        params = {p.name: p.value for p in self.get_parameters([
            'imu_topic', 'joint_topic', 'cmd_topic', 'odom_topic',
            'sync_queue', 'sync_slop',
            'odom_frame', 'base_frame',
            'init_theta', 'init_x', 'init_y', 'init_v', 'init_w', 'init_P_diag',
            'Q_diag', 'R_diag',
            'model_tau_v', 'model_tau_w', 'model_G_v', 'model_G_w',
            'wheel_radius', 'wheel_separation',
            'debug'
        ])}
        
        imu_topic = params['imu_topic']
        joint_topic = params['joint_topic']
        cmd_topic = params['cmd_topic']
        self.odom_topic = params['odom_topic']
            
        self.odom_frame = params['odom_frame']
        self.base_frame = params['base_frame']
        
        sync_queue = int(params['sync_queue'])
        sync_slop = float(params['sync_slop'])
        
        
        #-----------EKF state initialization-------------
        self.x = np.array([
            float(params['init_theta']),
            float(params['init_x']),
            float(params['init_y']),
            float(params['init_v']),
            float(params['init_w'])
        ])  # state vector: [x, y, theta, v, yaw_rate]
        
        self.P = np.diag(np.array(params['init_P_diag'], dtype=float).reshape(5))  
        self.Q_base = np.diag(np.array(params['Q_diag'], dtype=float).reshape(5))       
        self.R_full = np.diag(np.array(params['R_diag'], dtype=float).reshape(3))      
        
        self.model_tau_v = float(params['model_tau_v'])
        self.model_tau_w = float(params['model_tau_w'])
        self.model_G_v = float(params['model_G_v'])
        self.model_G_w = float(params['model_G_w'])
        
        self.r_w = float(params['wheel_radius']) 
        self.R = float(params['wheel_separation'])
        
        #-----------subscriptions and synchronization-------------
        self._cmd_mutex = threading.Lock()
        self._last_cmd = Twist()     # default zero cmd
        self._last_cmd_stamp = None  # builtin_interfaces/Time
        
        self.create_subscription(Twist, cmd_topic, self._on_cmd_vel, 10)      
        
        # message_filters subscribers + approximate time synchronizer
        self.imu_sub   = Subscriber(self, Imu, imu_topic)
        self.joint_sub = Subscriber(self, JointState, joint_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.imu_sub, self.joint_sub], 
            queue_size=sync_queue, 
            slop=sync_slop
        )
        self.sync.registerCallback(self._on_synced_measurements)
        
        # publisher for odometry / EKF output
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        
        self._last_sync_stamp = None  # builtin_interfaces/Time
        
        self.get_logger().info(
            f'EKF input ready. imu="{imu_topic}", joints="{joint_topic}", '
            f'cmd="{cmd_topic}", slop={sync_slop}s, queue={sync_queue}'
            f'wheel_radius={self.r_w}m, wheel_separation={self.R}m'
        )     
        
        # Debug logging   
        debug_flag = bool(params['debug'])
        # allow environment variable to enable debug when CLI flags are problematic
        if not debug_flag and os.environ.get('PROB_ROB_DEBUG', '0') == '1':
            debug_flag = True
        if debug_flag:
            # set this node's logger to DEBUG for easier debugging
            try:
                self.get_logger().set_level(LoggingSeverity.DEBUG)
                self.get_logger().info('Debug logging enabled for odometry_tracking')
            except Exception:
                # best-effort: continue even if setting level fails
                pass
        
        
    # ============ 回调函数 ============
    def _on_cmd_vel(self, msg: Twist):
        with self._cmd_mutex:
            self._last_cmd = msg
            # 用“节点当前时间”做记录即可；真正滤波用传感器时间戳为准
            self._last_cmd_stamp = self.get_clock().now().to_msg()
            
            
    #=========== 同步 imu + joint 回调函数 ============
    def _on_synced_measurements(self, imu_msg: Imu, joint_msg: JointState):
        # use imu_msg.header.stamp as time reference
        hdr_stamp = imu_msg.header.stamp
        # if the incoming stamp is all zeros, fall back to node clock
        if hdr_stamp.sec == 0 and hdr_stamp.nanosec == 0:
            t_now = self.get_clock().now()
        else:
            t_now = Time.from_msg(hdr_stamp)

        # calculate dt since last synchronized measurement (safety checks)
        if self._last_sync_stamp is None:
            dt = 0.0
        else:
            delta = t_now - self._last_sync_stamp
            # Duration.nanoseconds is an int (may be negative if clocks jump)
            dt = float(delta.nanoseconds) * 1e-9
            # guard against negative or absurd dt values
            if dt < 0.0:
                self.get_logger().warning(f'Negative dt computed ({dt:.6f}s); clamping to 0')
                dt = 0.0
            elif dt > 5.0:
                # very large gap — clamp and warn
                self.get_logger().warning(f'Large dt computed ({dt:.3f}s); clamping to 5.0s')
                dt = 5.0
        self._last_sync_stamp = t_now

        # debug: show stamps if debug logging enabled
        if self.get_logger().get_effective_level() <= LoggingSeverity.DEBUG:
            try:
                self.get_logger().debug(f'imu_stamp={hdr_stamp.sec}.{hdr_stamp.nanosec:09d} last_stamp={(self._last_sync_stamp.seconds_nanoseconds())}')
            except Exception:
                # ignore any formatting errors
                pass

        # get last cmd_vel
        with self._cmd_mutex:
            cmd = self._last_cmd
            
        # extract measurements
        imu_gyro_z = imu_msg.angular_velocity.z
        wl = None
        wr = None
        try:
            name_to_idx = {n: i for i, n in enumerate(joint_msg.name)}
            if 'wheel_left_joint' in name_to_idx:
                i = name_to_idx['wheel_left_joint']
                wl = joint_msg.velocity[i]
            if 'wheel_right_joint' in name_to_idx:
                i = name_to_idx['wheel_right_joint']
                wr = joint_msg.velocity[i]
        except Exception as e:
            self.get_logger().warn(f'JointState parsing error: {e}')
        
        # Debug Output
        self.get_logger().debug(
            f'dt={dt:.4f}s | gyro_z={imu_gyro_z:.3f} | '
            f'wl={wl} wr={wr} | '
            f'cmd: v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}'
        )
            
 
        # ----------- EKF pridict step -----------
        self.ekf_predict(cmd, dt)
        
        # ----------- EKF update step -----------
        self.ekf_update(imu_gyro_z, wl, wr)
        
        # ---------- Publish EKF odometry -----------
        self._publish_odom(imu_msg.header.stamp)
        
    
    # =========== EKF Predict ============
    def ekf_predict(self, cmd: Twist, dt: float):
        
        theta, px, py, v, w = self.x.flatten()
        u_v = float(cmd.linear.x)
        u_w = float(cmd.angular.z)
        
        # a_v, a_w per handout:  a = 0.1^(dt/tau)
        a_v = pow(0.1, dt / max(self.model_tau_v, 1e-6))
        a_w = pow(0.1, dt / max(self.model_tau_w, 1e-6))

        # ---- f(x,u) ----
        theta_n = theta + w * dt
        px_n    = px + v * dt * math.cos(theta)
        py_n    = py + v * dt * math.sin(theta)
        v_n     = a_v * v + self.model_G_v * (1.0 - a_v) * u_v
        w_n     = a_w * w + self.model_G_w * (1.0 - a_w) * u_w

        x_pred = np.array([theta_n, px_n, py_n, v_n, w_n], dtype=float).reshape(5, 1)
        
        # ---- Jacobian F ----
        J = np.eye(5)
        J[0, 4] = dt                         # dtheta_n/dw
        J[1, 0] = -v * dt * math.sin(theta)  # dpx_n/dtheta
        J[1, 1] = 1.0                        # dpx_n/dpx
        J[1, 3] = dt * math.cos(theta)       # dpx_n/dv
        J[2, 0] = v * dt * math.cos(theta)   # dpy_n/dtheta
        J[2, 2] = 1.0                        # dpy_n/dpy
        J[2, 3] = dt * math.sin(theta)       # dpy_n/dv
        J[3, 3] = a_v                        # dv_n/dv
        J[4, 4] = a_w                        # dw_n/dw
        
        # ---- B ----
        # B =
        # [0, 0;
        #  0, 0;
        #  0, 0;
        #  G_v*(1-a_v), 0;
        #  0, G_w*(1-a_w) ]
        
        # Process noise
        Q = self._Q(dt)

        # Propagate
        self.x = x_pred
        self.P = J @ self.P @ J.T + Q
        
    
    # =========== EKF Update ============
    def ekf_update(self, imu_gyro_z: float, wl: float, wr: float):
        rows = []
        z_list = []
        r_list = []
        
        # left wheel
        if wl is not None:
            rows.append([0.0, 0.0, 0.0, 1.0 / self.r_w, -self.R / self.r_w])
            z_list.append(wl)
            r_list.append(self.R_full[0, 0])
        
        # right wheel
        if wr is not None:
            rows.append([0.0, 0.0, 0.0, 1.0 / self.r_w, self.R / self.r_w])
            z_list.append(wr)
            r_list.append(self.R_full[1, 1])
            
        # imu gyro z
        if imu_gyro_z is not None:
            rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
            z_list.append(imu_gyro_z)
            r_list.append(self.R_full[2, 2])
        
        if not rows:
            # no measurements available
            return
        
        C = np.array(rows, dtype=float)                        # (m x 5)
        z = np.array(z_list, dtype=float).reshape(-1, 1)       # (m x 1)
        R = np.diag(np.array(r_list, dtype=float).reshape(-1)) # (m x m)
        
        # measurement prediction
        z_hat = C @ self.x         
        
        # innovation
        y = z - z_hat             
        
        # innovation covariance
        S = C @ self.P @ C.T + R   
        
        # Kalman gain
        K = self.P @ C.T @ np.linalg.inv(S)
        
        # state update
        self.x = self.x + K @ y
        
        I = np.eye(5)
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ R @ K.T
        
        # normalize theta to [-pi, pi]
        self.x[0, 0] = math.atan2(math.sin(self.x[0, 0]), math.cos(self.x[0, 0]))
    
    
    # =========== Publish Odometry ============
    def _publish_odom(self, stamp):
        theta, px, py, v, w = self.x.flatten()
        qx, qy, qz, qw = yaw_to_quaternion(theta)
        
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        
        # pose
        msg.pose.pose.position.x = float(px)
        msg.pose.pose.position.y = float(py)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        
        # twist: body frame
        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(w)
        
        #covariance mapping
        # initialize covariances large (1e3) and then map EKF P into relevant entries
        pose_cov = np.full((6, 6), 1e3, dtype=float)
        twist_cov = np.full((6, 6), 1e3, dtype=float)
        # map P -> pose(x,y,theta)
        pose_cov[0, 0] = self.P[1, 1]  # x
        pose_cov[1, 1] = self.P[2, 2]  # y
        pose_cov[5, 5] = self.P[0, 0]  # theta
        # map P -> twist(v, yaw_rate)
        twist_cov[0, 0] = self.P[3, 3]  # v
        twist_cov[5, 5] = self.P[4, 4]  # yaw_rate
        
        msg.pose.covariance = pose_cov.flatten().tolist()
        msg.twist.covariance = twist_cov.flatten().tolist()
        
        self.odom_pub.publish(msg)
    
    
    def _Q(self, dt: float) -> np.ndarray:
        """Process noise scaled by dt (simple)."""
        return self.Q_base * max(dt, 1e-3)
    
                   
    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    odometry_tracking = OdometryTracking()
    odometry_tracking.spin()
    odometry_tracking.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
