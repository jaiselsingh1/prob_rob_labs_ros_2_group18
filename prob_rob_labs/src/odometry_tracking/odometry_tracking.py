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
        self.declare_parameter('init_yaw', 0.0)
        self.declare_parameter('init_P_diag', [0.1, 0.1, 0.1, 0.1, 0.1]) #initial covariance diagonal
        #process noise Q (diagonal)
        self.declare_parameter('Q_diag', [0.1, 0.1, 0.1, 0.1, 0.1]) #[theta, x, y, v, yaw_rate]
        #measurement noise R (diagonal)
        self.declare_parameter('R_diag', [0.1, 0.1, 0.1]) #[wheel_left, wheel_right, gyro_z]
        #model parameters
        self.declare_parameter('model_tau_v',1.8)
        self.declare_parameter('model_tau_yaw',0.45)   
        self.declare_parameter('model_G_v',0.1)  
        self.declare_parameter('model_G_yaw',0.03)   
        # robot parameters
        self.declare_parameter('wheel_radius', 33)  # mm
        self.declare_parameter('wheel_separation', 143.5)  # mm
        #debug flag
        self.declare_parameter('debug', False)
        
        params = {p.name: p.value for p in self.get_parameters([
            'imu_topic', 'joint_topic', 'cmd_topic', 'odom_topic',
            'sync_queue', 'sync_slop',
            'odom_frame', 'base_frame',
            'init_theta', 'init_x', 'init_y', 'init_v', 'init_yaw', 'init_P_diag',
            'Q_diag', 'R_diag',
            'model_tau_v', 'model_tau_yaw', 'model_G_v', 'model_G_yaw',
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
            float(params['init_yaw'])
        ])  # state vector: [x, y, theta, v, yaw_rate]
        
        self.P = np.diag(np.array(params['init_P_diag'], dtype=float).reshape(5))  
        self.Q_base = np.diag(np.array(params['Q_diag'], dtype=float).reshape(5))       
        self.R_full = np.diag(np.array(params['R_diag'], dtype=float).reshape(3))      
        
        self.model_tau_v = float(params['model_tau_v'])
        self.model_tau_yaw = float(params['model_tau_yaw'])
        self.model_G_v = float(params['model_G_v'])
        self.model_G_yaw = float(params['model_G_yaw'])
        
        self.r_w = float(params['wheel_radius']) * 1e-3  # convert mm to meters
        self.R_half = float(params['wheel_separation']) * 1e-3 / 2.0 # convert mm to meters
        
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
        #use imu_msg.header.stamp as time reference
        t_now = Time.from_msg(imu_msg.header.stamp)
        
        # calculate dt since last synchronized measurement
        dt = None
        if self._last_sync_stamp is None:
            dt = 0.0
        else:
            dt = (t_now - self._last_sync_stamp).nanoseconds * 1e-9
        self._last_sync_stamp = t_now

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
        self._publish_odom(imu_msg.header.stamp,imu_gyro_z)
        
    
    # =========== EKF Predict ============
    def ekf_predict(self, cmd: Twist, dt: float):
        
        theta, px, py, v, w = self.x.flatten()
        u_v = float(cmd.linear.x)
        u_w = float(cmd.angular.z)
        
        # a_v, a_w per handout:  a = 0.1^(dt/tau)
        a_v = pow(0.1, dt / max(self.tau_v, 1e-6))
        a_w = pow(0.1, dt / max(self.tau_w, 1e-6))

        # ---- f(x,u) ----
        theta_n = theta + w * dt
        px_n    = px + v * dt * math.cos(theta)
        py_n    = py + v * dt * math.sin(theta)
        v_n     = a_v * v + self.G_v * (1.0 - a_v) * u_v
        w_n = a_w * w + self.G_w * (1.0 - a_w) * u_w

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
            rows.append([0.0, 0.0, 0.0, 1.0 / self.r_w, -self.R_half / self.r_w])
            z_list.append(wl)
            r_list.append(self.R_full[0, 0])
        
        # right wheel
        if wr is not None:
            rows.append([0.0, 0.0, 0.0, 1.0 / self.r_w, self.R_half / self.r_w])
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
    def _publish_odom(self, stamp, imu_gyro_z: float):
        pass
    
    
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
