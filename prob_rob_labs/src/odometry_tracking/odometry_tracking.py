import threading
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer



class OdometryTracking(Node):

    def __init__(self):
        super().__init__('odometry_tracking')
        
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        # convenient flag to enable debug logs from CLI or env
        self.declare_parameter('debug', False)
        self.declare_parameter('sync_queue', 20)
        self.declare_parameter('sync_slop', 0.05)

        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        joint_topic = self.get_parameter('joint_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        sync_queue = self.get_parameter('sync_queue').get_parameter_value().integer_value
        sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value
        debug_flag = bool(self.get_parameter('debug').get_parameter_value().bool_value)
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
        
        self._cmd_mutex = threading.Lock()
        self._last_cmd = Twist()     # 默认 0
        self._last_cmd_stamp = None  # builtin_interfaces/Time
        self.create_subscription(Twist, cmd_topic, self._on_cmd_vel, 10)      
          
        # message_filters.Subscriber forwards positional args into create_subscription
        # which can collide with the callback parameter; pass qos_profile as a
        # keyword to avoid 'multiple values for argument qos_profile'.
        self.imu_sub   = Subscriber(self, Imu, imu_topic)
        self.joint_sub = Subscriber(self, JointState, joint_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.imu_sub, self.joint_sub], 
            queue_size=sync_queue, 
            slop=sync_slop
        )
        self.sync.registerCallback(self._on_synced_measurements)
        
        self._last_sync_stamp = None  # builtin_interfaces/Time
        self.get_logger().info(
            f'EKF input ready. imu="{imu_topic}", joints="{joint_topic}", '
            f'cmd="{cmd_topic}", slop={sync_slop}s, queue={sync_queue}'
        )        
        
        
    def _on_cmd_vel(self, msg: Twist):
        with self._cmd_mutex:
            self._last_cmd = msg
            # 用“节点当前时间”做记录即可；真正滤波用传感器时间戳为准
            self._last_cmd_stamp = self.get_clock().now().to_msg()
            
    def _on_synced_measurements(self, imu_msg: Imu, joint_msg: JointState):
        t_now = Time.from_msg(imu_msg.header.stamp)
        
        dt = None
        if self._last_sync_stamp is None:
            dt = 0.0
        else:
            dt = (t_now - self._last_sync_stamp).nanoseconds * 1e-9
        self._last_sync_stamp = t_now

        # 取“最近一次”cmd_vel（粘性）
        with self._cmd_mutex:
            cmd = self._last_cmd
            
            
        wheel_vel_left = None
        wheel_vel_right = None
        try:
            name_to_idx = {n: i for i, n in enumerate(joint_msg.name)}
            if 'wheel_left_joint' in name_to_idx:
                i = name_to_idx['wheel_left_joint']
                wheel_vel_left = joint_msg.velocity[i]
            if 'wheel_right_joint' in name_to_idx:
                i = name_to_idx['wheel_right_joint']
                wheel_vel_right = joint_msg.velocity[i]
        except Exception as e:
            self.get_logger().warn(f'JointState parsing error: {e}')
            
        imu_gyro_z = imu_msg.angular_velocity.z
        
        # Debug Output
        self.get_logger().debug(
            f'dt={dt:.4f}s | gyro_z={imu_gyro_z:.3f} | '
            f'wl={wheel_vel_left} wr={wheel_vel_right} | '
            f'cmd: v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}'
        )
            
 
        # ============ 交给你的 EKF：predict / update ============
        # TODO: 在这里调用你自己的函数：
        # self.ekf_predict(cmd, dt)
        # self.ekf_update(imu_msg, joint_msg)
        # 然后 publish /ekf_odom（注意使用 imu/joint 的时间戳，而非 now()）
        
                   
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
