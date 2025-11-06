#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from prob_rob_msgs.msg import Point2DArrayStamped
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import LinkStates

HEARTBEAT_PERIOD = 0.5
NO_MEAS_TIMEOUT = 1.0
MIN_HEIGHT_PX = 4.0


class LandmarkPositioner(Node):
    """
    subscribes:
      /vision_<color>/corners      (Point2DArrayStamped)
      /camera/camera_info          (CameraInfo)
      /gazebo/link_states          (LinkStates)

    publishes:
      /vision_<color>/measurement  (PointStamped: x=range [m], y=bearing [rad], z=0)
      /vision_<color>/error        (PointStamped: x=dr [m], y=dtheta [rad], z=0)
    """

    COLOR_TO_LINK = {
        'red': 'landmark_1::link',
        'green': 'landmark_2::link',
        'yellow': 'landmark_3::link',
        'magenta': 'landmark_4::link',
        'cyan': 'landmark_5::link',
    }

    CAMERA_LINK = 'waffle_pi::camera_link'

    def __init__(self):
        super().__init__('landmark_positioner')
        self.log = self.get_logger()

        self.declare_parameter('landmark_color', 'cyan')
        self.declare_parameter('landmark_height', 0.5)

        self.landmark_color = (
            self.get_parameter('landmark_color')
            .get_parameter_value().string_value
        )
        self.landmark_height = float(
            self.get_parameter('landmark_height')
            .get_parameter_value().double_value
        )

        self.landmark_link_name = self.COLOR_TO_LINK.get(self.landmark_color)
        if self.landmark_link_name is None:
            self.log.error(f"unknown landmark color: {self.landmark_color}")
            raise RuntimeError("invalid landmark_color parameter")

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_link_states = None
        self.last_meas_time = None

        self.timer = self.create_timer(HEARTBEAT_PERIOD, self.heartbeat_cb)

        self.create_subscription(
            Point2DArrayStamped,
            f'/vision_{self.landmark_color}/corners',
            self.landmark_corners_cb,
            10,
        )
        self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_cb,
            10,
        )
        self.create_subscription(
            LinkStates,
            '/gazebo/link_states',
            self.link_states_cb,
            10,
        )

        meas_topic = f'/vision_{self.landmark_color}/measurement'
        self.meas_pub = self.create_publisher(PointStamped, meas_topic, 10)

        err_topic = f'/vision_{self.landmark_color}/error'
        self.err_pub = self.create_publisher(PointStamped, err_topic, 10)

        self.log.info(
            f'landmark_positioner for color={self.landmark_color} initialized'
        )

    def heartbeat_cb(self):
        if self.last_meas_time is None:
            return
        now = self.get_clock().now()
        dt = (now - self.last_meas_time).nanoseconds * 1e-9
        if dt > NO_MEAS_TIMEOUT:
            self.log.info(
                f"no valid measurement for {dt:.1f}s for color={self.landmark_color}"
            )

    def camera_info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def link_states_cb(self, msg: LinkStates):
        self.latest_link_states = msg

    def landmark_corners_cb(self, msg: Point2DArrayStamped):
        if None in (self.fx, self.fy, self.cx, self.cy):
            self.log.warning("no camera intrinsics yet, skipping measurement")
            return

        points = msg.points
        num_points = len(points)
        if num_points == 0:
            return

        xs = [p.x for p in points]
        ys = [p.y for p in points]

        min_y = min(ys)
        max_y = max(ys)
        height_px = max_y - min_y

        if height_px <= 0.0:
            self.log.warning("non positive landmark height, skipping")
            return
        if height_px < MIN_HEIGHT_PX:
            return

        sorted_idx = sorted(range(num_points), key=lambda i: ys[i])
        k = min(2, num_points)
        top_idx = sorted_idx[:k]
        bottom_idx = sorted_idx[-k:]

        x_top = sum(xs[i] for i in top_idx) / float(len(top_idx))
        x_bot = sum(xs[i] for i in bottom_idx) / float(len(bottom_idx))
        x_sym = 0.5 * (x_top + x_bot)

        horiz_skew = abs(x_top - x_bot)
        if horiz_skew > 0.7 * height_px:
            return

        theta = math.atan((self.cx - x_sym) / self.fx)

        cos_th = math.cos(theta)
        if abs(cos_th) < 1e-3:
            return

        d = self.landmark_height * self.fy / (height_px * cos_th)

        meas_msg = PointStamped()
        meas_msg.header.stamp = msg.header.stamp
        meas_msg.header.frame_id = self.CAMERA_LINK
        meas_msg.point.x = float(d)
        meas_msg.point.y = float(theta)
        meas_msg.point.z = 0.0
        self.meas_pub.publish(meas_msg)

        self.last_meas_time = self.get_clock().now()

        self.log.info(
            f"h_px={height_px:.1f}, x_sym={x_sym:.1f}, d={d:.3f}, theta={theta:.3f}"
        )

        self.publish_error(
            stamp=meas_msg.header.stamp,
            measured_d=d,
            measured_theta=theta,
        )

    def publish_error(self, stamp, measured_d: float, measured_theta: float):
        if self.latest_link_states is None:
            return

        names = self.latest_link_states.name
        if self.CAMERA_LINK not in names:
            return
        if self.landmark_link_name not in names:
            return

        cam_idx = names.index(self.CAMERA_LINK)
        lm_idx = names.index(self.landmark_link_name)

        cam_pose = self.latest_link_states.pose[cam_idx]
        lm_pose = self.latest_link_states.pose[lm_idx]

        cam_x = cam_pose.position.x
        cam_y = cam_pose.position.y
        cam_q = cam_pose.orientation
        cam_yaw = self.quaternion_to_yaw(
            cam_q.x, cam_q.y, cam_q.z, cam_q.w
        )

        lm_x = lm_pose.position.x
        lm_y = lm_pose.position.y

        dx = lm_x - cam_x
        dy = lm_y - cam_y

        d_true = math.hypot(dx, dy)
        ang_world = math.atan2(dy, dx)
        theta_true = self.normalize_angle(ang_world - cam_yaw)

        err_d = measured_d - d_true
        err_theta = self.normalize_angle(measured_theta - theta_true)

        err_msg = PointStamped()
        err_msg.header.stamp = stamp
        err_msg.header.frame_id = self.CAMERA_LINK
        err_msg.point.x = float(err_d)
        err_msg.point.y = float(err_theta)
        err_msg.point.z = 0.0
        self.err_pub.publish(err_msg)

        self.log.info(
            f"d_err={err_d:.3f}, theta_err={err_theta:.3f}"
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main():
    rclpy.init()
    node = LandmarkPositioner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
