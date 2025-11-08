#!/usr/bin/env python3
import math
import csv
import os

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

        # Log topic types for diagnostics so we can detect message-type mismatches
        try:
            topics = self.get_topic_names_and_types()
            topics_dict = {name: types for name, types in topics}
            want1 = f'/vision_{self.landmark_color}/corners'
            want2 = '/gazebo/link_states'
            self.log.debug(f"available topics: {list(topics_dict.keys())}")
            if want1 in topics_dict:
                self.log.info(f"topic {want1} types: {topics_dict[want1]}")
            else:
                self.log.info(f"topic {want1} not present at startup")
            if want2 in topics_dict:
                self.log.info(f"topic {want2} types: {topics_dict[want2]}")
            else:
                self.log.info(f"topic {want2} not present at startup")
        except Exception:
            # non-fatal: diagnostic logging only
            pass

        meas_topic = f'/vision_{self.landmark_color}/measurement'
        self.meas_pub = self.create_publisher(PointStamped, meas_topic, 10)

        err_topic = f'/vision_{self.landmark_color}/error'
        self.err_pub = self.create_publisher(PointStamped, err_topic, 10)

        self.heartbeat_timer = self.create_timer(
            HEARTBEAT_PERIOD, self.heartbeat_cb
        )

        # CSV recording parameters (no driving performed)
        # Default csv_file is empty here; if not provided we set it to the
        # workspace pictures folder below so we don't write to root or /tmp.
        self.declare_parameter('record_data', False)
        self.declare_parameter('csv_file', '')
        self.record_data = bool(self.get_parameter('record_data').get_parameter_value().bool_value)
        csv_param = str(self.get_parameter('csv_file').get_parameter_value().string_value)
        if csv_param == '' or csv_param is None:
            # default into workspace pictures folder (non-root)
            self.csv_file = f"/home/hqh/ros2_ws/src/prob_rob_labs_ros_2/pictures/landmark_data_{self.landmark_color}.csv"
        else:
            self.csv_file = csv_param
        self._csv_writer = None
        self._csv_handle = None

        if self.record_data:
            try:
                dirpath = os.path.dirname(self.csv_file)
                if dirpath and not os.path.exists(dirpath):
                    try:
                        os.makedirs(dirpath, exist_ok=True)
                    except Exception as e:
                        self.log.error(f"failed to create directory {dirpath}: {e}")
                        raise

                new_file = not os.path.exists(self.csv_file)
                # attempt to open file for append and test a write immediately
                self._csv_handle = open(self.csv_file, 'a', newline='')
                self._csv_writer = csv.writer(self._csv_handle)
                if new_file:
                    # write header
                    self._csv_writer.writerow(['stamp', 'measured_d', 'measured_theta', 'true_d', 'true_theta', 'err_d', 'err_theta', 'color'])
                    self._csv_handle.flush()

                # write a startup test row so we can verify filesystem writes
                try:
                    import time as _time
                    ts0 = _time.time()
                    # test row uses zeros for numeric fields; it's harmless and easy to spot
                    self._csv_writer.writerow([ts0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, str(self.landmark_color)])
                    self._csv_handle.flush()
                    self.log.info(f"recording measurements to {self.csv_file} (startup test row written)")
                except Exception as e:
                    self.log.error(f"failed to write startup test row to {self.csv_file}: {e}")
                    raise

                # diagnostic counter for how many rows we've written (includes the test row)
                self._rows_written = 1
            except Exception as e:
                self.log.error(f"failed to open csv file {self.csv_file}: {e}")
                self.record_data = False

        self.log.info(
            f'landmark_positioner initialized for color={self.landmark_color}'
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
        # Record measurement and error to CSV if enabled
        if getattr(self, '_csv_writer', None) and self.record_data:
            try:
                # prefer stamp.sec/nanosec but handle other representations
                sec = getattr(stamp, 'sec', None)
                nsec = getattr(stamp, 'nanosec', None)
                if sec is None:
                    sec = getattr(stamp, 'sec', 0)
                if nsec is None:
                    nsec = getattr(stamp, 'nsec', 0)
                ts = float(sec) + float(nsec) * 1e-9
                self._csv_writer.writerow([
                    ts,
                    float(measured_d),
                    float(measured_theta),
                    float(d_true),
                    float(theta_true),
                    float(err_d),
                    float(err_theta),
                    str(self.landmark_color),
                ])
                try:
                    self._csv_handle.flush()
                except Exception:
                    pass
                # increment and log diagnostics
                try:
                    self._rows_written += 1
                    if self._rows_written == 1:
                        self.log.info(f"first csv row written to {self.csv_file}")
                    elif (self._rows_written % 50) == 0:
                        self.log.info(f"{self._rows_written} rows written to {self.csv_file}")
                except Exception:
                    pass
            except Exception as e:
                self.log.error(f"failed to write csv row: {e}")

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

    def __del__(self):
        # ensure CSV handle closed on destruction
        try:
            if getattr(self, '_csv_handle', None):
                try:
                    self._csv_handle.close()
                except Exception:
                    pass
        except Exception:
            pass


def main():
    rclpy.init()
    node = LandmarkPositioner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # user requested shutdown
        pass
    except Exception as e:
        # log unexpected exceptions with traceback to aid debugging
        try:
            node.get_logger().error(f"Unhandled exception in spin: {e}", exc_info=True)
        except Exception:
            print(f"Unhandled exception in spin: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            # rclpy may already be shutdown; ignore
            pass


if __name__ == '__main__':
    main()
