import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float64
from geometry_msgs.msg import Twist


heartbeat_period = 0.1


class OpenDoorMoveRobot(Node):

    def __init__(self):
        super().__init__('open_door_move_robot')
        self.log = self.get_logger()

        # basic parameters
        self.declare_parameter('robot_speed', 0.5)
        self.robot_speed = float(self.get_parameter('robot_speed').value)

        # Bayesian decision parameters
        self.declare_parameter('measurement_threshold', 238.0)
        self.declare_parameter('z_open_when_below', True)
        self.declare_parameter('bayes_decision_threshold', 0.99)
        self.declare_parameter('prior_open', 0.5)

        # measurement model (P(z|x))
        self.declare_parameter('p_z_open_given_x_open', 0.947)
        self.declare_parameter('p_z_closed_given_x_open', 0.053)
        self.declare_parameter('p_z_open_given_x_closed', 0.042)
        self.declare_parameter('p_z_closed_given_x_closed', 0.958)

        # transition model and attempt rate
        self.declare_parameter('p_open_success', 0.167)
        self.declare_parameter('p_open_spontaneous', 0.0)
        self.declare_parameter('p_stay_open', 0.99)
        self.declare_parameter('attempt_interval', 0.5)

        # load parameters
        self.measurement_threshold = float(self.get_parameter('measurement_threshold').value)
        self.z_open_when_below = bool(self.get_parameter('z_open_when_below').value)
        self.bayes_decision_threshold = float(self.get_parameter('bayes_decision_threshold').value)
        self.belief_open = float(self.get_parameter('prior_open').value)

        self.p_z_open_given_x_open = float(self.get_parameter('p_z_open_given_x_open').value)
        self.p_z_closed_given_x_open = float(self.get_parameter('p_z_closed_given_x_open').value)
        self.p_z_open_given_x_closed = float(self.get_parameter('p_z_open_given_x_closed').value)
        self.p_z_closed_given_x_closed = float(self.get_parameter('p_z_closed_given_x_closed').value)

        self.p_open_success = float(self.get_parameter('p_open_success').value)
        self.p_open_spontaneous = float(self.get_parameter('p_open_spontaneous').value)
        self.p_stay_open = float(self.get_parameter('p_stay_open').value)
        self.attempt_interval = float(self.get_parameter('attempt_interval').value)

        # channels
        self.pub_door = self.create_publisher(Empty, '/door_open', 1)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.sub_feature = self.create_subscription(Float64, '/feature_mean', self.feature_mean_callback, 10)

        # runtime
        self.step = 0
        self.feature_mean_value = 0.0
        self.last_z_open = None
        self.last_z_time = 0.0
        self.last_attempt_time = 0.0

        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def feature_mean_callback(self, msg):
        self.feature_mean_value = msg.data
        if self.z_open_when_below:
            z_open = (self.feature_mean_value < self.measurement_threshold)
        else:
            z_open = (self.feature_mean_value >= self.measurement_threshold)
        self.last_z_open = bool(z_open)
        self.last_z_time = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        self.log.debug(f'feature_mean={self.feature_mean_value:.1f} z_open={self.last_z_open}')

    def heartbeat(self):
        now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        elapsed = now - self.start_time

        if self.step == 0:
            will_attempt = (now - self.last_attempt_time) >= self.attempt_interval and self.belief_open < self.bayes_decision_threshold
            prior = self.belief_open
            p01 = self.p_open_success if will_attempt else self.p_open_spontaneous
            p11 = self.p_stay_open
            predicted = p11 * prior + p01 * (1.0 - prior)

            posterior = predicted
            if self.last_z_open is not None and (now - self.last_z_time) < 1.0:
                if self.last_z_open:
                    p_z_given_open = self.p_z_open_given_x_open
                    p_z_given_closed = self.p_z_open_given_x_closed
                else:
                    p_z_given_open = self.p_z_closed_given_x_open
                    p_z_given_closed = self.p_z_closed_given_x_closed
                num = p_z_given_open * predicted
                den = num + p_z_given_closed * (1.0 - predicted)
                if den > 0.0:
                    posterior = num / den

            self.belief_open = posterior
            self.log.info(f'pred={predicted:.3f} post={posterior:.3f}')

            if self.belief_open >= self.bayes_decision_threshold:
                self.log.info(f'Belief {self.belief_open:.3f} >= {self.bayes_decision_threshold}, proceeding to move')
                self.step = 1
                self.start_time = now
            elif will_attempt:
                self.pub_door.publish(Empty())
                self.last_attempt_time = now
                self.log.info(f'Attempted /door_open (belief={self.belief_open:.3f})')

        elif self.step == 1:
            move_cmd = Twist()
            move_cmd.linear.x = float(self.get_parameter('robot_speed').value)
            self.pub_cmd_vel.publish(move_cmd)
            if elapsed >= 10.0:
                self.step = 2
                self.start_time = now

        elif self.step == 2:
            stop_cmd = Twist()
            self.pub_cmd_vel.publish(stop_cmd)
            self.step = 3
            self.start_time = now

        elif self.step == 3:
            if elapsed >= 3.0:
                self.log.info('Done')
                self.step = 4

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    node = OpenDoorMoveRobot()
    node.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
