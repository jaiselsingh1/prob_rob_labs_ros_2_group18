import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist


heartbeat_period = 0.1

t_door_open = 6.0
t_robot_move = 9.0
t_door_close = 3.0

class OpenDoorMoveRobot(Node):

    def __init__(self):
        super().__init__('open_door_move_robot')
        self.log = self.get_logger()

        # basic parameters
        self.declare_parameter('robot_speed', 0.5)
        self.robot_speed = self.get_parameter('robot_speed').get_parameter_value().double_value
        self.log.info(f'Robot speed this time: {self.robot_speed}')

        # Bayesian decision parameters
        # measurement_threshold: numerical cutoff to map feature_mean -> observed z (True means z=open)
        self.declare_parameter('measurement_threshold', 238.0)
        # if True, z=open when feature_mean < measurement_threshold (keeps legacy behavior)
        self.declare_parameter('z_open_when_below', True)
        self.declare_parameter('bayes_decision_threshold', 0.999)
        self.declare_parameter('prior_open', 0.5)
        # conditional probabilities (from your values)
        self.declare_parameter('p_z_open_given_x_open', 0.947)
        self.declare_parameter('p_z_closed_given_x_open', 0.053)
        self.declare_parameter('p_z_open_given_x_closed', 0.042)
        self.declare_parameter('p_z_closed_given_x_closed', 0.958)

        self.measurement_threshold = float(self.get_parameter('measurement_threshold').value)
        self.z_open_when_below = bool(self.get_parameter('z_open_when_below').value)
        self.bayes_decision_threshold = float(self.get_parameter('bayes_decision_threshold').value)
        self.belief_open = float(self.get_parameter('prior_open').value)
        # probabilities
        self.p_z_open_given_x_open = float(self.get_parameter('p_z_open_given_x_open').value)
        self.p_z_closed_given_x_open = float(self.get_parameter('p_z_closed_given_x_open').value)
        self.p_z_open_given_x_closed = float(self.get_parameter('p_z_open_given_x_closed').value)
        self.p_z_closed_given_x_closed = float(self.get_parameter('p_z_closed_given_x_closed').value)

        self.pub_door = self.create_publisher(Empty, '/door_open', 1)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_door_torque = self.create_publisher(Float64, '/hinged_glass_door/torque', 1)

        self.step = 0

        # store it as a class attribute
        self.feature_mean_value = 0.0 
        self.feature_mean = self.create_subscription(Float64, '/feature_mean', self.feature_mean_callback, 10)

        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)
    
    def feature_mean_callback(self, msg):
        # update measurement and Belief using Bayesian update
        self.feature_mean_value = msg.data
        # map measurement to observed z (open/closed)
        if self.z_open_when_below:
            z_open = (self.feature_mean_value < self.measurement_threshold)
        else:
            z_open = (self.feature_mean_value >= self.measurement_threshold)

        # compute likelihoods P(z | x=open) and P(z | x=closed)
        if z_open:
            p_z_given_x_open = self.p_z_open_given_x_open
            p_z_given_x_closed = self.p_z_open_given_x_closed
        else:
            p_z_given_x_open = self.p_z_closed_given_x_open
            p_z_given_x_closed = self.p_z_closed_given_x_closed

        prior = self.belief_open
        numerator = p_z_given_x_open * prior
        denominator = numerator + p_z_given_x_closed * (1.0 - prior)
        if denominator > 0.0:
            posterior = numerator / denominator
        else:
            posterior = prior

        # update belief (recursive Bayesian update)
        self.belief_open = posterior

        self.log.info(f'Feature mean: {self.feature_mean_value:.1f} z_open={z_open} belief_open={self.belief_open:.3f}')

    def heartbeat(self):
        # self.log.info('heartbeat')
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.start_time

        if self.step == 0:
            self.log.info('Step 1: Opening the door')
            self.pub_door_torque.publish(Float64(data=10.0))
            # Decide using Bayesian posterior probability
            if self.belief_open >= self.bayes_decision_threshold:
                self.pub_door.publish(Empty())
                self.log.info(f'Door open command published (belief_open={self.belief_open:.3f} >= {self.bayes_decision_threshold})')
                self.step += 1
                self.start_time = current_time
            # if elapsed >= t_door_open:
            #     self.step += 1
            #     self.start_time = current_time

        elif self.step == 1:
            self.log.info('Step 2: Moving the robot forward')
            move_cmd = Twist()

            move_cmd.linear.x = self.robot_speed
            # move_cmd.linear.x = 0.5

            self.pub_cmd_vel.publish(move_cmd)
            if elapsed >= 10.0:
                self.step += 1
                self.start_time = current_time

        elif self.step == 2:
            self.log.info('Step 3: Stopping the robot')
            stop_cmd = Twist()
            self.pub_cmd_vel.publish(stop_cmd)
            self.step += 1
            self.start_time = current_time

        elif self.step == 3:
            self.log.info('Step 4: Closing the door')
            self.pub_door_torque.publish(Float64(data=-10.0))
            if elapsed >= t_door_close:
                self.log.info('Done!!!')
                self.step += 1

        elif self.step == 4:
            pass

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    open_door_move_robot = OpenDoorMoveRobot()
    open_door_move_robot.spin()
    open_door_move_robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

