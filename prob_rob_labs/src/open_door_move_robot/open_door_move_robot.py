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

        self.pub_door = self.create_publisher(Empty, '/door_open', 1)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_door_torque = self.create_publisher(Float64, '/hinged_glass_door/torque', 1)

        self.step = 0
        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def heartbeat(self):
        # self.log.info('heartbeat')
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = current_time - self.start_time

        if self.step == 0:
            self.log.info('Step 1: Opening the door')
            self.pub_door_torque.publish(Float64(data=10.0))
            if elapsed >= t_door_open:
                self.step += 1
                self.start_time = current_time

        elif self.step == 1:
            self.log.info('Step 2: Moving the robot forward')
            move_cmd = Twist()
            move_cmd.linear.x = 0.5  # move forward
            self.pub_cmd_vel.publish(move_cmd)
            if elapsed >= t_robot_move:
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

