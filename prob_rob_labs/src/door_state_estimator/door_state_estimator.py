import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int32MultiArray
from collections import deque


heartbeat_period = 0.1
            
class DoorStateEstimator(Node):

    def __init__(self):
        super().__init__('door_state_estimator')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)
        self.sub_feature_mean = self.create_subscription(
            Float64,
            '/feature_mean',
            self.handle_mean_feature,
            1
        )
        self.feature_mean = None
        # publisher for counts: two ints [#above_threshold, #below_threshold]
        self.pub_threshold_error = self.create_publisher(
            Int32MultiArray,
            '/threshold_error',
            1
        )
        self.threshold_error = Int32MultiArray()
        self.threshold_error.data = [0, 0]
        self.feature_mean_list = deque(maxlen=1000)
        self.feature_mean_list.clear()
        self.threshold = 238

    def handle_mean_feature(self, msg):
        self.feature_mean = msg.data
        # append new measurement (deque handles FIFO maxlen=1000)
        self.feature_mean_list.append(msg.data)
        # compute counts based on current buffer (do not accumulate repeatedly)
        above = 0
        below = 0
        for v in self.feature_mean_list:
            if v > self.threshold:
                above += 1
            else:
                below += 1
        self.threshold_error.data = [above, below]
        self.pub_threshold_error.publish(self.threshold_error)

    def heartbeat(self):
        self.log.info('heartbeat')

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    door_state_estimator = DoorStateEstimator()
    door_state_estimator.spin()
    door_state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
