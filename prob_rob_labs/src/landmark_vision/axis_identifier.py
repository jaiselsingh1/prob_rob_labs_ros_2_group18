import rclpy
from rclpy.node import Node
from prob_rob_msgs.msg import Point2DArrayStamped

heartbeat_period = 0.1

class LandmarkAxisIdentifier(Node):

    def __init__(self):
        super().__init__('landmark_axis_identifier')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.declare_parameter('landmark_color', 'cyan')
        self.landmark_color = self.get_parameter('landmark_color').get_parameter_value().string_value
        self.sub = self.create_subscription(
            Point2DArrayStamped,
            f'/vision_{self.landmark_color}/corners',
            self.landmark_positioning_callback,
            10
        )

    def heartbeat(self):
        self.log.info('heartbeat')

    def landmark_positioning_callback(self, msg: Point2DArrayStamped):
        ''' Callback for landmark corner positions. 
        Consider the shape to be rectangular and the object is far away if the number of points is below 8.
        In rectangular case: 
            height = max y - min y
            vertical symetrical axis position = (max x + min x) / 2

        If number of points is more than 8, consider it cylindrical
        In cylindrical case:
            height = max y - min y
            vertical axis position = average x position of all points'''
        num_points = len(msg.points)
        if num_points < 8:
            xs = [point.x for point in msg.points]
            ys = [point.y for point in msg.points]
            height = max(ys) - min(ys)
            vertical_axis_position = (max(xs) + min(xs)) / 2
            self.log.info(f'Rectangular landmark detected: height={height:.2f}, vertical_axis_position={vertical_axis_position:.2f}')

        else:
            xs = [point.x for point in msg.points]
            ys = [point.y for point in msg.points]
            height = max(ys) - min(ys)
            vertical_axis_position = sum(xs) / num_points
            self.log.info(f'Cylindrical landmark detected: height={height:.2f}, vertical_axis_position={vertical_axis_position:.2f}')

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    landmark_axis_identifier = LandmarkAxisIdentifier()
    landmark_axis_identifier.spin()
    landmark_axis_identifier.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()