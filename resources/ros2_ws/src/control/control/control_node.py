import rclpy
from rclpy.node import Node
import argparse
import random

from std_msgs.msg import Float64 


class ControlNode(Node):
    def __init__(self, noise, rate):
        super().__init__('control_node')

        self.noise = noise
        self.rate = rate
        self.last_ideal_control = 0.0

        self.publisher = self.create_publisher(Float64, '/noisy_control', 10)

        self.subscriber = self.create_subscription(
            Float64,
            '/ideal_control',
            self.ideal_control_callback, 
            10
        )

        self.timer = self.create_timer(1.0 / self.rate, self.publish_noisy_control)
        self.get_logger().info(f"Control Node running at {self.rate} Hz")

    def ideal_control_callback(self, msg):
        self.last_ideal_control = msg.data
        self.get_logger().debug(f"Received ideal control: {self.last_ideal_control:.4f}")

    def publish_noisy_control(self):
        ideal_val = self.last_ideal_control   

        noisy_control_msg = Float64()
        noise_magnitude = np.abs(ideal_val) * self.noise        
        random_noise = random.uniform(-noise_magnitude, noise_magnitude)
        noisy_control_msg.data = ideal_val + random_noise
        
        self.publisher.publish(noisy_control_msg)
        self.get_logger().debug(f"Published noisy control: {noisy_control_msg.data:.4f}")

def main(args=None):
    parser = argparse.ArgumentParser(description='Control noise node')
    parser.add_argument('--noise', type=float, default=0.05, help='Control noise in percentage.')
    parser.add_argument('--rate', type=int, default=50, help='Control frequency.')
    
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    
    node = ControlNode(
        noise=cli_args.noise,
        rate=cli_args.rate,
    )
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
