import rclpy
from rclpy.node import Node
import argparse

from std_msgs.msg import Float64 
from geometry_msgs.msg import Vector3

class DynamicsNode(Node):
    def __init__(self, rate):
        super().__init__('dynamics_node')

        self.rate = rate
        self.dt = 1.0 / self.rate

        self.position = 0.0
        self.velocity = 0.0
        self.control_input = 0.0

        self.publisher = self.create_publisher(Vector3, '/state', 10)

        self.subscriber = self.create_subscription(
            Float64,
            '/noisy_control',
            self.control_input_callback, 
            10
        )

        self.timer = self.create_timer(self.dt, self.update_state)
        self.get_logger().info(f"Dynamics Node running at {self.rate} Hz; dt: {self.dt:.4f}s")

    def control_input_callback(self, msg):
        self.control_input = msg.data
        self.get_logger().debug(f"Received control_input: {self.control_input:.4f}")

    def update_state(self):
        self.velocity += self.control_input * self.dt
        self.velocity *= 0.99  # Add some damping        
        self.position += self.velocity * self.dt

        # Clip position to the bounds [-1.0, 1.0]
        self.position = np.clip(self.position, -1.0, 1.0)
        
        # If it hits a wall, dampen the velocity (like a bounce)
        if self.position == -1.0 or self.position == 1.0:
            self.velocity *= -0.5
            
        # Publish the state
        state_msg = Vector3()        
        state_msg.x = self.position
        state_msg.y = self.velocity
        state_msg.z = 0.0  # Unused
        self.publisher.publish(state_msg)
        self.get_logger().debug(f"Published state (Pos/Vel): {self.position:.4f}, {self.velocity:.4f}")

def main(args=None):
    parser = argparse.ArgumentParser(description='Dynamics Node')
    parser.add_argument('--rate', type=int, default=50, help='Control frequency.')
    
    cli_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    
    node = DynamicsNode(
        rate=cli_args.rate,
    )
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
