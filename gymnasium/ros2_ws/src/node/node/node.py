import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Empty
import numpy as np

class RobotNode(Node):
    """
    Simulates a 1D robot.
    - Subscribes to /action (Float64)
    - Subscribes to /robot_reset (Empty)
    - Publishes /observation (Float64)
    """
    def __init__(self):
        super().__init__('robot_node')
        self.position = 0.0
        self.velocity = 0.0
        self.last_action = 0.0

        # Create pub/sub
        self.sub_action = self.create_subscription(
            Float64, '/action', self.action_callback, 10)
        self.sub_reset = self.create_subscription(
            Empty, '/robot_reset', self.reset_callback, 10)
        self.pub_obs = self.create_publisher(Float64, '/observation', 10)
        
        # Physics timer (50Hz)
        self.timer = self.create_timer(0.02, self.step_physics)
        self.get_logger().info('Robot node started.')

    def action_callback(self, msg):
        # Clip action to be between -1.0 and 1.0
        self.last_action = np.clip(msg.data, -1.0, 1.0)

    def reset_callback(self, msg):
        self.get_logger().info('Robot reset.')
        self.position = 0.0
        self.velocity = 0.0
        self.last_action = 0.0
        # Publish initial state
        self.publish_observation()

    def step_physics(self):
        # Simple 1D physics: force -> velocity -> position
        self.velocity += self.last_action * 0.02 # Force applied over time step
        self.velocity *= 0.98 # Damping
        self.position += self.velocity * 0.02
        
        # Clip position
        self.position = np.clip(self.position, -10.0, 10.0)
        
        self.publish_observation()

    def publish_observation(self):
        obs_msg = Float64()
        obs_msg.data = self.position
        self.pub_obs.publish(obs_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()