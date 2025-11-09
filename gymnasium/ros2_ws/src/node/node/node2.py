import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, Empty
from std_srvs.srv import Trigger

class WorldNode(Node):
    """
    Manages the simulation state and rewards.
    - Subscribes to /observation (Float64)
    - Publishes /reward (Float64)
    - Publishes /done (Bool)
    - Publishes /robot_reset (Empty)
    - Provides /reset service (Trigger)
    """
    def __init__(self):
        super().__init__('world_node')
        self.target_position = 5.0
        self.step_count = 0
        self.max_steps = 250

        # Create pub/sub
        self.sub_obs = self.create_subscription(
            Float64, '/observation', self.obs_callback, 10)
        self.pub_reward = self.create_publisher(Float64, '/reward', 10)
        self.pub_done = self.create_publisher(Bool, '/done', 10)
        self.pub_robot_reset = self.create_publisher(Empty, '/robot_reset', 10)

        # Create reset service
        self.srv_reset = self.create_service(
            Trigger, '/reset', self.reset_callback)
        
        self.get_logger().info('World node started.')

    def obs_callback(self, msg):
        position = msg.data
        self.step_count += 1

        # Calculate reward
        distance = abs(position - self.target_position)
        reward = -distance  # Simple reward: negative distance

        # Check for done
        done = (distance < 0.1) or (self.step_count > self.max_steps)
        
        # Publish reward and done
        reward_msg = Float64()
        reward_msg.data = reward
        self.pub_reward.publish(reward_msg)

        done_msg = Bool()
        done_msg.data = done
        self.pub_done.publish(done_msg)

    def reset_callback(self, request, response):
        self.get_logger().info('World reset service called.')
        self.step_count = 0
        
        # (Could randomize target_position here)
        
        # Trigger the robot to reset
        self.pub_robot_reset.publish(Empty())

        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    node = WorldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()