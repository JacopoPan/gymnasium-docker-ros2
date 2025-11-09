import gymnasium as gym
from gymnasium import spaces
import docker
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool, Empty
from std_srvs.srv import Trigger
import numpy as np
import threading
import time
import os

class RosSimEnv(gym.Env, Node):
    """
    Gymnasium environment that manages a multi-container ROS2 simulation.
    
    It inherits from rclpy.node.Node to handle ROS2 communication.
    """
    
    def __init__(self):
        # Initialize the Node part of the class
        super().__init__('ros_sim_env_node')
        
        # Gym Env properties
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # Docker client
        self.docker_client = docker.from_env()
        self.containers = []
        self.network = None
        self.ros_domain_id = os.environ.get('ROS_DOMAIN_ID', '42') # Use host's domain ID

        # ROS2 communication
        self.action_pub = self.create_publisher(Float64, '/action', 10)
        self.reset_client = self.create_client(Trigger, '/reset')

        # Threading for ROS callbacks
        self.obs_val = 0.0
        self.reward_val = 0.0
        self.done_val = False
        self.data_ready = threading.Event()
        
        # Create subscribers in their own callback group
        self.sub_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.obs_sub = self.create_subscription(
            Float64, '/observation', self._obs_callback, 10, callback_group=self.sub_group)
        self.reward_sub = self.create_subscription(
            Float64, '/reward', self._reward_callback, 10, callback_group=self.sub_group)
        self.done_sub = self.create_subscription(
            Bool, '/done', self._done_callback, 10, callback_group=self.sub_group)

        # Start the rclpy.spin() in a separate thread
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self.spin_thread.start()

        self.get_logger().info("RosSimEnv initialized.")

    # --- ROS Callbacks ---
    def _obs_callback(self, msg):
        self.obs_val = msg.data

    def _reward_callback(self, msg):
        self.reward_val = msg.data

    def _done_callback(self, msg):
        self.done_val = msg.data
        self.data_ready.set() # Signal that a step/reset is complete

    # --- Docker Management ---
    def _start_containers(self):
        self.get_logger().info("Starting Docker containers...")
        try:
            # Create a network
            self.network = self.docker_client.networks.create("ros_mwe_net")
            
            env_vars = {'ROS_DOMAIN_ID': self.ros_domain_id}

            # Start world container
            world_cont = self.docker_client.containers.run(
                "ros-world-mwe",
                detach=True,
                network=self.network.name,
                name="ros_world_mwe",
                environment=env_vars
            )
            self.containers.append(world_cont)
            
            # Start robot container
            robot_cont = self.docker_client.containers.run(
                "ros-robot-mwe",
                detach=True,
                network=self.network.name,
                name="ros_robot_mwe",
                environment=env_vars
            )
            self.containers.append(robot_cont)
            
            self.get_logger().info("Containers started.")
        
        except Exception as e:
            self.get_logger().error(f"Failed to start containers: {e}")
            self.close() # Attempt cleanup
            raise

    # --- Gym API ---
    def reset(self, seed=None, options=None):
        # super().reset(seed=seed) # Not strictly needed for this MWE
        
        self.get_logger().info("Resetting environment...")
        self.close() # Clean up old containers
        self._start_containers()

        # Wait for the /reset service to be available
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting...')

        # Clear the event flag
        self.data_ready.clear()
        
        # Call the /reset service
        req = Trigger.Request()
        future = self.reset_client.call_async(req)
        
        # Wait for the service call to complete
        # Note: We don't spin here, as the spin_thread is handling it
        # We just wait for the future.
        while rclpy.ok() and not future.done():
            time.sleep(0.1)
        
        if future.result() is None:
            self.get_logger().error("Failed to call reset service")
            return None, {}

        # Wait for the first data packet (triggered by _done_callback)
        # This confirms the simulation has reset and published its first state
        if not self.data_ready.wait(timeout=10.0):
            self.get_logger().error("Timeout waiting for first data after reset.")
            return None, {}

        self.get_logger().info("Reset complete.")
        return np.array([self.obs_val], dtype=np.float32), {}

    def step(self, action):
        self.data_ready.clear()
        
        # Publish the action
        action_msg = Float64()
        action_msg.data = float(action[0])
        self.action_pub.publish(action_msg)
        
        # Wait for the simulation to process and publish results
        if not self.data_ready.wait(timeout=5.0):
            self.get_logger().warn("Step timeout: no data received.")
            # Return a default state (e.g., last known state)
            return np.array([self.obs_val], dtype=np.float32), 0.0, True, False, {"timeout": True}

        # The callbacks have updated self.obs_val, self.reward_val, self.done_val
        terminated = self.done_val
        truncated = False # We can add timeout logic from the world_node if needed
        reward = self.reward_val
        obs = np.array([self.obs_val], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.get_logger().info("Closing environment: stopping containers...")
        for container in self.containers:
            try:
                container.stop(timeout=5)
                container.remove()
            except docker.errors.NotFound:
                pass # Container already gone
            except Exception as e:
                self.get_logger().warn(f"Error stopping {container.name}: {e}")
        
        if self.network:
            try:
                self.network.remove()
            except docker.errors.NotFound:
                pass
            except Exception as e:
                self.get_logger().warn(f"Error removing network: {e}")

        self.containers = []
        self.network = None

    def render(self, mode='human'):
        # Not implemented for this MWE
        pass