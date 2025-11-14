import numpy as np
import gymnasium as gym
import docker
import zmq
import time
import struct

from docker.types import NetworkingConfig, EndpointConfig


class GDR2Env(gym.Env):
    """
    A simple 1D dynamical system (point mass).
    
    - State: [position, velocity] (2D)
    - Action: [force] (1D)
    - Goal: Stay at position 0.0
    """
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.max_steps = 1000  # Max steps per episode
        self.dt = 0.05         # Time step

        # Observation Space: [position, velocity]
        # position is in [-1, 1], velocity is in [-5, 5]
        self.obs_low = np.array([-1.0, -5.0], dtype=np.float32)
        self.obs_high = np.array([1.0, 5.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # Action Space: [force] between -1.0 and 1.0
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self.position = 0.0
        self.velocity = 0.0
        self.step_count = 0
        
        # Rendering
        self.render_mode = render_mode

        # Docker setup
        self.NETWORK_NAME = "gdr2-network"
        try:
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError("Could not connect to the Docker daemon. Ensure Docker is running.") from e
        print(f"Setting up Docker Network: {self.NETWORK_NAME}...")
        try:
            existing_network = self.client.networks.get(self.NETWORK_NAME)
            existing_network.remove()
            print(f"Existing network '{self.NETWORK_NAME}' removed.")
        except docker.errors.NotFound:
            pass
        ipam_pool = docker.types.IPAMPool(
            subnet='10.42.0.0/16',
            gateway='10.42.0.1'
        )
        ipam_config=docker.types.IPAMConfig(
            pool_configs=[ipam_pool]
        )
        self.network = self.client.networks.create(
            self.NETWORK_NAME, 
            driver="bridge",
            ipam=ipam_config
        )
        SIMULATION_IP = "10.42.0.20"
        DYNAMICS_IP = "10.42.0.30"
        print("Creating simulation-container...")
        self.simulation_container = self.client.containers.create(
            "gdr2-image:latest",
            name="simulation-container",
            tty=True,
            detach=True,
            auto_remove=True,
            environment={
                "ROS_DOMAIN_ID": "42",
                "TMUX_OPTS": "simulation",
            }
        )
        print(f"Connecting simulation-container to {self.NETWORK_NAME} with IP {SIMULATION_IP}...")
        self.network.connect(
            self.simulation_container,
            ipv4_address=SIMULATION_IP
        )
        self.simulation_container.start()
        print("Creating dynamics-container...")
        self.dynamics_container = self.client.containers.create(
            "gdr2-image:latest",
            name="dynamics-container",
            tty=True,
            detach=True,
            auto_remove=True,
            environment={
                "ROS_DOMAIN_ID": "42",
                "TMUX_OPTS": "dynamics",
            }
        )
        print(f"Connecting dynamics-container to {self.NETWORK_NAME} with IP {DYNAMICS_IP}...")
        self.network.connect(
            self.dynamics_container,
            ipv4_address=DYNAMICS_IP
        )
        self.dynamics_container.start()
        print("Docker setup complete. Containers are running.")

        # ZeroMQ Setup
        self.ZMQ_PORT = 5555
        self.ZMQ_IP = SIMULATION_IP # DYNAMICS_IP
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 10 * 1000) # 1000 ms = 1 seconds
        self.socket.connect(f"tcp://{self.ZMQ_IP}:{self.ZMQ_PORT}")
        print(f"ZeroMQ socket connected to {self.ZMQ_IP}:{self.ZMQ_PORT}")

    def _get_obs(self):
        return np.array([self.position, self.velocity], dtype=np.float32)

    def _get_info(self):
        return {"position": self.position, "velocity": self.velocity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding

        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            reset = 9999.0
            # Serialize the action and send the REQ
            action_payload = f"{reset}".encode('utf-8')
            self.socket.send(action_payload)    
            # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
            reply_bytes = self.socket.recv()
            # Deserialize
            unpacked = struct.unpack('ffii', reply_bytes)
            pos, vel, sec, nanosec = unpacked
            # print(f"Received reset state: Pos={pos}, Vel={vel} at time {sec}.{nanosec}")
            self.position = pos
            self.velocity = vel
        except zmq.error.Again:
            print("ZMQ Error: Reply from container timed out.")
        except ValueError:
            print("ZMQ Error: Reply format error. Received garbage state.")
        ###########################################################################################
        # Reset state to a random position near the center ########################################
        ###########################################################################################
        # self.position = self.np_random.uniform(low=-0.1, high=0.1)
        # self.velocity = 0.0
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        self.step_count = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        force = action[0]

        ###########################################################################################
        # ZeroMQ REQ/REP to the ROS2 sim ##########################################################
        ###########################################################################################
        try:
            # Serialize the action and send the REQ
            action_payload = f"{force}".encode('utf-8')
            self.socket.send(action_payload)    
            # Wait for the REP (synchronous block) this call will block until a reply is received or it times out
            reply_bytes = self.socket.recv()
            # Deserialize
            unpacked = struct.unpack('ffii', reply_bytes)
            pos, vel, sec, nanosec = unpacked
            # print(f"Received state: Pos={pos}, Vel={vel} at time {sec}.{nanosec}")
            self.position = pos
            self.velocity = vel
        except zmq.error.Again:
            print("ZMQ Error: Reply from container timed out.")
        except ValueError:
            print("ZMQ Error: Reply format error. Received garbage state.")
        ###########################################################################################
        # Simple Euler integration ################################################################
        ###########################################################################################
        # self.velocity += force * self.dt
        # self.velocity *= 0.99  # Add some damping
        # self.position += self.velocity * self.dt
        # # Clip position to the bounds [-1.0, 1.0]
        # self.position = np.clip(self.position, -1.0, 1.0)
        # # If it hits a wall, dampen the velocity (like a bounce)
        # if self.position == -1.0 or self.position == 1.0:
        #     self.velocity *= -0.5
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        self.step_count += 1
        
        # Calculate reward: Negative distance from the goal (position 0)
        reward = float(-np.abs(self.position))
        # Check for termination
        terminated = False  # This is a continuing task, never "terminates"
        # Check for truncation (episode ends due to time limit)
        truncated = self.step_count >= self.max_steps
        # Get obs and info
        obs = self._get_obs()
        info = self._get_info()
        
        # Handle rendering
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # Scale position from [-1, 1] to a 40-char width
        pos_int = int((self.position + 1.0) / 2.0 * 40)     
        # Create the display string
        display = ['-'] * 41
        display[pos_int] = 'O'  # The agent
        if pos_int != 20:
            display[20] = '|'       # The target (0.0)   
        # Print to console
        print(f"\r{''.join(display)}  Pos: {self.position:6.3f}, Vel: {self.velocity:6.3f}", end="")

    def close(self):
        if self.render_mode == "human":
            print()  # Add a newline after the final render
        
        # Docker clean-up (remove=True handles removal after stop)
        try:
            self.simulation_container.stop()
            print(f"Simulation container stopped.")
        except Exception:
            pass
        try:
            self.dynamics_container.stop()
            print(f"Dynamics container stopped.")
        except Exception:
            pass
        try:
            self.network.remove()
            print(f"Network {self.NETWORK_NAME} removed.")
        except Exception:
            pass

        # Close ZMQ resources
        if self.socket:
            self.socket.close(linger=0)
        if self.zmq_context:
            self.zmq_context.term()
