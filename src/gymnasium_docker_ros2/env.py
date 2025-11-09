import numpy as np
import gymnasium as gym

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

    def _get_obs(self):
        return np.array([self.position, self.velocity], dtype=np.float32)

    def _get_info(self):
        return {"position": self.position, "velocity": self.velocity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Handle seeding

        # Reset state to a random position near the center
        self.position = self.np_random.uniform(low=-0.1, high=0.1)
        self.velocity = 0.0
        self.step_count = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        force = action[0]
        
        # Simple Euler integration
        self.velocity += force * self.dt
        self.velocity *= 0.99  # Add some damping
        self.position += self.velocity * self.dt

        # Clip position to the bounds [-1.0, 1.0]
        self.position = np.clip(self.position, -1.0, 1.0)
        
        # If it hits a wall, dampen the velocity (like a bounce)
        if self.position == -1.0 or self.position == 1.0:
            self.velocity *= -0.5

        self.step_count += 1
        
        # Calculate reward: Negative distance from the goal (position 0)
        reward = -np.abs(self.position)
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
