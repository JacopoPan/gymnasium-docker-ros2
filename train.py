import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

# Import your custom environment
from gym_env.ros_sim_env import RosSimEnv

def main():
    # Initialize rclpy
    rclpy.init()
    
    # Create the environment
    # We pass the 'rclpy.Node' functionality to the environment
    # by letting it create its own node.
    try:
        env = RosSimEnv()

        # It's a good practice to check the environment
        # check_env(env)
        # print("Environment check passed!")
        # Note: check_env may fail due to the complex async nature
        # of this env. Manual testing is recommended.

        # Instantiate the agent
        model = PPO("MlpPolicy", env, verbose=1)
        
        # Train the agent
        print("Starting agent training...")
        model.learn(total_timesteps=10000)
        
        # Save the agent
        model.save("ppo_ros_sim")
        print("Model saved to ppo_ros_sim.zip")
        del model  # remove to demonstrate loading

        # Load and test the trained agent
        model = PPO.load("ppo_ros_sim")
        print("Model loaded. Starting evaluation...")
        
        obs, info = env.reset()
        for _ in range(200):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished. Resetting.")
                obs, info = env.reset()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        print("Shutting down environment and rclpy.")
        if 'env' in locals():
            env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()