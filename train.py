import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from gymnasium_docker_ros2.env import GDR2Env

def main():
    # Register the environment so we can create it with gym.make()
    gym.register(
        id="GDR2Env-v0",
        entry_point=GDR2Env,
    )
    env = gym.make("GDR2Env-v0", render_mode="human")

    try:
        # check_env(env) # Throws warning
        # check_env(env.unwrapped)
        sb3_check_env(env) # Reward not a float (?)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    print("Training agent...")
    model.learn(total_timesteps=30000)
    print("Training complete.")

    # Save the agent
    model_path = "ppo_agent.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Load and test the trained agent
    del model # remove to demonstrate loading
    model = PPO.load(model_path)

    print("\nTesting trained agent...")
    obs, info = env.reset()
    for _ in range(500): # Run for 500 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
    
    env.close()
    print("Test complete.")

if __name__ == '__main__':
    main()
