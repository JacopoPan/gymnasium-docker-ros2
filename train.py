import numpy as np
import gymnasium as gym
import argparse
import time

from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from gymnasium_docker_ros2.env import GDR2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["step", "speed", "learn"])
    args = parser.parse_args()

    # Register the environment so we can create it with gym.make()
    gym.register(
        id="GDR2Env-v0",
        entry_point=GDR2Env,
    )
    env = gym.make("GDR2Env-v0", render_mode="human")

    if args.mode == "step":
        obs, info = env.reset()
        print(f"Reset result -- Obs: {obs}")
        for i in range(8):
            user_input = input("Press Enter to step, 'r' then Enter to reset...")
            stripped_input = user_input.strip().lower()
            if stripped_input and stripped_input in ('r', 'reset'):
                obs, info = env.reset()
                print(f"\nReset result -- Obs: {obs}")
            else:
                rnd_action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(rnd_action)
                print(f"\nStep {i} -- action: {rnd_action} result -- Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        print("\nInitial test steps complete. Closing environment.")
        env.close()

    elif args.mode == "speed":
        STEPS = 10000
        print(f"Starting Speed Test ({STEPS} steps)")    
        obs, info = env.reset()
        start_time = time.time()        
        for _ in range(STEPS):
            action = env.action_space.sample()            
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        total_time = time.time() - start_time 
        print(f"\nAverage Step Time: {(total_time / STEPS) * 1000:.3f} ms")
        print(f"Throughput: {(STEPS / total_time):.2f} steps/second")
        print(f"Time for 1,000,000 steps: {((total_time * (1000000/STEPS))/3600):.2f} hours")
        print(f"Time to simulate 10 days at 50 Hz: {((10 * 24 * 60 * 60) * 50 * (total_time / STEPS) / 3600):.2f} hours")
        env.close()

    elif args.mode == "learn":
        try:
            # check_env(env) # Throws warning
            # check_env(env.unwrapped)
            sb3_check_env(env)
            print("\nEnvironment passes all checks!")
        except Exception as e:
            print(f"\nEnvironment has issues: {e}")

        env.reset()
        env.step(env.action_space.sample())
        env.reset()

        # Instantiate the agent
        model = PPO("MlpPolicy", env, verbose=1, device='cpu')

        # Train the agent
        print("Training agent...")
        model.learn(total_timesteps=40000)
        print("Training complete.")

        # Save the agent
        model_path = "ppo_agent.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Load and test the trained agent
        del model # remove to demonstrate loading
        model = PPO.load(model_path, device='cpu')

        print("\nTesting trained agent...")
        obs, info = env.reset()
        for _ in range(800): # Run for 800 steps
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Episode finished. Resetting.")
                obs, info = env.reset()
        
        env.close()

    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
