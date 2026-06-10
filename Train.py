from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium.envs.registration import register
import os

def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("Initializing Environment...")

    register(
        id = 'Robot-Nav2-Env', # env name
        entry_point = 'Robot_Env:RobotEscapeEnv'
    )

    raw_env = gym.make('Robot-Nav2-Env')

    check_env(raw_env)
    print("Environment check passed! No errors found.")

    env = Monitor(raw_env, log_dir)
    env = DummyVecEnv([lambda: env])

    print("Building the PPO Brain...")

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=4096,
        batch_size=64,
        tensorboard_log="./ppo_robot_tensorboard/" 
    )

    print("Starting Training Loop")
    # 6. Train the model
    model.learn(total_timesteps=4000000, tb_log_name="PPO_Run_1")

    model_path = "ppo_local_planner_v1"
    model.save(model_path)
    print(f"Training complete! Model saved successfully as {model_path}.zip")

if __name__ == "__main__":
    main()