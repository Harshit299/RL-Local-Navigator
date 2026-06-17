import os
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 

def main():

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("Initializing Environment...")

    # Prevent re-registration errors if running multiple times
    try:
        register(
            id='Robot-RL-Env',
            entry_point='Robot_Env:RobotEscapeEnv'
        )
    except Exception:
        pass

    raw_env = gym.make('Robot-RL-Env')

    check_env(raw_env)
    print("No errors found")

    env = Monitor(raw_env, log_dir)
    env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Training the PPO algorithm...")

    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1, 
        ent_coef=0.01,
        clip_range=0.1,
        learning_rate=0.0003,
        n_steps=8192,
        batch_size=256,
        tensorboard_log="./ppo_robot_tensorboard/" 
    )

    print("Starting Training Loop")
    model.learn(total_timesteps=2000000, tb_log_name="PPO_Run_1")

    model_path = "ppo_local_planner_v1"
    model.save(model_path)
    print(f"Model saved successfully as {model_path}.zip")

    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"Normalization stats saved successfully as {stats_path}")

if __name__ == "__main__":
    main()
