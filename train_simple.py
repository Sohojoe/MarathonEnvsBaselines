import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from gym_unity.envs import UnityEnv
import os
import psutil

def main():
    env_id = "hopper"
    # env_id = "walker"
    if psutil.MACOS:
        env_path = os.path.join('envs', env_id)
    elif psutil.WINDOWS:
        env_path = os.path.join('envs', env_id, 'Unity Environment.exe')
    env = UnityEnv(env_path)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    # Automatically normalize the input features
    # env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)
    env = VecNormalize(env)
    tensorboard_log = os.path.join("summaries", env_id)

    model = PPO2(MlpPolicy, env, 
        verbose=2, tensorboard_log=tensorboard_log
        )
    model.learn(total_timesteps=1000000)
    os.makedirs('models', exist_ok=True)
    model.save(os.path.join("models", "walker_ppo2_simple"))

if __name__ == '__main__':
    main()