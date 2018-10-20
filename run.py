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
        env_path = os.path.join('envs', env_id+'-run')
    elif psutil.WINDOWS:
        env_path = os.path.join('envs', env_id+'-run', 'Unity Environment.exe')
    env = UnityEnv(env_path)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    # Automatically normalize the input features
    # env = VecNormalize(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)


    # Load the trained agent
    model = PPO2.load(os.path.join("models", env_id))

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
