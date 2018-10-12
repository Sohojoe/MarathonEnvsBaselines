import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from gym_unity.envs import UnityEnv

# env = UnityEnv('./envs/Walker')
env = UnityEnv('./envs/Walker_x16', multiagent=True)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("./models/my-model")
