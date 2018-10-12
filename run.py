import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from gym_unity.envs import UnityEnv

env = UnityEnv('./envs/Walker')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)


# Load the trained agent
model = PPO2.load('./models/my-model')

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()