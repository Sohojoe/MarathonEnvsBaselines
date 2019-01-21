import argparse
import os

import gym
# import pybullet_envs
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack


from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
# parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                    type=int)
parser.add_argument('--n-envs', help='number of environments', default=1,
                    type=int)
parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                    type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                    type=int)
parser.add_argument('--no-render', action='store_true', default=False,
                    help='Do not render the environment (useful for tests)')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='Use deterministic actions')
parser.add_argument('--norm-reward', action='store_true', default=False,
                    help='Normalize reward if applicable (trained with VecNormalize)')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)

args = parser.parse_args()

env_id = args.env
algo = args.algo
folder = args.folder
model_path = os.path.join(folder, algo, env_id)

if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)

# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, env_id, args.exp_id)
else:
    log_path = os.path.join(folder, algo, env_id)

model_path = os.path.join(log_path, env_id) + '.pkl'

assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)

if algo in ['dqn', 'ddpg', 'sac']:
    args.n_envs = 1

set_global_seeds(args.seed)

is_atari = 'NoFrameskip' in env_id

stats_path = os.path.join(log_path, env_id)
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

log_dir = args.reward_log if args.reward_log != '' else None

env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                        stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                        should_render=not args.no_render,
                        hyperparams=hyperparams)


# env = create_test_env(env_id, n_envs=args.n_envs, n_agents=args.n_agents, is_atari=is_atari,
#                       stats_path=stats_path, norm_reward=args.norm_reward,
#                       seed=args.seed, log_dir=log_dir, should_render=not args.no_render)

model = ALGOS[algo].load(model_path)

obs = env.reset()

# Force deterministic for DQN and DDPG
deterministic = args.deterministic or algo in ['dqn', 'ddpg']

running_reward = 0.0
ep_len = 0
for _ in range(args.n_timesteps):
    action, _ = model.predict(obs, deterministic=deterministic)
    # Random Agent
    # action = [env.action_space.sample()]
    # Clip Action to avoid out of bound errors
    if isinstance(env.action_space, gym.spaces.Box):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, done, infos = env.step(action)
    if not args.no_render:
        env.render('human')
    running_reward += reward[0]
    ep_len += 1

    if args.n_envs == 1:
        # For atari the return reward is not the atari score
        # so we have to get it from the infos dict
        if is_atari and infos is not None and args.verbose >= 1:
            episode_infos = infos[0].get('episode')
            if episode_infos is not None:
                print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                print("Atari Episode Length", episode_infos['l'])

        if done and not is_atari and args.verbose >= 1:
            # NOTE: for env using VecNormalize, the mean reward
            # is a normalized reward when `--norm_reward` flag is passed
            print("Episode Reward: {:.2f}".format(running_reward))
            print("Episode Length", ep_len)
            running_reward = 0.0
            ep_len = 0

print("Episode Reward: {:.2f}".format(running_reward))
print("Episode Length", ep_len)

# Workaround for https://github.com/openai/gym/issues/893
if not args.no_render:
    if args.n_envs == 1 and not 'Bullet' in env_id and not is_atari:
        # DummyVecEnv
        # Unwrap env
        while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
            env = env.venv
        env.envs[0].env.close()
    else:
        # SubprocVecEnv
        env.close()
