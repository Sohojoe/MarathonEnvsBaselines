from collections import OrderedDict

import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv
from gym_unity.envs import UnityEnv


class UnityVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """
    
    def __init__(self, env_id):
        env = UnityEnv(env_id, multiagent=True)
        self.envs = [fn() for fn in [lambda: env]]
        # env = self.envs[0]
        # VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        VecEnv.__init__(self, env.number_agents, env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        else:
            subspaces = {None: obs_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        # actions = [self.action_space.sample() for agent in range(self.num_envs)]
        self.actions = actions.tolist()

    def step_wait(self):
        obs, rewards, dones, info = self.envs[0].step(self.actions)
        # self.buf_rews = rewards
        # self.buf_dones = dones
        # self.buf_infos = info
        for env_idx in range(self.num_envs):
            self.buf_rews[env_idx] = rewards[env_idx]
            self.buf_dones[env_idx] = dones[env_idx]
            # if self.buf_dones[env_idx]:
                # obs = self.envs[0].reset()[0]
            self._save_obs(env_idx, obs[env_idx])
            if len(info['brain_info'].action_masks) > 0:
                self.buf_infos[env_idx]['action_masks'] = info['brain_info'].action_masks[env_idx]
            if len(info['brain_info'].agents) > 0:
                self.buf_infos[env_idx]['agents'] = info['brain_info'].agents[env_idx]
            if len(info['brain_info'].local_done) > 0:
                self.buf_infos[env_idx]['local_done'] = info['brain_info'].local_done[env_idx]
            if len(info['brain_info'].max_reached) > 0:
                self.buf_infos[env_idx]['max_reached'] = info['brain_info'].max_reached[env_idx]
            if len(info['brain_info'].memories) > 0:
                self.buf_infos[env_idx]['memories'] = info['brain_info'].memories[env_idx]
            if len(info['brain_info'].previous_text_actions) > 0:
                self.buf_infos[env_idx]['previous_text_actions'] = info['brain_info'].previous_text_actions[env_idx]
            if len(info['brain_info'].previous_vector_actions) > 0:
                self.buf_infos[env_idx]['previous_vector_actions'] = info['brain_info'].previous_vector_actions[env_idx]
            if len(info['brain_info'].rewards) > 0:
                self.buf_infos[env_idx]['rewards'] = info['brain_info'].rewards[env_idx]
            if len(info['brain_info'].text_observations) > 0:
                self.buf_infos[env_idx]['text_observations'] = info['brain_info'].text_observations[env_idx]
            if len(info['brain_info'].vector_observations) > 0:
                self.buf_infos[env_idx]['vector_observations'] = info['brain_info'].vector_observations[env_idx]
            if len(info['brain_info'].visual_observations) > 0:
                self.buf_infos[env_idx]['visual_observations'] = info['brain_info'].visual_observations[env_idx]

        return (np.copy(self._obs_from_buf()), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        obs = self.envs[0].reset()
        for env_idx in range(self.num_envs):
            self._save_obs(env_idx, obs[env_idx])
        return np.copy(self._obs_from_buf())

    def close(self):
        return

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs
