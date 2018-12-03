from collections import OrderedDict

import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv
from gym_unity.envs import UnityEnv
import time
import csv
import os.path as osp
import json
import os


MarathonEnvs = {
    'MarathonHopperEnv-v0': 'hopper',
    'MarathonWalker2DEnv-v0': 'walker'
}

class UnityVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) the list of environments to vectorize
    """

    @staticmethod
    def GetFilePath(env_id, inference_mode=False, n_agents=1):
        import psutil
        env_name = MarathonEnvs[env_id]
        if inference_mode:
            env_name = env_name + '-run'
        elif n_agents is 16:
            env_name = env_name + '-x16'
        if psutil.MACOS:
            env_path = os.path.join('envs', env_name)
        elif psutil.WINDOWS:
            env_path = os.path.join('envs', env_name, 'Unity Environment.exe')
        return env_path
    
    def __init__(self, env_id, n_agents):
        env_path = UnityVecEnv.GetFilePath(env_id, n_agents=n_agents)
        print ("**** ", env_path)
        env = UnityEnv(env_path, multiagent=True)
        self.env = env
        env.num_envs = env.number_agents
        VecEnv.__init__(self, env.num_envs, env.observation_space, env.action_space)
        obs_space = env.observation_space

        # self.keys, shapes, dtypes = obs_space_info(obs_space)
        # self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        # self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        # self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        # Fake Monitor
        self.tstart = time.time()
        self.results_writer = ResultsWriter(
            "filename",
            header={"t_start": time.time(), 'env_id' : env.spec and env.spec.id},
            extra_keys=() + ()
        )
        self.reset_keywords = ()
        self.info_keywords = ()
        self.allow_early_resets = True
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # obs, self.buf_rews, self.buf_dones, self.buf_infos = self.env.step(action)
        obs, rews, dones, infos = self.env.step(self.actions.tolist())
        obs = np.stack(obs)
        rews = np.stack(rews)
        dones = np.stack(dones)
        self.fix_info(infos)
        self.update(obs[0], rews[0], dones[0], self.buf_infos[0])
        return (obs, rews, dones, self.buf_infos)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            self.results_writer.write_row(epinfo)
            self.rewards = []
            self.needs_reset = False
            if isinstance(info, dict):
                info['episode'] = epinfo

    def fix_info(self, info):
        for env_idx in range(self.num_envs):
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
        return self.buf_infos    


    def reset(self):
        self.reset_state()
        obs = self.env.reset()
        obs = np.stack(obs)
        return obs

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False

    # def _save_obs(self, e, obs):
    #     for k in self.keys:
    #         if k is None:
    #             self.buf_obs[k][e] = obs
    #         else:
    #             self.buf_obs[k][e] = obs[k]

    # def _obs_from_buf(self):
    #     return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [self.env.render(mode='rgb_array')]

    def render(self, mode='human'):
        super().render(mode=mode)

    def close(self):
        return

class ResultsWriter(object):
    def __init__(self, filename=None, header='', extra_keys=()):
        self.extra_keys = extra_keys
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith("monitor.csv"):
                if osp.isdir(filename):
                    filename = osp.join(filename, "monitor.csv")
                else:
                    filename = filename + "." + "monitor.csv"
            self.f = open(filename, "wt")
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()
