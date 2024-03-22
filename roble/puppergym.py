import os

import gin
import gymnasium
import gymnasium as gym
import numpy as np
import torch
from gymnasium import Wrapper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.wrappers import TimeLimit
from pybullet_envs.minitaur.envs_v2 import env_loader
from tqdm import tqdm

import puppersim


TIMELIMIT = 1_000

def make_pupper_task():
    CONFIG_DIR = puppersim.getPupperSimPath()
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "../puppersim/config", "pupper_pmtg.gin")
    #  _NUM_STEPS = 10000
    #  _ENV_RANDOM_SEED = 2

    import puppersim.data as pd
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    env = env_loader.load()

    class GymnasiumWrapper(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gymnasium.spaces.Box(low=env.observation_space.low, high=env.observation_space.high)
            self.action_space = gymnasium.spaces.Box(low=env.action_space.low, high=env.action_space.high)

        @property
        def render_mode(self):
            return "rgb_array"

        def reset(self, **kwargs):
            return self.env.reset(), {}

        def step(self, action):
            return convert_to_terminated_truncated_step_api(self.env.step(action))

        def render(self, render_mode=None):
            return self.env.render(mode=self.render_mode)

    env = GymnasiumWrapper(env)
    return env


def get_env_thunk(seed, sim2real_wrap, idx, capture_video, video_save_path, timelimit=TIMELIMIT):
    def thunk():
        env = make_pupper_task() # replace with gym.make todo @ guillaume et jaydan

        env = TimeLimit(env, TIMELIMIT)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{video_save_path}")

        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)

        env = sim2real_wrap(env)

        env.action_space.seed(seed)
        return env

    return thunk

def _identity(env):
    return env

def make_vector_env(seed, capture_video, video_save_path, num_vector=10, sim2real_wrap=_identity):
    envlist = []
    for i in range(num_vector):
        envlist.append(get_env_thunk(seed, sim2real_wrap, i, capture_video, video_save_path))
    envs = gym.vector.AsyncVectorEnv(envlist)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs


def evaluate(
        agent,
        make_env,
        video_save_path,
        eval_episodes: int=5,
        timelimit=1000
):
    print("EVALUATING")
    envs = make_env(0, True, video_save_path)

    with torch.no_grad():
        obs, _ = envs.reset()
        episodic_returns = []
        episodic_lengths = []
        while len(episodic_returns) < eval_episodes:
            for timestep in tqdm(range(TIMELIMIT + 1)):
                actions = agent.get_action(torch.Tensor(obs).to(agent.device))[0]
                next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info is None:
                            continue
                        if "episode" not in info:
                            continue
                        print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                        episodic_returns += [info["episode"]["r"]]
                        episodic_lengths += [info["episode"]["l"]]
            obs = next_obs

    episodic_returns = [float(i) for i in episodic_returns]
    episodic_lengths = [float(i) for i in episodic_lengths]
    return np.array(episodic_returns).mean(), np.array(episodic_lengths).mean()
