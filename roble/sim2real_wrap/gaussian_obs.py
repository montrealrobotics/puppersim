import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, WrapperObsType


class GaussianObsWrapper(ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        return self._clip_obs(self._noise_obs(observation))

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def _clip_obs(self, obs):
        return np.clip(obs, a_min=self.observation_space.low, a_max=self.observation_space.high)

    def _noise_obs(self, obs):
        return np.random.normal(loc=obs, scale=self.scale)
