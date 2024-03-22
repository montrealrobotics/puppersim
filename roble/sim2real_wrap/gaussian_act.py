import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, WrapperObsType, ActionWrapper, WrapperActType, ActType


class GaussianActWrapper(ActionWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def action(self, action: WrapperActType) -> ActType:
        return self._clip_act(self._noise_act(action))
        pass

    def _clip_act(self, act):
        return np.clip(act, a_min=self.action_space.low, a_max=self.action_space.high)

    def _noise_act(self, act):
        return np.random.normal(loc=act, scale=self.scale)
