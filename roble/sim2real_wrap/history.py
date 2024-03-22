import collections

import gymnasium
import numpy as np
from gym import Wrapper


class HistoryWrapper(Wrapper):
    def __init__(self, env, length):
        super().__init__(env)
        self.length = length

        low, high = env.observation_space.low, env.observation_space.high
        low = np.array([[low] * length]).squeeze().flatten()
        high = np.array([[high] * length]).squeeze().flatten()

        self.observation_space = gymnasium.spaces.Box(low=low,
                                                      high=high)
        self._reset_buf()

    def _reset_buf(self):
        self._buf = collections.deque(maxlen=self.length)
        for _ in range(self.length+1):
            self._buf.append(self.env.observation_space.sample() * 0)

    def _make_observation(self):
        ret = np.concatenate(list(self._buf)).squeeze().flatten()
        return ret

    def reset(self, **kwargs):
        self._reset_buf()
        ret = super(HistoryWrapper, self).reset(**kwargs)
        self._buf.append(ret[0])
        return self._make_observation(), ret[1]

    def step(self, action):
        ret = super(HistoryWrapper, self).step(action)
        self._buf.append(ret[0])

        return self._make_observation(), *ret[1:]

class HistoryWrapperSTUDENT(Wrapper):
    def __init__(self, env, length):
        super().__init__(env)
        self.length = length

        # todo ??? obs space

        self._reset_buf()

    def _reset_buf(self):
        # todo ???
        pass

    def _make_observation(self):
        # todo ???
        pass

    def reset(self, **kwargs):
        self._reset_buf()
        ret = super(HistoryWrapper, self).reset(**kwargs)

        # todo ???
        #ret = self._make_observation()
        return ret

    def step(self, action):
        ret = super(HistoryWrapper, self).step(action)

        # todo ??? replace obs with the history?
        # add obs to self.buffer and then
        # obs = self._make_observation()

        return ret
