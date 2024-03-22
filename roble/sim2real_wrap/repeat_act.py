import numpy as np
from gymnasium.core import Wrapper


class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, min_repeat, max_repeat):
        super().__init__(env)
        self.min_repeat= min_repeat
        self.max_repeat= max_repeat

        assert min_repeat <= max_repeat
        assert min_repeat > 0

    def _sample_num_repeat(self):
        return np.random.uniform(self.min_repeat, self.max_repeat)

    def step(self, action):
        for _ in range(self._sample_num_repeat()):
            ret = super(ActionRepeatWrapper, self).step(action)
        return ret
