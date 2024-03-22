# adds sim2real wrappers
from roble.sim2real_wrap.gaussian_act import GaussianActWrapper
from roble.sim2real_wrap.gaussian_obs import GaussianObsWrapper
from roble.sim2real_wrap.history import HistoryWrapper
from roble.sim2real_wrap.last_action import LastActionWrapper


thunk_sim2real_wrap = None

def make_thunk(cfg):
    global thunk_sim2real_wrap

    def sim2real_wrap(env):
        env = HistoryWrapper(env, length=2) # todo
        env = LastActionWrapper(env)
        env = GaussianObsWrapper(env, scale=0.01)
        env = GaussianActWrapper(env, scale=0.01)
        return env

    thunk_sim2real_wrap = sim2real_wrap
    return thunk_sim2real_wrap