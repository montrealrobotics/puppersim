"""A task to teach pupper to stand still"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import gin
from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils
from puppersim import pupper_v2

@gin.configurable
class SimpleStandTask(task_interface.Task):
    def __init__(self, 
                 weight=1.0,
                 min_com_height=0.0,
                 terminal_condition=terminal_conditions.default_terminal_condition_for_minitaur):
        
        self.weight = weight
        self._min_com_height = min_com_height
        self._terminal_condition = terminal_condition
        self._env = None
        self._step_count = 0
        self.TARGET_MOTOR_ANGLES = np.array(pupper_v2.Pupper.get_neutral_motor_angles())

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        self._env = env

    def update(self, env):
        del env

    def reward(self, env):
        del env

        self._step_count += 1
        env = self._env

        current_motor_angles = self._env.robot.motor_angles
        reward = -(np.sum(np.abs(self.TARGET_MOTOR_ANGLES - current_motor_angles)))

        return reward * self.weight

    def done(self, env):
        del env
        position = env_utils.get_robot_base_position(self._env.robot)
        if self._min_com_height and position[2] < self._min_com_height:
            return True
        return self._terminal_condition(self._env)


    @property
    def step_count(self):
        return self._step_count
