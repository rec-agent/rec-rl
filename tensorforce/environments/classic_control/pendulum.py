# -*- coding: utf-8 -*-

"""
    desc : the pendulum emulator
    create: 2017.12.11
    @author: sam.dm
"""

import numpy as np
import tensorforce.core.lib.env_seeding as seeding
from tensorforce.environments import Environment

class Pendulum(Environment):
    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05

        self.high = np.array([1., 1., self.max_speed])

        self.seed()

    def __str__(self):
        return "Pendulum"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def execute(self,actions):
        th, thdot = self.state # th := theta

        g, m, l = 10.0, 1.0, 1.0
        dt = self.dt

        action = np.clip(actions, -self.max_torque, self.max_torque)[0]
        costs = self._angle_normalize(th)**2 + .1*thdot**2 + .001*(action**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*action) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), False, -costs

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    @property
    def state_space(self):
        state = dict(shape=3, type='float')

        return state

    @property
    def action_space(self):
        action = dict(type='float', min_value=-self.max_torque, max_value=self.max_torque)

        return action

    def state_contains(self, state):
        cons = [np.abs(x)<=y for x,y,z in zip(state, self.high)]
        return all(cons)

    def action_contains(self, action):
        cons = np.abs(action[0]) <= self.max_torque

        return cons

    def close(self):
        self.state = None
