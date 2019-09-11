# -*- coding:utf-8 -*-

"""
    desc: classic cart-pole.
    create: 2017.12.11
    author: @sam.dm
"""


import math
import numpy as np
import tensorforce.core.lib.env_seeding as seeding
from tensorforce.environments import Environment


class CartPole(Environment):

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.seed()
        self.state = None
        self.steps_beyond_done = None

    def __str__(self):
        return "CartPole"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def execute(self, actions):
        assert self._action_contains(actions), "%r (%s) invalid"%(actions, type(actions))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if actions==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print("You are calling 'step()' even though this environment    \
                       has already returned done = True. You should always call \
                       'reset()' once you receive 'done = True' -- any further  \
                       steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), done, reward

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def close(self):
        self.state = None
        self.steps_beyond_done = None

    def _state_contains(self, state):
        cons = [np.abs(x)<=y for x,y,z in zip(state, self.high)]

        return all(cons)

    def _action_contains(self, action):
        cons = action>=0 and action < 2

        return cons

    @property
    def state_space(self):
        state = dict(shape=4, type='float')

        return state

    @property
    def action_space(self):
        action = dict(type='int', num_actions=2)

        return action
