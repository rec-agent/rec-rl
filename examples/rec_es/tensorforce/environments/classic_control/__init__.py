# -*- coding:utf-8 -*-

"""
    desc: classic control environments.
    create: 2017.12.19
    modified by @sam.dm
"""


from tensorforce.environments.classic_control.cart_pole import CartPole
from tensorforce.environments.classic_control.pendulum import Pendulum

environments = dict(
    cart_pole=CartPole,
    pendulum=Pendulum,
)

__all__ = ['Pendulum', 'CartPole']
