"""
    desc: SGD and Adam optimizer numpy implement.
    create: 2018.01.18
    @author: sam.dm
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorforce import util, TensorforceError
from tensorforce.core.lib import schedules


def from_spec(spec, kwargs=None):
    """
    Creates an optimizer from a specification dict.
    """
    optimizer = util.get_object(
        obj=spec,
        predefined_objects=optimizers,
        kwargs=kwargs
    )
    assert isinstance(optimizer, Optimizer)
    return optimizer


class Optimizer(object):
    def __init__(self, dim):
        self.dim = dim
        self.t = 0

    def update(self, grad):
        self.t += 1
        step = self._compute_step(grad)
        return step

    def _compute_step(self, grad):
        raise NotImplementedError


class Momentum(Optimizer):
    def __init__(self, dim, learning_rate, momentum=0.9, lr_schedule=None):
        Optimizer.__init__(self, dim)
        if lr_schedule is not None:
            lr_schedule['value'] = learning_rate
            self.decay_obj = schedules.from_spec(lr_schedule)
        self.lr_schedule = lr_schedule
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.learning_rate, self.momentum = learning_rate, momentum

    def _compute_step(self, globgrad):
        self.v = self.momentum * self.v + (1. - self.momentum) * grad
        if self.lr_schedule is not None:
            self.learning_rate = self.decay_obj(self.t)
        step = -self.learning_rate * self.v
        return step


class Adam(Optimizer):
    def __init__(self, dim, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, lr_schedule=None):
        Optimizer.__init__(self, dim)
        if lr_schedule is not None:
            lr_schedule['value'] = learning_rate
            self.decay_obj = schedules.from_spec(lr_schedule)
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        if isinstance(self.learning_rate, list):
            self.learning_rate = np.asarray(self.learning_rate, dtype=np.float32).flatten()
            assert self.learning_rate.size == self.dim
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, grad):
        if self.lr_schedule is not None:
            self.learning_rate = self.decay_obj(self.t)
        a = self.learning_rate * (np.sqrt(1 - self.beta2 ** self.t) /
                             (1 - self.beta1 ** self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) *grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

optimizers = {"adam": Adam, "momentum": Momentum}
