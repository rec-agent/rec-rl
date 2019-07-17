# -*- coding: utf-8 -*-

"""
    desc: learning rate decayer.
    created: 2017.12.27
    @author: sam.dm
"""

import tensorflow as tf
from tensorforce import util, TensorforceError


def from_spec(spec, kwargs=None):
    lr_schedule = util.get_object(
        obj=spec,
        predefined_objects=lr_schedulers,
        kwargs=kwargs
    )
    assert isinstance(lr_schedule, DecaySchedule)

    return lr_schedule


def add_lr_decay(spec, global_step, kwargs=None):
    """
    Creates an learning rate decayed instance from a optimizer specification dict.
    """

    def parse_decay_conf(optimizer_spec, global_step):
        lr = optimizer_spec['learning_rate']
        lr_schedule = optimizer_spec['lr_schedule']
        if lr_schedule is None:
            del optimizer_spec['lr_schedule']
            return optimizer_spec
        lr_schedule['global_step'] = global_step
        lr_decay_obj = from_spec(lr_schedule)
        optimizer_spec['learning_rate'] = lr_decay_obj(value=lr)
        pop_value = optimizer_spec.pop('lr_schedule', None)
        return optimizer_spec

    if 'optimizer' in spec:
        optimizer_spec = spec['optimizer']
        if 'learning_rate' in optimizer_spec and 'lr_schedule' in optimizer_spec:
            spec['optimizer'] = parse_decay_conf(optimizer_spec, global_step)

    elif 'learning_rate' in spec and 'lr_schedule' in spec:
        spec = parse_decay_conf(spec, global_step)

    return spec

class DecaySchedule(object):

    def __call(self, value):

        raise NotImplementedError()

class Constant(DecaySchedule):
    def __init__(self, global_step=None):
        """
            decayed_value = value
        """

        self._global_step = global_step

    def __call__(self, value):

        return value


class TFExponentialDecay(DecaySchedule):
    def __init__(self, global_step, decay_steps=20000, decay_rate=0.96, staircase=False):
        """
            decayed_value = value * decay_rate ^ (global_step / decay_steps)
        """

        self._global_step = global_step
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def __call__(self, value):

        decayed_value = tf.train.exponential_decay(value, self._global_step,
                                       self._decay_steps, self._decay_rate, self._staircase)
        return decayed_value


class TFInverseTimeDecay(DecaySchedule):
    def __init__(self, global_step, decay_steps=20000, decay_rate=0.96, staircase=False):
        """
            decayed_value = value / (1 + decay_rate * t)
        """

        self._global_step = global_step
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def __call__(self, value):

        decayed_value = tf.train.inverse_time_decay(value, self._global_step,
                                       self._decay_steps, self._decay_rate, self._staircase)
        return decayed_value


class TFNaturalExpDecay(DecaySchedule):
    def __init__(self, global_step, decay_steps=20000, decay_rate=0.96, staircase=False):
        """
            decayed_value = value * exp(-decay_rate * (global_step / decay_steps))
        """

        self._global_step = global_step
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def __call__(self, value):

        decayed_value = tf.train.natural_exp_decay(value, self._global_step,
                                       self._decay_steps, self._decay_rate, self._staircase)
        return decayed_value


class TFPolynomialDecay(DecaySchedule):
    def __init__(self, global_step, decay_steps=20000, final_value=0.0001,
            power=1.0, cycle=False):
        """
        global_step = min(global_step, decay_steps)
        decayed_final_value = (final_value - final_value) *
                        (1 - global_step / decay_steps) ^ (power) +
                        final_value
        """

        self._global_step = global_step
        self._decay_steps = decay_steps
        self._final_value = final_value
        self._power = power
        self._cycle = cycle

    def __call__(self, value):
        decayed_value = tf.train.polynomial_decay(value, self._global_step,
                                       self._decay_steps, self._final_value,
                                       self._power, self._cycle)
        return decayed_value

class LinearDecay(DecaySchedule):
    def __init__(self, global_step, max_decay_steps=20000, final_value=0.0001):
        """
        decayed_value = init_value + (global_step / max_decay_steps) * (
                       init_value - final_value)
        """

        self._global_step = global_step
        self._max_decay_steps = tf.constant(value=max_decay_steps, dtype=tf.int32)
        self._final_value = final_value
        self._first_pass = True

    def __call__(self, value):
        if self._first_pass:
            self._init_value = value
            self._first_pass = False

        self.fraction = tf.minimum(tf.divide(self._global_step, self._max_decay_steps), 1.0)

        return self._init_value + tf.multiply(self.fraction, self._final_value - self._init_value)


lr_schedulers = {
    "constant": Constant,
    "exp_decay": TFExponentialDecay,
    "natural_exp_decay": TFNaturalExpDecay,
    "inverse_time_decay": TFInverseTimeDecay,
    "polynomial_decay": TFPolynomialDecay,
    "linear_decay": LinearDecay
}
