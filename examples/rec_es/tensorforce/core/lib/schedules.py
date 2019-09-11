# -*- coding: utf-8 -*-

"""
    desc: schedule type, eg: learning rate, priority beta epsilon etc.
    created: 2017.12.11
    @author: sam.dm
"""
import math
from tensorforce import util, TensorforceError


def from_spec(spec, kwargs=None):
    lr_schedule = util.get_object(
        obj=spec,
        predefined_objects=lr_schedulers,
        kwargs=kwargs
    )
    assert isinstance(lr_schedule, Schedule)

    return lr_schedule


class Schedule(object):
    def __call__(self, global_step):
        """
        Value of the schedule at time t
        """

        raise NotImplementedError()


class Constant(Schedule):
    def __init__(self, value):
        """
        Value remains constant over time.
        Args:
            value: float, Constant value of the schedule
        """

        self._value = value

    def __call__(self, global_step):

        return self._value


class PiecewiseDecay(Schedule):
    def __init__(self, endpoints, outside_value=None):
        """
        Piecewise decay schedule.
        Args:
            endpoints: [(int, int)], list of pairs (time, value) meanining that schedule should output
                value when t==time. All the values for time must be sorted in an increasing order.
            outside_value: float, if the value is requested outside of all the intervals sepecified in
                endpoints this value is returned.
        """

        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def _linear_interpolation(self, l, r, alpha):

        return l + alpha * (r - l)

    def __call__(self, global_step):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= global_step and global_step < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None

        return self._outside_value


class LinearDecay(Schedule):
    def __init__(self, value, max_decay_steps, final_value):
        """
        Linear interpolation between initial_value and final_value over schedule_timesteps.
        Args:
            max_timesteps: int, Number of max schedule timesteps.
            value: float, initial output value
            final_value: float, final output value
        """

        self._max_decay_steps = max_decay_steps
        self._initial_value = value
        self._final_value = final_value

    def __call__(self, global_step):
        fraction = min(float(global_step) / self._max_decay_steps, 1.0)

        return self._initial_value + fraction * (self._final_value - self._initial_value)


class ExponentialDecay(Schedule):
    def __init__(self, value, decay_steps, decay_rate, staircase=False):
        """
            decayed_value = value * decay_rate ^ (global_step / decay_steps)
        """

        self._value = value
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def __call__(self, global_step):
        p = float(global_step) / self._decay_steps
        if self._staircase:
             p = math.floor(p)

        return self._value * math.pow(self._decay_rate, p)

class PolynomialDecay(Schedule):
    def __init__(self, value, decay_steps, final_value=0.0001, power=1.0, cycle=False):
        """
        global_step = min(global_step, decay_steps)
        decayed_value = (value - final_value) *
                          (1 - global_step / decay_steps) ^ (power) +
                          final_value
        If cycle is True then a multiple of decay_steps is used, the first one
        that is bigger than global_steps.

        decay_steps = decay_steps * ceil(global_step / decay_steps)
        decayed_value = (value - final_value) * (1 - global_step / decay_steps) ^ (power) +
                        final_value
        """

        self._value = value
        self._decay_steps = decay_steps
        self._final_value = final_value
        self._power = power
        self._cycle = cycle

    def __call__(self, global_step):
        if self._cycle:
            if global_step == 0:
                multiplier = 1.0
            else:
                multiplier = math.ceil(global_step / self._decay_steps)
            decay_steps = self._decay_steps * multiplier
        else:
            decay_steps = self._decay_steps
            global_step = min(global_step, self._decay_steps)

        p = float(global_step) / decay_steps

        return (self._value - self._final_value) * math.pow(
            1 - p, self._power) + self._final_value

class NaturalExpDecay(Schedule):
    def __init__(self, value, decay_steps, decay_rate, staircase=False):
        """
        decayed_value = value * exp(-decay_rate * global_step)
        """

        self._value = value
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def __call__(self, global_step):
        p = float(global_step) / self._decay_steps
        if self._staircase:
            p = math.ceil(p)
        exponent = math.exp(-self._decay_rate * p)

        return self._value * exponent

class InverseTimeDecay(Schedule):
    def __init__(self, value, decay_steps, decay_rate, staircase=False):
        """
          decayed_value = value / (1 + decay_rate * global_step / decay_step)

          if staircase is True, as:
          decayed_value = value / (1 + decay_rate * floor(global_step / decay_step))
        """

        self._value = value
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def __call__(self, global_step):
        p = float(global_step) / self._decay_steps
        if self._staircase:
            p = math.ceil(p)
        denom = 1.0 + p * self._decay_rate

        return self._value / denom

lr_schedulers = {
    "constant": Constant,
    "exp_decay": ExponentialDecay,
    "natural_exp_decay": NaturalExpDecay,
    "inverse_time_decay": InverseTimeDecay,
    "polynomial_decay": PolynomialDecay,
    "linear_decay": LinearDecay
}
