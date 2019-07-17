# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from random import randrange
import numpy as np

from tensorforce import util
from tensorforce.core.memories import Memory


class Replay(Memory):
    """
    Replay memory to store observations and sample mini batches for training from.
    """

    def __init__(self, states_spec, actions_spec, capacity, random_sampling=True):
        super(Replay, self).__init__(states_spec=states_spec, actions_spec=actions_spec)
        self.capacity = capacity
        self.states = {name: np.zeros((capacity,) + tuple(state['shape']), dtype=util.np_dtype(state['type']))
            for name, state in states_spec.items()}
        self.next_states = {name: np.zeros((capacity,) + tuple(state['shape']), dtype=util.np_dtype(state['type']))
            for name, state in states_spec.items()}
        self.internals, self.next_internals = None, None
        self.actions = {name: np.zeros((capacity,) + tuple(action['shape']), dtype=util.np_dtype(action['type']))
            for name, action in actions_spec.items()}
        self.terminal = np.zeros((capacity,), dtype=util.np_dtype('bool'))
        self.reward = np.zeros((capacity,), dtype=util.np_dtype('float'))

        self.size = 0
        self.index = 0
        self.random_sampling = random_sampling

    def add_observation(self, states, internals, actions, terminal, reward, next_states, next_internals):
        if self.internals is None:
            self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal in internals]
        if self.next_internals is None:
            self.next_internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal in next_internals]

        for name, state in states.items():
            self.states[name][self.index] = state
        for name, next_state in next_states.items():
            self.next_states[name][self.index] = next_state
        for n, internal in enumerate(internals):
            self.internals[n][self.index] = internal
        for n, next_internal in enumerate(next_internals):
            self.next_internals[n][self.index] = next_internal
        for name, action in actions.items():
            self.actions[name][self.index] = action
        self.reward[self.index] = reward
        self.terminal[self.index] = terminal

        if self.size < self.capacity:
            self.size += 1
        self.index = (self.index + 1) % self.capacity

    def get_batch(self, batch_size):
        """
        Samples a batch of the specified size by selecting a random start/end point and returning
        the contained sequence or random indices depending on the field 'random_sampling'.

        Args:
            batch_size: The batch size
            next_states: A boolean flag indicating whether 'next_states' values should be included

        Returns: A dict containing states, actions, rewards, terminals, internal states (and next states)

        """
        indices = np.random.randint(self.size - 1, size=batch_size)
        terminal = self.terminal.take(indices)

        states = {name: state.take(indices, axis=0) for name, state in self.states.items()}
        internals = [internal.take(indices, axis=0) for internal in self.internals]
        actions = {name: action.take(indices, axis=0) for name, action in self.actions.items()}
        terminal = self.terminal.take(indices)
        reward = self.reward.take(indices)
        next_states = {name: state.take(indices, axis=0) for name, state in self.next_states.items()}
        next_internals = [internal.take(indices, axis=0) for internal in self.next_internals]

        batch = dict(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward,
            next_states=next_states, next_internals=next_internals)
        return batch

    def set_memory(self, states, internals, actions, terminal, reward, next_states, next_internals):
        """
        Convenience function to set whole batches as memory content to bypass
        calling the insert function for every single experience.

        """
        self.size = len(terminal)

        if len(terminal) == self.capacity:
            # Assign directly if capacity matches size.
            for name, state in states.items():
                self.states[name] = np.asarray(state)
            for name, state in next_states.items():
                self.next_states[name] = np.asarray(state)
            self.internals = [np.asarray(internal) for internal in internals]
            self.next_internals = [np.asarray(internal) for internal in next_internals]
            for name, action in actions.items():
                self.actions[name] = np.asarray(action)
            self.terminal = np.asarray(terminal)
            self.reward = np.asarray(reward)
            # Filled capacity to point of index wrap
            self.index = 0

        else:
            # Otherwise partial assignment.
            if self.internals is None:
                self.internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal
                                  in internals]
            if self.next_internals is None:
                self.next_internals = [np.zeros((self.capacity,) + internal.shape, internal.dtype) for internal
                                  in next_internals]

            for name, state in states.items():
                self.states[name][:len(state)] = state
            for name, state in next_states.items():
                self.next_states[name][:len(state)] = state
            for n, internal in enumerate(internals):
                self.internals[n][:len(internal)] = internal
            for n, next_internal in enumerate(next_internals):
                self.next_internals[n][:len(internal)] = next_internal
            for name, action in actions.items():
                self.actions[name][:len(action)] = action
            self.terminal[:len(terminal)] = terminal
            self.reward[:len(reward)] = reward
            self.index = len(terminal)

    def update_batch(self, idxes, priorities):
        pass
