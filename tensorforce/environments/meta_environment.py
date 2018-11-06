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

from tensorforce.environments.environment import Environment
from tensorforce.exception import TensorforceError


class MetaEnvironment(Environment):
    """
    Base class for unified IO interface
    """

    def __init__(self, config):
        super(MetaEnvironment, self).__init__()
        self._parse(config)

    def _parse(self, config):
        """
        Base class for configuration parsing
        """
        # get the type of IO,
        # optional params are ('Table','DataHub','Gym','Universe','UserDef')
        if not 'env_type' in config:
            raise TensorforceError('can not find env_type in configuration')
        self.env_type = config['env_type']

        if 'env' in config:
            self.env_conf = config['env']
        else:
            raise TensorforceError('can not find env config')

        # whether task is in mode of interaction
        # default mode is non-interaction
        self.interactive = False
        if 'interactive' in self.env_conf:
            self.interactive = self.env_conf['interactive']


    def parse_env_config(self):
        """
        IO specific parsing function
        """
        raise NotImplementedError()

    def get_input_tensor(self):
        """
        Init a dict of single state_input tensor,action tensor,reward tensor
        or a dict of state_input tensor if multi-states are provided
        the return will be used to initialize the agent
        """
        raise NotImplementedError()

    def read(self):
        """
        Read a batch data for model update
        this method only be used in mode of non-interaction
        Call execute in mode of interaction
        """
        raise NotImplementedError()

    def should_stop(self):
        """
        In mode of non-interaction,
        should_stop() will be called in Runner.consumer() to determine whether to end the trianing loop
        this method only be used in mode of non-interaction
        """
        raise NotImplementedError()