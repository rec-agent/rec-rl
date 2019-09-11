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


from tensorforce.environments.environment import Environment
from tensorforce.environments.meta_environment import MetaEnvironment
from tensorforce.environments.classic_control import CartPole
from tensorforce.environments.classic_control import Pendulum
import six

environments = dict(
    cart_pole=CartPole,
    pendulum=Pendulum
)

__all__ = ['Environment', 'MetaEnvironment', 'Pendulum', 'CartPole']