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
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorforce import util
from tensorforce.core.preprocessing import Preprocessor


class Standardize(Preprocessor):
    """
    Standardize state. Subtract mean and divide by standard deviation.
    """

    def __init__(self, mean=None, var=None, across_batch=False, scope='standardize', summary_labels=()):
        self.across_batch = across_batch
        self.mean = mean
        self.var = var

        super(Standardize, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_process(self, tensor):
        if self.mean is not None and self.var is not None:
            return (tensor - self.mean) / (self.var + util.epsilon)

        if self.across_batch:
            axes = tuple(range(util.rank(tensor)))
        else:
            axes = tuple(range(1, util.rank(tensor)))

        mean, variance = tf.nn.moments(x=tensor, axes=axes, keep_dims=True)
        return (tensor - mean) / tf.maximum(x=tf.sqrt(variance), y=util.epsilon)
