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

"""
Collection of custom layer implementations. We prefer not to use contrib-layers to retain full control over shapes and internal states.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import sqrt
import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
import tensorforce.core.networks


class Layer(object):
    """
    Base class for network layers.
    """

    def __init__(self, num_internals=0, scope='layer', summary_labels=None):
        self.num_internals = num_internals
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()
        self.all_variables = dict()
        self.summaries = list()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                self.all_variables[name] = variable
                if kwargs.get('trainable', True) and not name.startswith('optimization'):
                    self.variables[name] = variable
                    if 'variables' in self.summary_labels:
                        summary = tf.summary.histogram(name=name, values=variable)
                        self.summaries.append(summary)
            return variable

        self.apply = tf.make_template(
            name_=(scope + '/apply'),
            func_=self.tf_apply,
            custom_getter_=custom_getter
        )
        self.regularization_loss = tf.make_template(
            name_=(scope + '/regularization-loss'),
            func_=self.tf_regularization_loss,
            custom_getter_=custom_getter
        )

    def tf_apply(self, x, update):
        """
        Creates the TensorFlow operations for applying the layer to the given input.

        Args:
            x: Layer input tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Layer output tensor.
        """
        raise NotImplementedError

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the layer regularization loss.

        Returns:
            Regularization loss tensor.
        """
        return None

    def internals_input(self):
        """
        Returns the TensorFlow placeholders for internal state inputs.

        Returns:
            List of internal state input placeholders.
        """
        return list()

    def internals_init(self):
        """
        Returns the TensorFlow tensors for internal state initializations.

        Returns:
            List of internal state initialization tensors.
        """
        return list()

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the layer.

        Returns:
            List of variables.
        """
        if include_non_trainable:
            return [self.all_variables[key] for key in sorted(self.all_variables)]
        else:
            return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the layer.

        Returns:
            List of summaries.
        """
        return self.summaries

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates a layer from a specification dict.
        """
        layer = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.networks.layers,
            kwargs=kwargs
        )
        assert isinstance(layer, Layer)
        return layer


class Nonlinearity(Layer):
    """
    Non-linearity layer applying a non-linear transformation.
    """

    def __init__(self, name='relu', scope='nonlinearity', summary_labels=()):
        """
        Non-linearity layer.

        Args:
            name: Non-linearity name, one of 'elu', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'tanh' or 'none'.
        """
        self.name = name
        super(Nonlinearity, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if self.name == 'elu':
            x = tf.nn.elu(features=x)

        elif self.name == 'none':
            x = tf.identity(input=x)

        elif self.name == 'relu':
            x = tf.nn.relu(features=x)
            if 'relu' in self.summary_labels:
                non_zero = tf.cast(x=tf.count_nonzero(input_tensor=x), dtype=tf.float32)
                size = tf.cast(x=tf.reduce_prod(input_tensor=tf.shape(input=x)), dtype=tf.float32)
                summary = tf.summary.scalar(name='relu', tensor=(non_zero / size))
                self.summaries.append(summary)

        elif self.name == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            negative = alpha * tf.nn.elu(features=x)
            x = scale * tf.where(condition=(x >= 0.0), x=x, y=negative)

        elif self.name == 'sigmoid':
            x = tf.sigmoid(x=x)

        elif self.name == 'softmax':
            x = tf.nn.softmax(logits=x)

        elif self.name == 'softplus':
            x = tf.nn.softplus(features=x)

        elif self.name == 'tanh':
            x = tf.nn.tanh(x=x)

        else:
            raise TensorforceError('Invalid non-linearity: {}'.format(self.name))

        return x


class Dropout(Layer):
    """
    Dropout layer. If using dropout, add this layer after inputs and after dense layers. For
    LSTM, dropout is handled independently as an argument. Not available for Conv2d yet.
    """

    def __init__(self, rate=0.0, scope='dropout', summary_labels=()):
        self.rate = rate
        super(Dropout, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        return tf.cond(
            pred=update,
            true_fn=(lambda: tf.nn.dropout(x=x, keep_prob=(1.0 - self.rate))),
            false_fn=(lambda: x)
        )

class Layernorm(Layer):
    """
    Layer normalization. Can be used as a normalizer function for conv2d and fully_connected.
    [Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton](https://arxiv.org/abs/1607.06450)
    """

    def __init__(self, center=True, scale=True, activation_fn=None, scope='layer_norm', summary_labels=()):
        self.center = center
        self.scale = scale
        self.activation_fn = activation_fn
        self.variables_collections = [scope]
        super(Layernorm, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        return tf.contrib.layers.layer_norm(
            x,
            center=self.center,
            scale=self.scale,
            activation_fn=self.activation_fn,
            variables_collections=self.variables_collections)

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Layernorm, self).get_variables(include_non_trainable=include_non_trainable)
        if include_non_trainable:
            norm_variables = tf.get_collection(self.variables_collections[0])
            return layer_variables + norm_variables
        else:
            return layer_variables

class Flatten(Layer):
    """
    Flatten layer reshaping the input.
    """

    def __init__(self, scope='flatten', summary_labels=()):
        super(Flatten, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        return tf.reshape(tensor=x, shape=(-1, util.prod(util.shape(x)[1:])))

class Identity(Layer):
    """
    Output the same tensor as the input.
    """

    def __init__(self, scope='identity', summary_labels=()):
        super(Identity, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        return tf.identity(x)


class Pool2d(Layer):
    """
    2-dimensional pooling layer.
    """

    def __init__(
        self,
        pooling_type='max',
        window=2,
        stride=2,
        padding='SAME',
        scope='pool2d',
        summary_labels=()
    ):
        """
        2-dimensional pooling layer.

        Args:
            pooling_type: Either 'max' or 'average'.
            window: Pooling window size, either an integer or pair of integers.
            stride: Pooling stride, either an integer or pair of integers.
            padding: Pooling padding, one of 'VALID' or 'SAME'.
        """
        self.pooling_type = pooling_type
        if isinstance(window, int):
            self.window = (1, window, window, 1)
        elif len(window) == 2:
            self.window = (1, window[0], window[1], 1)
        else:
            raise TensorforceError('Invalid window {} for pool2d layer, must be of size 2'.format(window))
        if isinstance(stride, int):
            self.stride = (1, stride, stride, 1)
        elif len(window) == 2:
            self.stride = (1, stride[0], stride[1], 1)
        else:
            raise TensorforceError('Invalid stride {} for pool2d layer, must be of size 2'.format(stride))
        self.padding = padding
        super(Pool2d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if self.pooling_type == 'average':
            x = tf.nn.avg_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        elif self.pooling_type == 'max':
            x = tf.nn.max_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)

        else:
            raise TensorforceError('Invalid pooling type: {}'.format(self.name))

        return x


class Embedding(Layer):
    """
    Embedding layer.
    """

    def __init__(
        self,
        indices,
        size,
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='embedding',
        summary_labels=()
    ):
        """
        Embedding layer.

        Args:
            indices: Number of embedding indices.
            size: Embedding size.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        self.indices = indices
        self.size = size
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        super(Embedding, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        stddev = min(0.1, sqrt(1.0 / self.size))
        weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.weights = tf.get_variable(
            name='embeddings',
            shape=(self.indices, self.size),
            dtype=tf.float32,
            initializer=weights_init
        )
        return tf.nn.embedding_lookup(params=self.weights, ids=x)

    def tf_regularization_loss(self):
        regularization_loss = super(Embedding, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.weights))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.weights)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None


class Linear(Layer):
    """
    Linear fully-connected layer.
    """

    def __init__(
        self,
        size,
        weights=None,
        bias=True,
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='linear',
        summary_labels=()
    ):
        """
        Linear layer.

        Args:
            size: Layer size.
            weights: Weight initialization, random if None.
            bias: Bias initialization, random if True, no bias added if False.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        self.size = size
        self.weights_init = weights
        self.bias_init = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        super(Linear, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update=False):
        if util.rank(x) != 2:
            raise TensorforceError(
                'Invalid input rank for linear layer: {}, must be 2.'.format(util.rank(x))
            )

        if self.size is None:  # If size is None than Output Matches Input, required for Skip Connections
            self.size = x.shape[1].value

        weights_shape = (x.shape[1].value, self.size)

        if self.weights_init is None:
            stddev = min(0.1, sqrt(2.0 / (x.shape[1].value + self.size)))
            self.weights_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)

        elif isinstance(self.weights_init, float):
            if self.weights_init == 0.0:
                self.weights_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, list):
            self.weights_init = np.asarray(self.weights_init, dtype=np.float32)
            if self.weights_init.shape != weights_shape:
                raise TensorforceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights_init.shape, weights_shape)
                )
            self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, np.ndarray):
            if self.weights_init.shape != weights_shape:
                raise TensorforceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights_init.shape, weights_shape)
                )
            self.weights_init = tf.constant_initializer(value=self.weights_init, dtype=tf.float32)

        elif isinstance(self.weights_init, tf.Tensor):
            if util.shape(self.weights_init) != weights_shape:
                raise TensorforceError(
                    'Weights shape {} does not match expected shape {} '.format(self.weights_init.shape, weights_shape)
                )

        bias_shape = (self.size,)

        if isinstance(self.bias_init, bool):
            if self.bias_init:
                self.bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.bias_init = None

        elif isinstance(self.bias_init, float):
            if self.bias_init == 0.0:
                self.bias_init = tf.zeros_initializer(dtype=tf.float32)
            else:
                self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, list):
            self.bias_init = np.asarray(self.bias_init, dtype=np.float32)
            if self.bias_init.shape != bias_shape:
                raise TensorforceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias_init.shape, bias_shape)
                )
            self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, np.ndarray):
            if self.bias_init.shape != bias_shape:
                raise TensorforceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias_init.shape, bias_shape)
                )
            self.bias_init = tf.constant_initializer(value=self.bias_init, dtype=tf.float32)

        elif isinstance(self.bias_init, tf.Tensor):
            if util.shape(self.bias_init) != bias_shape:
                raise TensorforceError(
                    'Bias shape {} does not match expected shape {} '.format(self.bias_init.shape, bias_shape)
                )

        if isinstance(self.weights_init, tf.Tensor):
            self.weights = self.weights_init
        else:
            self.weights = tf.get_variable(
                name='W',
                shape=weights_shape,
                dtype=tf.float32,
                initializer=self.weights_init
            )

        x = tf.matmul(a=x, b=self.weights)

        if self.bias_init is None:
            self.bias = None

        else:
            if isinstance(self.bias_init, tf.Tensor):
                self.bias = self.bias_init
            else:
                self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=self.bias_init)

            x = tf.nn.bias_add(value=x, bias=self.bias)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Linear, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.weights))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.weights)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None


class Dense(Layer):
    """
    Dense layer, i.e. linear fully connected layer with subsequent non-linearity.
    """

    def __init__(
        self,
        size=None,
        weights=None,
        bias=True,
        activation='tanh',
        l2_regularization=0.0,
        l1_regularization=0.0,
        skip=False,
        scope='dense',
        summary_labels=()
    ):
        """
        Dense layer.

        Args:
            size: Layer size, if None than input size matches the output size of the layer
            bias: If true, bias is added.
            activation: Type of nonlinearity.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
            skip: Add skip connection like ResNet (https://arxiv.org/pdf/1512.03385.pdf),
                  doubles layers and ShortCut from Input to output
        """
        self.skip = skip
        if self.skip and size is not None:
            raise TensorforceError(
                'Dense Layer SKIP connection needs Size=None, uses input shape '
                'sizes to create skip connection network, please delete "size" parameter'
            )

        self.linear = Linear(
            size=size,
            weights=weights,
            bias=bias,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels
        )
        if self.skip:
            print("SKIP ENABLED")
            self.linear_skip = Linear(
                size=size,
                weights=weights,
                bias=bias,
                l2_regularization=l2_regularization,
                l1_regularization=l1_regularization,
                summary_labels=summary_labels
            )
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Dense, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        xl1 = self.linear.apply(x=x, update=update)
        xl1 = self.nonlinearity.apply(x=xl1, update=update)
        if self.skip:
            xl2 = self.linear_skip.apply(x=xl1, update=update)
            xl2 = self.nonlinearity.apply(x=(xl2 + x), update=update)  #add input back in as skip connection per paper
        else:
            xl2 = xl1

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=xl2)
            self.summaries.append(summary)

        return xl2

    def tf_regularization_loss(self):
        regularization_loss = super(Dense, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.linear.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if self.skip:
            regularization_loss = self.linear_skip.regularization_loss()
            if regularization_loss is not None:
                losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Dense, self).get_variables(include_non_trainable=include_non_trainable)
        linear_variables = self.linear.get_variables(include_non_trainable=include_non_trainable)
        if self.skip:
            linear_variables = linear_variables \
                               + self.linear_skip.get_variables(include_non_trainable=include_non_trainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + linear_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(Dense, self).get_summaries()
        linear_summaries = self.linear.get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + linear_summaries + nonlinearity_summaries


class Dueling(Layer):
    """
    Dueling layer, i.e. Duel pipelines for Exp & Adv to help with stability
    """

    def __init__(
        self,
        size,
        bias=False,
        activation='none',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='dueling',
        summary_labels=()
    ):
        """
        Dueling layer.

        [Dueling Networks] (https://arxiv.org/pdf/1511.06581.pdf)
        Implement Y = Expectation[x] + (Advantage[x] - Mean(Advantage[x]))

        Args:
            size: Layer size.
            bias: If true, bias is added.
            activation: Type of nonlinearity.
            l2_regularization: L2 regularization weight.
            l1_regularization: L1 regularization weight.
        """
        # Expectation is broadcast back over advantage values so output is of size 1
        self.expectation_layer = Linear(
            size=1, bias=bias,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels
        )
        self.advantage_layer = Linear(
            size=size,
            bias=bias,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            summary_labels=summary_labels
        )
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Dueling, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update=False):
        expectation = self.expectation_layer.apply(x=x, update=update)
        advantage = self.advantage_layer.apply(x=x, update=update)
        mean_advantage = tf.reduce_mean(input_tensor=advantage, axis=1, keep_dims=True)

        x = expectation + advantage - mean_advantage

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Dueling, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.expectation_layer.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.advantage_layer.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Dueling, self).get_variables(include_non_trainable=include_non_trainable)
        expectation_layer_variables = self.expectation_layer.get_variables(include_non_trainable=include_non_trainable)
        advantage_layer_variables = self.advantage_layer.get_variables(include_non_trainable=include_non_trainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + expectation_layer_variables + advantage_layer_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(Dueling, self).get_summaries()
        expectation_layer_summaries = self.expectation_layer.get_summaries()
        advantage_layer_summaries = self.advantage_layer.get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + expectation_layer_summaries + advantage_layer_summaries + nonlinearity_summaries


class Conv1d(Layer):
    """
    1-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='conv1d',
        summary_labels=()
    ):
        """
        1D convolutional layer.

        Args:
            size: Number of filters
            window: Convolution window size
            stride: Convolution stride
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        self.window = window
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Conv1d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if util.rank(x) != 3:
            raise TensorforceError('Invalid input rank for conv1d layer: {}, must be 3'.format(util.rank(x)))

        filters_shape = (self.window, x.shape[2].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        x = tf.nn.conv1d(value=x, filters=self.filters, stride=self.stride, padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Conv1d, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Conv1d, self).get_variables(include_non_trainable=include_non_trainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(Conv1d, self).get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + nonlinearity_summaries


class Conv2d(Layer):
    """
    2-dimensional convolutional layer.
    """

    def __init__(
        self,
        size,
        window=3,
        stride=1,
        padding='SAME',
        bias=True,
        activation='relu',
        l2_regularization=0.0,
        l1_regularization=0.0,
        scope='conv2d',
        summary_labels=()
    ):
        """
        2D convolutional layer.

        Args:
            size: Number of filters
            window: Convolution window size, either an integer or pair of integers.
            stride: Convolution stride, either an integer or pair of integers.
            padding: Convolution padding, one of 'VALID' or 'SAME'
            bias: If true, a bias is added
            activation: Type of nonlinearity
            l2_regularization: L2 regularization weight
            l1_regularization: L1 regularization weight
        """
        self.size = size
        if isinstance(window, int):
            self.window = (window, window)
        elif len(window) == 2:
            self.window = tuple(window)
        else:
            raise TensorforceError('Invalid window {} for conv2d layer, must be of size 2'.format(window))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.nonlinearity = Nonlinearity(name=activation, summary_labels=summary_labels)
        super(Conv2d, self).__init__(scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update):
        if util.rank(x) != 4:
            raise TensorforceError('Invalid input rank for conv2d layer: {}, must be 4'.format(util.rank(x)))

        filters_shape = self.window + (x.shape[3].value, self.size)
        stddev = min(0.1, sqrt(2.0 / self.size))
        filters_init = tf.random_normal_initializer(mean=0.0, stddev=stddev, dtype=tf.float32)
        self.filters = tf.get_variable(name='W', shape=filters_shape, dtype=tf.float32, initializer=filters_init)
        stride_h, stride_w = self.stride if type(self.stride) is tuple else (self.stride, self.stride)
        x = tf.nn.conv2d(input=x, filter=self.filters, strides=(1, stride_h, stride_w, 1), padding=self.padding)

        if self.bias:
            bias_shape = (self.size,)
            bias_init = tf.zeros_initializer(dtype=tf.float32)
            self.bias = tf.get_variable(name='b', shape=bias_shape, dtype=tf.float32, initializer=bias_init)
            x = tf.nn.bias_add(value=x, bias=self.bias)

        x = self.nonlinearity.apply(x=x, update=update)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x

    def tf_regularization_loss(self):
        regularization_loss = super(Conv2d, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        if self.l2_regularization > 0.0:
            losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.filters))
            if self.bias is not None:
                losses.append(self.l2_regularization * tf.nn.l2_loss(t=self.bias))

        if self.l1_regularization > 0.0:
            losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.filters)))
            if self.bias is not None:
                losses.append(self.l1_regularization * tf.reduce_sum(input_tensor=tf.abs(x=self.bias)))

        regularization_loss = self.nonlinearity.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        layer_variables = super(Conv2d, self).get_variables(include_non_trainable=include_non_trainable)
        nonlinearity_variables = self.nonlinearity.get_variables(include_non_trainable=include_non_trainable)

        return layer_variables + nonlinearity_variables

    def get_summaries(self):
        layer_summaries = super(Conv2d, self).get_summaries()
        nonlinearity_summaries = self.nonlinearity.get_summaries()

        return layer_summaries + nonlinearity_summaries


class InternalLstm(Layer):
    """
    Long short-term memory layer for internal state management.
    """

    def __init__(self, size, dropout=None, scope='internal_lstm', summary_labels=()):
        """
        LSTM layer.

        Args:
            size: LSTM size.
            dropout: Dropout rate.
        """
        self.size = size
        self.dropout = dropout
        super(InternalLstm, self).__init__(num_internals=1, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update, state):
        if util.rank(x) != 2:
            raise TensorforceError(
                'Invalid input rank for internal lstm layer: {}, must be 2.'.format(util.rank(x))
            )

        state = tf.contrib.rnn.LSTMStateTuple(c=state[:, 0, :], h=state[:, 1, :])

        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.size)

        if self.dropout is not None:
            keep_prob = tf.cond(pred=update, true_fn=(lambda: 1.0 - self.dropout), false_fn=(lambda: 1.0))
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell, output_keep_prob=keep_prob)

        x, state = self.lstm_cell(inputs=x, state=state)

        internal_output = tf.stack(values=(state.c, state.h), axis=1)

        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        return x, (internal_output,)

    def internals_input(self):
        return super(InternalLstm, self).internals_input() \
               + [tf.placeholder(dtype=tf.float32, shape=(None, 2, self.size))]

    def internals_init(self):
        return super(InternalLstm, self).internals_init() + [np.zeros(shape=(2, self.size))]


class Lstm(Layer):

    def __init__(self, size, dropout=None, scope='lstm', summary_labels=(), return_final_state=True):
        """
        LSTM layer.

        Args:
            size: LSTM size.
            dropout: Dropout rate.
        """
        self.size = size
        self.dropout = dropout
        self.return_final_state = return_final_state
        super(Lstm, self).__init__(num_internals=0, scope=scope, summary_labels=summary_labels)

    def tf_apply(self, x, update, sequence_length=None):
        if util.rank(x) != 3:
            raise TensorforceError('Invalid input rank for lstm layer: {}, must be 3.'.format(util.rank(x)))

        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.size)
        if 'activations' in self.summary_labels:
            summary = tf.summary.histogram(name='activations', values=x)
            self.summaries.append(summary)

        x, state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=x,
            sequence_length=sequence_length,
            dtype=tf.float32
        )

        # This distinction is so we can stack multiple LSTM layers
        if self.return_final_state:
            return tf.concat(values=(state.c, state.h), axis=1)
        else:
            return x
