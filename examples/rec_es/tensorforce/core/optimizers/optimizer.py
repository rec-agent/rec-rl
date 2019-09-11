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

import tensorflow as tf

from tensorforce import util, TensorforceError
import tensorforce.core.optimizers


class Optimizer(object):
    """
    Generic TensorFlow optimizer which minimizes a not yet further specified expression, usually
    some kind of loss function. More generally, an optimizer can be considered as some method of
    updating a set of variables.
    """

    def __init__(self, summaries=None, summary_labels=None):
        """
        Creates a new optimizer instance.
        """
        self.variables = dict()
        self.summaries = summaries
        if summary_labels is None:
            self.summary_labels = dict()
        else:
            self.summary_labels = summary_labels

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if not registered:
                assert kwargs.get('trainable', False)
                self.variables[name] = variable
            return variable

        # TensorFlow function
        self.step = tf.make_template(
            name_='step',
            func_=self.tf_step,
            custom_getter=custom_getter
        )

    def tf_step(self, time, variables, **kwargs):
        """
        Creates the TensorFlow operations for performing an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            **kwargs: Additional arguments depending on the specific optimizer implementation.
                For instance, often includes `fn_loss` if a loss function is optimized.

        Returns:
            List of delta tensors corresponding to the updates for each optimized variable.
        """
        raise NotImplementedError

    def minimize(self, time, variables, **kwargs):
        """
        Performs an optimization step.

        Args:
            time: Time tensor.
            variables: List of variables to optimize.
            **kwargs: Additional optimizer-specific arguments. The following arguments are used
                by some optimizers:
                - fn_loss: A callable returning the loss of the current model.
                - fn_kl_divergence: A callable returning the KL-divergence relative to the
                    current model.
                - return_estimated_improvement: Returns the estimated improvement resulting from
                    the natural gradient calculation if true.
                - fn_reference: A callable returning the reference values necessary for comparison.
                - fn_compare: A callable comparing the current model to the reference model given
                    by its values.
                - source_variables: List of source variables to synchronize with.
                - global_variables: List of global variables to apply the proposed optimization
                    step to.


        Returns:
            The optimization operation.
        """
        # Add training variable gradient histograms/scalars to summary output
        #if 'gradients' in self.summary_labels:
        if any(k in self.summary_labels for k in ['gradients', 'gradients_histogram', 'gradients_scalar']):
            valid = True
            if isinstance(self, tensorforce.core.optimizers.TFOptimizer):
                gradients = self.optimizer.compute_gradients(kwargs['fn_loss']())
            elif isinstance(self.optimizer, tensorforce.core.optimizers.TFOptimizer):
                ## This section handles "Multi_step" and may handle others
                #  if failure is found, add another elif to handle that case
                gradients = self.optimizer.optimizer.compute_gradients(kwargs['fn_loss']())
            else:
                # Didn't find proper gradient information
                valid = False

            # Valid gradient data found, create summary data items
            if valid:
                for grad, var in gradients:
                    if grad is not None:
                        if any(k in self.summary_labels for k in ['gradients','gradients_scalar']):
                            axes = list(range(len(grad.shape)))
                            mean, var = tf.nn.moments(grad,axes)
                            summary = tf.summary.scalar(name='gradients/' + var.name+ "/mean", tensor=mean)
                            self.summaries.append(summary)
                            summary = tf.summary.scalar(name='gradients/' + var.name+ "/variance", tensor=var)
                            self.summaries.append(summary)
                        if any(k in self.summary_labels for k in ['gradients', 'gradients_histogram']):
                            summary = tf.summary.histogram(name='gradients/' + var.name, values=grad)
                            self.summaries.append(summary)

        deltas = self.step(time=time, variables=variables, **kwargs)
        with tf.control_dependencies(control_inputs=deltas):
            return tf.no_op()

    def get_variables(self):
        """
        Returns the TensorFlow variables used by the optimizer.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]

    @staticmethod
    def from_spec(spec, kwargs=None):
        """
        Creates an optimizer from a specification dict.
        """
        optimizer = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.optimizers.optimizers,
            kwargs=kwargs
        )
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def apply_step(self, variables, deltas):
        """
        Applies step deltas to variable values.

        Args:
            variables: List of variables.
            deltas: List of deltas of same length.

        Returns:
            The step-applied operation.
        """
        if len(variables) != len(deltas):
            raise TensorforceError("Invalid variables and deltas lists.")
        return tf.group(*(variable.assign_add(delta=delta) for variable, delta in zip(variables, deltas)))
