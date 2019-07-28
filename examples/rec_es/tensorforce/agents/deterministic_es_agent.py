# -*- coding: utf-8 -*-

"""
    desc: Evolution stratege agent.
    created: 2017.01.23
    @author: cuiqing.cq
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np

from tensorforce.agents import Agent
from tensorforce import util, TensorforceError
from tensorforce.models import DeterministicESModel


class DeterministicESAgent(Agent):
    """
    Evolution Strategy as a Scalable Alternative to Reinforcement Learning
    [Tim Salimans, Jonathan Ho, et al., 2017]
    (https://arxiv.org/abs/1703.03864).

    Use DeterministicESModel which does not have the distribution layer.
    """

    def __init__(
        self,
        env,
        states_spec,
        actions_spec,
        network_spec,
        device=None,
        session_config=None,
        scope='deterministic_es',
        saver_spec=None,
        summary_spec=None,
        distributed_spec=None,
        optimizer=None,
        states_preprocessing_spec=None,
        explorations_spec=None,
        reward_preprocessing_spec=None,
        distributions_spec=None,
        entropy_regularization=None,
        max_episode_timesteps=None,
        batch_size=1000,
        noise_stddev=0.02,
        eval_prob=0.01,
        l2_coeff=0.01,
        train_iters=1000,
        seed_range=1000000,
        repeat_actions=1,
        batch_data=None
    ):

        """
        Args:
            states_spec: Dict containing at least one state definition. In the case of a single state,
               keys `shape` and `type` are necessary. For multiple states, pass a dict of dicts where each state
               is a dict itself with a unique name as its key.
            actions_spec: Dict containing at least one action definition. Actions have types and either `num_actions`
                for discrete actions or a `shape` for continuous actions. Consult documentation and tests for more.
            network_spec: List of layers specifying a neural network via layer types, sizes and optional arguments
                such as activation or regularization. Full examples are in the examples/configs folder.
            device: Device string specifying model device.
            session_config: optional tf.ConfigProto with additional desired session configurations
            scope: TensorFlow scope, defaults to agent name (e.g. `dqn`).
            saver_spec: Dict specifying automated saving. Use `directory` to specify where checkpoints are saved. Use
                either `seconds` or `steps` to specify how often the model should be saved. The `load` flag specifies
                if a model is initially loaded (set to True) from a file `file`.
            summary_spec: Dict specifying summaries for TensorBoard. Requires a 'directory' to store summaries, `steps`
                or `seconds` to specify how often to save summaries, and a list of `labels` to indicate which values
                to export, e.g. `losses`, `variables`. Consult neural network class and model for all available labels.
            distributed_spec: Dict specifying distributed functionality. Use `parameter_server` and `replica_model`
                Boolean flags to indicate workers and parameter servers. Use a `cluster_spec` key to pass a TensorFlow
                cluster spec.
            states_preprocessing_spec: Optional list of states preprocessors to apply to state
                (e.g. `image_resize`, `grayscale`).
            explorations_spec: Optional dict specifying action exploration type (epsilon greedy
                or Gaussian noise).
            reward_preprocessing_spec: Optional dict specifying reward preprocessing.
            distributions_spec: Optional dict specifying action distributions to override default distribution choices.
                Must match action names.
            entropy_regularization: Optional positive float specifying an entropy regularization value.
            batch_size: Int specifying number of samples collected via `observe` before an update is executed.
            batch_data: Input data tensor, which is for table environment
            repeat_actions: Int specifying the times of repearting actions to better estimate the reward
        """

        if network_spec is None:
            raise TensorforceError("No network_spec provided.")

        self.env = env
        self.network_spec = network_spec
        self.device = device
        self.session_config = session_config
        self.scope = scope
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.distributed_spec = distributed_spec
        self.states_preprocessing_spec = states_preprocessing_spec
        self.explorations_spec = explorations_spec
        self.reward_preprocessing_spec = reward_preprocessing_spec
        self.distributions_spec = distributions_spec
        self.entropy_regularization = entropy_regularization
        self.batch_size=batch_size
        self.max_episode_timesteps = max_episode_timesteps
        self.noise_stddev = noise_stddev
        self.eval_prob = eval_prob
        self.l2_coeff = l2_coeff
        self.train_iters = train_iters
        self.seed_range = seed_range
        self.repeat_actions = repeat_actions
        self.batch_data = batch_data

        if optimizer is None:
            self.optimizer = dict(
                type='adam',
                learning_rate=0.01
            )
        else:
            self.optimizer = optimizer

        super(DeterministicESAgent, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            batched_observe=None
        )

    def run_worker(self):
        # Start running on all workers.
        #self.model定义在父类中
        self.model.update()

    def initialize_model(self):
        return DeterministicESModel(
            env=self.env,
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
            network_spec=self.network_spec,
            device=self.device,
            session_config=self.session_config,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            optimizer=self.optimizer,
            states_preprocessing_spec=self.states_preprocessing_spec,
            explorations_spec=self.explorations_spec,
            reward_preprocessing_spec=self.reward_preprocessing_spec,
            distributions_spec=self.distributions_spec,
            entropy_regularization=self.entropy_regularization,
            batch_size=self.batch_size,
            max_episode_timesteps=self.max_episode_timesteps,
            noise_stddev=self.noise_stddev,
            eval_prob=self.eval_prob,
            l2_coeff=self.l2_coeff,
            train_iters=self.train_iters,
            seed_range=self.seed_range,
            repeat_actions=self.repeat_actions,
            batch_data=self.batch_data
        )
