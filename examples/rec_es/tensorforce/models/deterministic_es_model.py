# -*- coding: utf-8 -*-

"""
    desc: Evolution stratege as a scalable alternative to reinforcement learning.
    created: 2017.02.23
    @author: cuiqing.cq
"""

import math
import time
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import datetime

from tensorforce import util, TensorforceError
from tensorforce.core.lib import optimizers
from tensorforce.models import Model
from tensorforce.core.networks import Network

class DeterministicESModel(Model):
    '''
    Similar to ESModel, except this does not have the distribution layer.
    '''

    def __init__(
        self,
        env,
        states_spec,
        actions_spec,
        network_spec,
        device,
        session_config,
        scope,
        saver_spec,
        summary_spec,
        distributed_spec,
        optimizer,
        states_preprocessing_spec,
        explorations_spec,
        reward_preprocessing_spec,
        distributions_spec,
        entropy_regularization,
        batch_size,
        max_episode_timesteps,
        noise_stddev,
        eval_prob,
        l2_coeff,
        train_iters,
        seed_range,
        repeat_actions,
        batch_data
    ):
        assert optimizer is not None
        self.optimizer_spec = optimizer

        self.network_spec = network_spec
        self.max_episode_timesteps = max_episode_timesteps
        if isinstance(noise_stddev, list):
            self.noise_stddev = np.asarray(noise_stddev, dtype=np.float32).flatten()
            self.divide_by_norm = True
        else:
            self.noise_stddev = noise_stddev
            self.divide_by_norm = False
        self.eval_prob = eval_prob
        self.l2_coeff = l2_coeff
        self.train_iters = train_iters
        self.seed_range = seed_range
        self.repeat_actions = repeat_actions

        if distributed_spec is not None and 'cluster_spec' in distributed_spec:
            self.num_workers = distributed_spec.get('cluster_spec').num_tasks('worker')
        else:
            self.num_workers = 1

        if distributed_spec is not None and 'task_index' in distributed_spec:
            self.task_index = distributed_spec.get('task_index')
        else:
            self.task_index = 0

        self.batch_size = int(batch_size / self.num_workers)
        self.eval_len = max(1, int(self.batch_size * self.eval_prob)) if self.eval_prob > 0 else 0

        # Ensure local mini-batch size wont't be odd
        self.batch_size += self.batch_size % 2
        self.vec_len = int(self.batch_size / 2.0)

        # I/O specification
        self.io_spec = None
        if batch_data is not None:
            self.io_spec = dict(table=True)
            if isinstance(batch_data, dict):
                self.io_spec['tensor'] = True
                if 'states' in batch_data and 'actions' not in batch_data:
                    self.io_spec['interactive'] = True

        if len(actions_spec) > 1:
            raise TensorforceError('DeterministicESModel only support single action case')
        for name, action in actions_spec.items():
            if action['type'] != 'float':
                raise TensorforceError('DeterministicESModel only support continuous action case, action type must be float')

        super(DeterministicESModel, self).__init__(
            states_spec=states_spec,
            actions_spec=actions_spec,
            #network_spec=network_spec,
            device=device,
            session_config=session_config,
            scope=scope,
            saver_spec=saver_spec,
            summary_spec=summary_spec,
            distributed_spec=distributed_spec,
            optimizer=None,
            discount=None,
            variable_noise=None,
            states_preprocessing_spec=states_preprocessing_spec,
            explorations_spec=explorations_spec,
            reward_preprocessing_spec=reward_preprocessing_spec,
            batch_data=batch_data
        )

        self.env = env

        # debug
        self.rollout_num = 0

    def initialize(self, custom_getter):
        super(DeterministicESModel, self).initialize(custom_getter)

        # Network
        self.network = Network.from_spec(
            spec=self.network_spec,
            kwargs=dict(summary_labels=self.summary_labels)
        )

        # Network internals
        self.internals_input.extend(self.network.internals_input())
        self.internals_init.extend(self.network.internals_init())

        # Seed
        collection = self.graph.get_collection(name='noise_seed')
        if len(collection) == 0:
            self.seed = tf.get_variable(
                name='noise_seed',
                shape=(self.num_workers, self.vec_len),
                dtype=util.tf_dtype('int'),
                initializer=tf.zeros_initializer(
                    dtype=util.tf_dtype('int'))
            )
            self.graph.add_to_collection(name='noise_seed', value=self.seed)
        else:
            assert len(collection) == 1
            self.seed = collection[0]

        # Score
        collection = self.graph.get_collection(name='evolution_score')
        if len(collection) == 0:
            self.score = tf.get_variable(
                name='evolution_score',
                shape=(self.num_workers, 2*self.vec_len),
                dtype=util.tf_dtype('float'),
                initializer=tf.zeros_initializer(
                dtype=util.tf_dtype('float'))
            )
            self.graph.add_to_collection(name='evolution_score', value=self.score)
        else:
            assert len(collection) == 1
            self.score = collection[0]

        # Evaluation score
        collection = self.graph.get_collection(name='evaluation_score')
        if len(collection) == 0:
            self.eval_score = tf.get_variable(
                name='evaluation_score',
                shape=(self.num_workers, self.eval_len),
                dtype=util.tf_dtype('float'),
                initializer=tf.zeros_initializer(
                dtype=util.tf_dtype('float'))
            )
            self.graph.add_to_collection(name='evaluation_score', value=self.eval_score)
        else:
            assert len(collection) == 1
            self.eval_score = collection[0]

        self.doors = {}
        self.locks = {}
        self.lock_collection = self.graph.get_collection(name="sync_var")
        if len(self.lock_collection) == 0:
            with tf.variable_scope('sync_var'):
                for i in range(1, self.num_workers):
                    self.doors[i] = tf.get_variable(
                        name="sync_point_%d"%i,
                        dtype=util.tf_dtype('int'),
                        initializer=tf.constant(0, dtype=tf.int32)
                    )
                    self.locks[i] = tf.get_variable(
                        name="lock_flag_%d"%i,
                        dtype=util.tf_dtype('int'),
                        initializer=tf.constant(1, dtype=tf.int32)
                    )
                    self.graph.add_to_collection(name='sync_var', value=self.doors[i])
                    self.graph.add_to_collection(name='sync_var', value=self.locks[i])
        else:
            assert len(self.lock_collection) == 2 * (self.num_workers - 1)
            for i in range(1, self.num_workers):
                self.doors[i] = self.lock_collection[2*(i-1)]
                self.locks[i] = self.lock_collection[2*(i-1)+1]

        self.lock_collection = self.graph.get_collection(name="sync_var")

        # Seed and score placeholder
        self.seed_ph = tf.placeholder(dtype=tf.int32, shape=(self.vec_len,), name='seed_ph')
        self.score_ph = tf.placeholder(dtype=tf.float32, shape=(2*self.vec_len,), name='score_ph')
        self.eval_score_ph = tf.placeholder(dtype=tf.float32, shape=(self.eval_len,), name='evaluation_score_ph')

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, update):
        # Actually do nothing here
        return None

    def tf_actions_and_internals(self, states, internals, update, deterministic):
        actions = dict()
        embedding, internals = self.network.apply(x=states, internals=internals, update=update, return_internals=True)

        # it should only have one action here, assert in __init__ function
        for name, action in self.actions_spec.items():
            if 'min_value' in action and 'max_value' in action:
                actions[name] = tf.clip_by_value(t=embedding, clip_value_min=action['min_value'], clip_value_max=action['max_value'])
            else:
                actions[name] = embedding

        return actions, internals

    def create_output_operations(self, states, internals, actions, terminal, reward, update, deterministic):
        super(DeterministicESModel, self).create_output_operations(states, internals, actions, terminal, reward, update, deterministic)

        assign_op1 = tf.assign(self.seed[self.task_index], self.seed_ph)
        assign_op2 = tf.assign(self.score[self.task_index], self.score_ph)
        assign_op3 = tf.assign(self.eval_score[self.task_index], self.eval_score_ph)
        self.assign_op = tf.group(*(assign_op1, assign_op2, assign_op3))

        self.close_ops = {}
        self.open_ops = {}
        self.release_locks = {}
        self.reset_locks = {}
        for i in range(1, self.num_workers):
            self.close_ops[i] = tf.assign(self.doors[i], tf.constant(0))
            self.open_ops[i] = tf.assign(self.doors[i], tf.constant(i))
            self.release_locks[i] = tf.assign(self.locks[i], tf.constant(0))
            self.reset_locks[i] = tf.assign(self.locks[i], tf.constant(1))

        # Policy params assignment
        local_policy_vars = self.network.get_variables()
        self.policy_variables = OrderedDict()
        for v in local_policy_vars:
            self.policy_variables[v.op.node_def.name] = v
        self.policy_placeholders = dict()
        self.assignment_nodes = []

        # Create new placeholders to put in custom weights.
        for k, var in self.policy_variables.items():
            self.policy_placeholders[k] = tf.placeholder(var.value().dtype,
                                                  var.get_shape().as_list())
            self.assignment_nodes.append(var.assign(self.policy_placeholders[k]))

        self.policy_num_params = sum([np.prod(variable.shape.as_list())
                               for _, variable
                               in self.policy_variables.items()])
        # Optimizer
        self.optimizer_spec['dim'] = self.policy_num_params
        self.es_optimizer = optimizers.from_spec(self.optimizer_spec)

    def get_variables(self, include_non_trainable=False):
        model_variables = super(DeterministicESModel, self).get_variables(include_non_trainable=include_non_trainable)
        network_variables = self.network.get_variables(include_non_trainable=include_non_trainable)
        global_shared_var = [self.seed, self.score, self.eval_score] + self.lock_collection
        if include_non_trainable:
            return model_variables + network_variables + global_shared_var
        else:
            return model_variables + network_variables

    def get_summaries(self):
        model_summaries = super(DeterministicESModel, self).get_summaries()
        network_summaries = self.network.get_summaries()

        return model_summaries + network_summaries

    def get_weights(self):
        return np.concatenate([v.eval(session=self.monitored_session).flatten()
                               for v in self.policy_variables.values()])

    def set_weights(self, new_weights):
        shapes = [v.get_shape().as_list() for v in self.policy_variables.values()]
        arrays = util.unflatten(new_weights, shapes)
        placeholders = [self.policy_placeholders[k]
                        for k, v in self.policy_variables.items()]
        self.monitored_session.run(self.assignment_nodes,
                      feed_dict=dict(zip(placeholders, arrays)))

    def rollout(self, deterministic=False):
        step = 0
        rewards = []
        internals = []
        unique_state = len(self.states_spec) == 1
        unique_action = len(self.actions_spec) == 1
        
        for i in range(self.repeat_actions):
            reward_sum = 0.0
            states = self.env.reset()

            while True:
                if unique_state:
                    states = {"state": np.asarray(states)}
                else:
                    states = {name: np.asarray(state) for name, state in states.items()}
                actions, internals, _ = self.act(states, [], deterministic)
                if unique_action:
                    states, terminal, reward = self.env.execute(next(iter(actions.values())))
                else:
                    states, terminal, reward = self.env.execute(actions)

                reward_sum += reward
                step += 1
                if terminal or (self.max_episode_timesteps is not None and step >= self.max_episode_timesteps):
                    break
            rewards.append(reward_sum)
        rewards = np.array(np.average(rewards), dtype=np.float32)

        '''
        # debug
        if self.rollout_num % 100 == 0:
            print('rollout number:', self.rollout_num)
            print('pageid:', self.monitored_session.run(states_bak))
            print('weights:', self.monitored_session.run(self.weights))
            print('action:', next(iter(actions.values())))
            print('reward:', rewards[0])
        '''
        self.rollout_num += 1
        return rewards, step

    def do_rollouts(self, params, evaluate=False):
        noise_indices, returns, sign_returns, lengths = [], [], [], []
        eval_returns, eval_lengths = [], []

        # Run with parameter perturbations.
        seed = np.random.randint(self.seed_range)
        noise = np.random.RandomState(seed).randn(len(params)).astype(np.float32)
        perturbation = self.noise_stddev * noise

        self.set_weights(params + perturbation)
        rewards_pos, lengths_pos = self.rollout(deterministic=True)
        # print('!!!!!!')
        # print(rewards_pos)

        self.set_weights(params - perturbation)
        rewards_neg, lengths_neg = self.rollout(deterministic=True)

        noise_indices.append(seed)
        returns.append([rewards_pos.sum(), rewards_neg.sum()])
        sign_returns.append(
            [np.sign(rewards_pos).sum(), np.sign(rewards_neg).sum()])
        lengths.append([lengths_pos, lengths_neg])

        if evaluate:
            # Run with no perturbation.
            self.set_weights(params)
            rewards, length = self.rollout(deterministic=True)
            eval_returns.append(rewards.sum())
            eval_lengths.append(length)

        return Result(noise_indices=noise_indices,
            noisy_returns=returns,
            sign_noisy_returns=sign_returns,
            noisy_lengths=lengths,
            eval_returns=eval_returns,
            eval_lengths=eval_lengths)

    def collect_results(self, min_episodes, params):
        num_episodes, num_timesteps = 0, 0
        num_eval_episodes = 0
        results = []
        while num_episodes < min_episodes: #or num_timesteps < min_timesteps:
            if num_eval_episodes < self.eval_len:
                result = self.do_rollouts(params, evaluate=True)
                num_eval_episodes += 1
            else:
                result = self.do_rollouts(params, evaluate=False)

            # Get the results of the rollouts.
            results.append(result)
            num_episodes += sum([len(pair) for pair
                             in result.noisy_lengths])
            num_timesteps += sum([sum(pair) for pair
                             in result.noisy_lengths])

            if num_episodes >= 100 and num_episodes % 100 == 0 or num_episodes == min_episodes:
                print("Collected %d episodes %d timesteps so far this iter" % (
                    num_episodes, num_timesteps))

        return results, num_episodes, num_timesteps

    def open_the_doors(self):
        for door_id in self.doors.keys():
            self.monitored_session.run(self.open_ops[door_id])
            self.monitored_session.run(self.doors[door_id])

    def wait_for_all(self):
        double_check = 0
        while True:
            s = 0
            release_list = []
            for door_id in self.doors.keys():
                flag = self.monitored_session.run(self.locks[door_id])
                res = None
                if flag > 0:
                    res = self.monitored_session.run(self.doors[door_id])
                else:
                    release_list.append(door_id)
                    res = 0
                s += res
            if s == 0:
                double_check += 1
            if s == 0 and double_check == 2:
                break

    def m_sync(self):
        while self.doors[self.task_index].eval(
                session=self.monitored_session) != self.task_index:
            lock_state = self.locks[self.task_index].eval(session=self.monitored_session)
            if lock_state == 0:
                return

        self.monitored_session.run(self.close_ops[self.task_index])
        while self.doors[self.task_index].eval(
                session=self.monitored_session) != self.task_index:
            lock_state = self.locks[self.task_index].eval(session=self.monitored_session)
            if lock_state == 0:
                return
        self.monitored_session.run(self.close_ops[self.task_index])

    def sync_barrier(self):
        if self.task_index == 0:
            self.open_the_doors()
            self.wait_for_all()
            self.open_the_doors()
            self.wait_for_all()
        else:
            self.m_sync()

    def release_lock(self):
        # Release locks in case of getting incorrect state
        if self.task_index == 0:
            for i in range(1,self.num_workers):
                self.monitored_session.run(self.release_locks[i])
        else:
            self.monitored_session.run(self.release_locks[self.task_index])

    def get_direction(self, seed):
        noise = np.random.RandomState(seed).randn(self.policy_num_params).astype(np.float32)
        if self.divide_by_norm:
            perturbation = self.noise_stddev * noise
            direction = perturbation / np.linalg.norm(perturbation)
            return direction
        else:
            return noise

    def update(self):

        if self.task_index != 0:
            self.monitored_session.run(self.reset_locks[self.task_index])

        # debug
        self.weights = self.network.layers[0].weights
        # print('!!!!!!!')
        # print(self.weights)

        episodes_so_far = 0
        timesteps_so_far = 0
        tstart = time.time()
        sync_time = 0.0
        theta = self.get_weights()
        # print('!!!!!!!!')
        # print(theta)
        for iters_so_far in range(self.train_iters):

            step_tstart = time.time()

            print("********** Iteration %i ************"%iters_so_far)
            results, num_episodes, num_timesteps = self.collect_results(self.batch_size, theta)

            all_noise_indices = []
            all_training_returns = []
            all_training_lengths = []
            all_eval_returns = []
            all_eval_lengths = []

            # Loop over the results.
            for result in results:
                all_eval_returns += result.eval_returns
                all_eval_lengths += result.eval_lengths

                all_noise_indices += result.noise_indices
                all_training_returns += result.noisy_returns
                all_training_lengths += result.noisy_lengths

            assert (len(all_eval_returns) == len(all_eval_lengths) == self.eval_len)
            assert (len(all_noise_indices) == len(all_training_returns) ==
                    len(all_training_lengths) == self.vec_len)

            episodes_so_far += num_episodes
            timesteps_so_far += num_timesteps

            # Assemble the results.
            eval_returns = np.array(all_eval_returns)
            eval_lengths = np.array(all_eval_lengths)
            noise_indices = np.array(all_noise_indices)
            noisy_returns = np.array(all_training_returns)
            noisy_lengths = np.array(all_training_lengths)

            row_seeds = noise_indices.reshape((self.vec_len,))
            row_scores = noisy_returns.reshape((2*self.vec_len,))
            row_eval_scores = eval_returns.reshape((self.eval_len,))
            self.monitored_session.run(self.assign_op, feed_dict={
                self.seed_ph: row_seeds,
                self.score_ph: row_scores,
                self.eval_score_ph: row_eval_scores}
                )

            step_sync_tstart = time.time()
            self.sync_barrier()
            step_sync_tend = time.time()
            sync_time += step_sync_tend - step_sync_tstart

            noise_indices = self.seed.eval(session=self.monitored_session)
            noisy_returns = self.score.eval(session=self.monitored_session)
            eval_returns = self.eval_score.eval(session= self.monitored_session)
            noise_indices = noise_indices.reshape((int(self.num_workers*self.vec_len),))
            noisy_returns = noisy_returns.reshape((int(self.num_workers*self.vec_len), 2))

            # Process the returns.
            proc_noisy_returns = util.compute_centered_ranks(noisy_returns)

            # Compute and take a step.
            g, count = util.batched_weighted_sum(
                proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
                (self.get_direction(index) for index in noise_indices),
                slice_size=500
            )

            g /= noisy_returns.size

            assert (g.shape == (self.policy_num_params,) and
                g.dtype == np.float32 and
                count == len(noise_indices))

            # Reset the permutated weights.
            self.set_weights(theta)

            # Compute the new weights theta.
            step = self.es_optimizer.update(
                -g + self.l2_coeff * theta)

            # Update ratio
            update_ratio = float(np.linalg.norm(step) / np.linalg.norm(theta))
            theta = theta + step

            step_tend = time.time()

            #print('eval_returns:', eval_returns)
            #print('noisy_returns:', noisy_returns)
            print("total rollout number|  %d" % self.rollout_num)
            print("EvalEpisodesThisIter|  %d" % eval_returns.size)
            print("EvalEpRewMean       |  %.4f" % (eval_returns.mean() if len(eval_returns) > 0 else 0,))
            print("EvalEpRewStd        |  %.2f" % (eval_returns.std() if len(eval_returns) > 0 else 0,))
            print("EvalEpLenMean       |  %.2f" % (eval_lengths.mean() if len(eval_returns) > 0 else 0,))
            print("EpRewMean           |  %.4f" % noisy_returns.mean())
            print("EpRewStd            |  %.2f" % noisy_returns.std())
            print("EpLenMean           |  %.2f" % noisy_lengths.mean())
            print("UpdateRatio         |  %.4f" % update_ratio)
            print("EpisodesThisIter    |  %d"   % noisy_lengths.size)
            print("EpisodesSoFar       |  %d"   % episodes_so_far)
            print("TimestepsThisIter   |  %d"   % noisy_lengths.sum())
            print("TimestepsSoFar      |  %d"   % timesteps_so_far)
            print("TimeElapsedThisIter |  %.2f" % (step_tend - step_tstart))
            print("SyncTimeThisIter    |  %.2f" % (step_sync_tend - step_sync_tstart))
            print("TimeElapsed         |  %.2f" % (step_tend - tstart))
            print("SyncTimeTotal       |  %.2f" % sync_time)

            print(datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S'))
            print('current weights:', self.monitored_session.run(self.weights))
            if iters_so_far % 10 == 0:
                print('optimizer grad square:', self.es_optimizer.v)
            print('optimizer grad square max: {}, min: {}, mean: {}'.format(
                self.es_optimizer.v.max(),
                self.es_optimizer.v.min(),
                self.es_optimizer.v.mean()
            ))
            print('optimizer grad max: {}, min: {}, mean: {}'.format(
                self.es_optimizer.m.max(),
                self.es_optimizer.m.min(),
                self.es_optimizer.m.mean()
            ))

        print("Worker %d finished training." % self.task_index)
        self.release_lock()

class Result(object):
    def __init__(self, noise_indices, noisy_returns, sign_noisy_returns,
        noisy_lengths, eval_returns, eval_lengths):
        self.noise_indices = noise_indices
        self.noisy_returns = noisy_returns
        self.sign_noisy_returns = sign_noisy_returns
        self.noisy_lengths = noisy_lengths
        self.eval_returns = eval_returns
        self.eval_lengths = eval_lengths
