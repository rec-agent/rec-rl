from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import os
import sys
path = os.path.abspath('../../')
sys.path.append(path)

from tensorforce.environments.meta_environment import MetaEnvironment
import tensorforce.util as utl
from tensorforce.exception import TensorforceError
from rec_input_fn_local import input_fn as input_fn_local

def _invert_permutation(tensor):
    '''wrapper for matrix'''
    return tf.cast(tf.map_fn(tf.invert_permutation, tensor), tf.float32)

def _gather(param, indices):
    '''wrapper for matrix'''
    return tf.map_fn(lambda x : tf.gather(x[0], x[1]), (param, indices), dtype=param.dtype)

class RecTableEnv(MetaEnvironment):
    '''
    ODPS Table env for gul ranking scenario.
    '''
    def __init__(self, config):
        config['env_type'] = 'odps_table'
        #read config from json, load 'env' into RecTableEnv
        super(RecTableEnv, self).__init__(config)

        # parse more config
        self.parse_env_config()

        self._version = '0.1'

        self.sess = None

    def __str__(self):
        return 'RecTableEnv({})'.format(self._version)

    def parse_env_config(self):
        """
        Obtain table name,schema and partition
        """
        print('env config:', self.env_conf)

        # get worker_num and worker_id, if not has key, then set default value as 1 and 0 in get()
        self.worker_num = self.env_conf.get('worker_num', 1)
        self.worker_id = self.env_conf.get('worker_id', 0)

        # get table name 获取数据集地址
        if 'tables' not in self.env_conf:
            raise TensorforceError("Can't find tables in configuration")
        self.tables = self.env_conf['tables']
        self.epoch = self.env_conf.get('epoch', None)
        self.batch_size = self.env_conf.get('batch_size', 100)
        self.capacity = self.env_conf.get('capacity', 4 * self.batch_size)
        self.max_pageid = self.env_conf.get('max_pageid', 7)
        self.discount_base = self.env_conf.get('discount_base', 0.8)
        self.local_mode = self.env_conf.get('local_mode', False)
        self.alipay_coef = self.env_conf.get('alipay_coef', 1.0)
        self.reward_shaping_method = self.env_conf.get('reward_shaping_method', None)
        self.alipay_threshold = self.env_conf.get('alipay_threshold', 0.0)
        self.alipay_penalty = self.env_conf.get('alipay_penalty', 0.0)

        '''
        ranking_formula_type 0: ctr * cvr^a * price^b
        ranking_formula_type 1: (ctr * cvr^a * price^b) * matchtype_weight
        ranking_formula_type 2: (a * ctr + ctr * cvr^b * price^c) * matchtype_weight
        ranking_formula_type 3: (a * ctr + b * cvr + ctr * cvr^c * price^d) * matchtype_weight
        ranking_formula_type 4: (a * ctr + b * ctr * cvr + ctr * cvr^c * price^d) * matchtype_weight
        '''
        #默认ranking function的type是0
        self.ranking_formula_type = self.env_conf.get('ranking_formula_type', 0)
        self.feature_include_hour_power = self.env_conf.get('feature_include_hour_power', False)
        self.feature_include_age_gender = self.env_conf.get('feature_include_age_gender', False)


        self.states_spec = {}
        #设置当前state的feature为pageid，做完one-hot后，维度为8
        feature_dim = self.max_pageid + 1
        if self.feature_include_hour_power:
            feature_dim += 32
        if self.feature_include_age_gender:
            feature_dim += 12

        self.states_spec['state'] = {
            'type': 'float',
            'shape': (feature_dim,)
        }

        self.actions_spec = {}
        if self.ranking_formula_type == 0:
            action_shape = 2
        elif self.ranking_formula_type == 1:
            action_shape = 6
        elif self.ranking_formula_type == 2:
            action_shape = 7
        elif self.ranking_formula_type in (3, 4):
            action_shape = 8
        else:
            raise TensorforceError("Invalid ranking formula type " + str(self.ranking_formula_type))

        self.actions_spec['action'] = {
            'type': 'float',
            #因为默认ranking function是type0，因此只需要两个action
            'shape': (action_shape,),
            'min_value': -1.0,
            'max_value': 2.0
        }

        print('states:', self.states)
        print('actions:', self.actions)

    def set_up(self):
        #local_mode从json文件中读取，且默认为True
        if self.local_mode:
            print('load data in local mode')
            #读取数据，并赋给batch data
            self.batch_data = input_fn_local(
                name='table_env',
                tables=self.tables,
                num_epochs=self.epoch,
                num_workers=self.worker_num,
                worker_id=self.worker_id,
                batch_size=self.batch_size
            )
            # print('!!!!!!')
            # print(self.tables)
            # print(self.epoch)
            # print(self.worker_num)
            # print(self.worker_id)
            # print(self.batch_size)
            # print(self.batch_data)
            self.device = ("/job:localhost/replica:0/task:%d" % self.worker_id) if self.worker_id != -1 else 0
        else:
            self.batch_data = input_fn_local(
                name='table_env',
                tables=self.tables,
                num_epochs=self.epoch,
                num_workers=self.worker_num,
                worker_id=self.worker_id,
                batch_size=self.batch_size,
                capacity=self.capacity
            )
            self.device = ("/job:worker/task:%d" % self.worker_id) if self.worker_id != -1 else 0
        # print('!!!!!!')
        # print(self.batch_data['ctr'])
        # with tf.Session() as sess:
            # sess.run(self.batch_data['ctr'].initializer)
            # print((self.batch_data['ctr']).eval())
        with tf.variable_scope(name_or_scope='table_env') as scope:
            with tf.device(device_name_or_function = self.device):
                self.build_graph()

    def get_input_tensor(self):
        """
        Get the input tensor for agent
        """
        data = {}
        data['states'] = {}
        data['states']['states'] = self.states_tensor

        return data

    def set_session(self, session):
        self.sess = session

    def update(self):
        if self.sess is None:
            raise TensorforceError("self.session is None")

        self.sess.run([self.batch_data, self.assign_cache_ops])
        # print('!!!!!!!!')
        # print((self.batch_data['ctr']).eval(session=self.sess))

    def reset(self):
        self.update()

        return self.states_tensor

    def build_graph(self):
        self.cache_data = {}

        self.cache_data['pageid'] = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32),
                                                trainable=False,
                                                name='pageid_var')
        self.cache_data['ctr'] = tf.Variable(tf.zeros([self.batch_size, 50], dtype=tf.float32),
                                             trainable=False,
                                             name='ctr_var')
        self.cache_data['cvr'] = tf.Variable(tf.zeros([self.batch_size, 50], dtype=tf.float32),
                                             trainable=False,
                                             name='cvr_var')
        self.cache_data['price'] = tf.Variable(tf.zeros([self.batch_size, 50], dtype=tf.float32),
                                               trainable=False,
                                               name='price_var')
        self.cache_data['click'] = tf.Variable(tf.zeros([self.batch_size, 50], dtype=tf.float32),
                                               trainable=False,
                                               name='click_var')
        self.cache_data['pay'] = tf.Variable(tf.zeros([self.batch_size, 50], dtype=tf.float32),
                                             trainable=False,
                                             name='pay_var')


        if self.feature_include_hour_power:
            self.cache_data['hour'] = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32),
                                                    trainable=False,
                                                    name='hour_var')
            self.cache_data['power'] = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32),
                                                    trainable=False,
                                                    name='power_var')
            hour = self.cache_data['hour']
            power = self.cache_data['power']
        if self.feature_include_age_gender:
            self.cache_data['age'] = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32),
                                                    trainable=False,
                                                    name='age_var')
            self.cache_data['gender'] = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32),
                                                    trainable=False,
                                                    name='gender_var')
            age = self.cache_data['age']
            gender = self.cache_data['gender']

        
        if self.ranking_formula_type in (1, 2, 3, 4):
            self.cache_data['matchtype'] = tf.Variable(tf.zeros([self.batch_size, 50], dtype=tf.int32),
                                                 trainable=False,
                                                 name='matchtype_var')
            matchtype = self.cache_data['matchtype']

        self.assign_cache_ops = {}
        for tensor_name in self.cache_data.keys():
            self.assign_cache_ops[tensor_name] = tf.assign(self.cache_data[tensor_name], self.batch_data[tensor_name], name=tensor_name + 'assign_cache')
        # print('!!!!!!!')
        # print(self.assign_cache_ops)
        
        ctr = self.cache_data['ctr']
        cvr = self.cache_data['cvr']
        price = self.cache_data['price']
        click = self.cache_data['click']
        pay = self.cache_data['pay']

        self.actions_input = tf.placeholder(tf.float32, shape=None, name='env_action')
        # print('!!!!!!!')
        # print(self.actions_input)

        offset = 0
        if self.ranking_formula_type in (2, 3, 4):
            ctr_weight = tf.reshape(self.actions_input[:,0], (-1,1))
            offset += 1
        if self.ranking_formula_type in (3, 4):
            cvr_weight = tf.reshape(self.actions_input[:,offset], (-1,1))
            offset += 1

        cvr_power = tf.reshape(self.actions_input[:,offset], (-1,1))
        # print('!!!!!!!')
        # print(cvr_power)
        price_power = tf.reshape(self.actions_input[:,1 + offset], (-1,1))
        # print(price_power)

        rank_score = ctr * tf.pow(cvr,cvr_power) * tf.pow(price,price_power)                                                                                                                                                   
        if self.ranking_formula_type == 2:
            rank_score = rank_score + ctr * ctr_weight
        elif self.ranking_formula_type == 3:
            rank_score = rank_score + ctr * ctr_weight + cvr * cvr_weight
        elif self.ranking_formula_type == 4:
            rank_score = rank_score + ctr * ctr_weight + ctr * cvr * cvr_weight

        if self.ranking_formula_type in (1, 2, 3, 4):
            matchtype_params = self.actions_input[:, 2 + offset : 6 + offset]
            i2i_param = tf.ones([self.batch_size, 1], tf.float32)
            full_matchtype_params = tf.concat([i2i_param, matchtype_params], axis=1)
            matchtype_weights = _gather(full_matchtype_params, matchtype)
            rank_score = rank_score * matchtype_weights

        sorted_rank_score, sorted_index = tf.nn.top_k(rank_score, k=50, sorted=True)
        # tf.invert_permutation only support 1-D vector, wrap it for matrix
        perm_index = _invert_permutation(sorted_index)
        pos_discount = tf.pow(self.discount_base, perm_index)

        discounted_click = click * pos_discount
        discounted_pay = pay * pos_discount

        self.pv_discount_click = tf.reduce_sum(discounted_click, 1)
        self.pv_discount_click_mean = tf.reduce_mean(self.pv_discount_click, 0)
        self.pv_discount_pay = tf.reduce_sum(discounted_pay, 1)
        self.pv_discount_pay_mean = tf.reduce_mean(self.pv_discount_pay, 0)

        #clip_by_value可以将一个张量中的数值限制在一个范围之内,此处是[0，7],cache_data包含一个batch的数据，所以大小是batch_size
        pageid = tf.clip_by_value(self.cache_data['pageid'], 0, self.max_pageid)
        # print(self.cache_data['pageid'])
        # print(pageid)
        self.pageid_onehot = tf.one_hot(pageid, depth=self.max_pageid + 1, dtype=tf.float32)
        # print('!!!!!!!')
        # print(self.pageid_onehot.eval())
        feature_list = [self.pageid_onehot]

        # print(feature_list)
        if self.feature_include_hour_power:
            self.hour_onehot = tf.one_hot(hour, depth=24, dtype=tf.float32)
            self.power_onehot = tf.one_hot(power, depth=8, dtype=tf.float32)
            feature_list.append(self.hour_onehot)
            feature_list.append(self.power_onehot)
        if self.feature_include_age_gender:
            self.age_onehot = tf.one_hot(age, depth=9, dtype=tf.float32)
            self.gender_onehot = tf.one_hot(gender, depth=3, dtype=tf.float32)
            feature_list.append(self.age_onehot)
            feature_list.append(self.gender_onehot)

        if len(feature_list) == 1:
            self.states_tensor = self.pageid_onehot
        else:
            self.states_tensor = tf.concat(feature_list, 1)

        print('build graph done')


    #返回走一步之后的下一个状态，是否终止，以及reward
    def execute(self, actions):
        """
        Interact with the environment
        if set interactive to True, env.execute will apply an action to the environment and
        get an observation after the action

        actions are batch_size * 3 tensor

        return (next_state, step_reward, terminal)
        """
        step_click, step_pay = self.sess.run([self.pv_discount_click_mean, self.pv_discount_pay_mean], feed_dict={self.actions_input: actions})

        return (None, True, self.get_reward(step_click, step_pay))

    def get_reward(self, click, pay):
        if self.reward_shaping_method is None:
            return click + pay
        elif self.reward_shaping_method == 'weighting':
            return click + self.alipay_coef * pay
        elif self.reward_shaping_method == 'penalty':
            if pay >= self.alipay_threshold:
                return click + pay
            else:
                return click + pay - self.alipay_penalty * (self.alipay_threshold - pay)

    def close(self):
        pass

    @property
    def states(self):
        return self.states_spec

    @property
    def actions(self):
        return self.actions_spec

if __name__ == '__main__':
    import json
    with open('rec_env_config_local.json', 'rb') as fp:
        config = json.load(fp=fp)
    print('config:', config)
    action_val = tf.constant(np.array([[1,1,1], [1,1,1], [1,0.83,0.83], [1,0.67,0.67], [1,0.5,0.5], [1,0.33,0.33], [1,0.17,0.17], [1,0.0,0.0]], dtype=np.float32))
    env = RecTableEnv(config)
    sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    env.set_session(sess)
    env.set_up()
    cur_action = tf.matmul(env.pageid_onehot, action_val)

    ###print(env.pageid_onehot) (100,8) 因为batchsize = 100
    ###print(action_val) (8,3)
    ###print(cur_action) (100,3)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        for i in range(4):
            print('pageid_onehot:', sess.run([env.reset()]))
            print('pageid cached:', sess.run(env.cache_data['pageid']))
            print('pageid cached again:', sess.run(env.cache_data['pageid']))
            cur_action_val = sess.run(cur_action)
            print('cur_action:', cur_action_val)
            #print('cur_action again:', sess.run(cur_action))
            next_state, terminal, reward = env.execute(cur_action_val)
            print('next state:', next_state)
            print('reward:', reward)

    except tf.errors.OutOfRangeError:
        print('data is out of range')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
