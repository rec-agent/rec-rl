# -*- coding: utf-8 -*-
"""
Created on 2018-07-13

@author: xinru.yxr
"""

from odps.udf import BaseUDTF
from odps.udf import annotate
import numpy as np

@annotate('*->double,double,double,double,double,double,double,double,double')
class Processor(BaseUDTF):
    def __init__(self):
        self.actions = None
    def process(self, action, pageid, hour, power, ctr, cvr, price, isclick, pay,
        ranking_formula_type=1, matchtype=None, age=None, gender=None, discount=0.8):
        '''
        calculate the reward of specified action

        action is the format of '1,1,1;...;0.8,0.8,0.8'
        '''
        for field in [ctr, cvr, price, isclick, pay]:
            if field is None or field.strip() == '':
                return

        if self.actions is None:
            actions = self.parse_action(action)
        else:
            actions = self.actions
        #print 'action length: ', len(actions)

        if pageid > len(actions):
            raise RuntimeError('pageid {} exceed the length of action {}'.format(pageid, len(actions)))

        pageid_act = np.asarray(actions[pageid])
        act = pageid_act
        if len(actions) > 12:
            hour_act = np.asarray(actions[12 + hour])
            power_act = np.asarray(actions[12 + 24 + power])
            act += (hour_act + power_act)
        if len(actions) > 44:
            assert age is not None and gender is not None, 'age and gender can not be none when act rows > 40'
            age_act = np.asarray(actions[44 + age])
            gender_act = np.asarray(actions[53 + gender])
            act += (age_act + gender_act)

        ctr_val = self.parse_seq(ctr)
        cvr_val = self.parse_seq(cvr)
        price_val = self.parse_seq(price)
        click_val = self.parse_seq(isclick)
        pay_val = self.parse_seq(pay)

        if len(ctr_val) != len(cvr_val) or len(ctr_val) != len(price_val) or \
                len(ctr_val) != len(click_val) or len(ctr_val) != len(pay_val):
            print ctr, cvr, price, isclick, pay
            raise Exception('ctr, cvr, price, isclick, pay vector length are not equal')

        offset = 0
        if ranking_formula_type in (2, 3, 4):
            ctr_weight = act[0]
            offset += 1
        if ranking_formula_type in (3, 4):
            cvr_weight = act[offset]
            offset += 1
        cvr_power = act[offset]
        price_power = act[1 + offset]

        rank_score = ctr_val * np.power(cvr_val, cvr_power) * np.power(price_val, price_power)

        '''
        ranking_formula_type 0: ctr * cvr^a * price^b
        ranking_formula_type 1: (ctr * cvr^a * price^b) * matchtype_weight
        ranking_formula_type 2: (a * ctr + ctr * cvr^b * price^c) * matchtype_weight
        ranking_formula_type 3: (a * ctr + b * cvr + ctr * cvr^c * price^d) * matchtype_weight
        ranking_formula_type 4: (a * ctr + b * ctr * cvr + ctr * cvr^c * price^d) * matchtype_weight
        '''

        if ranking_formula_type in (0, 1):
            pass
        elif ranking_formula_type == 2:
            rank_score = rank_score + ctr_val * ctr_weight
        elif ranking_formula_type == 3:
            rank_score = rank_score + ctr_val * ctr_weight + cvr_val * cvr_weight
        elif ranking_formula_type == 4:
            rank_score = rank_score + ctr_val * ctr_weight + ctr_val * cvr_val * cvr_weight
        else:
            raise Exception('ranking_formula_type {} is invalid'.format(ranking_formula_type))


        if ranking_formula_type in (1, 2, 3, 4):
            assert matchtype is not None, 'matchtype should not be None when ranking_formula_type in (1,2)'
            assert act.size >= 6, 'act length {} must be >= 6 in matchtype case!'.format(act.size)
            matchtype_val = self.parse_seq(matchtype, dtype=np.int32)
            matchtype_act = act[2 + offset : 6 + offset]
            # set i2i_w to 1.0
            matchtype_act = np.concatenate(([1.0], matchtype_act))
            matchtype_weights = np.take(matchtype_act, matchtype_val)
            rank_score = rank_score * matchtype_weights

        sorted_click_pay = sorted(zip(rank_score, click_val, pay_val), key=lambda t: -t[0])


        #Calculate precision, recall, map. By xinru.yxr
        sorted_click_by_score = sorted(zip(rank_score, click_val), key=lambda t: -t[0])
        total_num_click = sum(t[1] for t in sorted_click_by_score)
        top20_sorted_click_by_score = sorted_click_by_score[:20]
        top20_total_num_click = sum(t[1] for t in top20_sorted_click_by_score) * 1.0
        
        precision = top20_total_num_click / 20   
        if (total_num_click > 0):
            recall = top20_total_num_click / sum(t[1] for t in sorted_click_by_score)
        else:
            recall = 0
        
        count_1 = 0.
        map_ = 0.
        for i in range(20):
            if top20_sorted_click_by_score[i][1] > 0:
                count_1 = count_1 + 1.
                map_ = map_ + count_1 / (i + 1)
        if (count_1 > 0):
            map_ = map_ / count_1

        # reward = click_val + np.log1p(pay_val)
        # reward = click_val + pay_val
        #sorted_reward = [x for _, x in sorted(zip(rank_score, reward), key=lambda t: -t[0])]
        #sorted_reward = np.asarray(sorted_reward, dtype=np.float32)
        sorted_click = [t[1] for t in sorted_click_pay]
        sorted_click = np.asarray(sorted_click, dtype=np.float32)
        sorted_pay = [t[2] for t in sorted_click_pay]
        sorted_pay = np.asarray(sorted_pay, dtype=np.float32)
        sorted_reward = sorted_click + sorted_pay

        discount_val = np.power(discount, np.arange(len(rank_score)))
        discounted_click = np.sum(sorted_click * discount_val)
        discounted_pay = np.sum(sorted_pay * discount_val)
        discounted_reward = np.sum(sorted_reward * discount_val)

        sorted_totalvalue = sorted_click * ctr_val * price_val + sorted_pay
        discounted_totalvalue = np.sum(sorted_totalvalue * discount_val)


        click_pos_sum = self.get_rank_pos(rank_score, click_val)
        pay_pos_sum = self.get_rank_pos(rank_score, pay_val)

        self.forward(float(discounted_click),
                     float(discounted_pay),
                     float(discounted_reward),
                     click_pos_sum,
                     pay_pos_sum,
                     float(precision),
                     float(recall),
                     float(map_),
                     float(discounted_totalvalue))

    def get_rank_pos(self, rank_score, id_vec):
        comb = zip(rank_score, id_vec)
        sorted_comb = sorted(comb, key=lambda t: -t[0])

        pos_sum = 0.0
        for index, kvp in enumerate(sorted_comb):
            if kvp[1] > 0:
                pos_sum += index
        return pos_sum

    def parse_action(self, s):
        '''
        action is the format of '1,1,1;...;0.8,0.8,0.8'
        '''
        tokens = s.strip().split(';')
        actions = [map(float, t.split(',')) for t in tokens]
        return actions

    def parse_seq(self, s, dtype=np.float32):
        tokens = s.strip().split(',')
        if dtype == np.float32:
            return np.asarray(map(float, tokens), dtype=np.float32)
        elif dtype == np.int32:
            return np.asarray(map(int, tokens), dtype=np.int32)
        else:
            raise Exception(str(dtype) + 'not supported')