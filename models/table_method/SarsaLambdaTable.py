import json
import os

import pandas as pd

from models.table_method import BaseTable


class SarsaLambdaTable(BaseTable):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.7):

        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.table.copy()

    def check_state_exist(self, state):
        if state not in self.table.index:
            to_be_append = pd.Series([0] * len(self.actions), index=self.table.columns, name=state)
            self.table = self.table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, state, action, R, state_, action_):

        state = self.convert2str(state)
        state_ = self.convert2str(state_)


        self.check_state_exist(state_)

        q_predict = self.table.loc[state, action]
        q_target = R + self.gamma * self.table.loc[state_, action_]
        diff = q_target - q_predict

        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1

        self.table += self.lr * diff * self.eligibility_trace

        self.eligibility_trace *= self.gamma * self.lambda_

        return diff

    def save(self, episode, path):
        table = self.table.to_json()
        eligibility_trace = self.eligibility_trace.to_json()
        info_dict = {
            'table': table,
            'eligibility_trace': eligibility_trace,
            'episode': episode
        }
        if not os.path.exists(path):
            os.makedirs(path)
        json.dump(info_dict, open(f'{path}/model.json', 'w'))

    def load(self, path):
        info_json = json.load(open(f'{path}/model.json', 'r'))
        self.table = pd.read_json(info_json['table'])
        self.eligibility_trace = pd.read_json(info_json['eligibility_trace'])
        return int(info_json['episode'])

