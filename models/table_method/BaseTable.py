import json
import os

import numpy as np
import pandas as pd

class BaseTable():

	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, e_increase=3e-4):

		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.e_increase = e_increase

		self.table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, observation):
		''' Choose an action based on current obeservation '''
		observation = self.convert2str(observation)

		self.check_state_exist(observation)

		if np.random.uniform() < self.epsilon:
			state_action = self.table.loc[observation, :]
			action = np.random.choice(state_action[state_action==np.max(state_action)].index)
		else:
			action = np.random.choice(self.actions)

		if self.epsilon < 1.:
			self.epsilon += self.e_increase

		return action

	def learn(self, *args):
		pass

	def check_state_exist(self, state):
		''' Check is state exist, if not exist, insert into table '''
		if state not in self.table.index:
			self.table = self.table.append(
				pd.Series([0]*len(self.actions), index=self.table.columns, name=state))

	def convert2str(self, item):
		''' Convert list or tuple input as str '''
		if isinstance(item, list) or isinstance(item, tuple)\
				or isinstance(item, np.ndarray):
			return str(item)
		else:
			return item

	def save(self, episode, path):
		''' Save model paramters '''
		table = self.table.to_json()
		info_dict = {
			'table': table,
			'episode': episode
		}
		if not os.path.exists(path):
			os.makedirs(path)

		json.dump(info_dict, open(f'{path}/model.json', 'w'))

	def load(self, path):
		''' Load model parameters '''
		info_json = json.load(open(f'{path}/model.json', 'r'))
		self.table = pd.read_json(info_json['table'])
		return int(info_json['episode'])
