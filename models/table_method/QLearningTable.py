from models.table_method import BaseTable


class QLearningTable(BaseTable):

	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.7):

		super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

	def learn(self, state, action, R, state_):

		state = self.convert2str(state)

		state_ = self.convert2str(state_)

		self.check_state_exist(state_)

		q_predict = self.table.loc[state, action]

		q_target = R + self.gamma * self.table.loc[state_, :].max()

		self.table.loc[state, action] += self.lr * (q_target - q_predict)

		return abs(q_predict - q_target)