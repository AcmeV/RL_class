import os
import json

import torch
import numpy as np
from torch import nn

from models.dqn.net import QNetwork


class BaseDQN():
    def __init__(self, n_actions, n_features, n_hidden=20,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=64,
                 e_greedy_increment=None, device='cpu'):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.device = device

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [state, action, reward, state_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

    def _build_net(self):
        self.q_eval = QNetwork(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.q_target = QNetwork(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss().to(self.device)

    def store_transition(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        state = self._conver_input(state)
        state_ = self._conver_input(state_)
        transition = np.hstack((state, [action, reward], state_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):

        observation = self._conver_input(observation)

        observation = torch.Tensor(observation[np.newaxis, :]).to(self.device)

        if np.random.uniform() < self.epsilon:
            actions_values = self.q_eval(observation)

            action = np.argmax(actions_values.cpu().data.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        pass

    def _conver_input(self, input):
        return np.array(input)

    def save(self, episode, path):
        if not os.path.exists(path):
            os.makedirs(path)


        info_dict = {
            'episode': episode,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'epsilon_max': self.epsilon_max,
            'epsilon_increment': self.epsilon_increment,
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }

        json.dump(info_dict, open(f'{path}/info.json', 'w'))
        # np.save(f'{path}/memory.npy', self.memory)
        torch.save(self.q_eval.state_dict(), f'{path}/q_eval.params')
        torch.save(self.q_target.state_dict(), f'{path}/q_target.params')

    def load(self, path):
        # self.memory = np.load(f'{path}/memory.npy')
        self.q_target.load_state_dict(torch.load(f'{path}/q_target.params', map_location=self.device))
        self.q_eval.load_state_dict(torch.load(f'{path}/q_eval.params', map_location=self.device))
        info_json = json.load(open(f'{path}/info.json', 'r'))

        self.epsilon = info_json['epsilon']
        self.episode = info_json['episode']
        self.gamma = info_json['gamma']
        self.batch_size = info_json['batch_size']
        self.epsilon_max = info_json['epsilon_max']
        self.epsilon_increment = info_json['epsilon_increment']
        self.epsilon = info_json['epsilon']
        self.learn_step_counter = info_json['learn_step_counter']

        return int(info_json['episode'])