import torch
import numpy as np
from torch import nn

from models.dqn.BaseDQN import BaseDQN
from models.dqn.net import QDuelingNetwork


class DuelingDQN(BaseDQN):
    def __init__(self, n_actions, n_features, n_hidden=20,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32,
                 e_greedy_increment=None, device='cpu'):

        super(DuelingDQN, self).__init__(
            n_actions, n_features, n_hidden, learning_rate,
            reward_decay, e_greedy, replace_target_iter,
            memory_size, batch_size, e_greedy_increment, device)

    def _build_net(self):
        self.q_eval = QDuelingNetwork(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.q_target = QDuelingNetwork(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss().to(self.device)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_next is used for getting which action would be choosed by target network in state s_(t+1)
        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:]).to(self.device)), \
                         self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]).to(self.device))
        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.cpu().data.numpy().copy()).to(self.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1]).to(self.device)

        # torch.max(data, dim)[0]: max_value in each dimension
        # torch.max(data, dim)[1]: max_value's index in each dimension
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]

        loss = self.loss_func(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment \
            if self.epsilon < self.epsilon_max \
            else self.epsilon_max
        self.learn_step_counter += 1

        return loss.item()