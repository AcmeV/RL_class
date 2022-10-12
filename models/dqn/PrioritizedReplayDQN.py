import torch
import numpy as np

from models.dqn.BaseDQN import BaseDQN

np.random.seed(1)
torch.manual_seed(1)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # store transition in self.data
        self.update(tree_idx, p)  # add p to the tree
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # left kid of the parent node
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # kid node is out of the tree, so parent is the leaf node
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):  # stored as (s, a, r, s_) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):

        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max=1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculation ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.cpu().data, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class PrioritizedReplayDQN(BaseDQN):
    def __init__(self, n_actions, n_features, n_hidden=20,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32,
                 e_greedy_increment=None, device='cpu'):

        super(PrioritizedReplayDQN, self).__init__(
            n_actions, n_features, n_hidden, learning_rate,
            reward_decay, e_greedy, replace_target_iter,
            memory_size, batch_size, e_greedy_increment, device)

        self.memory = Memory(capacity=memory_size)

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_))
        # have high priority for newly arrived transition
        self.memory.store(transition)

        # larger TD error has higher priority
        # q_next, q_eval = self.q_target(torch.Tensor(s[np.newaxis, :])), \
        #                  self.q_eval(torch.Tensor(s_[np.newaxis, :]))
        #
        # error = abs(r + torch.max(q_next, 1)[0].item() - q_eval[0, a].item())
        # self.memory.store(error, transition)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # sample batch memory from all memory
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

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

        self.abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)
        loss = torch.mean(torch.mean(torch.Tensor(ISWeights).to(self.device) * (q_target - q_eval) ** 2, dim=1))
        self.memory.batch_update(tree_idx, self.abs_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment \
            if self.epsilon < self.epsilon_max \
            else self.epsilon_max
        self.learn_step_counter += 1

        return loss.item()