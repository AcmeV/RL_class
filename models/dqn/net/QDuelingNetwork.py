import torch
from torch import nn


class QDuelingNetwork(nn.Module):

    def __init__(self, n_features, n_hiddens, n_actions):

        super(QDuelingNetwork, self).__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(n_features, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU())

        self.value_net = nn.Linear(n_hiddens, 1)

        self.advantage_net = nn.Linear(n_hiddens, n_actions)

    def forward(self, X):

        X = self.pre_net(X)

        value = self.value_net(X)
        advantages = self.advantage_net(X)
        out = value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

        return out