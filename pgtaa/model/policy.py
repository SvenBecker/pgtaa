import torch
import torch.nn as nn
from torch.distributions import Dirichlet
from pgtaa.model.net import build_mlp


class Policy(nn.Module):
    def __init__(self,
                 observation_space: tuple,
                 action_space: tuple,
                 layers: list,
                 load_model: str=None
                 ):
        super(Policy, self).__init__()
        if load_model:
            self.network = torch.load(load_model)
        else:
            self.network = self.network = build_mlp(
                in_dim=observation_space,
                out_dim=action_space,
                layers=layers)

    def forward(self, x):
        pass


class DirichletPolicy(Policy):
    def __init__(self,
                 observation_space: tuple,
                 action_space: tuple,
                 layers: list,
                 load_model: str=None
                 ):
        super(DirichletPolicy, self).__init__(
            observation_space,
            action_space,
            layers,
            load_model)
        self.distribution = None
        self.action = None

    def get_action(self, observation, mode: str="train"):
        alphas = self.network(observation)
        self.distribution = Dirichlet(alphas)
        if mode == "train":
            self.action = self.distribution.sample()
        else:
            # else evaluation mode
            self.action = self.distribution.mean
        return self.action

    def log_prob(self):
        # return the negative log probability
        return self.distribution.log_prob(self.action)

    def forward(self, x):
        pass
