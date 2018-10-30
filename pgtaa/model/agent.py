import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pgtaa.model.net import build_mlp
from pgtaa.model.policy import DirichletPolicy
from pgtaa.model.memory import Memory


class Agent:
    def __init__(self,
                 state_space: tuple=None,
                 action_space: tuple=None,
                 layers: list=None,
                 batch_size: int=1,
                 lr: float=1e-3,
                 eps: float=1e-7,
                 gamma: float=0.98,
                 memory_capacity: int= 100,
                 load_model: str=None
                 ):

        """
        All agents inherit from this class. The class method
        update have to be overridden in the subclasses.


        :param state_space: state space dimension => input size for the neural net
        :param action_space: action space dimension => output size size for the neural net
        :param layers: number of cells on each layer
        :param batch_size: batch size for network updates
        :param lr: learning rate for optimizer
        :param eps: epsilon parameter for optimizer
        :param gamma: discount factor for future rewards
        :param load_model: path to pretrained agent if there is one
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy = DirichletPolicy(state_space, action_space, layers, load_model)
        self.optimizer = optim.Adam(self.policy.parameters(), lr, eps)
        self.memory = Memory(memory_capacity)

        self.timestep = None
        self.reward = None

    def __str__(self):
        return str(self.__class__.__name__)

    def observe(self, state: np.ndarray, reward: float, benchmark_reward: float):
        # using a fixed bias for estimating the advantage
        advantage = reward - benchmark_reward
        NotImplementedError

    def update(self):
        NotImplementedError

    def experience(self, observation, action, reward, next_observation):
        NotImplementedError

    def act(self, observation, mode="train"):
        if mode == "train":
            self.policy.train()
            action = self.policy(observation)
        else:
            self.policy.eval()
            action = self.policy.mean
        return action

    @classmethod
    def from_dict(self, dict):
        pass

    def save(self, path):
        """
        :param path: defines the path + file where the agent will be saved to
        """
        torch.save(self.policy, path)


class PPOAgent(Agent):
    def __init__(self,
                 state_space: tuple = None,
                 action_space: tuple = None,
                 layers: list = None,
                 batch_size: int = 1,
                 lr: float = 1e-3,
                 eps: float = 1e-7,
                 gamma: float = 0.98,
                 load_model: str = None,
                 memory_capacity: int = 100,
                 clip_param: float=0.2,
                 entropy_factor: float= 0.
                 ):
        """
        See Superclass
        :param clip_param: clipping value if given, else kl_penalty
        :param entropy_factor: factor 
        """
        super(PPOAgent, self).__init__(
            state_space,
            action_space,
            layers,
            batch_size,
            lr,
            eps,
            gamma,
            memory_capacity,
            load_model
        )
        self.clip_param = clip_param
        self.entropy_factor = entropy_factor

    def update(self):
        # updates network after x number of time steps
        self.optimizer.step()
        pass

    def observe(self, state: np.ndarray, reward: float, benchmark_reward: float):

        pass

    def clip(self, x):
        return torch.clamp(x, -self.clip_param, self.clip_param)

    def kl_penalty(self, x):
        return x

    def ppo_loss(self, action_loss, entropy):
        loss = action_loss - entropy * self.entropy_factor
        return loss.backward()


