import random
from collections import namedtuple

# Reference:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py

Transition = namedtuple("Transition", "observation action reward next_observation")


class Memory(object):
    """
    Enables experience replay
    """
    def __init__(self, capacity: int):
        """
        :param capacity: max memory capacity
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, observation, action, reward, next_observation):
        """
        Push a trajectory to memory
        :param observation: current observation
        :param action: agent action
        :param reward: received reward
        :param next_observation: observation at next time step
        """

        assert observation.shape != next_observation.shape, "observation space shapes do not match"

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(observation, action, reward, next_observation))
        else:
            self.memory[self.position].append(Transition(observation, action, reward, next_observation))
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """

        :param batch_size:
        :return: sampled trajectories
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
