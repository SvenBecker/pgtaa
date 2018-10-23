import numpy as np

"""
Environment <-> Runner <-> Agent <-> Model
Each agent has to provide at least the following methods:
- act(states, deterministic) -> returns a dict of actions the agent wants to perform
- observe(reward, terminal) for not causing errors
- close() and reset() also for not causing errors
"""


class BasicAgent(object):
    def __init__(self, action_shape=None):
        self.action_shape = action_shape

    def __str__(self):
        return str(self.__class__.__name__)

    def act(self, state, deterministic=True):
        # has to be overridden
        raise NotImplementedError

    def observe(self, terminal, reward):
        """
        This agent don't really have to observe anything because their policy is fixed and not
        being influenced by state rewards. But for using this in the runner class some
        methods have to be implemented to not cause some error.
        """
        pass

    def restore_model(self, **kwargs):
        pass

    def close(self):
        pass

    def reset(self):
        pass


class BaHAgent(BasicAgent):
    """
    Buy and Hold Agent:
    For each asset the agent performs same action resulting in no further asset weight changes.
    """
    def __init__(self, action_shape=None):
        super(BaHAgent, self).__init__(action_shape=action_shape)
        self.actions = [1 for _ in range(self.action_shape[0])]

    def act(self, state, deterministic=True):
        return self.actions


class RandomActionAgent(BasicAgent):
    """
    This agent does random discrete or continues actions.
    """
    def __init__(self, action_shape=None, discrete=False):
        super(RandomActionAgent, self).__init__(action_shape=action_shape)
        self.discrete = discrete

    def act(self, state, deterministic=True):
        if self.discrete:
            actions = np.random.randint(low=0, high=100, size=self.action_shape[0])
        else:
            actions = np.random.random(self.action_shape[0])
        return actions
