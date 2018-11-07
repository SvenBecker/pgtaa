import pandas as pd
import numpy as np
from pgtaa.core.optimize import WeightOptimize


class Env:
    def __init__(self, data: np.ndarray, seed: int):
        self.data = data
        np.random.seed(seed)
        
    def observation(self):
        NotImplementedError
        
    def step(self, action: np.ndarray):
        #reward, info = self._step(action)
        #return reward, info
        pass
    
    def reset(self):
        return self._reset()
    
    def _step(self, action):
        NotImplementedError
    
    def _reset(self):
        NotImplementedError
        
    def __str__(self):
        return str(self.__class__.__name__)
    
    def __len__(self):
        return len(self.data)
    
    @property
    def action_space(self):
        NotImplementedError
    
    @property
    def state_space(self):
        NotImplementedError
        

class PortfolioEnv(Env):
    def __init__(
        self, 
        data: np.ndarray,
        nb_assets: int = 8,
        episodes: int = 100,
        horizon: int = 30,
        window_size: int = 100,
        portfolio_value: float = 1000.,
        risk_aversion: float = 1.,
        costs: float = 0.025,
        seed: int = 42
    ):
        super(PortfolioEnv, self).__init__(data, seed)
        self.nb_assets = nb_assets
        #self.horizon = horizon
        #self.window_size = window_size
        self.portfolio_value = portfolio_value
        self.risk_aversion = risk_aversion
        self.costs = costs
        self.dl = DataLoader(data, nb_assets, episodes, horizon, window_size)
        
    def observation(self):
        weights, var_covar, returns, mean, areturn = ()
        
    @classmethod
    def from_config_spec(cls, data, mode="train"):
        import pgtaa.config as cfg  
        if mode == "train":
            episodes = cfg.TRAIN_EPISODES
        else:
            episodes = cfg.TEST_EPISODES
        return cls(data, cfg.NB_ASSETS, episodes, cfg.HORIZON, cfg.WINDOW_SIZE, 
                   cfg.PORTFOLIO_INIT_VALUE, cfg.RISK_AVERSION, cfg.COSTS, cfg.SEED)
        
    @property
    def action_space(self):
        return self.nb_assets,
    
    @property
    def state_space(self):
        return int(0.5 * self.nb_assets * (self.nb_assets + 7)),
    
    
class DataLoader:
    def __init__(self, 
                 data: np.ndarray, 
                 nb_assets: int, 
                 episodes: int,
                 horizon: int, 
                 window_size: int
                ):
        
        self.data = data
        self.horizon = horizon
        self.nb_assets =nb_assets
        self.episodes = episodes
        self.window_size = window_size
        
    def init_batches(self):
        pass
        
    def get_batch(self):
        print(self.episodes)
