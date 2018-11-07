import pandas as pd
import numpy as np
from pgtaa.core.optimize import WeightOptimize


class Env:
    def __init__(self, data: np.ndarray, seed: int):
        """
        Base class for environment objects. Most of it's methods have to be overridden in the subclasses.
        
        Arguments:
            init_portfolio_value {np.ndarray} -- historical data
            seed {int} -- number for random seed
        """
        self.data = data
        np.random.seed(seed)
        
    def observation(self):
        NotImplementedError
        
    def step(self, action: np.ndarray):
        #reward, info = self._step(action)
        #return reward, info
        return self._step(action)
    
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
        """
        Main environment for portfolio management. RL agents are being trained on this environment. 
        
        Arguments:
            data {np.ndarray} -- train or test set of historical data
        
        Keyword Arguments:
            nb_assets {int} -- number of assets in the portfolio (default: {8})
            horizon {int} -- investment horizon (default: {20})
            episodes {int} -- number of training/testing episodes (default: {100})
            window_size {int} -- evaluation window (number of previous days + current day) (default: {100})
            epochs {int} -- number of training epochs, if testing epochs=1 (default: {1})
            risk_aversion {float} -- rate of risk aversion (default: {1.0})
            costs {float} -- rate of composure to risk (default: {0.0025})
            seed {int} -- number for random seed setting (default: {42})
            val_eps {int} -- number of validation episodes, if testing val_eps=None (default: {None})
            predictors {list} -- list of market/asset price predictors (default: {None})
        """
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
    def from_config_spec(cls, data, train_mode=True):
        """
        Reads the config.py specs and initializes the PortfolioEnv based on this file. 
        
        Arguments:
            data {np.ndarray} -- train or test set of historical data
        
        Keyword Arguments:
            train_mode {bool} -- evaluation or training mode (default: {True})

        Returns:
            PortfolioEnv -- initializes PortfolioEnv
        """
        import pgtaa.config as cfg  
        if train_mode:
            episodes = cfg.TRAIN_EPISODES
        else:
            episodes = cfg.TEST_EPISODES
        
        return cls(data, cfg.NB_ASSETS, episodes, cfg.HORIZON, cfg.WINDOW_SIZE, 
                   cfg.PORTFOLIO_INIT_VALUE, cfg.RISK_AVERSION, cfg.COSTS, cfg.SEED)
        
    @property
    def action_space(self):
        # return action space shape
        return self.nb_assets,
    
    @property
    def state_space(self):
        # return state space shape
        return int(0.5 * self.nb_assets * (self.nb_assets + 7)),
    
    
class DataLoader:
    def __init__(self, 
                 data: np.ndarray, 
                 nb_assets: int, 
                 episodes: int,
                 horizon: int, 
                 window_size: int
                ):
        """
        DataLoader for the PortfolioEnv.  
        
        Arguments:
            data {np.ndarray} -- train or test set of historical data
        
        Keyword Arguments:
            nb_assets {int} -- number of assets in the portfolio (default: {8})
            episodes {int} -- number of training/testing episodes (default: {100})
            horizon {int} -- investment horizon (default: {20})
            window_size {int} -- evaluation window (number of previous days + current day) (default: {100})
        """
        self.data = data
        self.horizon = horizon
        self.nb_assets =nb_assets
        self.episodes = episodes
        self.window_size = window_size
        
    def init_batches(self):
        pass
        
    def get_batch(self):
        print(self.episodes)
