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
        reward, info = self._step(action)
        return reward, info
    
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


class PortfolioInit(object):
    def __init__(self,
                 data: np.ndarray,
                 nb_assets: int,
                 horizon: int,
                 episodes: int,
                 window_size: int,
                 epochs: int = 1,
                 risk_aversion: float=1.0,
                 val_eps: int=None
                 ):
        """
        :param data: evaluation dataframe
        :param episodes: number of training/testing episodes
        :param epochs: number of training epochs, if testing epochs=1
        :param window: evaluation window (number of previous days + current day)
        :param val_eps: number of validation episodes, if testing val_eps=None
        """
        self.data = data
        self.assets = data[:, :8]
        self.episodes = episodes
        self.epochs = epochs
        self.horizon = horizon
        self.window_size = window_size
        self.nb_assets = nb_assets
        self.risk_aversion = risk_aversion

        # random permutation of episode starting point
        episode_starts = np.random.permutation(range(self.window_size, len(data) - self.horizon))
        self.episode_starts = episode_starts[:episodes]
        self.windows, self.init_weights, self.preds = self._get_windows(*self._build_windows())

        #self.val_window = self.episode_window[self.episodes:]
        #self.episode_window = self.episode_window[:self.episodes]

    def _get_windows(self, window, weights, pred):
        epoch_permutations = [np.random.permutation(self.episodes) for _ in range(self.epochs)]
        windows = []
        init_weights = []
        preds = []
        for i in range(self.epochs):
            windows.append(window[epoch_permutations[i]])
            init_weights.append(weights[epoch_permutations[i]])
            #preds.append(pred[epoch_permutations[i]])
        # windows has the shape (epochs, nb_epsides, horizon, window_size, columns)     4D
        # init_weights has the shape (epochs, nb_episodes, columns)                     2D
        # preds has the shape (epochs, nb_episodes, horizon, columns)                   3D
        return np.array(windows), np.array(init_weights), np.array(preds)

    def _build_windows(self):
        # each window has horizon times subwindows
        w_episodes = []
        init_weights = []
        predictions = []
        for episode in self.episode_starts:
            ws = []
            assets = self.assets[episode - self.window_size: episode]
            weight = WeightOptimize(covariance_matrix=np.cov(assets.T), asset_returns=assets, risk_aversion=self.risk_aversion).optimize_weights()
            for s in range(self.horizon):
                ws.append(self.data[episode - self.window_size + s : episode + s])
            # TODO: Add model predictions
            w_episodes.append(ws)
            init_weights.append(weight)
        # w_episodes has the shape (nb_epsides, horizon, window_size, columns)  4D
        # init_weights has the shape (nb_episodes, columns)                     2D
        # predictions has the shape (nb_episodes, horizon, columns)             3D
        return np.array(w_episodes), np.array(init_weights), np.array(predictions)