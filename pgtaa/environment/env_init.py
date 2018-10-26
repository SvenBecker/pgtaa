import pandas as pd
import numpy as np
from pgtaa.core.optimize import WeightOptimize


class PortfolioInit(object):
    def __init__(self,
                 df: pd.DataFrame,
                 assets: list,
                 episodes: int,
                 epochs: int,
                 seed: int,
                 window: int,
                 risk_aversion: float=1.0,
                 val_eps: int=None
                 ):
        """
        :param df: evaluation dataframe
        :param episodes: number of training/testing episodes
        :param epochs: number of training epochs, if testing epochs=1
        :param seed: random seed
        :param window: evaluation window (number of previous days + current day)
        :param val_eps: number of validation episodes, if testing val_eps=None
        """
        np.random.seed(seed)
        self.df = df
        self.episodes = episodes
        self.epochs = epochs
        self.window = window
        self.assets = assets
        self.risk_aversion = risk_aversion
        episode_starts = np.random.permutation(range(self.window, len(df)))
        self.episode_starts = episode_starts[:episodes]
        self.val_episodes = episode_starts[episodes:episodes + val_eps]
        self.episode_window = self._build_windows()
        self.val_window = self.episode_window[self.episodes:]
        self.episode_window = self.episode_window[:self.episodes]
        self.episode_windows = [np.random.permutation(self.episode_window)
                                for _ in range(self.epochs)]
        self.init_weights = {}
        self.predictions = {}

    def _build_windows(self):
        window = []
        for episode in (self.episode_starts + self.val_episodes):
            w = self.df.iloc[episode-self.window:episode + 1]
            assets = w.iloc[, :self.assets].values
            # TODO: Add model predictions
            self.init_weights[episode] = WeightOptimize(
                covariance_matrix=np.cov(assets),
                asset_returns=assets,
                risk_aversion=self.risk_aversion).optimize_weights()
            window.append(w)
        return window

    def get_windows(self):
        # episode_windows has the shape:
        # (epochs x episodes x time steps x state rows x state columns)

        # val_window has the shape:
        # (episode x time_steps x state rows x state columns)

        # init_weights is a dictionary having the pairs:
        # (episode_start : initial weights)
        return self.episode_windows, self.val_window, self.init_weights
