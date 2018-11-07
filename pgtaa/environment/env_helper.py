import numpy as np  
import pandas as pd
from pgtaa.core.optimize import WeightOptimize


class Portfolio:
    """
    Provides some useful utility for basic portfolio calculations.
    Keeps track of portfolio development (like variance, return, sharpe ratio etc.).
    Has to be called for each new episode.
    """
    def __init__(
            self,
            init_portfolio_value: float=100.0,
            risk_aversion: float=1.0,
            costs: float=0.025
    ):

        self.init_portfolio_value = init_portfolio_value
        self.portfolio_value = init_portfolio_value
        self.costs = costs
        self.risk_aversion = risk_aversion
        self.weights = None
        self.new_weights = None
        self.covariance = None
        self.portfolio_return = 0.
        self.cost = 0.
        self.variance = 0.
        self.sharpe = 0.

    def __str__(self):
        return str(self.__class__.__name__)

    def update(self, actions):
        # update portfolio weights based on given actions
        self.new_weights = actions

        # calculate weight difference
        _weight_diff = np.array(self.new_weights) - np.array(self.weights)

        # estimate costs for trading
        self.cost = self._get_cost(_weight_diff)

        # update portfolio value
        self.portfolio_value -= self.cost

        return self.new_weights, self.cost, self.portfolio_value

    def step(self, asset_returns, covariance):
        """
        Args:
            :param asset_returns: (list) asset returns price_t / price_{t-1} - 1
            :param covariance: (object) variance-covariance matrix based on current window
        :return: reward, weights, portfolio value
        """
        # step forward and get new window
        self.covariance = covariance

        # get new weights
        self.weights = self._get_weights(asset_returns)

        # calculate portfolio return
        self.portfolio_return = np.dot(self.new_weights, asset_returns)

        # calculate new portfolio variance based on updated window
        self.variance = self._get_variance()

        # get reward based on reward function
        step_reward = self._get_reward()

        # update portfolio value
        self.portfolio_value *= (self.portfolio_return + 1)

        # calculate sharpe ratio
        self.sharpe = self._sharpe_ratio()

        return step_reward, self.weights, self.portfolio_value

    def reset(self, weights, covariance):
        # resets the portfolio value and the asset weights to their initial value
        self.portfolio_value = self.init_portfolio_value
        self.weights = weights
        self.covariance = covariance
        self.variance = self._get_variance()
        self.sharpe = self._sharpe_ratio()

    def _get_weights(self, asset_returns):
        # change of portfolio weights given possible deviations in asset returns.
        # multiplication by broadcating
        return asset_returns * self.new_weights / np.sum(asset_returns * self.new_weights)

    def _get_cost(self, weight_diff):
        # cost for trading based on trading volume
        cost = 0
        sum_weight_diff = sum(abs(weight_diff))     # sum over absolute weight difference
        if round(sum_weight_diff, 5) != 0:
            cost += sum_weight_diff * self.portfolio_value * self.costs
        return cost

    def _get_reward(self):
        # returns the perceived reward (utility) of the portfolio.
        return self.portfolio_return - self.risk_aversion / 2 * self.variance - \
               self.cost / self.portfolio_value

    def _sharpe_ratio(self, risk_free=0.0):
        # reward-to-Variability-Ratio (R_p - R_f) / sigma_R_p
        # risk free rate on daily returns can be assumed to be zero
        return (self.portfolio_value / self.init_portfolio_value - 1 - risk_free) \
                / np.sqrt(self.variance)

    def _get_variance(self):
        # returns the portfolio variance
        return self.weights @ self.covariance @ self.weights


class PortfolioInit(object):
    def __init__(self,
                 data: np.ndarray,
                 nb_assets: int,
                 horizon: int,
                 episodes: int,
                 window_size: int,
                 epochs: int = 1,
                 risk_aversion: float=1.0,
                 val_eps: int=None,
                 predictors: list=None
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
        self.predictors = predictors

        # random permutation of episode starting point
        episode_starts = np.random.permutation(range(self.window_size, len(data) - self.horizon))
        self.episode_starts = episode_starts[:episodes]
        self.windows, self.init_weights, self.preds = self._get_windows(*self._build_windows())
        np.save("window", self.windows)
        np.save("weights", self.init_weights)
        np.save("preds", self.preds)

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
        # windows has the shape (epochs, nb_epsides, horizon, window_size, columns)     5D
        # init_weights has the shape (epochs, nb_episodes, columns)                     3D
        # preds has the shape (epochs, nb_episodes, horizon, predictors, columns)       4D
        return np.array(windows), np.array(init_weights), np.array(preds)

    def _build_windows(self):
        # each window has horizon times subwindows
        w_episodes = []
        init_weights = []
        predictions = []
        for episode in self.episode_starts:
            ws = []
            prd = []
            assets = self.assets[episode - self.window_size: episode]
            weight = WeightOptimize(covariance_matrix=np.cov(assets.T), asset_returns=assets, risk_aversion=self.risk_aversion).optimize_weights()
            for s in range(self.horizon):
                w = self.data[episode - self.window_size + s : episode + s]
                ws.append(w)
                # TODO: Add model predictions
                prd.append([predictor.predict(w) for predictor in self.predictors])
            w_episodes.append(ws)
            init_weights.append(weight)
        # w_episodes has the shape (nb_epsides, horizon, window_size, columns)      4D
        # init_weights has the shape (nb_episodes, columns)                         2D
        # predictions has the shape (nb_episodes, horizon, num_predictors, columns) 3D
        return np.array(w_episodes), np.array(init_weights), np.array(predictions)