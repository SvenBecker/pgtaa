import numpy as np  
import pandas as pd
from pgtaa.core.optimize import WeightOptimize


class Portfolio:

    def __init__(
            self,
            init_portfolio_value: float=100.0,
            risk_aversion: float=1.0,
            costs: float=0.025
    ):
        """
        Provides some useful utility for basic portfolio calculations.
        Keeps track of portfolio development (like variance, return, sharpe ratio etc.).
        Has to be called for each new episode.
        
        Keyword Arguments:
            init_portfolio_value {float} -- initial portfolio value (default: {100.0})
            risk_aversion {float} -- rate of risk aversion (default: {1.0})
            costs {float} -- transaction costs as a percentage of total transaction volume (default: {0.025})
        """

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

    def update(self, actions: np.ndarray):
        """Updates the environment based on agent action and returns the next state observation 
        to the agent.
        
        Arguments:
            actions {np.ndarray} -- agent actions for each asset
        
        Returns:
            [type] -- [description]
        """

        # update portfolio weights based on given actions
        self.new_weights = actions

        # calculate weight difference
        _weight_diff = np.array(self.new_weights) - np.array(self.weights)

        # estimate costs for trading
        self.cost = self._get_cost(_weight_diff)

        # update portfolio value
        self.portfolio_value -= self.cost

        return self.new_weights, self.cost, self.portfolio_value

    def step(self, asset_returns: np.ndarray, covariance: np.ndarray):
        """Make a step in time and return the resulting reward asset returns and asset weights.

        Arguments:
            asset_returns {np.ndarray} -- asset returns price_t / price_{t-1} - 1
            covariance {np.ndarray} -- variance-covariance matrix based on current window
        
        Returns:
            tuple -- reward, weights, portfolio value
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

    def _get_weights(self, asset_returns: np.ndarray):
        """Natural change of portfolio weights given possible deviations in asset returns.
        
        Arguments:
            asset_returns {np.ndarray} -- linear returns for each asset 
        
        Returns:
            np.ndarray -- weights for the next timestep
        """
        return asset_returns * self.new_weights / np.sum(asset_returns * self.new_weights)

    def _get_cost(self, weight_diff: np.ndarray):
        """Cost for trading based on total trading volume in relation to the portfolio value.
        
        Arguments:
            weight_diff {np.ndarray} -- difference between old asset weights and reallocated weights
        
        Returns:
            float -- transaction cost based on transaction volume and portfolio value
        """

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
        """Reward-to-Variability-Ratio (Sharpe Ratio).
        Keyword Arguments:
            risk_free {float} -- risk free rate, assumed to be zero because of very small investment horizons (default: {0.0})
        
        Returns:
            float -- sharpe ration
        """
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
        Initializes the portfolio by calculating episode windows. Those windows contain
        historical data windows for each timestep (evaluation window with a fixed size + 
        current timestep), initial episode start asset weights as well as predictions for each timestep 
        
        Arguments:
            data {np.ndarray} -- train or test set of historical data
            nb_assets {int} -- number of assets in the portfolio
            horizon {int} -- investment horizon
            episodes {int} -- number of training/testing episodes
            window_size {int} -- evaluation window (number of previous days + current day)
        
        Keyword Arguments:
            epochs {int} -- number of training epochs, if testing epochs=1 (default: {1})
            risk_aversion {float} -- rate of risk aversion (default: {1.0})
            val_eps {int} -- number of validation episodes, if testing val_eps=None (default: {None})
            predictors {list} -- list of market/asset price predictors (default: {None})
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

    def _get_windows(self, window: np.ndarray, weights: np.ndarray, pred: np.ndarray):
        """Given multiple training epochs, for each epoch the sequence of episodes will be randomly shuffeled
        to (hopefully) improve agent training.
        
        Arguments:
            window {np.ndarray} -- data window for each episode
            weights {np.ndarray} -- inital weights for each episode
            pred {np.ndarray} -- prediction for each episode and each timestep
        
        Returns:
            tuple -- multiple arrays where epoch has been added as a new dimension
        """

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
        """Creates episode windows containing historical data as well market predictions 
        for each episode and each timestep. Furthermore ptimized initial weights for each 
        episode will be calculated.
        
        Returns:
            tuple -- multiple arrays
        """

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