import numpy as np
import pandas as pd
from collections import namedtuple

_Step = namedtuple('Step', ['state', 'reward', 'info'])

_PortfolioInfo = namedtuple('Portfolio',
                            ['weights',
                             'old_weights',
                             'new_weights',
                             'init_weights',
                             'asset_returns',
                             'portfolio_return',
                             'predictions',
                             'portfolio_value',
                             'new_portfolio_value',
                             'old_portfolio_value',
                             'portfolio_variance',
                             'sharpe_ratio',
                             'transaction_costs'])


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
