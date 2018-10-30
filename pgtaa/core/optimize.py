import numpy as np
from scipy.optimize import minimize


class WeightOptimize:

    """
    Class for calculating semi-optimized asset weights based on given historical returns and asset covariance matrix.
    Assumption: expected return = historical mean return -> semi-optimized but probably better than random or naive.
    """

    def __init__(
            self,
            asset_returns: np.ndarray,
            covariance_matrix: np.ndarray,
            risk_aversion: float=1.0
    ):

        self.asset_returns = asset_returns
        self.covariance_matrix = covariance_matrix
        self.nb_assets = asset_returns.shape[1]
        self.risk_aversion = risk_aversion
        self.expected_return = np.mean(self.asset_returns, axis=0)

    def objective_function(self, x):

        # utility = expected return - exposure to risk
        return self.risk_aversion / 2 * float((np.matmul(np.matmul(x.T, self.covariance_matrix), x))) - \
               float(np.matmul(x.T, self.expected_return))

    def optimize_weights(self, method='SLSQP'):

        """
        Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        method: optimization method (see link above)
        x0: initial guess

        returns: optimized weights for each asset
        """

        # bnds: lower and upper bound for the weights -> between 0 and 1
        bnds = tuple((0, 1) for _ in range(self.nb_assets))

        # using naive weights for initial guess
        x0 = np.array([1 / self.nb_assets for _ in range(self.nb_assets)])
        """
        # the sum of all weights has to be equal one
        constrain = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # calculate 'optimal' portfolio weights
        solution = minimize(self.objective_function, x0, method=method, constraints=constrain, bounds=bnds)
        """
        solution = minimize(self.objective_function, x0,
                            method=method,
                            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            bounds=bnds)

        return solution.x
