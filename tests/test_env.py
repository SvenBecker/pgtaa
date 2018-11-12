import numpy as np
import pandas as pd
from pgtaa.environment.env import PortfolioEnv, PortfolioInit
from pgtaa.core.utils import read_data
from pgtaa.config import *


def test_portfolio_init():
    df = pd.read_csv(TRAIN_CSV, nrows=1000, parse_dates=True, index_col=0)
    pinit= PortfolioInit()
    assert pinit.windows.shape[:-1] != (EPOCHS, TRAIN_EPISODES, HORIZON, WINDOW_SIZE), "sizes for windows do not match"
    return


def tets_run_env():
    pass