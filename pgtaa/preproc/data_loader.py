import pandas as pd 
import numpy as np 
from pgtaa.config import *
from core.optimize import WeightOptimize

def get_compounded_returns(pct_returns):
    pass

def get_linear_returns(assets: pd.DataFrame):
    lt = assets.pct_change(1)
    lt5 = assets.pct_change(5)
    lt10 = assets.pct_change(19)
    return lt, lt5, lt10


class DataLoader(object):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # portfolio assets are assumed to be in the first x columns
        self.assets = df.iloc[:,:NB_ASSETS]

    def __getattribute__(self, x):
        return self.assets.iloc[x]
