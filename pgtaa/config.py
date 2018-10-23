import datetime
import json
import pprint
from pathlib import Path

ROOT_DIR = Path(".")

with open('config.json') as f:
    config = json.load(f)

# data
ASSETS = list(config["data"]["assets"].keys())
NB_ASSETS = len(ASSETS)
_start = list(map(int, config["data"]["date"]["start"].split("-")))
START = datetime.date(*_start)
_end = list(map(int, config["data"]["date"]["end"].split("-")))
END = datetime.date(*_end)
YAHOO_DATA = config["data"]["yahoo_data"]
FRED_DATA = config["data"]["fred_data"]

# environment
RISK_AVERSION = config["environment"]["risk_aversion"]
PORTFOLIO_INIT_VALUE = config["environment"]["risk_aversion"]
HORIZON = config["environment"]["horizon"]
WINDOW_SIZE = config["environment"]["window_size"]
TRAIN_TEST_SPLIT = config["environment"]["train_test_split"]


# agent train
AGENT_LR = config["agent"]["train"]["learning_rate"]
AGENT_TRAIN_BATCH_SIZE = config["agent"]["train"]["batch_size"]
AGENT_EPISODES = config["agent"]["train"]["episodes"]
AGENT_EPOCHS = config["agent"]["train"]["epochs"]

# agent test
AGENT_TEST_EPISODES = config["agent"]["test"]["episodes"]


def show_config():
    pprint.pprint(config)

"""
# portfolio parameters
ASSETS = ('IWD', 'IWF', 'IWO', 'IWN', 'EFA', 'IEMG', 'TIP', 'GOVT', 'GLD', 'VNQ')   # asset symbols
START_DATE = datetime.date(2013, 1, 1)      # start data for historical data
END_DATE = datetime.date(2018, 10, 23)      # end date for historical data
PORTFOLIO_VALUE = 1000.                     # initial portfolio value
RISK_AVERSION = 1.                          # constant rate of relative risk aversion
NB_ASSETS = len(ASSETS)                     # number of assets

# environment parameters
WINDOW_SIZE = 100       # length of historical data provided on each time step
COST_SELLING = 0.0025   # (transaction-)costs for selling assets based on trading volume
COST_BUYING = 0.0025    # (transaction-)costs for buying assets based on trading volume
COST_FIX = 0            # fix costs for trading (absolute)
TRAIN_SPLIT = 0.75      # train/test data split

# run parameters
EPOCHS = 200            # number of max episodes runs
EPISODES = 100         # number of episodes on each epoch
HORIZON = 20            # max number of time steps for each episode

# directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT_DIR, 'run')
RUN_TMP = os.path.join(RUN_DIR, 'tmp')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
AGENT_CONFIG = os.path.join(MODEL_DIR, 'agent_config')
NET_CONFIG = os.path.join(MODEL_DIR, 'net_config')
ENV_DIR = os.path.join(ROOT_DIR, 'env')
DATA_DIR = os.path.join(ENV_DIR, 'data')
DATABASE_DIR = os.path.join(DATA_DIR, 'database')
IMAGE_DIR = os.path.join(ROOT_DIR, 'img')
"""

for k, _ in YAHOO_DATA.items():
    print(k)
