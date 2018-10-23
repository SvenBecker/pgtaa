import datetime
import json
import pprint
from pathlib import Path

# load json config file and parse arguments
with open('config.json') as f:
    config = json.load(f)

# ------------------------ data --------------------------------- #

# portfolio assets
ASSETS = list(config["data"]["assets"].keys())

# number of assets
NB_ASSETS = len(ASSETS)

# start date for historical data
_start = list(map(int, config["data"]["date"]["start"].split("-")))
START = datetime.date(*_start)

# end date for historical data
_end = list(map(int, config["data"]["date"]["end"].split("-")))
END = datetime.date(*_end)

# data symbols from yahoo and fred
YAHOO_DATA = config["data"]["yahoo_data"]
FRED_DATA = config["data"]["fred_data"]

# --------------------- environment------------------------------- #

# constant rate of relative risk aversion
RISK_AVERSION = config["environment"]["risk_aversion"]

PORTFOLIO_INIT_VALUE = config["environment"]["risk_aversion"]

# max number of time steps for each episode
HORIZON = config["environment"]["horizon"]

# length of historical data provided on each time step
WINDOW_SIZE = config["environment"]["window_size"]

TRAIN_TEST_SPLIT = config["environment"]["train_test_split"]

# (transaction-)costs for selling assets based on trading volume
COST_SELLING = 0.0025
# (transaction-)costs for buying assets based on trading volume
COST_BUYING = 0.0025
# fix costs for trading (absolute)
COST_FIX = 0

# --------------------------- agent------------------------------ #

# agent train
AGENT_LR = config["agent"]["train"]["learning_rate"]
AGENT_TRAIN_BATCH_SIZE = config["agent"]["train"]["batch_size"]
AGENT_EPISODES = config["agent"]["train"]["episodes"]
AGENT_EPOCHS = config["agent"]["train"]["epochs"]

# agent validate
AGENT_VALID_EPISODES = config["agent"]["valid"]["episodes"]

# agent test
AGENT_TEST_EPISODES = config["agent"]["test"]["episodes"]

# ---------------- files, folders and paths --------------------- #

ROOT_DIR = Path(".")

# files
ENV_CSV = ROOT_DIR / "environment" / "data" / "environment.csv"

# folders
MODEL_DIR = ROOT_DIR / "model"

# --------------------------------------------------------------- #


def show_config():
    pprint.pprint(config)
