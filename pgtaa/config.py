import datetime
import json
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# load json config file and parse arguments
with open(os.path.join(ROOT_DIR, 'config.json')) as f:
    config = json.load(f)


def show_config():
    from pprint import pprint
    pprint(config)

# ------------------------ data --------------------------------- #

# portfolio assets
ASSETS = list(config["data"]["assets"].keys())
ASSET_NAMES = list(config["data"]["assets"].values())
NB_ASSETS = len(ASSETS)

# start date for historical data
__start = list(map(int, config["data"]["date"]["start"].split("-")))
START = datetime.date(*__start)

# end date for historical data
__end = list(map(int, config["data"]["date"]["end"].split("-")))
END = datetime.date(*__end)

# data symbols for Federal Reserve St. Louis Economic Data
FRED_DATA = config["data"]["fred_data"]

# data symbols for Yahoo Finance
YAHOO_DATA = config["data"]["yahoo_data"]

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

# folders
ENV_DIR = os.path.join(ROOT_DIR, "environment")
DATA_DIR = os.path.join(ENV_DIR, "data")
PRED_DIR = os.path.join(ENV_DIR, "saves")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
AGENT = os.path.join(MODEL_DIR, "saves")
BOARD = os.path.join(MODEL_DIR, "board")

# files
ENV_CSV = os.path.join(DATA_DIR, "environment.csv")
ASSETS_CSV = os.path.join(DATA_DIR, "assets.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# --------------------------------------------------------------- #
