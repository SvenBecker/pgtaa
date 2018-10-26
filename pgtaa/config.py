import datetime
import json
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# load json config file and parse arguments, try self modified json file first
try:
    with open(os.path.join(ROOT_DIR, 'mod_config.json')) as f:
        config = json.load(f)
except FileNotFoundError:
    with open(os.path.join(ROOT_DIR, 'config.json')) as f:
        config = json.load(f)


def show_config():
    from pprint import pprint
    pprint(config)


def build_mlp(cells):
    mlp = {"layer_" + str(i): c for i, c in enumerate(cells)}
    return mlp


def build_agent():
    pass


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

# data symbols for the Investors Exchange
IEX_DATA = config["data"]["iex_data"]

# data symbols for the Moscow Exchange
MOEX_DATA = config["data"]["moex_data"]

# data symbols for Stooq Index Data
STOOQ_DATA = config["data"]["stooq_data"]

# --------------------- environment------------------------------- #

# (transaction-)costs for selling assets based on trading volume
COST_SELLING = 0.0025
# (transaction-)costs for buying assets based on trading volume
COST_BUYING = 0.0025
# fix costs for trading (absolute)
COST_FIX = 0

# number of training epochs
EPOCHS = config["environment"]["epochs"]

# relative transaction costs
COSTS = config["environment"]["costs"]

# number of training episodes
TRAIN_EPISODES = config["environment"]["train_episodes"]

# number of validation episodes
VALID_EPISODES = config["environment"]["val_episodes"]

# number of test episodes
TEST_EPISODES = config["environment"]["test_episodes"]

# constant rate of relative risk aversion
RISK_AVERSION = config["environment"]["risk_aversion"]

# initial portfolio for each trading episode
PORTFOLIO_INIT_VALUE = config["environment"]["risk_aversion"]

# max number of time steps for each episode
HORIZON = config["environment"]["horizon"]

# length of historical data provided on each time step
WINDOW_SIZE = config["environment"]["window_size"]

# training and test set split size
TRAIN_TEST_SPLIT = config["environment"]["train_test_split"]

# --------------------------- agent------------------------------ #

# agent train
AGENT_LR = config["agent"]["train"]["learning_rate"]
AGENT_TRAIN_BATCH_SIZE = config["agent"]["train"]["batch_size"]

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


if __name__ == "__main__":
    from argparse import ArgumentParser

    def overwrite_env_config(configuration: str, value):
        with open(os.path.join(ROOT_DIR, 'config.json')) as f:
            config = json.load(f)
        config["environment"][configuration] = value
        with open('mod_config.json', 'w') as fp:
            json.dump(config, fp)

    parser = ArgumentParser()
    parser.add_argument('-s', '--split', help="Train test split", type=float, default=TRAIN_TEST_SPLIT)
    parser.add_argument('-e', '--epochs', type=int,
                        help="Number of epochs", default=EPOCHS)
    parser.add_argument('-t', '--train-episodes', type=int,
                        help="Number of training episodes per epoch", default=TRAIN_EPISODES)
    parser.add_argument('-ve', '--val-episodes', type=int,
                        help="Number of validation episodes", default=VALID_EPISODES)
    parser.add_argument('-te', '--test-episodes', type=int,
                        help="Number of test episodes", default=TEST_EPISODES)
    parser.add_argument('-w', '--window', type=int,
                        help="Window size", default=WINDOW_SIZE)
    parser.add_argument('-hz', '--horizon', type=int,
                        help="Investment horizon / number of time steps for each episode", default=HORIZON)
    parser.add_argument('-r', '--risk-aversion', type=float,
                        help="Rate of exposure to risk", default=HORIZON)
    parser.add_argument('-p', '--portfolio-init-value', type=float,
                        help="Initial portfolio value", default=PORTFOLIO_INIT_VALUE)
    parser.add_argument('-c', '--costs', type=float,
                        help="Transaction costs (w.r.t. transaction volume)", default=COSTS)
    arguments = parser.parse_args()
    for arg in vars(arguments):
        overwrite_env_config(arg, getattr(arguments, arg))
