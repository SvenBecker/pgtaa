import datetime
import json
import os

# root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# load json config file and parse arguments, try self modified json file first
try:
    with open(os.path.join(ROOT_DIR, 'mod_config.json')) as f:
        config = json.load(f)
except FileNotFoundError:
    with open(os.path.join(ROOT_DIR, 'config.json')) as f:
        config = json.load(f)

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

# training and test set split size
SEED = config["environment"]["seed"]

# list of market predictors
PREDICTOR = []

# --------------------------- agent ----------------------------- #

AGENT = config["agent"]["type"]
BATCH_SIZE = config["agent"]["batch_size"]
CLIP = config["agent"]["clip"]

# ------------------------- network ----------------------------- #

LAYERS = config["network"]["layers"]
LR = config["network"]["learning_rate"]
OPTIMIZER = config["network"]["optimizer"]


# ---------------- files, folders and paths --------------------- #

# folders
ENV_DIR = os.path.join(ROOT_DIR, "environment")
DATA_DIR = os.path.join(ROOT_DIR, "data")
PRED_DIR = os.path.join(ENV_DIR, "saves")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
AGENT_SAVES = os.path.join(MODEL_DIR, "saves")
BOARD = os.path.join(MODEL_DIR, "board")

# files
ENV_CSV = os.path.join(DATA_DIR, "environment.csv")
ASSETS_CSV = os.path.join(DATA_DIR, "assets.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# --------------------------------------------------------------- #


def show_config():
    from pprint import pprint
    pprint(config)


def overwrite_env_config(configuration: str, value):
    """
    You can overwrite some configurations either directly in the config.json file or through this function.
    :param configuration: configuration setting which should be replaced
    :param value: replacing value
    """
    with open(os.path.join(ROOT_DIR, 'config.json')) as f:
        config = json.load(f)
    config["environment"][configuration] = value
    with open('mod_config.json', 'w') as fp:
        json.dump(config, fp)
        print("Changes have been made to the config file. "
              "This changes will be saved in mod_config.json.")


def remove_config():
    # removes the modified config file
    file = os.path.join(ROOT_DIR, "mod_config.json")
    if os.path.exists(file):
        os.remove(file)
        print("File mod_config.json has been removed")
    else:
        print("The config file does not exist")


def get_scaler():
    # returns a standard scaler fitted on the the training data
    from pgtaa.core.utils import read_data
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(read_data(TRAIN_CSV, nb_assets=NB_ASSETS, lin_return=True))


# --------------------------------------------------------------- #

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-sp', '--split', help="Train test split (decimal)", type=float, default=TRAIN_TEST_SPLIT)
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
    parser.add_argument('-s', '--seed', help="Set random seed", type=int, default=SEED)
    env_args = parser.parse_args()

    for arg in vars(env_args):
        overwrite_env_config(arg, getattr(env_args, arg))
