import sys
import time
import logging
from tqdm import tqdm
import pandas as pd
import pandas_datareader as web

from pgtaa.core.colorized import ColourHandler, color_bar
from pgtaa.config import *


logging.basicConfig(
    level=logging.DEBUG,
    filename=os.path.join(DATA_DIR, "data_load.log"),
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
#ch = logging.StreamHandler(sys.stdout)
ch = ColourHandler()
# modify stream handler log format
#formatter_ch = logging.Formatter('%(levelname)s - %(message)s')
formatter_ch = logging.Formatter('%(message)s')
ch.setFormatter(formatter_ch)
# set stream handler log level
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class RequestData:
    def __init__(
        self,
        symbols: list,
        start: datetime=None,
        end: datetime=None,
        names: list=None,
        source: str=None
    ):
        """
        Args:
            :param symbols: (list) containing ticker symbol; ['IWD', 'GLD']
            :param source: data source; 'yahoo', 'fred', 'google'
            :param start: (object) start date object
            :param end: (object) end date object
            :param names: (list) column names for data frame object
        """
        self.symbols = self.names = list(symbols)
        self.source = source
        self.start = start
        self.end = end

        if names is not None:
            self.names = list(names)

        logger.debug(f'Collecting data for {self.names} from {self.source}')
        logger.debug(f'Time frame: {self.start} - {self.end}\n')

        self.ds = self._concat_data()

    def get_data(self, symbol, name):
        start = time.time()
        while True:
            try:
                # start an api request
                if self.source == "fred":
                    r = web.DataReader(symbol, self.source, self.start, self.end)
                elif self.source == "yahoo":
                    r = web.DataReader(symbol, self.source, self.start, self.end)['Adj Close']
                else:
                    raise ValueError("No valid data source given!")
                logger.info(f'   Data for {name} has been collected')
                # break loop if request has been successful
                break
            except Exception as e:
                logger.error(e)
                # sleep for 1 second and restart request
                time.sleep(1)
                if time.time() - start > 300:
                    # if single request takes more than 5 minutes exit process
                    logger.critical(f'Request for {name} has failed!')
                    sys.exit()
        return r

    def _concat_data(self):
        # concatenate data
        data = []
        with tqdm(total=len(self.symbols),
                  bar_format=color_bar("white")) as pbar:
            for symbol, name in zip(self.symbols, self.names):
                data.append(self.get_data(symbol, name))
                pbar.update(1)
        ds = pd.DataFrame(pd.concat(data, axis=1))
        ds.columns = self.names
        logger.debug(f'Data shape: {ds.shape}\n')
        return ds


def upgrade_ds(df):
    assets = df.iloc[:, :NB_ASSETS]

    # linear return
    pct_1d = assets.pct_change(1)

    pct_10d = assets.pct_change(10)
    creturn_10d = pct_1d.cumsum()
    creturn_10d = assets




def main():
    __start = time.time()

    portfolio_ds = RequestData(ASSETS, source='yahoo', names=ASSET_NAMES, start=START, end=END).ds

    # the feature data set is primarily for training additional market predictor models
    # data was obtained from https://fred.stlouisfed.org/
    feature_ds = RequestData(list(FRED_DATA.keys()), source='fred', start=START, end=END,
                             names=FRED_DATA.values()).ds

    logger.debug(f'Data request runtime: {time.time() - __start}')

    # concatenate yahoo and fred data and interpolate missing data
    feature_ds.interpolate(method='linear', inplace=True)

    # concatenate asset data and additional data + drop non trading days like weekends
    environment = pd.DataFrame(pd.concat([portfolio_ds, feature_ds], axis=1)).dropna()

    # saves the environment data as a csv file
    environment.to_csv(ENV_CSV)
    logger.debug(f'Done collecting data. Environment shape: {environment.shape}')
    logger.debug(f'Data has been saved to {ENV_CSV}')


if __name__ == '__main__':
    main()
