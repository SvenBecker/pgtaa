import sys
import time
import logging
import pandas as pd
import pandas_datareader as web
from pgtaa.config import *


logging.basicConfig(
    level=logging.INFO,
    filename="data_load.log",
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)

# add stream handler which prints to stderr
ch = logging.StreamHandler()

# modify stream handler log format
formatter_ch = logging.Formatter('%(levelname)s - %(message)s')

ch.setFormatter(formatter_ch)

# set stream handler log level
ch.setLevel(logging.WARNING)

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

        logger.info(f'Collecting data for {self.names} from {self.source}')
        logger.info(f'Time frame:{self.start} - {self.end}')

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
                logger.info(f'Data for {name} has been collected')
                # break loop if request has been successful
                break
            except Exception as e:
                logger.error(e)

                # sleep for 1 second and restart request
                time.sleep(1)
                if time.time() - start > 300:
                    # if single request takes more than 5 minutes exit process
                    logger.critical(f'Request for {name} has failed')
                    sys.exit()
        return r

    def _concat_data(self):
        # concatenate data
        data = []
        for symbol, name in zip(self.symbols, self.names):
            data.append(self.get_data(symbol, name))
        ds = pd.DataFrame(pd.concat(data, axis=1))
        ds.columns = self.names
        logger.info(f'Data shape: {ds.shape}')
        return ds


if __name__ == '__main__':

    __start = time.time()

    portfolio_ds = RequestData(ASSETS, source='yahoo', start=START, end=END).ds
    yahoo_ds = RequestData(list(YAHOO_DATA.keys()), source='yahoo', start=START, end=END,
                           names=pd.read_csv(config.DATABASE_DIR + '/yahoo.csv')['Name']).ds
    fred_ds = RequestData(list(FRED_DATA.keys()), source='fred', start=START, end=END,
                          names=pd.read_csv(config.DATABASE_DIR + '/fred.csv')['Name']).ds
    logger.info(f'Request time: {time.time() - __start}')

    # saves an csv file (is being used to do some visualization)
    portfolio_ds.to_csv('asset_price.csv')

    # for the environment only the asset daily returns are required + drop first row
    portfolio_ds = portfolio_ds.pct_change(1).dropna()

    # concatenate yahoo and fred data and interpolate missing data
    feature_ds = pd.DataFrame(pd.concat([yahoo_ds, fred_ds], axis=1)).interpolate(method='linear')

    # concatenate asset data and additional data + drop non trading days like weekends
    environment = pd.DataFrame(pd.concat([portfolio_ds, feature_ds], axis=1)).dropna()

    # saves the environment data as a csv file
    environment.to_csv(config.ENV_DATA)
    logger.info(f'Done collecting data. Environment shape: {environment.shape}')
    logger.info(f'Data has been saved to {config.ENV_DATA}')
    print(environment.head(5))
