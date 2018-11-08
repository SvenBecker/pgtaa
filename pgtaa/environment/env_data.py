import os
import sys
import time
import logging
import pandas as pd
import pandas_datareader as web
from tqdm import tqdm
from datetime import datetime

from pgtaa.core.colorized import ColourHandler, color_bar
import pgtaa.config as cfg


logging.basicConfig(
    level=logging.DEBUG,
    filename=os.path.join(cfg.DATA_DIR, "data.log"),
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
ch = ColourHandler()
# modify stream handler log format
formatter_ch = logging.Formatter('%(message)s')
ch.setFormatter(formatter_ch)
# set stream handler log level
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# all supported data source websites
data_sources = {
    "fred": cfg.FRED_DATA, 
    "yahoo": cfg.YAHOO_DATA, 
    "iex": cfg.IEX_DATA, 
    "moex": cfg.MOEX_DATA, 
    "stooq": cfg.STOOQ_DATA}

class RequestData:
    def __init__(
        self,
        symbols: list,
        start: datetime=None,
        end: datetime=None,
        names: list=None,
        source: str=None
    ):
        """Downloads requested data from multiple data sources like Yahoo Finance or FRED.
        
        Arguments:
            symbols {list} -- list with ticker symbols; ['IWD', 'GLD', ...]
        
        Keyword Arguments:
            start {datetime} -- start date for historical data (default: {None})
            end {datetime} -- end date for historical data (default: {None})
            names {list} -- column names for data frame object (default: {None})
            source {str} -- data source for requests (default: {None})
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

    def _get_data(self, symbol, name):
        start = time.time()
        while True:
            try:
                # start an api request
                if self.source == "fred":
                    r = web.DataReader(symbol, self.source, self.start, self.end)
                elif self.source == "yahoo":
                    r = web.DataReader(symbol, self.source, self.start, self.end)['Adj Close']
                elif self.source == "iex":
                    r = web.DataReader(symbol, self.source, self.start, self.end)['close']
                elif self.source == "moex":
                    r = web.DataReader(symbol, self.source, self.start, self.end)['CLOSE']
                elif self.source == "stooq":
                    r = web.DataReader(symbol, self.source, self.start, self.end)['Close']
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
                data.append(self._get_data(symbol, name))
                pbar.update(1)
        ds = pd.DataFrame(pd.concat(data, axis=1))
        ds.columns = self.names
        logger.debug(f'Data shape: {ds.shape}\n')
        return ds


def _availability_check(data, source):
    if len(data) is 0:
        return pd.DataFrame()
    else:
        return RequestData(list(data.keys()), source=source,
                           start=cfg.START, end=cfg.END, names=data.values()).ds


def main(save: bool=True):
    __start = time.time()

    ds = []
    for source, data in data_sources.items():
        ds.append(_availability_check(data, source))

    logger.debug(f'Data request runtime: {time.time() - __start}')

    feature_ds = pd.concat(ds, axis=1)
    portfolio_ds = feature_ds[cfg.ASSET_NAMES]
    feature_ds.drop(cfg.ASSET_NAMES, axis=1, inplace=True)

    # concatenate yahoo and fred data and interpolate missing data
    feature_ds.interpolate(method='linear', inplace=True)

    # concatenate asset data and additional data + drop non trading days like weekends
    environment = pd.DataFrame(pd.concat([portfolio_ds, feature_ds], axis=1)).dropna()

    # data split into a train and a test set
    train = environment.iloc[:int(cfg.TRAIN_TEST_SPLIT * len(environment))]
    test = environment.iloc[int(cfg.TRAIN_TEST_SPLIT * len(environment)):]
    if save:
        environment.to_csv(cfg.ENV_CSV)
        train.to_csv(cfg.TRAIN_CSV)
        test.to_csv(cfg.TEST_CSV)
        logger.debug(f'Files have been saved to:'
                     f'\n{cfg.ENV_CSV}\n{cfg.TRAIN_CSV}\n{cfg.TEST_CSV}')
    logger.debug(f'Environment shape: {environment.shape}')
    logger.debug(f'Train set size {train.shape}')
    logger.debug(f'Test set size {test.shape}')


if __name__ == '__main__':
    main()
