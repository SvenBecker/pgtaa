import pandas as pd
import numpy as np


def df_info(df: pd.DataFrame,
            show_rows: int = 0,
            horizontal: bool = True,
            percentiles: tuple = (.25, .5, .75),
            includes: tuple = (bool, int, float, np.object, pd.Categorical, np.datetime64, np.timedelta64),
            excludes: tuple = (),
            selected_cols: list = None
            ):
    """

    :param df: pd.DataFrame object which should be analysed
    :param show_rows: number of rows to display, if -1 all rows will be displayed
    :param horizontal: horizontal or vertical printing (cols <-> rows)
    :param percentiles: percentile ranks of given data to show
    :param includes: data types which should be included into analysis,
        valid values: bool, int, float, np.object, pd.Categorical, np.datetime64, np.timedelta64
    :param excludes: data types which should be excluded from analysis,
        valid values: bool, int, float, np.object, pd.Categorical, np.datetime64, np.timedelta64
    :param selected_cols: specify columns for analysis (if selected the includes and excludes arguments are disabled)
    :return: Pandas DataFrame object containing information about given DataFrame
    """
    df = df.copy()

    if show_rows == -1:
        show_rows = len(df)

    # you can either exclude some data types or specify columns to
    # reduce the numbers of columns which are being analyzed
    if selected_cols:
        df = df.loc[:, selected_cols]
    else:
        includes = tuple(set(includes).difference(excludes))
        for col in df:
            if not isinstance(col, includes):
                df.drop([col], axis=1, inplace=True)

    if len(df.columns) == 0: raise ValueError('DataFrame is empty!')

    # data types
    types = pd.DataFrame(df.dtypes, columns=["dtype"])

    # count missing values
    nans = pd.DataFrame(df.isnull().sum(), columns=["missing"])

    # description of the dataframe (mean, median, std, min-max values, frequency, uniques etc)
    descriptions = [df[col].describe(percentiles=percentiles, include="all").drop(["count"]) for col in df]
    descriptions = pd.concat(descriptions, axis=1, sort=False)

    # show the first few rows depending on how many you want to show
    head = df.head(show_rows)

    # display resulting df either vertically or horizontally
    if horizontal:
        info_df = pd.concat([types, nans, descriptions.T, head.T], axis=1, sort=False)
    else:
        info_df = pd.concat([types.T, nans.T, descriptions, head], axis=0, sort=False)

    # show all rows and columns, no matter how large the actual dataframe is
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        # do not permanently change pandas settings -> with statement
        try:
            from IPython.core.display import display
            display(info_df)
        except ImportError:
            print(info_df)


if __name__ == "__main__":
    df = pd.DataFrame(dict(s=["s", "b"], i=[9, 54353], f=[432.56, 212.3], b=[True, False]))
    df_info(df, percentiles=(0.5,), horizontal=False, show_rows=-1)
