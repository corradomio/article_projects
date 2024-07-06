import logging.config
import logging.handlers
import warnings
import pandas
import pandasx as pdx
import sktimex
import stdlib
import torchx.nn.timeseries as tnn

from sktime.forecasting.arima import AutoARIMA
from sktimex.forecasting.lnn import LNNLinearForecaster
from sktimex.forecasting.rnn import RNNLinearForecaster
from sktimex.forecasting.cnn import CNNLinearForecaster
from sktimex.forecasting.linear import LinearForecaster
from sktimex.forecasting.skorch import SkorchForecaster

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

TARGET = 'import_kg'


# ---------------------------------------------------------------------------

def main():

    df = pdx.read_data(
        # "data/vw_food_import_aed_pred.csv",
        "data/vw_food_import_aed_train_test.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),    # datetime format different from 'kg'
        binhot='imp_month',
        na_values=['(null)'],
        ignore=['item_country', 'imp_date'],
        numeric=["import_aed", ],
        index=['item_country', 'imp_date'],
        dropna='item_country'
    )

    df.fillna(0, inplace=True)

    df = pdx.groups_select(df, values=['ANIMAL FEED~ARGENTINA'], drop=True)

    model = SkorchForecaster(
        module=tnn.tide.TiDE
    )

    pass


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
