import logging.config
import logging.handlers
import warnings

import pandas
from sktime.forecasting.arima import AutoARIMA

import pandasx as pdx
import sktimex
import stdlib
from sktimex.forecasting.cnn import CNNLinearForecaster
from sktimex.forecasting.darts.linear import Linear as DartsLinearForecaster
from sktimex.forecasting.linear import LinearForecaster
from sktimex.forecasting.lnn import LNNLinearForecaster
from sktimex.forecasting.rnn import RNNLinearForecaster
from sktimex.forecasting.scikit import ScikitForecaster

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

log = None


TARGET = 'import_kg'


# ---------------------------------------------------------------------------

def use_scikit_forecaster(df_past, df_hist, df_test):

    # X, y = pdx.xy_split(df, target=TARGET)
    # X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    model = ScikitForecaster(
        window_length=24,
        prediction_length=12
    )

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict_history(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


def use_linear_forecaster(df_past, df_hist, df_test):

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    model = LinearForecaster(
        lags=24,
        tlags=1,
    )

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict_history(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


def use_nnlin_forecaster(df_past, df_hist, df_test):

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    model = LNNLinearForecaster(
        lags=24,
        tlags=12,

        hidden_size=48,
        activation='relu'
    )

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict_history(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


def use_rnn_forecaster(df_past, df_hist, df_test, flavour=None):

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    model = RNNLinearForecaster(
        flavour=flavour,
        lags=24,
        tlags=12,
        method='standard',
        max_epochs=20000,
        patience=100,
        hidden_size=48,
        activation='relu'
    )

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict_history(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


def use_cnn_forecaster(df_past, df_hist, df_test, flavour=None):

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    model = CNNLinearForecaster(
        flavour=flavour,
        lags=24,
        tlags=12,
        method='standard',
        max_epochs=20000,
        patience=100,
        hidden_size=48,
        activation='relu'
    )

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict_history(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


def use_sktime_arima_forecaster(df_past, df_hist, df_test, flavour=None):

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    # model = ScikitForecaster(
    #     estimator=qualified_name(AutoARIMA)
    # )
    model = AutoARIMA()

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict_history(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


def use_darts_arima_forecaster(df_past, df_hist, df_test, flavour=None):

    X_train, y_train = pdx.xy_split(df_past, target=TARGET)
    X_hist, y_hist, X_test, y_test = pdx.xy_split(df_hist, df_test, target=TARGET)

    model = DartsLinearForecaster(
        lags=36,
        output_chunk_length=6
    )

    model.fit(y_train, X_train)

    fh = stdlib.lrange(1, 13)
    y_pred = model.predict(fh, X=X_test, yh=y_hist, Xh=X_hist)

    sktimex.utils.plot_series(y_train, y_hist, y_test, y_pred, labels=['train', 'hist', 'test', 'pred'])
    sktimex.utils.show()
    return
# end


# ---------------------------------------------------------------------------

def main():
    global log
    log = logging.getLogger('main')

    df = pdx.read_data(
        # "data/vw_food_import_aed_pred.csv",
        "data/vw_food_import_aed_train_test.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),    # datetime format different from 'kg'
        # categorical='imp_month',
        binhot='imp_month',
        na_values=['(null)'],
        ignore=['item_country', 'imp_date'],
        numeric=["import_aed", ],
        index=['item_country', 'imp_date'],
        dropna='item_country'
    )

    assert not df.isnull().values.any()

    # plot_time_series(df)
    # groups_list = pdx.groups_list(df, sort=True)

    df = pdx.groups_select(df, 'ANIMAL FEED~ARGENTINA', drop=True)

    df_past, df_future = pdx.train_test_split(df, train_size=.50)
    df_hist, df_test = pdx.train_test_split(df_future, test_size=12)

    use_scikit_forecaster(df_past, df_hist, df_test)
    # use_linear_forecaster(df_past, df_hist, df_test)
    # use_nnlin_forecaster(df_past, df_hist, df_test)
    # use_rnn_forecaster(df_past, df_hist, df_test)
    # use_sktime_arima_forecaster(df_past, df_hist, df_test)
    # use_darts_arima_forecaster(df_past, df_hist, df_test)

    pass


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
