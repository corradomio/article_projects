#
# Train & Predict Time series
#
import logging.config
import logging.handlers
import warnings

import pandas
from sktime.forecasting.base import ForecastingHorizon

import pandasx as pdx
import sktimex.forecasting as sktf
from sktimex.forecasting.cnn import CNNLinearForecaster
from sktimex.forecasting.lnn import LNNLinearForecaster
from sktimex.forecasting.rnn import RNNLinearForecaster
from common import *


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

TARGET = "import_kg"
USED_LIBRARY = "sktnn"
RESULT_FILE = "wape_skt_nn.csv"


# ---------------------------------------------------------------------------

def compare_models(g, Xtr, ytr, Xte, yte):
    item_area = g[0]

    print(item_area)

    # 1) normalize all values in the range (0,1, 0.9)
    xscaler = pdx.preprocessing.MinMaxScaler(threshold=0.1, clip=True)
    yscaler = pdx.preprocessing.MinMaxScaler(threshold=0.1, clip=True)

    Xtr_scaled = xscaler.fit_transform(Xtr)
    ytr_scaled = yscaler.fit_transform(ytr)
    Xte_scaled = xscaler.transform(Xte)
    yte_scaled = yscaler.transform(yte)
    fh = ForecastingHorizon(yte.index)

    #
    # sklean.linear_model.LinearRegression
    #

    # -- LNNLinearForecaster

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=1
    )
    use_model(g, "lnnlin-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=3
    )
    use_model(g, "lnnlin-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=6
    )
    use_model(g, "lnnlin-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=12
    )
    use_model(g, "lnnlin-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -- LNNLinearForecaster[200]

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=1,
        hidden_size=150,
        activation="relu"
    )
    use_model(g, "lnnlin2-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=3,
        hidden_size=150,
        activation="relu"
    )
    use_model(g, "lnnlin2-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=6,
        hidden_size=150,
        activation="relu"
    )
    use_model(g, "lnnlin2-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.lnn.LNNLinearForecaster(
        lags=24,
        tlags=12,
        hidden_size=150,
        activation="relu"
    )
    use_model(g, "lnnlin2-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -- RNNLinearForecaster[rnn]

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=1,
        flavour="rnn"
    )
    use_model(g, "rnnlin-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=3,
        flavour="rnn"
    )
    use_model(g, "rnnlin-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=6,
        flavour="rnn"
    )
    use_model(g, "rnnlin-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=12,
        flavour="rnn"
    )
    use_model(g, "rnnlin-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -- RNNLinearForecaster[gru]

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=1,
        flavour="gru"
    )
    use_model(g, "grulin-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=3,
        flavour="gru"
    )
    use_model(g, "grulin-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=6,
        flavour="gru"
    )
    use_model(g, "grulin-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=12,
        flavour="gru"
    )
    use_model(g, "grulin-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -- RNNLinearForecaster[lstm]

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=1,
        flavour="lstm"
    )
    use_model(g, "lstmlin-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=3,
        flavour="lstm"
    )
    use_model(g, "lstmlin-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=6,
        flavour="lstm"
    )
    use_model(g, "lstmlin-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=12,
        flavour="lstm"
    )
    use_model(g, "lstmlin-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -- CNNLinearForecaster

    linear = sktf.cnn.CNNLinearForecaster(
        lags=24,
        tlags=1
    )
    use_model(g, "cnnlin-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.cnn.CNNLinearForecaster(
        lags=24,
        tlags=3
    )
    use_model(g, "cnnlin-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.cnn.CNNLinearForecaster(
        lags=24,
        tlags=6
    )
    use_model(g, "cnnlin-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktf.cnn.CNNLinearForecaster(
        lags=24,
        tlags=12
    )
    use_model(g, "cnnlin-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --
    pass
# end


# ---------------------------------------------------------------------------

def main():
    global log
    log = logging.getLogger('main')

    # -------------------------------------------
    # vw_food_import_train_test_newfeatures (352)
    # -------------------------------------------
    # "item_country","imp_month","imp_date","import_kg","prod_kg","avg_retail_price_src_country",
    # "producer_price_tonne_src_country","crude_oil_price","sandp_500_us","sandp_sensex_india","shenzhen_index_china",
    # "nikkei_225_japan","max_temperature","mean_temperature","min_temperature","vap_pressure","evaporation",
    # "rainy_days"
    #
    df = pdx.read_data(
        "data/vw_food_import_train_test_newfeatures.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),  # datetime format different from 'kg'
        # categorical='imp_month',
        binhot='imp_month',
        na_values=['(null)'],
        ignore=['item_country', 'imp_date', "prod_kg", "avg_retail_price_src_country",
                "producer_price_tonne_src_country"],
        numeric=[TARGET],
        index=['item_country', 'imp_date'],
    )

    df.fillna(0, inplace=True)

    # df = pdx.groups_select(df, values=['FISH - FROZEN~TAIWAN'], drop=False)

    df_tr, df_te = pdx.train_test_split(df, test_size=12)

    # all groups
    Xa_tr, ya_tr, Xa_te, ya_te = pdx.xy_split(df_tr, df_te, target=TARGET)

    # splitted by group
    Xg_tr = pdx.groups_split(Xa_tr)
    yg_tr = pdx.groups_split(ya_tr)
    Xg_te = pdx.groups_split(Xa_te)
    yg_te = pdx.groups_split(ya_te)

    # list of groups
    groups = pdx.groups_list(df)

    for g in groups:
        Xtr = Xg_tr[g]
        ytr = yg_tr[g]
        Xte = Xg_te[g]
        yte = yg_te[g]

        compare_models(g, Xtr, ytr, Xte, yte)

        pass
    pass

    # csvx.save_csv("results_wape.csv", RESULTS_WAPE[1:], header=RESULTS_WAPE[0])
    pass
# end


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
