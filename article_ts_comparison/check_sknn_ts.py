
# Train & Predict Time series

import logging.config
import logging.handlers
import warnings

import pandas
from sktime.forecasting.base import ForecastingHorizon

import pandasx as pdx
import sktimex.forecasting as sktf
from sktimex.forecasting.rnn import RNNLinearForecaster
from sktimex.forecasting.cnn import CNNLinearForecaster
from sktimex.forecasting.lnn import LNNLinearForecaster
import stdlib.csvx as csvx
from sklearnx.metrics import weighted_absolute_percentage_error

# import stdlib.loggingx as logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)


TARGET = 'import_kg'

RESULTS_WAPE = [
    ["item_area", "model", "wape"]
]

RESULT_FILE = "wape_sknn.csv"


# ---------------------------------------------------------------------------

def use_model(g, name, model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_true, fh):
    item_area = g[0]

    model.fit(y=ytr_scaled, X=Xtr_scaled)
    yte_predicted = model.predict(fh, X=Xte_scaled)

    wape = weighted_absolute_percentage_error(yte_true, yte_predicted)

    RESULTS_WAPE.append([item_area, name, wape])
    csvx.save_csv(RESULT_FILE, RESULTS_WAPE[1:], header=RESULTS_WAPE[0])
# end


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
    use_model(g, "lnnlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # linear = sktf.lnn.LNNLinearForecaster(
    #     lags=24,
    #     tlags=6
    # )
    # use_model(g, "lnnlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    #
    # linear = sktf.lnn.LNNLinearForecaster(
    #     lags=24,
    #     tlags=12
    # )
    # use_model(g, "lnnlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # -- RNNLinearForecaster[rnn]

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=1,
        flavour="rnn"
    )
    use_model(g, "rnnlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # linear = sktf.rnn.RNNLinearForecaster(
    #     lags=24,
    #     tlags=6,
    #     flavour="rnn"
    # )
    # use_model(g, "rnnlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    #
    # linear = sktf.rnn.RNNLinearForecaster(
    #     lags=24,
    #     tlags=12,
    #     flavour="rnn"
    # )
    # use_model(g, "rnnlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # -- RNNLinearForecaster[gru]

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=1,
        flavour="gru"
    )
    use_model(g, "grulinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # linear = sktf.rnn.RNNLinearForecaster(
    #     lags=24,
    #     tlags=6,
    #     flavour="gru"
    # )
    # use_model(g, "grulinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    #
    # linear = sktf.rnn.RNNLinearForecaster(
    #     lags=24,
    #     tlags=12,
    #     flavour="gru"
    # )
    # use_model(g, "grulinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # -- RNNLinearForecaster[lstm]

    linear = sktf.rnn.RNNLinearForecaster(
        lags=24,
        tlags=1,
        flavour="lstm"
    )
    use_model(g, "lstmlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # linear = sktf.rnn.RNNLinearForecaster(
    #     lags=24,
    #     tlags=6,
    #     flavour="lstm"
    # )
    # use_model(g, "lstmlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    #
    # linear = sktf.rnn.RNNLinearForecaster(
    #     lags=24,
    #     tlags=12,
    #     flavour="lstm"
    # )
    # use_model(g, "lstmlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # -- CNNLinearForecaster

    linear = sktf.cnn.CNNLinearForecaster(
        lags=24,
        tlags=1
    )
    use_model(g, "cnnlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

    # linear = sktf.cnn.CNNLinearForecaster(
    #     lags=24,
    #     tlags=6
    # )
    # use_model(g, "cnnlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    #
    # linear = sktf.cnn.CNNLinearForecaster(
    #     lags=24,
    #     tlags=12
    # )
    # use_model(g, "cnnlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

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
