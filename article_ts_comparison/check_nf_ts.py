
# Train & Predict Time series

import logging.config
import logging.handlers
import warnings
import pandas
import sktime.forecasting.arima
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import sktimex.forecasting.linear as sktfl
import sktimex.forecasting.nf as sktnf
import sktimex.forecasting.nf.nlinear
import sktimex.forecasting.nf.autoformer
import sktimex.forecasting.nf.bitcn
import sktimex.forecasting.nf.deepnpts
import sktimex.forecasting.nf.dilated_rnn
import sktimex.forecasting.nf.fedformer
import sktimex.forecasting.nf.gru
import sktimex.forecasting.nf.informer
import sktimex.forecasting.nf.itrasformer
import sktimex.forecasting.nf.lstm
import sktimex.forecasting.nf.mlp
import sktimex.forecasting.nf.nbeats
import sktimex.forecasting.nf.nbeatsx
import sktimex.forecasting.nf.nhits
import sktimex.forecasting.nf.patchtst
import sktimex.forecasting.nf.rnn
import sktimex.forecasting.nf.tcn
import sktimex.forecasting.nf.tft
import sktimex.forecasting.nf.tide
import sktimex.forecasting.nf.timesnet
import sktimex.forecasting.nf.vanillatransformer

from sktime.forecasting.base import ForecastingHorizon
from sklearnx.metrics import weighted_absolute_percentage_error
import stdlib.csvx as csvx
import pandasx as pdx
import sktimex
import os
from stdlib.tprint import tprint

# import stdlib.loggingx as logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)


TARGET = 'import_kg'

RESULTS_WAPE = [
    ["item_area", "model", "wape"]
]

RESULT_FILE = "wape_nf_models.csv"


# ---------------------------------------------------------------------------

def use_model(g, name, model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_true, fh):
    item_area = g[0]
    tprint(f"... {item_area}/{name}")

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

    # ----

    try:
        linear = sktnf.nlinear.NLinear(
            input_size=24,
            h=1
        )
        use_model(g, "nlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        linear = sktnf.nlinear.NLinear(
            input_size=24,
            h=3
        )
        use_model(g, "nlinear-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        linear = sktnf.nlinear.NLinear(
            input_size=24,
            h=6
        )
        use_model(g, "nlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        linear = sktnf.nlinear.NLinear(
            input_size=24,
            h=12
        )
        use_model(g, "nlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    # ----

    try:
        # model = sktnf.autoformer.Autoformer(
        #     input_size=24,
        #     h=6
        # )
        # use_model(g, "autoformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
        #
        # model = sktnf.autoformer.Autoformer(
        #     input_size=24,
        #     h=12
        # )
        # use_model(g, "autoformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.fedformer.FEDformer(
            input_size=24,
            h=6
        )
        use_model(g, "fedformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.fedformer.FEDformer(
            input_size=24,
            h=12
        )
        use_model(g, "fedformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.informer.Informer(
            input_size=24,
            h=6
        )
        use_model(g, "informer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.informer.Informer(
            input_size=24,
            h=12
        )
        use_model(g, "informer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    # ---

    try:
        model = sktnf.bitcn.BiTCN(
            input_size=24,
            h=6
        )
        use_model(g, "bitcn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.bitcn.BiTCN(
            input_size=24,
            h=12
        )
        use_model(g, "bitcn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.deepnpts.DeepNPTS(
            input_size=24,
            h=6
        )
        use_model(g, "deepnpts-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.deepnpts.DeepNPTS(
            input_size=24,
            h=12
        )
        use_model(g, "deepnpts-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.dilated_rnn.DilatedRNN(
            input_size=24,
            h=6
        )
        use_model(g, "dilated_rnn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.dilated_rnn.DilatedRNN(
            input_size=24,
            h=12
        )
        use_model(g, "dilated_rnn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    # ----

    try:
        model = sktnf.itrasformer.iTransformer(
            input_size=24,
            h=6
        )
        use_model(g, "itrasformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.itrasformer.iTransformer(
            input_size=24,
            h=12
        )
        use_model(g, "itrasformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    # ----

    try:
        model = sktnf.gru.GRU(
            input_size=24,
            h=6
        )
        use_model(g, "gru-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.gru.GRU(
            input_size=24,
            h=12
        )
        use_model(g, "gru-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.rnn.RNN(
            input_size=24,
            h=6
        )
        use_model(g, "rnn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.rnn.RNN(
            input_size=24,
            h=12
        )
        use_model(g, "rnn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.lstm.LSTM(
            input_size=24,
            h=6
        )
        use_model(g, "lstm-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.lstm.LSTM(
            input_size=24,
            h=12
        )
        use_model(g, "lstm-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    # ----

    try:
        model = sktnf.nbeats.NBEATS(
            input_size=24,
            h=6
        )
        use_model(g, "nbeats-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.nbeats.NBEATS(
            input_size=24,
            h=12
        )
        use_model(g, "nbeats-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.nbeatsx.NBEATSx(
            input_size=24,
            h=6
        )
        use_model(g, "nbeatsx-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.nbeatsx.NBEATSx(
            input_size=24,
            h=12
        )
        use_model(g, "nbeatsx-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.nhits.NHITS(
            input_size=24,
            h=6
        )
        use_model(g, "nhits-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.nhits.NHITS(
            input_size=24,
            h=12
        )
        use_model(g, "nhits-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.patchtst.PatchTST(
            input_size=24,
            h=6
        )
        use_model(g, "patchtst-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.patchtst.PatchTST(
            input_size=24,
            h=12
        )
        use_model(g, "patchtst-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    # ----

    try:
        model = sktnf.tcn.TCN(
            input_size=24,
            h=6
        )
        use_model(g, "tcn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.tcn.TCN(
            input_size=24,
            h=12
        )
        use_model(g, "tcn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.tide.TiDE(
            input_size=24,
            h=6
        )
        use_model(g, "tide-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.tide.TiDE(
            input_size=24,
            h=12
        )
        use_model(g, "tide-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.timesnet.TimesNet(
            input_size=24,
            h=6
        )
        use_model(g, "timesnet-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)

        model = sktnf.timesnet.TimesNet(
            input_size=24,
            h=12
        )
        use_model(g, "timesnet-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh)
    except:
        pass

    pass


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
        ignore=["item_country", "imp_date",
                "prod_kg", "avg_retail_price_src_country", "producer_price_tonne_src_country"],
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
