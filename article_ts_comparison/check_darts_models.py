
# Train & Predict Time series

import logging.config
import logging.handlers
import warnings

import pandas
from sktime.forecasting.base import ForecastingHorizon

import pandasx as pdx
import sktimex.forecasting.darts as sktdts
import sktimex.forecasting.darts.dlinear
import sktimex.forecasting.darts.nlinear
import sktimex.forecasting.darts.nbeats
import sktimex.forecasting.darts.nhits
import sktimex.forecasting.darts.rnn_model
import sktimex.forecasting.darts.tcn_model
import sktimex.forecasting.darts.tft_model
import sktimex.forecasting.darts.tide_model
import sktimex.forecasting.darts.transformer_model
import sktimex.forecasting.darts.tsmixer_model
from common import *


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

TARGET = 'import_kg'
USED_LIBRARY = "darts"
RESULT_FILE = "wape_darts_models.csv"


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

    # ----

    linear = sktdts.dlinear.DLinearModel(
        input_chunk_length=24,
        output_chunk_length=1
    )
    use_model(g, "dlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.dlinear.DLinearModel(
        input_chunk_length=24,
        output_chunk_length=3
    )
    use_model(g, "dlinear-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.dlinear.DLinearModel(
        input_chunk_length=24,
        output_chunk_length=6
    )
    use_model(g, "dlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.dlinear.DLinearModel(
        input_chunk_length=24,
        output_chunk_length=12
    )
    use_model(g, "dlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.nbeats.NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,
        n_epochs=100
    )
    use_model(g, "nbeats-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nbeats.NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=3,
        n_epochs=100
    )
    use_model(g, "nbeats-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nbeats.NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=6,
        n_epochs=100
    )
    use_model(g, "nbeats-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nbeats.NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
        n_epochs=100
    )
    use_model(g, "nbeats-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.nhits.NHiTSModel(
        input_chunk_length=24,
        output_chunk_length=1
    )
    use_model(g, "nhits-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nhits.NHiTSModel(
        input_chunk_length=24,
        output_chunk_length=3
    )
    use_model(g, "nhits-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nhits.NHiTSModel(
        input_chunk_length=24,
        output_chunk_length=6
    )
    use_model(g, "nhits-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nhits.NHiTSModel(
        input_chunk_length=24,
        output_chunk_length=12
    )
    use_model(g, "nhits-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.nlinear.NLinearModel(
        input_chunk_length=24,
        output_chunk_length=1
    )
    use_model(g, "nlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nlinear.NLinearModel(
        input_chunk_length=24,
        output_chunk_length=3
    )
    use_model(g, "nlinear-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nlinear.NLinearModel(
        input_chunk_length=24,
        output_chunk_length=6
    )
    use_model(g, "nlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.nlinear.NLinearModel(
        input_chunk_length=24,
        output_chunk_length=12
    )
    use_model(g, "nlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=1,
        model='RNN'
    )
    use_model(g, "rnn-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=1,
        model='GRU'
    )
    use_model(g, "gru-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=1,
        model='LSTM'
    )
    use_model(g, "lstm-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=3,
        model='RNN'
    )
    use_model(g, "rnn-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=3,
        model='GRU'
    )
    use_model(g, "gru-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=3,
        model='LSTM'
    )
    use_model(g, "lstm-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=6,
        model='RNN'
    )
    use_model(g, "rnn-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=6,
        model='GRU'
    )
    use_model(g, "gru-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=6,
        model='LSTM'
    )
    use_model(g, "lstm-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # -

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=12,
        model='RNN'
    )
    use_model(g, "rnn-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=12,
        model='GRU'
    )
    use_model(g, "gru-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.rnn_model.RNNModel(
        input_chunk_length=24,
        output_chunk_length=12,
        model='LSTM'
    )
    use_model(g, "lstm-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.tcn_model.TCNModel(
        input_chunk_length=24,
        output_chunk_length=1,
    )
    use_model(g, "tcn-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tcn_model.TCNModel(
        input_chunk_length=24,
        output_chunk_length=3,
    )
    use_model(g, "tcn-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tcn_model.TCNModel(
        input_chunk_length=24,
        output_chunk_length=6,
    )
    use_model(g, "tcn-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tcn_model.TCNModel(
        input_chunk_length=24,
        output_chunk_length=12,
    )
    use_model(g, "tcn-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.tft_model.TFTModel(
        input_chunk_length=24,
        output_chunk_length=1,
        # n_epochs=100
    )
    use_model(g, "tft-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tft_model.TFTModel(
        input_chunk_length=24,
        output_chunk_length=3,
        # n_epochs=100
    )
    use_model(g, "tft-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tft_model.TFTModel(
        input_chunk_length=24,
        output_chunk_length=6,
        # n_epochs=100
    )
    use_model(g, "tft-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tft_model.TFTModel(
        input_chunk_length=24,
        output_chunk_length=12,
        # n_epochs=50
    )
    use_model(g, "tft-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.tide_model.TiDEModel(
        input_chunk_length=24,
        output_chunk_length=1
    )
    use_model(g, "tide-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tide_model.TiDEModel(
        input_chunk_length=24,
        output_chunk_length=3
    )
    use_model(g, "tide-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tide_model.TiDEModel(
        input_chunk_length=24,
        output_chunk_length=6
    )
    use_model(g, "tide-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tide_model.TiDEModel(
        input_chunk_length=24,
        output_chunk_length=12
    )
    use_model(g, "tide-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.transformer_model.TransformerModel(
        input_chunk_length=24,
        output_chunk_length=1
    )
    use_model(g, "transformer-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.transformer_model.TransformerModel(
        input_chunk_length=24,
        output_chunk_length=3
    )
    use_model(g, "transformer-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.transformer_model.TransformerModel(
        input_chunk_length=24,
        output_chunk_length=6
    )
    use_model(g, "transformer-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.transformer_model.TransformerModel(
        input_chunk_length=24,
        output_chunk_length=12
    )
    use_model(g, "transformer-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    linear = sktdts.tsmixer_model.TSMixerModel(
        input_chunk_length=24,
        output_chunk_length=1
    )
    use_model(g, "tsmixer-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tsmixer_model.TSMixerModel(
        input_chunk_length=24,
        output_chunk_length=3
    )
    use_model(g, "tsmixer-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tsmixer_model.TSMixerModel(
        input_chunk_length=24,
        output_chunk_length=6
    )
    use_model(g, "tsmixer-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktdts.tsmixer_model.TSMixerModel(
        input_chunk_length=24,
        output_chunk_length=12
    )
    use_model(g, "tsmixer-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

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
