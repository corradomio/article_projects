
# Train & Predict Time series

import logging.config
import logging.handlers
import warnings

import pandas
from sktime.forecasting.base import ForecastingHorizon

import pandasx as pdx
import sktimex.forecasting.nf as sktnf
import sktimex.forecasting.nf.autoformer
import sktimex.forecasting.nf.bitcn
import sktimex.forecasting.nf.deepnpts
import sktimex.forecasting.nf.dilated_rnn
import sktimex.forecasting.nf.fedformer
import sktimex.forecasting.nf.gru
import sktimex.forecasting.nf.informer
import sktimex.forecasting.nf.itrasformer
import sktimex.forecasting.nf.lstm
import sktimex.forecasting.nf.nbeats
import sktimex.forecasting.nf.nbeatsx
import sktimex.forecasting.nf.nhits
import sktimex.forecasting.nf.nlinear
import sktimex.forecasting.nf.patchtst
import sktimex.forecasting.nf.rnn
import sktimex.forecasting.nf.tcn
import sktimex.forecasting.nf.tft
import sktimex.forecasting.nf.tide
import sktimex.forecasting.nf.timesnet
import sktimex.forecasting.nf.vanillatransformer
from common import *


# import stdlib.loggingx as logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

TARGET = 'import_kg'
USED_LIBRARY = "nf"
RESULT_FILE = "wape_nf_models.csv"


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

    linear = sktnf.nlinear.NLinear(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "nlinear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktnf.nlinear.NLinear(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "nlinear-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktnf.nlinear.NLinear(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "nlinear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktnf.nlinear.NLinear(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "nlinear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- VERY SLOW

    model = sktnf.autoformer.Autoformer(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "autoformer-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.autoformer.Autoformer(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "autoformer-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.autoformer.Autoformer(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "autoformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.autoformer.Autoformer(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "autoformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- VERY SLOW

    model = sktnf.fedformer.FEDformer(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "fedformer-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.fedformer.FEDformer(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "fedformer-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.fedformer.FEDformer(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "fedformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.fedformer.FEDformer(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "fedformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- VERY SLOW

    model = sktnf.informer.Informer(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "informer-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.informer.Informer(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "informer-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.informer.Informer(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "informer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.informer.Informer(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "informer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.bitcn.BiTCN(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "bitcn-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.bitcn.BiTCN(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "bitcn-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.bitcn.BiTCN(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "bitcn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.bitcn.BiTCN(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "bitcn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.deepnpts.DeepNPTS(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "deepnpts-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.deepnpts.DeepNPTS(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "deepnpts-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.deepnpts.DeepNPTS(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "deepnpts-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.deepnpts.DeepNPTS(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "deepnpts-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.dilated_rnn.DilatedRNN(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "dilated_rnn-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.dilated_rnn.DilatedRNN(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "dilated_rnn-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.dilated_rnn.DilatedRNN(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "dilated_rnn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.dilated_rnn.DilatedRNN(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "dilated_rnn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.itrasformer.iTransformer(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "itrasformer-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.itrasformer.iTransformer(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "itrasformer-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.itrasformer.iTransformer(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "itrasformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.itrasformer.iTransformer(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "itrasformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.gru.GRU(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "gru-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.rnn.RNN(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "rnn-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.lstm.LSTM(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "lstm-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --

    model = sktnf.gru.GRU(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "gru-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.rnn.RNN(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "rnn-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.lstm.LSTM(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "lstm-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --

    model = sktnf.gru.GRU(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "gru-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.rnn.RNN(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "rnn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.lstm.LSTM(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "lstm-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --

    model = sktnf.gru.GRU(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "gru-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.rnn.RNN(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "rnn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.lstm.LSTM(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "lstm-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.nbeats.NBEATS(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "nbeats-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nbeatsx.NBEATSx(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "nbeatsx-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --

    model = sktnf.nbeats.NBEATS(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "nbeats-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nbeatsx.NBEATSx(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "nbeatsx-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --

    model = sktnf.nbeats.NBEATS(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "nbeats-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nbeatsx.NBEATSx(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "nbeatsx-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # --

    model = sktnf.nbeats.NBEATS(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "nbeats-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nbeatsx.NBEATSx(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "nbeatsx-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.nhits.NHITS(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "nhits-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nhits.NHITS(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "nhits-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nhits.NHITS(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "nhits-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.nhits.NHITS(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "nhits-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.patchtst.PatchTST(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "patchtst-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.patchtst.PatchTST(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "patchtst-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.patchtst.PatchTST(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "patchtst-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.patchtst.PatchTST(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "patchtst-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ----

    model = sktnf.tcn.TCN(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "tcn-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tcn.TCN(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "tcn-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tcn.TCN(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "tcn-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tcn.TCN(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "tcn-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- VERY SLOW

    model = sktnf.tft.TFT(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "tft-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tft.TFT(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "tft-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tft.TFT(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "tft-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tft.TFT(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "tft-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- SLOW

    model = sktnf.tide.TiDE(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=100
    )
    use_model(g, "tide-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tide.TiDE(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=100
    )
    use_model(g, "tide-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tide.TiDE(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=100
    )
    use_model(g, "tide-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.tide.TiDE(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=100
    )
    use_model(g, "tide-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- VERY SLOW

    model = sktnf.timesnet.TimesNet(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=50
    )
    use_model(g, "timesnet-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.timesnet.TimesNet(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=50
    )
    use_model(g, "timesnet-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.timesnet.TimesNet(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=50
    )
    use_model(g, "timesnet-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.timesnet.TimesNet(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=50
    )
    use_model(g, "timesnet-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    # ---- SLOW

    model = sktnf.vanillatransformer.VanillaTransformer(
        input_size=24,
        h=1,
        early_stop_patience_steps=10,
        val_size=1,
        max_steps=50
    )
    use_model(g, "vanillatransformer-1", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.vanillatransformer.VanillaTransformer(
        input_size=24,
        h=3,
        early_stop_patience_steps=10,
        val_size=3,
        max_steps=50
    )
    use_model(g, "vanillatransformer-3", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.vanillatransformer.VanillaTransformer(
        input_size=24,
        h=6,
        early_stop_patience_steps=10,
        val_size=6,
        max_steps=50
    )
    use_model(g, "vanillatransformer-6", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    model = sktnf.vanillatransformer.VanillaTransformer(
        input_size=24,
        h=12,
        early_stop_patience_steps=10,
        val_size=12,
        max_steps=50
    )
    use_model(g, "vanillatransformer-12", model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

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

    # csvx.save_csv("results_wape.csv", RESULTS_WAPE[1:], header=RESULTS_WAPE[0])
    pass
# end


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
