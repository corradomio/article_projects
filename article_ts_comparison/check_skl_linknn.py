# Train & Predict Time series

import logging.config
import logging.handlers
import warnings

import pandas
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.base import ForecastingHorizon
import pandasx as pdx
import sktimex.forecasting.linear as sktfl
from common import *


# import stdlib.loggingx as logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

TARGET = 'import_kg'
USED_LIBRARY = "skl"
RESULT_FILE = "wape_skl_linknn.csv"


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
    linear = sktfl.LinearForecaster(
        estimator=LinearRegression,
        lags=24, tlags=1
    )
    use_model(g, "linear-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=LinearRegression,
        lags=24, tlags=3
    )
    use_model(g, "linear-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=LinearRegression,
        lags=24, tlags=6
    )
    use_model(g, "linear-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=LinearRegression,
        lags=24, tlags=12
    )
    use_model(g, "linear-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=LinearRegression,
        lags=24, tlags=12, flatten=True
    )
    use_model(g, "linear-12-flatten", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    #
    # sklean.linear_model.KNeighborsRegressor
    #
    linear = sktfl.LinearForecaster(
        estimator=KNeighborsRegressor,
        lags=24, tlags=1
    )
    use_model(g, "knn-1", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=KNeighborsRegressor,
        lags=24, tlags=3
    )
    use_model(g, "knn-3", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=KNeighborsRegressor,
        lags=24, tlags=6
    )
    use_model(g, "knn-6", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=KNeighborsRegressor,
        lags=24, tlags=12
    )
    use_model(g, "knn-12", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

    linear = sktfl.LinearForecaster(
        estimator=KNeighborsRegressor,
        lags=24, tlags=12, flatten=True
    )
    use_model(g, "knn-12-flatten", linear, Xtr_scaled, ytr_scaled, Xte_scaled, yte_scaled, fh, USED_LIBRARY, RESULT_FILE)

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
