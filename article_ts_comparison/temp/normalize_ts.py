import logging.config
import logging.handlers
import warnings
import pandas
import pandasx as pdx
import sktimex
import os
from stdlib.tprint import tprint

# import stdlib.loggingx as logging
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)

log = None


TARGET = 'import_kg'


# ---------------------------------------------------------------------------

def plot_ts(df, group, method):
    tprint(group)
    gname = group[0].replace('/', '_').replace(' ', '_')
    y = df[TARGET]
    sktimex.utils.plot_series(y, labels=['data'], title=group[0])

    fdir = f"plots/{method}"
    fname = f"{fdir}/{gname}.png"
    os.makedirs(fdir, exist_ok=True)

    sktimex.utils.savefig(fname, dpi=300)
    sktimex.utils.close()


def scale_ts(X, method):
    mms = pdx.preprocessing.MinMaxScaler(
        method=method,
        sp=12
    )
    X_scaled = mms.fit_transform(X)
    return X_scaled


def clip_ts(X):
    clip = pdx.OutlierTransformer(
        sp=12
    )
    X_clipped = clip.fit_transform(X)
    return X_clipped


def transform_ts(X, group):
    for method in ['global', 'linear', 'poly1']:
        plot_ts(X, group, 'plain')
        X_clipped = clip_ts(X)
        X_scaled = scale_ts(X_clipped, method)
        plot_ts(X_scaled, group, method)
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
        dropna='item_country'
    )

    pdx.groups_apply(df, transform_ts)
    pass


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
