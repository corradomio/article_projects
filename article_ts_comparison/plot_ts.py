import logging.config
import logging.handlers

import pandasx as pdx
import sktimex
from stdlib.tprint import tprint


log = None
TARGET = 'import_kg'


# ---------------------------------------------------------------------------
# D:\Projects.github\python_projects\check_timeseries_nn\plots\import_aed
#

def plot_ts(df, group: tuple):
    # group: <item>~<country>
    # reverse the order
    item_area = group[0]
    parts = item_area.split("~")
    area_item =f"{parts[1]}~{parts[0]}"

    tprint(item_area)
    gname = area_item.replace('/', '_').replace(' ', '_')
    y = df[TARGET]
    sktimex.utils.plot_series(y, labels=['data'], title=item_area)

    fname = f"../article_ts_comparison_data/plots.352/{gname}.png"
    sktimex.utils.savefig(fname, dpi=300)
    sktimex.utils.close()
    return
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
    # ------------------------------------
    # tb_food_import_features_month (4190)
    # ------------------------------------
    # "item","country","imp_date","import_kg","export_kg","reexport_kg","prod_kg","retail_price_unit",
    # "avg_retail_price_src_country","producer_price_tonne_src_country","crude_oil_price","sandp_500_us",
    # "sandp_sensex_india","shenzhen_index_china","nikkei_225_japan","max_temperature","mean_temperature",
    # "min_temperature","vap_pressure","evaporation","rainy_days"

    df = pdx.read_data(
        "data/vw_food_import_train_test_newfeatures.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),    # datetime format different from 'kg'
        # categorical='imp_month',
        binhot='imp_month',
        na_values=['(null)'],
        ignore=['item_country', 'imp_date'],
        numeric=[TARGET],
        index=['item_country', 'imp_date'],
        dropna='item_country'
    )

    # df = pdx.read_data(
    #     "data/tb_food_import_features_month_item_country.csv",
    #     datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),  # datetime format different from 'kg'
    #     # categorical='imp_month',
    #     # na_values=['(null)'],
    #     ignore=['item_country', 'imp_date', "export_kg", "reexport_kg", "prod_kg", "retail_price_unit",
    #             "avg_retail_price_src_country", "producer_price_tonne_src_country"],
    #     numeric=[TARGET],
    #     index=['item_country', 'imp_date']
    # )

    pdx.groups_apply(df, plot_ts)
    pass


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
