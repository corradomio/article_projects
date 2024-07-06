import logging.config
import logging.handlers
import pandasx as pdx


TARGET = 'import_kg'


def check_heteroschedastic(df, group):
    if pdx.is_heteroschedastic(df, target=TARGET):
        print("is_heteroschedastic:", group)


def check_spike(df, group):
    if pdx.is_spike(df, target=TARGET, outlier_std=6, outliers=0.1, detrend=True):
        print("is_spike:", group)


def main():

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

    # assert not df.isnull().values.any()

    # df = pdx.groups_select(df, 'ANIMAL FEED~CANADA', drop=True)
    # print(pdx.is_spike(df, target=TARGET))

    # pdx.groups_apply(df, check_spike)
    pdx.groups_apply(df, check_heteroschedastic)


    pass


if __name__ == '__main__':
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("main").info("Logging initialized")
    main()
