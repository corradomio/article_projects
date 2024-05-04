# import logging.config
import stdlib.loggingx as logging
import pandasx as pdx

log = logging.getLogger("main")


def main():
    log.info("main ...")

    df = pdx.read_data(
        "data/vw_food_import_train_test_newfeatures.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]'),
        binhot="imp_month",
        na_values=['(null)'],
        ignore=["prod_kg", "avg_retail_price_src_country", "producer_price_tonne_src_country",
                "item_country", "imp_date"
                ],
        numeric=["import_kg", ],
        index=["item_country", "imp_date"],
    )
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.info("Logging configured")
    main()
