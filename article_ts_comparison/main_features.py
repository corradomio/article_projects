import stdlib.loggingx as logging
import pandasx as pdx


def main():
    log = logging.getLogger("main")
    log.info("main ...")

    df = pdx.read_data(
        "data/tb_food_import_features_month.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]'),
        na_values=['(null)'],
        ignore=["export_kg", "prod_kg", "retail_price_unit", "avg_retail_price_src_country",
                "producer_price_tonne_src_country", "reexport_kg",
                'country', 'item', 'imp_date'
                ],
        numeric=["import_kg", ],
        index=['country', 'item', 'imp_date'],
        dropna=['country', 'item']
    )

    df_groups = pdx.groups_split(df)

    log.info("done")
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.info("Logging configured")
    main()
