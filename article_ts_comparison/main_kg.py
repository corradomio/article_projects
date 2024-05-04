import stdlib.loggingx as logging
import pandasx as pdx


def main():
    log = logging.getLogger("main")
    log.info("main ...")

    df = pdx.read_data(
        "data/vw_food_import_pred.csv",
        datetime=('imp_date', '[%Y/%m/%d:%H:%M:%S %p]'),
        categorical='imp_month',
        na_values=('(null)'),
        ignore=[],
        numeric=["import_kg", ],
        index=['item_country', 'imp_date'],
        # dropna='item_country'
    )

    # df = pdx.nan_drop(df, columns='item_country')
    pdx.nan_drop(df, columns='item_country', inplace=True)

    df, dfnan = pdx.nan_split(df, columns='import_kg')
    X_train, y_train, X_pred, y_pred = pdx.xy_split(df, dfnan, target='import_kg')

    df_groups = pdx.groups_split(df)

    log.info("done")


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.info("Logging configured")
    main()
