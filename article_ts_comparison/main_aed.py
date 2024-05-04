import stdlib.loggingx as logging
import pandasx as pdx


def apply_train_pred(group, Xy_all):
    log = logging.getLogger("apply")
    log.debug(f'{group[0]}')

    X_train, y_train, X_pred, y_pred = Xy_all

    return
# end


def main():
    log = logging.getLogger("main")
    log.info("main ...")

    df = pdx.read_data(
        "data/vw_food_import_aed_pred.csv",
        datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]'),    # datetime format different from 'kg'
        # categorical='imp_month',
        binhot='imp_month',
        na_values=['(null)'],
        ignore=['item_country', 'imp_date'],
        numeric=["import_aed", ],
        index=['item_country', 'imp_date'],
        dropna='item_country'
    )

    df, dfnan = pdx.nan_split(df, columns='import_aed')
    # X_train, y_train, X_pred, y_pred = pdx.xy_split(df, dfnan, target='import_aed')

    df_groups = pdx.groups_split(df)
    dfnan_groups = pdx.groups_split(dfnan)
    # X_train, y_train, X_pred, y_pred = pdx.xy_split(df_groups, dfnan_groups, target='import_kg')

    for group in df_groups:
        dfg = df_groups[group]
        dfn = dfnan_groups[group]
        Xy_all = pdx.xy_split(dfg, dfn, target='import_aed')

        apply_train_pred(group, Xy_all)

    log.info("done")


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.info("Logging configured")
    main()
