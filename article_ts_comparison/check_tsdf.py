import pandas as pd
import pandasx as pdx
import numpy as np

from stdlib import lrange
from sktimex.forecasting.zero import ZeroForecaster
from sktimex.forecasting.linear import LinearForecaster
from sktimex.forecasting.scikit import ScikitForecaster


def main():
    ncols = 4
    index1 = pd.date_range('2024-01-01', periods=100, freq='MS')
    index = pdx.date_range('2024-01-01', periods=100, freq='M')
    data = np.ones((len(index), ncols), dtype=np.float64)
    columns = [f"c{i+1}" for i in range(ncols)]

    df = pd.DataFrame(data=data, index=index, columns=columns)
    # df = pd.DataFrame(data=data, columns=columns)

    train, test = pdx.train_test_split(df, train_size=75)
    X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target=['c3', 'c4'])

    # cf = ConstantForecaster(const_value=42)
    # cf = LinearForecaster(
    #     lags=10,
    #     tlags=5
    # )
    cf = ScikitForecaster(
        window_length=10,
        prediction_length=5
    )
    cf.fit(y=y_train, X=X_train)

    n_pred = len(X_test)
    # y1_pred = cf.predict(fh=lrange(1, n_pred+1), X=X_test)
    y2_pred = cf.predict(fh=y_test.index, X=X_test)

    pass


if __name__ == '__main__':
    main()