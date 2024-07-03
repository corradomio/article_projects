import pandas as pd
import numpy as np
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class FirstForecaster(BaseForecaster):
    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    def __init__(self, param1=0):
        super().__init__()
        self.param1 = param1

    def _fit(self, y, X=None, fh=None):
        return self

    def _predict(self, fh, X=None):
        _y_shape = self._y.shape
        if len(_y_shape) == 1:
            y_pred = np.zeros(len(fh))
        else:
            y_pred = np.zeros((len(fh),) + _y_shape[1:])
        return y_pred

    def get_params(self, deep=True):
        params = super().get_params(deep=True)
        return params


class SecondForecaster(FirstForecaster):

    def __init__(self, param1=1, param2=2):
        super().__init__()
        self.param2 = param2

    def get_params(self, deep=True):
        params = super().get_params(deep=True)
        return params


def main():
    it = pd.date_range(start='2024-01-01', periods=24, freq='M')
    ip = pd.date_range(start='2026-01-01', periods=12, freq='M')

    Nx = 4
    Ny = 2
    Nt = 24
    Np = 12
    xc = [f"x{i + 1}" for i in range(Nx)]
    yc = [f"y{i + 1}" for i in range(Ny)]

    xt = np.zeros((Nt, Nx), dtype=float)
    yt = np.zeros((Nt, Ny), dtype=float)
    xp = np.zeros((Np, Nx), dtype=float)
    yp = np.zeros((Np, Ny), dtype=float)

    fhr = ForecastingHorizon(range(1, 12 + 1))
    fha = ForecastingHorizon(ip)

    Xt = pd.DataFrame(data=xt, columns=xc, index=it)
    Yt = pd.DataFrame(data=yt, columns=yc, index=it)
    Xp = pd.DataFrame(data=xp, columns=xc, index=ip)
    Yp = pd.DataFrame(data=yp, columns=yc, index=ip)

    zf = SecondForecaster()
    zf.fit(Yt, Xt)
    Ypr = zf.predict(fhr)
    pass


if __name__ == "__main__":
    main()
