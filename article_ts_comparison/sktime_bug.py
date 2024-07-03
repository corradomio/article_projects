import pandas as pd
import numpy as np
import sktime
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class ZeroForecaster(BaseForecaster):
    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        print(type(y))
        return self

    def _predict(self, fh, X=None):
        _y_shape = self._y.shape
        if len(_y_shape) == 1:
            y_pred = np.zeros(len(fh))
        else:
            y_pred = np.zeros((len(fh),) + _y_shape[1:])
        return y_pred


def main():
    print(sktime.__version__)
    
    it = pd.date_range(start='2024-01-01', periods=24, freq='MS')
    ip = pd.date_range(start='2026-01-01', periods=12, freq='MS')

    Nx = 4
    Ny = 2
    Nt = 24
    Np = 12
    xc = [f"x{i+1}" for i in range(Nx)]
    yc = [f"y{i+1}" for i in range(Ny)]

    xt = np.zeros((Nt, Nx), dtype=float)
    yt = np.zeros((Nt, Ny), dtype=float)
    xp = np.zeros((Np, Nx), dtype=float)
    yp = np.zeros((Np, Ny), dtype=float)

    fhr = ForecastingHorizon(range(1, 12+1))
    fha = ForecastingHorizon(ip)

    Xt = pd.DataFrame(data=xt, columns=xc, index=it)
    Yt = pd.DataFrame(data=yt, columns=yc, index=it)
    Xp = pd.DataFrame(data=xp, columns=xc, index=ip)
    Yp = pd.DataFrame(data=yp, columns=yc, index=ip)

    print("Xt.index: ", Xt.index)
    print("Yt.index: ", Yt.index)
    print("Xp.index: ", Xp.index)
    print("Yp.index: ", Yp.index)

    zf = ZeroForecaster()
    zf.fit(Yt, Xt)

    Ypr = zf.predict(fhr)
    print(Ypr.index)
    Ypr = zf.predict(fha)
    print(Ypr.index)
    Ypr = zf.predict(fhr, X=Xp)
    print(Ypr.index)
    Ypr = zf.predict(fha, X=Xp)
    print(Ypr.index)
    pass


if __name__ == "__main__":
    main()
