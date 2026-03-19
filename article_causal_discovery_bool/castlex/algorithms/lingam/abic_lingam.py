import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class ABICLiNGAM(lingam.ABICLiNGAM, BaseLingamLearner):
    def __init__(
            self,
            beta=1.0,
            lam=0.05,
            acyc_order=None,
            seed=0,
            max_outer=100,
            tol_h=1e-8,
            min_causal_effect=0.05,
            min_error_covariance=0.05,
            rho_max=1e16,
            inner_start=1,
            inner_growth=1,
            inner_tol=1e-4,
    ):
        super().__init__(
            beta=beta,
            lam=lam,
            acyc_order=acyc_order,
            seed=seed,
            max_outer=max_outer,
            tol_h=tol_h,
            min_causal_effect=min_causal_effect,
            min_error_covariance=min_error_covariance,
            rho_max=rho_max,
            inner_start=inner_start,
            inner_growth=inner_growth,
            inner_tol=inner_tol,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass

