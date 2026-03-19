import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class VARLiNGAM(lingam.VARLiNGAM, BaseLingamLearner):
    def __init__(
            self,
            lags=1,
            criterion="bic",
            prune=True,
            ar_coefs=None,
            lingam_model=None,
            random_state=None,
    ):
        super().__init__(
            lags=lags,
            criterion=criterion,
            prune=prune,
            ar_coefs=ar_coefs,
            lingam_model=lingam_model,
            random_state=random_state,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrices[0])
        pass


