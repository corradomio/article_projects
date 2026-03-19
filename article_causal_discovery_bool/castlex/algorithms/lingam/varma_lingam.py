import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class VARMALiNGAM(lingam.VARMALiNGAM, BaseLingamLearner):
    def __init__(
            self,
            order=(1, 1),
            criterion="bic",
            prune=True,
            max_iter=100,
            ar_coefs=None,
            ma_coefs=None,
            lingam_model=None,
            random_state=None,
    ):
        super().__init__(
            order=order,
            criterion=criterion,
            prune=prune,
            max_iter=max_iter,
            ar_coefs=ar_coefs,
            ma_coefs=ma_coefs,
            lingam_model=lingam_model,
            random_state=random_state,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrices[0])
        pass


