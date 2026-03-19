import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class HighDimDirectLiNGAM(lingam.HighDimDirectLiNGAM, BaseLingamLearner):
    def __init__(self, J=3, K=4, alpha=0.5, estimate_adj_mat=True, random_state=None):
        super().__init__(
            J=J,
            K=K,
            alpha=alpha,
            estimate_adj_mat=estimate_adj_mat,
            random_state=random_state
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


