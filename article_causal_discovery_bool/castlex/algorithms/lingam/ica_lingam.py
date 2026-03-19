import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class ICALiNGAM(lingam.ICALiNGAM, BaseLingamLearner):
    def __init__(self, random_state=None, max_iter=1000):
        super().__init__(
            max_iter=max_iter,
            random_state=random_state
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


