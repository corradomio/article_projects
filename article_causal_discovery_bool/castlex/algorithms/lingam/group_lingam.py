import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class GroupLiNGAM(lingam.GroupLiNGAM, BaseLingamLearner):
    def __init__(self, alpha=0.01):
        super().__init__(alpha=alpha)

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


