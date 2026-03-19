import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class RESIT(lingam.RESIT, BaseLingamLearner):
    def __init__(
            self,
            regressor,
            random_state=None,
            prior_knowledge=None,
            alpha=0.01
    ):
        super().__init__(
            regressor=regressor,
            random_state=random_state,
            prior_knowledge=prior_knowledge,
            alpha=alpha
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


