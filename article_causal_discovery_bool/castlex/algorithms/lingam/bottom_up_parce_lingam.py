import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class BottomUpParceLiNGAM(lingam.BottomUpParceLiNGAM, BaseLingamLearner):
    def __init__(
            self,
            random_state=None,
            alpha=0.1,
            regressor=None,
            prior_knowledge=None,
            independence="hsic",
            ind_corr=0.5,
    ):
        super().__init__(
            random_state=random_state,
            alpha=alpha,
            regressor=regressor,
            prior_knowledge=prior_knowledge,
            independence=independence,
            ind_corr=ind_corr,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


