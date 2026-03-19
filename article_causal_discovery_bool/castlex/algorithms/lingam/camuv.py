import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class CAMUV(lingam.CAMUV, BaseLingamLearner):
    def __init__(
            self,
            alpha=0.01,
            num_explanatory_vals=2,
            independence="hsic",
            ind_corr=0.5,
            prior_knowledge=None,
    ):
        super().__init__(
            alpha=alpha,
            num_explanatory_vals=num_explanatory_vals,
            independence=independence,
            ind_corr=ind_corr,
            prior_knowledge=prior_knowledge,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


