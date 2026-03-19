import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix


class RCD(lingam.RCD, BaseLingamLearner):
    def __init__(
            self,
            max_explanatory_num=2,
            cor_alpha=0.01,
            ind_alpha=0.01,
            shapiro_alpha=0.01,
            MLHSICR=False,
            bw_method="mdbs",
            independence="hsic",
            ind_corr=0.5,
    ):
        super().__init__(
            max_explanatory_num=max_explanatory_num,
            cor_alpha=cor_alpha,
            ind_alpha=ind_alpha,
            shapiro_alpha=shapiro_alpha,
            MLHSICR=MLHSICR,
            bw_method=bw_method,
            independence=independence,
            ind_corr=ind_corr,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


