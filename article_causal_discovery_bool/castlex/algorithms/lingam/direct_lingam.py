import lingam
from .base import BaseLingamLearner
from .utils import to_causal_matrix

class DirectLiNGAM(lingam.DirectLiNGAM, BaseLingamLearner):
    def __init__(
            self,
            random_state=None,
            prior_knowledge=None,
            apply_prior_knowledge_softly=False,
            measure="pwling",
    ):
        super().__init__(
            random_state=random_state,
            prior_knowledge=prior_knowledge,
            apply_prior_knowledge_softly=apply_prior_knowledge_softly,
            measure=measure,
        )

    def learn(self, data, columns=None, **kwargs):
        super().fit(data)
        self._causal_matrix = to_causal_matrix(self._adjacency_matrix)
        pass


