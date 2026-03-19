import numpy as np
from causallearn.search.FCMBased import lingam

from .base import BaseCLLearner

def _to_causal_matrix(adjacency_matrix: np.ndarray, threashold=0.1):
    n = adjacency_matrix.shape[0]
    causal_matrix = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i,j] >= threashold:
                causal_matrix[i,j] = 1
            elif adjacency_matrix[i,j] <= -threashold:
                causal_matrix[j,i] = 1
    return causal_matrix


class ICALiNGAM(BaseCLLearner):
    def __init__(self,
                 threshold=0.1,
                 random_state=None,
                 max_iter=1000
    ):
        super().__init__()
        self.threshold = threshold
        self.random_state = random_state
        self.max_iter = max_iter
        self.adjacency_matrix_ = None
        self.causal_matrix = None

    def learn(self, data, columns=None, **kwargs):
        model = lingam.ICALiNGAM(self.random_state, self.max_iter)
        model.fit(data)
        self.adjacency_matrix_ = model.adjacency_matrix_
        self.causal_matrix = _to_causal_matrix(model.adjacency_matrix_, self.threshold)
        pass


class DirectLiNGAM(BaseCLLearner):
    def __init__(self,
                 threshold=0.1,
                 random_state=None,
                 prior_knowledge=None,
                 apply_prior_knowledge_softly=False,
                 measure='pwling'
    ):
        super().__init__()
        self.threshold = threshold
        self.random_state = random_state
        self.prior_knowledge = prior_knowledge
        self.apply_prior_knowledge_softly = apply_prior_knowledge_softly
        self.measure = measure

    def learn(self, data, columns=None, **kwargs):
        model = lingam.DirectLiNGAM(self.random_state,
                                    self.prior_knowledge,
                                    self.apply_prior_knowledge_softly,
                                    self.measure)
        model.fit(data)
        self.adjacency_matrix_ = model.adjacency_matrix_
        self.causal_matrix = _to_causal_matrix(model.adjacency_matrix_, self.threshold)
        pass


class VARLiNGAM(BaseCLLearner):
    def __init__(self,
                 lags=1,
                 criterion='bic',
                 prune=False,
                 ar_coefs=None,
                 lingam_model=None,
                 threshold=0.1,
                 random_state=None
    ):
        super().__init__()
        self.threshold = threshold
        self.random_state = random_state
        self.lags = lags
        self.criterion = criterion
        self.prune = prune
        self.ar_coefs = ar_coefs
        self.lingam_model = lingam_model

    def learn(self, data, columns=None, **kwargs):
        model = lingam.VARLiNGAM(self.lags, self.criterion, self.prune, self.ar_coefs, self.lingam_model, self.random_state)
        model.fit(data)
        self.adjacency_matrix_ = model.adjacency_matrices_
        self.causal_matrix = _to_causal_matrix(model.adjacency_matrices_[0], self.threshold)
        pass
