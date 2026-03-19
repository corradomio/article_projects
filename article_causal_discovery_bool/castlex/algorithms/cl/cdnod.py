from typing import Optional

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *

from .base import BaseCLLearner
from .utils import from_causal_graph


class CDNOD(BaseCLLearner):
    def __init__(self,
                 # c_indx: np.ndarray,
                 alpha: float = 0.05,
                 indep_test: str = fisherz,
                 stable: bool = True,
                 uc_rule: int = 0,
                 uc_priority: int = 2,
                 mvcdnod: bool = False,
                 correction_name: str = 'MV_Crtn_Fisher_Z',
                 background_knowledge: Optional[BackgroundKnowledge] = None,
                 verbose: bool = False,
                 show_progress: bool = False,
                 **kwargs
    ):
        super().__init__()
        # self.c_indx = c_indx
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.mvcdnod = mvcdnod
        self.correction_name = correction_name
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.show_progress = show_progress
        self.kwargs = kwargs

    def learn(self, data, columns=None, **kwargs):
        X = data[:,:-1]
        y = data[:,-1:]
        graph: CausalGraph = cdnod(
            X,
            y,
            alpha=self.alpha,
            indep_test=self.indep_test,
            stable=self.stable,
            uc_rule=self.uc_rule,
            uc_priority=self.uc_priority,
            mvcdnod=self.mvcdnod,
            correction_name=self.correction_name,
            background_knowledge=self.background_knowledge,
            verbose=self.verbose,
            show_progress=self.show_progress,
            **self.kwargs
        )
        causal_matrix = from_causal_graph(graph)
        self._causal_matrix = causal_matrix
        pass

