from causallearn.search.FCMBased.PNL.PNL import PNL as pnlPNL

from .base import BaseCLLearner


class PNL(BaseCLLearner):
    def __init__(self,
    ):
        super().__init__()

    def learn(self, data, columns=None, **kwargs):
        pnl = pnlPNL()
        pass

