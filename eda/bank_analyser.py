from utils import BasedAnalyzer


class BankAnalyzer(BasedAnalyzer):
    def __init__(self, dataset,cfg):
        super(BankAnalyzer, self).__init__(dataset,cfg)
