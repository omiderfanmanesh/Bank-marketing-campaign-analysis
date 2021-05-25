from sklearn.svm import SVC

from based_model import BasedModel


class SVM(BasedModel):
    def __init__(self, cfg):
        super(SVM, self).__init__(cfg=cfg)
        params = {
            'C': cfg.SVM.C,
            'kernel': cfg.SVM.KERNEL,
            'gamma': cfg.SVM.GAMMA,
            'verbose': cfg.SVM.VERBOSE
        }
        super.model = SVC(**params)
