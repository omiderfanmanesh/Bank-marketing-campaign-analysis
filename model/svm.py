from sklearn.svm import SVC, SVR

from model.based import BasedModel
from model.based import TrainingMode


class SVM(BasedModel):
    def __init__(self, cfg):
        super(SVM, self).__init__(cfg=cfg)
        self._params = {
            # 'C': cfg.SVM.C,
            'kernel': cfg.SVM.KERNEL,
            'gamma': cfg.SVM.GAMMA,
            'verbose': cfg.SVM.VERBOSE
        }

        self._training_mode = cfg.SVM.MODE
        if self._training_mode == TrainingMode.CLASSIFICATION:
            self.model = SVC(**self._params)
        elif self._training_mode == TrainingMode.REGRESSION:
            self.model = SVR(**self._params)

    def get_model(self):
        return self.model

