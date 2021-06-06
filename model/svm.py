from sklearn.svm import SVC, SVR

from model.based import BasedModel
from model.based import TrainingMode


class SVM(BasedModel):
    def __init__(self, cfg):
        super(SVM, self).__init__(cfg=cfg)
        self._params = {
            'C': cfg.SVM.C,
            'kernel': cfg.SVM.KERNEL,
            'degree': cfg.SVM.DEGREE,
            'gamma': cfg.SVM.GAMMA,
            'coef0': cfg.SVM.COEF0,
            'shrinking': cfg.SVM.SHRINKING,
            'probability': cfg.SVM.PROBABILITY,
            'tol': cfg.SVM.TOL,
            'cache_size': cfg.SVM.CACHE_SIZE,
            'class_weight': cfg.SVM.CLASS_WEIGHT,
            'verbose': cfg.SVM.VERBOSE,
            'max_iter': cfg.SVM.MAX_ITER,
            'decision_function_shape': cfg.SVM.DECISION_FUNCTION_SHAPE,
            'break_ties': cfg.SVM.BREAK_TIES,
            'random_state': cfg.SVM.RANDOM_STATE

        }

        self._training_mode = cfg.SVM.MODE
        if self._training_mode == TrainingMode.CLASSIFICATION:
            self.model = SVC(**self._params)
        elif self._training_mode == TrainingMode.REGRESSION:
            self.model = SVR(**self._params)
        self.name = cfg.SVM.NAME

        for _k in cfg.SVM.HYPER_PARAM_TUNING:
            _param = cfg.SVM.HYPER_PARAM_TUNING[_k]

            if _param is not None:
                _param = [*_param]
                self.fine_tune_params[_k.lower()] = [*_param]
