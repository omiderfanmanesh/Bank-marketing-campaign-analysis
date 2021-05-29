from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler

from data.based import BasedDataset
from data.based.scale_types import ScaleTypes


class Scalers:
    def __init__(self, cfg, dataset: BasedDataset):
        self._cfg = cfg
        self._dataset = dataset

    def __get_scaler(self, scale_type):
        scl = None
        scl_name = None
        if scale_type == ScaleTypes.MIN_MAX:
            scl = MinMaxScaler()
            scl_name = 'min_max_scaler'
        elif scale_type == ScaleTypes.STANDARD:
            scl = StandardScaler()
            scl_name = 'min_max_scaler'
        elif scale_type == ScaleTypes.MAX_ABS:
            scl = MaxAbsScaler()
            scl_name = 'max_abs_scaler'
        elif scale_type == ScaleTypes.ROBUST:
            scl = RobustScaler()
            scl_name = 'robust_scaler'

        return scl, scl_name

    def __scaler(self, scl, X_train, X_test=None):
        scl.fit(X_train)  # Apply transform to both the training set and the test set.
        train_scale = scl.transform(X_train)
        test_scale = None
        if X_test is not None:
            test_scale = scl.transform(X_test)

        return train_scale, test_scale

    def do_scale(self, X_train, X_test):
        scl_type = self._cfg.SCALER
        scl, scl_name = self.__get_scaler(scale_type=scl_type)
        train_scale, test_scale = self.__scaler(scl=scl, X_train=X_train, X_test=X_test)
        return train_scale, test_scale
