from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler

from data.based.scale_types import ScaleTypes


class Scalers:
    def __init__(self, cfg):
        self._cfg = cfg

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

    def do_scale(self, data=None, X_train=None, X_test=None):
        if data is None:
            return self.__do_scale_by(X_train=X_train, X_test=X_test)
        else:
            return self.__do_scale_all(data=data)

    def __do_scale_by(self, X_train=None, X_test=None):
        scl_type = self._cfg.SCALER
        scl, scl_name = self.__get_scaler(scale_type=scl_type)
        train_scale, test_scale = self.__scaler_by(scl=scl, X_train=X_train, X_test=X_test)
        return train_scale, test_scale

    def __do_scale_all(self, data):
        scl_type = self._cfg.SCALER
        scl, scl_name = self.__get_scaler(scale_type=scl_type)
        data_scale = self.__scaler_all(scl=scl, data=data)
        return data_scale

    def __scaler_by(self, scl, X_train=None, X_test=None):
        scl.fit(X_train)  # Apply transform to both the training set and the test set.
        train_scale = scl.transform(X_train)
        test_scale = None
        if X_test is not None:
            test_scale = scl.transform(X_test)

        return train_scale, test_scale

    def __scaler_all(self, scl, data: DataFrame):
        scl.fit(data.values)  # Apply transform to both the training set and the test set.
        train_scale = scl.transform(data.values)
        return train_scale
