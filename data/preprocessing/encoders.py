from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data.based.encoder_enum import EncoderTypes


class Encoders:
    def __init__(self, cdg):
        self._cfg = cdg

    def __get_encoder(self, encoder_type, col):
        le = None
        le_name = None
        if encoder_type == EncoderTypes.LABEL:
            le = LabelEncoder()
            le_name = 'label_encoding'
        elif encoder_type == EncoderTypes.ORDINAL:
            le = OrdinalEncoder(cols=[col])
            le_name = 'ordinal_encoding'
        elif encoder_type == EncoderTypes.ONE_HOT:
            le = OneHotEncoder(cols=[col])
            le_name = 'one_hot_encoding'
        elif encoder_type == EncoderTypes.BINARY:
            le = BinaryEncoder(cols=[col])
            le_name = 'binary_encoding'

        return le, le_name

    def __encoder(self, enc, data=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None):
        if data is None and y is None:
            return self.__encoder_by(enc=enc, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        else:
            return self.__encoder_all(enc=enc, data=data, y=y)

    def __encoder_by(self, enc, X_train=None, X_test=None, y_train=None, y_test=None):
        if isinstance(enc, LabelEncoder):
            enc.fit(X_train)
            train_enc = enc.transform(X_train)
            test_enc = None
            if X_test is not None:
                test_enc = enc.transform(X_test)
        else:
            enc.fit(X_train, y_train)
            train_enc = enc.transform(X_train, y_train)
            test_enc = None
            if X_test is not None:
                test_enc = enc.transform(X_test, y_test)
        return train_enc, test_enc

    def __encoder_all(self, enc, data: DataFrame, y: DataFrame):
        if isinstance(enc, LabelEncoder):
            enc.fit(data)
            data_enc = enc.transform(data)
        else:
            enc.fit(data, y)
            data_enc = enc.transform(data, y)
        return data_enc

    def do_encode(self, data=None, y=None, X_train=None, X_test=None, y_train=None, y_test=None):
        if data is None and y is None:
            return self.__encode_by(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        else:
            return self.__encode_all(data=data, y=y)

    def __encode_by(self, X_train=None, X_test=None, y_train=None, y_test=None):
        for col in tqdm(self._cfg.ENCODER):
            encode_type = self._cfg.ENCODER[col]
            col = col.lower()
            enc, enc_name = self.__get_encoder(encoder_type=encode_type, col=col)
            if encode_type == EncoderTypes.LABEL:
                train_val = X_train[col].values
                test_val = X_test[col].values
                X_train[col], X_test[col] = self.__encoder_by(enc=enc, X_train=train_val, X_test=test_val)
            else:
                X_train, X_test = self.__encoder_by(enc=enc, X_train=X_train, X_test=X_test, y_train=y_train,
                                                    y_test=y_test)

        return X_train, X_test

    def __encode_all(self, data, y):
        for col in tqdm(self._cfg.ENCODER):
            encode_type = self._cfg.ENCODER[col]
            col = col.lower()
            enc, enc_name = self.__get_encoder(encoder_type=encode_type, col=col)
            if encode_type == EncoderTypes.LABEL:
                train_val = data[col].values
                data[col] = self.__encoder_all(enc=enc, data=train_val, y=y)
            else:
                data = self.__encoder_all(enc=enc, data=data, y=y)
        return data
