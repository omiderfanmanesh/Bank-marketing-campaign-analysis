from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data.based import BasedDataset
from data.based.encoder_enum import EncoderTypes


class Encoders:
    def __init__(self, cdg, dataset: BasedDataset):
        self._dataset = dataset
        self._cfg = cdg

    def __get_encoder(self, encoder_type, col):
        le = None
        le_name = None
        if encoder_type == EncoderTypes.LABEL:
            le = LabelEncoder()
            le_name = 'label_encoding'
        elif encoder_type == EncoderTypes.ORDINAL:
            le = OrdinalEncoder()
            le_name = 'ordinal_encoding'
        elif encoder_type == EncoderTypes.ONE_HOT:
            le = OneHotEncoder()
            le_name = 'one_hot_encoding'
        elif encoder_type == EncoderTypes.BINARY:
            le = BinaryEncoder(cols=[col])
            le_name = 'binary_encoding'

        return le, le_name

    def categorical_feature_mapping(self, col, mapping_value):
        new_col = self.dataset.generate_new_column_name(col=col, prefix='mapping')
        self.df[new_col] = self.df[col].map(mapping_value)

    def __encoder(self, enc, X_train, X_test=None, y_train=None, y_test=None):
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

    def do_encode(self, X_train, X_test, y_train, y_test):
        for col in tqdm(self._cfg.ENCODER):
            encode_type = self._cfg.ENCODER[col]
            col = col.lower()
            enc, enc_name = self.__get_encoder(encoder_type=encode_type, col=col)
            if encode_type == EncoderTypes.LABEL:
                train_val = X_train[col].values
                test_val = X_test[col].values
                X_train[col], X_test[col] = self.__encoder(enc=enc, X_train=train_val, X_test=test_val)
            else:
                X_train, X_test = self.__encoder(enc=enc, X_train=X_train, X_test=X_test, y_train=y_train,
                                                 y_test=y_test)
        return X_train, X_test

    @property
    def target(self):
        return self._dataset.target

    @property
    def df(self):
        return self._dataset.df

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
