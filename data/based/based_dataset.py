#  Copyright (c) 2021.
#

import random

import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder, OrdinalEncoder, BinaryEncoder
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from tqdm import tqdm

from data.based.encoder_enum import EncoderTypes
from data.based.file_types import FileTypes
from data.based.scale_types import ScaleTypes
from data.based.transformers_enums import TransformersType

seed = 2021
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


class BasedDataset:

    def __init__(self, cfg, dataset_type):

        self._cfg = cfg
        self.dataset_type = dataset_type
        self.dataset_address = cfg.DATASET.DATASET_ADDRESS
        self.target = cfg.DATASET.TARGET
        self.dataset_description_file = cfg.DATASET.DATASET_BRIEF_DESCRIPTION

        if self.dataset_description_file is not None:
            self.about = self.__open_txt_file(self.dataset_description_file)

        self.load_dataset()
        self.origin_df = self.df.copy()

    def load_dataset(self):
        if self.dataset_type == FileTypes.CSV:
            self.df = self.__create_csv_dataframe()
        else:
            raise ValueError('dataset should be CSV file')

    def __transform(self, data, trans_type):
        try:
            if trans_type == TransformersType.LOG:
                data = np.log(data)
            elif trans_type == TransformersType.SQRT:
                data = np.sqrt(data)
            elif trans_type == TransformersType.BOX_PLOT:
                data = stats.boxcox(data)[0]
        except Exception as e:
            print('transform can not be applied')

        return data

    def __categorical_features(self, data=None):
        if data is None:
            return self.df.select_dtypes(include=['object']).columns.tolist()
        else:
            return data.select_dtypes(include=['object']).columns.tolist()

    def __numerical_features(self, data=None):
        if data is None:
            return self.df.select_dtypes(exclude=['object']).columns.tolist()
        else:
            return data.select_dtypes(exclude=['object']).columns.tolist()

    def select_columns(self, data, cols=None, just_numerical=True):
        if cols is None and just_numerical:
            cols = self.__numerical_features(data=data)
        return data[cols]

    def __target_encoding(self):
        le = LabelEncoder()
        targets = le.fit_transform(self.df[self.target])
        return targets

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

    def __categorical_feature_mapping(self, col, mapping_value):
        new_col = self.__generate_new_column_name(col=col, prefix='mapping')
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

    def split_to(self, test_size=0.10, val_size=0.10, has_validation=False, random_state=seed):
        X, y = self.__samples_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if has_validation:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size,
                                                              random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test

    def __generate_new_column_name(self, col, prefix):
        return '{}_{}'.format(col, prefix)

    def __samples_and_labels(self):
        data = self.df.copy()

        y = self.__target_encoding()
        X = data.drop(self.target, axis=1)

        return X, y

    def __create_csv_dataframe(self):
        return pd.read_csv(self.dataset_address, delimiter=';')

    def __open_txt_file(self, desc):
        return open(desc, 'r').read()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df

    @property
    def origin_df(self):
        return self._origin_df

    @origin_df.setter
    def origin_df(self, value):
        self._origin_df = value

    @property
    def dataset_address(self):
        return self._dataset_address

    @dataset_address.setter
    def dataset_address(self, address):
        self._dataset_address = address

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        self._dataset_type = value

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target

    @property
    def about(self):
        return self._about

    @about.setter
    def about(self, about):
        self._about = about

    @property
    def dataset_description_file(self):
        return self._dataset_description_file

    @dataset_description_file.setter
    def dataset_description_file(self, value):
        self._dataset_description_file = value
