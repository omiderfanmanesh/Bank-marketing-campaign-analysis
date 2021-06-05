#  Copyright (c) 2021.
#

import random

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from sklearn.model_selection import train_test_split

from data.based.file_types import FileTypes
from data.based.transformers_enums import TransformersType

seed = 2021
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


class BasedDataset:

    def __init__(self, cfg, dataset_type, target_encoding=True):

        self._cfg = cfg
        self.dataset_type = dataset_type
        self.dataset_address = cfg.DATASET.DATASET_ADDRESS
        self.target_col = cfg.DATASET.TARGET
        self.dataset_description_file = cfg.DATASET.DATASET_BRIEF_DESCRIPTION

        if self.dataset_description_file is not None:
            self.about = self.__open_txt_file(self.dataset_description_file)

        self.load_dataset()
        self.df = self.df_main.copy()
        self.pca = None
        self.encoded_data = None
        self.scaled_data = None

    def load_dataset(self):
        if self.dataset_type == FileTypes.CSV:
            self.df_main = self.__create_csv_dataframe()
        else:
            raise ValueError('dataset should be CSV file')

    def transform(self, data, trans_type):
        _min = min(data)
        if _min <= 0:
            data = data + 1 - _min
        try:
            if trans_type == TransformersType.LOG:
                data = np.log(data)
            elif trans_type == TransformersType.SQRT:
                data = np.sqrt(data)
            elif trans_type == TransformersType.BOX_COX:
                data = stats.boxcox(data)[0]
        except Exception as e:
            print('transform can not be applied')

        return data

    def categorical_features(self, data=None):
        if data is None:
            return self.df.select_dtypes(include=['object']).columns.tolist()
        else:
            return data.select_dtypes(include=['object']).columns.tolist()

    def numerical_features(self, data=None):
        if data is None:
            return self.df.select_dtypes(exclude=['object']).columns.tolist()
        else:
            return data.select_dtypes(exclude=['object']).columns.tolist()

    def select_columns(self, data, cols=None, just_numerical=True):
        if cols is None and just_numerical:
            cols = self.numerical_features(data=data)
        return data[cols]

    def split_to(self, test_size=0.10, val_size=0.10, has_validation=False, use_pca=False, random_state=seed):
        _X, _y = self.__samples_and_labels(use_pca=use_pca)
        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=test_size, random_state=random_state)
        if has_validation:
            _X_train, X_val, _y_train, y_val = train_test_split(_X_train, _y_train, test_size=val_size,
                                                                random_state=random_state)
            return _X_train, X_val, _X_test, _y_train, y_val, _y_test
        else:
            return _X_train, _X_test, _y_train, _y_test

    def generate_new_column_name(self, col, prefix):
        return '{}_{}'.format(col, prefix)

    def __samples_and_labels(self, use_pca=False):
        _X = None
        if use_pca:
            if self.pca is not None:
                _X = self.pca.copy()
            else:
                print('pca data frame is not provided')
        else:
            _X = self.df.copy().drop(labels=[self.target_col], axis=1)

        _y = self.df[self.target_col].copy()

        return _X, _y

    def __create_csv_dataframe(self):
        return pd.read_csv(self.dataset_address, delimiter=';')

    def __open_txt_file(self, desc):
        return open(desc, 'r').read()

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: DataFrame):
        self._df = df

    @property
    def pca(self):
        return self._pca

    @pca.setter
    def pca(self, value):
        self._pca = value

    @property
    def df_main(self):
        return self._df_main

    @df_main.setter
    def df_main(self, value):
        self._df_main = value

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
    def target_col(self):
        return self._target_col

    @target_col.setter
    def target_col(self, target):
        self._target_col = target

    @property
    def targets(self):
        return self.df_main[self.target_col]

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

    @property
    def encoded_data(self):
        return self._encoded_data

    @encoded_data.setter
    def encoded_data(self, value):
        self._encoded_data = value

    @property
    def scaled_data(self):
        return self._scaled_data

    @scaled_data.setter
    def scaled_data(self, value):
        self._scaled_data = value
