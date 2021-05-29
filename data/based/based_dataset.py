#  Copyright (c) 2021.
#

import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data.based.file_types import FileTypes
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

    def split_to(self, test_size=0.10, val_size=0.10, has_validation=False, random_state=seed):
        X, y = self.__samples_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if has_validation:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size,
                                                              random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test

    def generate_new_column_name(self, col, prefix):
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

    def __target_encoding(self):
        le = LabelEncoder()
        targets = le.fit_transform(self.df[self.target])
        return targets

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
