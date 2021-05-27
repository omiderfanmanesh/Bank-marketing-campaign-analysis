import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.FileType import FileType

seed = 2021
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


class BasedDataset:

    def __init__(self, dataset_type, dataset_address=None, target=None, dataset_description_file=None):
        self.dataset_type = dataset_type
        self.dataset_address = dataset_address
        self.target = target
        self.dataset_description_file = dataset_description_file

        if self.dataset_description_file is not None:
            self.about = self.open_txt_file(self.dataset_description_file)

        self.load_dataset()

    def load_dataset(self):
        if self._dataset_type == FileType.CSV:
            self.df = self.create_csv_dataframe()
        else:
            raise ValueError('dataset should be CSV file')

    def categorical_features(self):
        return self.df.select_dtypes(include=['object']).columns.tolist()

    def numerical_features(self):
        return self.df.select_dtypes(exclude=['object']).columns.tolist()

    def __encoding(self, col):
        le = LabelEncoder()
        targets = le.fit_transform(self.df[col])
        return targets

    def samples_and_labels(self, num_features=True):
        data = self.df.copy()

        y = data[self.target]
        X = data.drop(self.target, axis=1)

        if num_features:
            cols = self.numerical_features()
            X = X[cols]

        return X, y

    def create_csv_dataframe(self):
        return pd.read_csv(self.dataset_address, delimiter=';')

    def open_txt_file(self, desc):
        return open(desc, 'r').read()

    def scaler(self, scaler, data, test=None):
        scaler.fit(data)  # Apply transform to both the training set and the test set.
        train_scale = scaler.transform(data)
        if test is not None:
            test_scale = scaler.fit_transform(test)

        return train_scale, test_scale, scaler

    def split_to(self, test_size=0.10, val_size=0.10, has_validation=False, num_features=True, random_state=seed):
        X, y = self.samples_and_labels(num_features=num_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if has_validation:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size,
                                                              random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df

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
    def dataset_type(self, type):
        self._dataset_type = type

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
