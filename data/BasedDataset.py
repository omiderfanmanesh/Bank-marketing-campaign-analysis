import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random

seed = 2021
np.random.seed(seed)
random.seed(seed)


class BasedDataset:

    def __init__(self, is_csv=True, address=None, target=None):
        self.is_csv = is_csv
        self.address = address
        self.target = target

    def load(self):
        if self.is_csv:
            self.df = self.create_csv_dataframe()
        else:
            raise ValueError('dataset should be CSV file')

        return self.df

    def head(self):
        self.df.head()

    def description(self):
        print("--------------- description ---------------")
        print(self.df.describe().T)
        print("-------------------------------------------")
        print("--------------- nan Values ----------------")
        print(self.is_nan())
        print("-------------------------------------------")
        print("--------------- duplicates ----------------")
        print(self.duplicates())
        print("-------------------------------------------")

    def is_nan(self):
        num_nan = self.df.isna().sum()
        return num_nan

    def duplicates(self):
        dup = self.df.duplicated().sum()
        return dup

    def samples_and_labels(self):
        data = self.df.clone()
        y = data[self.target]
        X = data.drop(self.target)
        return X, y

    def split(self, has_validation=True, test_size=0.10, val_size=0.10):
        X, y = self.samples_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        if has_validation:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test

    def create_csv_dataframe(self):
        return pd.read_csv(self.address)

    @property
    def df(self):
        return self.df

    @df.setter
    def df(self, df):
        self._df = df

    @property
    def address(self):
        return self.address

    @address.setter
    def address(self, address):
        self.address = address

    @property
    def is_csv(self):
        return self.is_csv

    @is_csv.setter
    def is_csv(self, is_csv):
        self.is_csv = is_csv

    @property
    def target(self):
        return self.target

    @target.setter
    def target(self, target):
        self.target = target
