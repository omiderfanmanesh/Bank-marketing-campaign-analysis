from pprint import pprint

import pandas as pd


class BasedAnalyzer:
    def __init__(self, dataset, cfg):
        self._dataset = dataset
        self._df = dataset.df

    def head(self):
        self.df.head()

    def description(self):
        if self.dataset.dataset_description_file is not None:
            print("--------------- about dataset  -----------------")
            print(self.dataset.about)
            print('\n')

        print("--------------- description.txt ----------------")
        pprint(self.info())
        print('\n')

        print("--------------- description.txt ----------------")
        pprint(self.describe_dataframe())
        print('\n')

        print("--------------- nan Values -----------------")
        print(self.missing_values().head(20))
        print('\n')

        print("--------------- duplicates -----------------")
        print('Total number of duplicates: ', self.duplicates())
        print('\n')

        print("------ Numerical/Categorical Features ------")
        print('Numerical Features: {}'.format(self.numerical_features()))
        print('number of Numerical Features: {}'.format(self.numerical_features().__len__()))
        print('Categorical Features: {}'.format(self.categorical_features()))
        print('number of Categorical Features: {}'.format(self.categorical_features().__len__()))
        print('\n')

        print("--------------- skew & kurt -----------------")
        print('calculate skewness and kurtosis of numerical features')
        print(self.skew_kurt())
        print(
            '\n* skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable '
            'about its mean. \nnegative skew commonly indicates that the tail is on the left side of the distribution, '
            'and positive skew indicates that the tail is on the right.\n ')
        print('* kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random '
              'variable. Like skewness,\n kurtosis describes the shape of a probability distribution and there are '
              'different ways of quantifying it for a theoretical distribution \nand corresponding ways of estimating '
              'it from a sample from a population.')
        print('\n')

        print("----------------- quantiles -----------------")
        print(self.quantiles())
        print('\n')

        print("----------------- is target balanced? -----------------")
        print(self.count_by(self.target))
        print('\n')

    def count_by(self, col):
        return self.df[col].value_counts()

    def categorical_features(self):
        return self.df.select_dtypes(include=['object']).columns.tolist()

    def numerical_features(self):
        return self.df.select_dtypes(exclude=['object']).columns.tolist()

    def missing_values(self):
        total = self.df.isnull().sum().sort_values(ascending=False)
        percent = (self.df.isnull().sum() / self.df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

    def duplicates(self):
        dup = self.df.duplicated().sum()
        return dup

    def describe_dataframe(self):
        return self.df.describe().T

    def unique_values(self, col):
        return self.df[col].unique()

    def info(self):
        return self.df.info()

    def skew_kurt(self):
        kurt = self.df.kurt()
        skew = self.df.skew()

        return pd.concat([skew, kurt], axis=1, keys=['skew', 'kurt']).sort_values(by=['skew'], ascending=False)

    def quantiles(self):
        return self.df.quantile([.1, .25, .5, .75], axis=0).T

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
