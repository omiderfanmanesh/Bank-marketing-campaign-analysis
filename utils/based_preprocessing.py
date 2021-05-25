import numpy as np
from sklearn.model_selection import train_test_split

seed = 2021
np.random.seed(seed)


class BasedPreprocessing:
    def __init__(self, dataset, cfg):
        self._dataset = dataset
        self._df = dataset.df

    def scaler(self, scaler, data, test=None):
        scaler.fit(data)  # Apply transform to both the training set and the test set.
        train_scale = scaler.transform(data)
        if test is not None:
            test_scale = scaler.fit_transform(test)

        return train_scale, test_scale, scaler

    def split_dataset(self, X, y, test_size=0.1, val=True, val_size=0.1):
        if val:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=test_size,
                                                                random_state=seed, shuffle=True)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=val_size,
                                                              random_state=seed, shuffle=True)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=test_size,
                                                                random_state=seed, shuffle=True)
            return X_train, X_test, y_train, y_test

    @property
    def target(self):
        return self.dataset.target

    @property
    def df(self):
        return self.dataset.df

    @property
    def dataset(self):
        return self._dataset
