#  Copyright (c) 2021.
#

from pprint import pprint
from time import time

import pandas as pd
# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback

from .metric_types import MetricTypes


class BasedModel:
    def __init__(self, cfg):
        self._model = None
        self._metric_function = cfg.METRIC


    def train(self, X_train, y_train):
        print('start training...')
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        print('evaluation...')
        y_p = self.model.predict(X_test)
        score = self.metric(y_test, y_p)
        print(f'score is {score}')
        return

    def metric(self, y_true, y_pred):
        metric_type = self._metric_function
        if metric_type == MetricTypes.F1_SCORE_BINARY:
            return f1_score(y_true, y_pred, average="binary")
        elif metric_type == MetricTypes.F1_SCORE_MACRO:
            return f1_score(y_true, y_pred, average="micro")
        elif metric_type == MetricTypes.F1_SCORE_MACRO:
            return f1_score(y_true, y_pred, average="macro")
        elif metric_type == MetricTypes.F1_SCORE_WEIGHTED:
            return f1_score(y_true, y_pred, average="weighted")
        elif metric_type == MetricTypes.F1_SCORE_SAMPLE:
            return f1_score(y_true, y_pred, average="sample")
        elif metric_type == MetricTypes.PRECISION:
            return precision_score(y_true, y_pred)
        elif metric_type == MetricTypes.RECALL:
            return recall_score(y_true, y_pred)
        elif metric_type == MetricTypes.ACCURACY:
            return accuracy_score(y_true, y_pred)

    def hyper_parameter_tuning(self, params, X, y, title):
        opt = BayesSearchCV(**params)
        best_params = self.report_best_params(opt, X, y, title,
                                              callbacks=[VerboseCallback(100),
                                                         DeadlineStopper(60 * 10)])
        return best_params

    def report_best_params(self, optimizer, X, y, title, callbacks=None):
        """
        A wrapper for measuring time and performances of different optmizers

        optimizer = a sklearn or a skopt optimizer
        X = the training set
        y = our target
        title = a string label for the experiment
        """
        start = time()
        if callbacks:
            optimizer.fit(X, y, callback=callbacks)
        else:
            optimizer.fit(X, y)
        d = pd.DataFrame(optimizer.cv_results_)
        best_score = optimizer.best_score_
        best_score_std = d.iloc[optimizer.best_index_].std_test_score
        best_params = optimizer.best_params_
        print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
               + u"\u00B1" + " %.3f") % (time() - start,
                                         len(optimizer.cv_results_['params']),
                                         best_score,
                                         best_score_std))
        print('Best parameters:')
        pprint(best_params)
        print()
        return best_params

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
