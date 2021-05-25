from pprint import pprint
from time import time

import pandas as pd
# Metrics
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback


class BasedModel:
    def train_model(self, X_tr, y_tr, X_te, y_te):
        print('start training...')
        self.model.fit(X_tr, y_tr)
        print('evaluation...')
        y_p = self.model.predict(X_te)
        score = self.evaluate(y_te, y_p)
        print(f'score is {score}')
        return self.model, score

    def evaluate(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

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
