import warnings

from data.bank import Bank
from model.based import BasedModel

warnings.simplefilter(action='ignore', category=FutureWarning)


def do_train(cfg, model: BasedModel, dataset: Bank):
    X_train, X_test, y_train, y_test = dataset.split_to()
    print(X_train.columns)
    X_train, X_test = dataset.do_encode(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    X_train = dataset.select_columns(data=X_train)
    print(X_train.columns)
    model.train(X_train=X_train, y_train=y_train)
    model.evaluate(X_test=X_test, y_test=y_test)
