import warnings

from data.bank import Bank
from data.preprocessing import Encoders, Scalers, PCA
from model.based import BasedModel

warnings.simplefilter(action='ignore', category=FutureWarning)


def do_train(cfg, model: BasedModel, dataset: Bank, encoder: Encoders, scaler: Scalers, pca: PCA):
    if pca is None:
        X_train, X_test, y_train, y_test = dataset.split_to()
        X_train, X_test = encoder.do_encode(X_train=X_train, X_test=X_test, y_train=y_train,
                                            y_test=y_test)
        X_train = dataset.select_columns(data=X_train)
        X_test = dataset.select_columns(data=X_test)
        X_train, X_test = scaler.do_scale(X_train=X_train, X_test=X_test)
    else:
        _data = encoder.do_encode(data=dataset.df, y=dataset.targets.values)
        _data = scaler.do_scale(data=_data)
        pca_df = pca.do_pca(data=_data)
        dataset.pca = pca_df
        X_train, X_test, y_train, y_test = dataset.split_to(use_pca=True)

    model.train(X_train=X_train, y_train=y_train)
    model.evaluate(X_test=X_test, y_test=y_test)
