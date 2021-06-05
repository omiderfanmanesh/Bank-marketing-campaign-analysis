import warnings

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from data.based.based_dataset import BasedDataset
from data.preprocessing import Encoders, Scalers, PCA
from model.based import BasedModel
from model.based.tuning_mode import TuningMode

warnings.simplefilter(action='ignore', category=FutureWarning)


def do_train(cfg, model: BasedModel, dataset: BasedDataset, encoder: Encoders, scaler: Scalers, pca: PCA):
    if pca is None:
        dataset.df[dataset.target_col] = encoder.custom_encoding(dataset.df, col=cfg.DATASET.TARGET,
                                                                 encode_type=cfg.ENCODER.Y)
        X_train, X_test, y_train, y_test = dataset.split_to()
        if encoder is not None:
            X_train, X_test = encoder.do_encode(X_train=X_train, X_test=X_test, y_train=y_train,
                                                y_test=y_test)
        X_train = dataset.select_columns(data=X_train)
        X_test = dataset.select_columns(data=X_test)
        if scaler is not None:
            X_train, X_test = scaler.do_scale(X_train=X_train, X_test=X_test)
    else:
        if encoder is not None:
            dataset[dataset.target_col] = encoder.custom_encoding(dataset.df, col=cfg.DATASET.TARGET,
                                                                  encode_type=cfg.ENCODER.Y)
            _data = encoder.do_encode(data=dataset.df, y=dataset.targets.values)
        else:
            _data = dataset.select_columns(data=dataset.df)

        if scaler is not None:
            _data = scaler.do_scale(data=_data)

        pca_df = pca.do_pca(data=_data)
        dataset.pca = pca_df
        X_train, X_test, y_train, y_test = dataset.split_to(use_pca=True)

    model.train(X_train=X_train, y_train=y_train)
    model.evaluate(X_test=X_test, y_test=y_test)


def do_cross_val(cfg, model: BasedModel, dataset: BasedDataset, encoder: Encoders, scaler: Scalers, pca: PCA):
    _X, _y = encoder.do_encode(data=dataset.df, y=dataset.targets.values)

    _data = dataset.select_columns(data=dataset.df)

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    scores = cross_val_score(model.model, _X.values, _y, scoring='accuracy', cv=cv, n_jobs=-1)
    for s in scores:
        print('{} is {}'.format(cfg.METRIC, s))


def do_fine_tune(cfg, model: BasedModel, dataset: BasedDataset, encoder: Encoders, scaler: Scalers,
                 method=TuningMode.GRID_SEARCH):
    _X_train, _X_val, _X_test, _y_train, _y_val, _y_test = dataset.split_to(has_validation=True)
    _X_train, X_test = encoder.do_encode(X_train=_X_train, X_test=_X_val, y_train=_y_train,
                                         y_test=_y_val)

    _X_train = dataset.select_columns(data=_X_train)
    _X_val = dataset.select_columns(data=_X_val)
    _X_train, _X_val = scaler.do_scale(X_train=_X_train, X_test=_X_val)

    model.hyper_parameter_tuning(X=_X_train, y=_y_train, title=model.name, method=method)
