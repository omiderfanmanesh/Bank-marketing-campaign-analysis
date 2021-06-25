#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from configs import cfg
from data.loader import load
from data.preprocessing import Encoders, Scalers, PCA
from engine.trainer import do_fine_tune, do_cross_val, do_train
from model import DecisionTree, LogisticRegression, SVM, RandomForest
from model.based import Model
from model.based.tuning_mode import TuningMode
from utils import RuntimeMode


def main():
    bank = load(cfg)
    bank.load_dataset()
    bank.age()
    bank.duration()
    bank.drop_cols()

    model_selection = cfg.BASIC.MODEL
    if model_selection == Model.SVM:
        model = SVM(cfg=cfg)
    elif model_selection == Model.DECISION_TREE:
        model = DecisionTree(cfg=cfg)
    elif model_selection == Model.RANDOM_FOREST:
        model = RandomForest(cfg=cfg)
    elif model_selection == Model.LOGISTIC_REGRESSION:
        model = LogisticRegression(cfg=cfg)

    encoder = Encoders(cdg=cfg)
    scaler = Scalers(cfg=cfg)
    pca = None
    if cfg.BASIC.PCA:
        pca = PCA(cfg=cfg)

    runtime_mode = cfg.BASIC.RUNTIME_MODE
    if runtime_mode == RuntimeMode.TRAIN:
        do_train(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca)
    elif runtime_mode == RuntimeMode.CROSS_VAL:
        do_cross_val(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca)
    elif runtime_mode == RuntimeMode.TUNING:
        do_fine_tune(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler,
                     method=TuningMode.GRID_SEARCH)
    if runtime_mode == RuntimeMode.FEATURE_IMPORTANCE:
        do_train(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca, feature_importance=True)

if __name__ == '__main__':
    main()
