from configs import cfg
from data.loader import load
from data.preprocessing import Encoders, Scalers, PCA
from engine.trainer import do_fine_tune
from model import DecisionTree
from model.based.tuning_mode import TuningMode


def main():
    bank = load(cfg)
    bank.load_dataset()
    bank.age()
    bank.duration()
    model = DecisionTree(cfg=cfg)

    encoder = Encoders(cdg=cfg)
    scaler = Scalers(cfg=cfg)
    pca = None
    if cfg.BASIC.PCA:
        pca = PCA(cfg=cfg)

    # do_train(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler, pca=pca)
    do_fine_tune(cfg=cfg, dataset=bank, model=model, encoder=encoder, scaler=scaler,
                 method=TuningMode.BAYES_SEARCH)


if __name__ == '__main__':
    main()
