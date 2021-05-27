from configs import cfg
from data.loader import load
from engine.trainer import do_train
from model import SVM


def main():
    bank = load(cfg)
    bank.load_dataset()
    model = SVM(cfg=cfg)
    do_train(cfg=cfg, dataset=bank, model=model)


if __name__ == '__main__':
    main()
