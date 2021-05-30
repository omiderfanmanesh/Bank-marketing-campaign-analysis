import warnings

from configs import cfg
from data import load
from data.preprocessing import Encoders, Scalers
from eda.bank_analyser import BankAnalyzer
from eda.bank_plot import BasedPlot

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    _bank = load(cfg)
    _bank.load_dataset()
    _analyzer = BankAnalyzer(dataset=_bank, cfg=cfg)
    _analyzer.description()

    encoder = Encoders(cdg=cfg)
    scaler = Scalers(cfg=cfg)

    _data = encoder.do_encode(data=_bank.df, y=_bank.targets.values)
    _bank.encoded_data = _data
    _data = scaler.do_scale(data=_data)
    _bank.scaled_data = _data

    _plots = BasedPlot(dataset=_bank, cfg=cfg)
    _plots.corr(data=_bank.encoded_data)


if __name__ == '__main__':
    main()
