import warnings

from configs import cfg
from data import load
from data.preprocessing import Encoders, Scalers
from eda.bank_analyser import BankAnalyzer
from eda.bank_plot import BankPlots

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    bank = load(cfg)
    bank.load_dataset()

    encoder = Encoders(cdg=cfg)
    scaler = Scalers(cfg=cfg)

    analyzer = BankAnalyzer(dataset=bank, cfg=cfg)
    plots = BankPlots(dataset=bank, cfg=cfg)

    # analyzer.description()

    # plots.kernel_density_estimation(x='balance', y='age')
    plots.duration()
    bank.duration()
    analyzer.duration()

    # analyzer.description()
    # _data = encoder.do_encode(data=bank.df, y=bank.targets.values)
    # bank.encoded_data = _data
    # _data = scaler.do_scale(data=_data)
    # bank.scaled_data = _data
    # plots.corr(data=bank.scaled_data)


if __name__ == '__main__':
    main()
