from configs import cfg
from data import load
from eda.bank_analyser import BankAnalyzer
from eda.bank_plot import BasedPlot


def main():
    bank = load(cfg)
    bank.load_dataset()
    analyzer = BankAnalyzer(dataset=bank, cfg=cfg)
    analyzer.description()

    plots = BasedPlot(dataset=bank, cfg=cfg)
    plots.numerical_features_scatter_plot()


if __name__ == '__main__':
    main()
