from data import load
from configs import cfg
from bank_analyser import BankAnalyzer
from bank_plot import BasedPlot
from utils.transformers_enums import TransformersEnum

def main():
    bank = load(cfg)
    bank.load_dataset()
    analyzer = BankAnalyzer(dataset=bank, cfg=cfg)
    analyzer.description()

    plots = BasedPlot(dataset=bank, cfg=cfg)
    plots.numerical_features_violin_plot()


if __name__ == '__main__':
    main()
