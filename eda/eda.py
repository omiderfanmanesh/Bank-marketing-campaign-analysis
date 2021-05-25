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
    plots.dist_plot_by_col('age')
    plots.dist_plot('age',trans=TransformersEnum.BOX_PLOT)
    plots.box_plot_by_col('age',trans=TransformersEnum.SQRT)
    plots.box_plot('age')
    plots.violin_plot('age')
if __name__ == '__main__':
    main()
