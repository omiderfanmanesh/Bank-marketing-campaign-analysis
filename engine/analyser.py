from data.based.based_dataset import BasedDataset
from data.preprocessing import Encoders, Scalers
from eda.based.based_analyzer import BasedAnalyzer
from eda.based.based_plots import BasedPlot


def do_analysis(dataset: BasedDataset, plots: BasedPlot, analyzer: BasedAnalyzer, encoder: Encoders, scaler: Scalers):
    # analyzer.description()

    # plots.rel(x='balance', y='age')
    # plots.duration()
    # dataset.duration()
    # analyzer.duration()
    analyzer.y()
    plots.y()
    # analyzer.description()
    # _data = encoder.do_encode(data=dataset.df, y=dataset.targets.values)
    # dataset.encoded_data = _data
    # _data = scaler.do_scale(data=_data)
    # dataset.scaled_data = _data
    # plots.corr(data=dataset.scaled_data)