import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from utils.transformers_enums import TransformersEnum


class BasedPlot:
    def __init__(self, dataset, cfg):
        self._dataset = dataset
        self._df = dataset.df

    def __transform(self, data, trans_type):
        try:
            if trans_type == TransformersEnum.LOG:
                data = np.log(data)
            elif trans_type == TransformersEnum.SQRT:
                data = np.sqrt(data)
            elif trans_type == TransformersEnum.BOX_PLOT:
                data = stats.boxcox(data)[0]
        except:
            print('transform can not be applied')

        return data

    def __dist_plot(self, params):
        sns.displot(**params)
        plt.show()

    def dist_plot_by_col(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.__transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'color': "green",
            'kde': True,
            'bins': 120,
            'hue': self.target
        }

        self.__dist_plot(params)

    def dist_plot(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.__transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'color': "green",
            'kde': True,
            'bins': 120
        }

        self.__dist_plot(params)

    def __box_plot(self, params):
        sns.boxplot(**params)
        sns.despine(offset=10, trim=True)
        plt.show()

    def box_plot_by_col(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.__transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'orient': "h",
            'y': self.target
        }

        self.__box_plot(params)

    def box_plot(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.__transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'orient': "h"
        }

        self.__box_plot(params)

    def __violin_plot(self, params):
        sns.violinplot(**params)
        plt.show()

    def violin_plot_by_col(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.__transform(df[col], trans)

        params = {
            'data': df,
            'x': self.target,
            'y': col,
            'hue': self.target
        }

        self.__violin_plot(params)

    def violin_plot(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.__transform(df[col], trans)

        params = {
            'data': df,
            'y': col,
            'hue': self.target
        }

        self.__violin_plot(params)

    @property
    def target(self):
        return self.dataset.target

    @property
    def df(self):
        return self.dataset.df

    @property
    def dataset(self):
        return self._dataset
