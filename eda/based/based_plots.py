#  Copyright (c) 2021.
#

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="white")


class BasedPlot:
    def __init__(self, dataset, cfg):
        self._dataset = dataset
        self._df = dataset.df

    def __scatter_plot(self, df, features, trans=None):
        g = sns.PairGrid(df, vars=features, hue=self.target)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        plt.show()

    def __plot_feature_distribution(self, df, col, is_numerical=True, trans=None):
        df[col] = self.dataset.transform(df[col], trans)
        if is_numerical:
            g = sns.FacetGrid(df, col=self.target, hue=self.target)
            g.map(sns.histplot, col)
        else:
            sns.countplot(data=df, x=col)
        plt.show()

    def numerical_features_distribution(self, trans=None):
        features = self.dataset.numerical_features()
        for f in features:
            self.__plot_feature_distribution(self.df.copy(), f, is_numerical=True, trans=trans)

    def categorical_features_distribution(self, trans=None):
        features = self.dataset.categorical_features()
        for f in features:
            self.__plot_feature_distribution(self.df.copy(), f, is_numerical=False, trans=trans)

    def numerical_features_scatter_plot(self, trans=None):
        features = self.dataset.numerical_features()
        self.__scatter_plot(self.df.copy(), features=features, trans=trans)

    def numerical_features_box_plot(self, trans=None):
        features = self.dataset.numerical_features()
        for f in features:
            self.box_plot_by_col(col=f, trans=trans)

    def numerical_features_violin_plot(self, trans=None):
        features = self.dataset.numerical_features()
        for f in features:
            self.violin_plot_by_col(col=f, trans=trans)

    def __dist_plot(self, params):
        sns.displot(**params)
        plt.show()

    def __dist_sub_plot(self, df):
        sub = ceil(len(self.df.columns) / 3)
        fig, axes = plt.subplots(nrows=sub, ncols=3)
        axes = axes.flatten()
        for ax, col in zip(axes, df.columns):
            sns.displot(df[col], ax=ax)
        plt.show()

    def dist_plot_numerical_columns(self):
        numerical = self.dataset.numerical_features()
        self.__dist_sub_plot(self.df[numerical])

    def dist_plot_by_col(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.dataset.transform(df[col], trans)

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
        df[col] = self.dataset.transform(df[col], trans)

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
        df[col] = self.dataset.transform(df[col], trans)

        params = {
            'data': df,
            'x': col,
            'orient': "h",
            'y': self.target
        }

        self.__box_plot(params)

    def box_plot(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.dataset.transform(df[col], trans)

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
        df[col] = self.dataset.transform(df[col], trans)

        params = {
            'data': df,
            'x': self.target,
            'y': col,
            'hue': self.target
        }

        self.__violin_plot(params)

    def violin_plot(self, col, trans=None):
        df = self.df.copy()
        df[col] = self.dataset.transform(df[col], trans)

        params = {
            'data': df,
            'y': col,
            'hue': self.target
        }

        self.__violin_plot(params)

    def corr(self, data, all_methods=True):
        if all_methods:
            for method in ["pearson", "spearman", "kendall"]:
                self.__corr_plot(data=data, method=method)
        else:
            self.__corr_plot(data=data)

    def __corr_plot(self, data, method='pearson'):
        _corr = data.corr(method=method)
        _mask = np.triu(np.ones_like(_corr, dtype=bool))
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        _cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        _heat_map = sns.heatmap(_corr, mask=_mask, cmap=_cmap, vmax=.3, center=0,
                                square=True, linewidths=.5, cbar_kws={"shrink": .5})
        _heat_map.set_title('Feature Correlation by {}'.format(method))
        plt.show()

    @property
    def target(self):
        return self.dataset.target_col

    @property
    def df(self):
        return self.dataset.df

    @property
    def dataset(self):
        return self._dataset
