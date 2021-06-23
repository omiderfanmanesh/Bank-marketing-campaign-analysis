import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;
from sklearn.decomposition import PCA as skl_pca

sns.set()


class PCA:
    def __init__(self, cfg):
        self.n_components = cfg.PCA.N_COMPONENTS
        self.pca = skl_pca(n_components=self.n_components, random_state=cfg.BASIC.RAND_STATE)

    def do_pca(self, data):
        _components = self.pca.fit_transform(data)
        print('Explained variance: %.4f' % self.pca.explained_variance_ratio_.sum())
        print('Individual variance contributions:')
        for j in range(self.pca.n_components_):
            print(self.pca.explained_variance_ratio_[j])

        _columns = ['pc' + str(i + 1) for i in range(self.pca.n_components_)]
        _pca_df = pd.DataFrame(data=_components
                               , columns=_columns)

        return _pca_df

    def plot_pca(self, X, y):
        X['y'] = y
        sns.pairplot(X, hue="y", height=2.5)
        plt.show()
