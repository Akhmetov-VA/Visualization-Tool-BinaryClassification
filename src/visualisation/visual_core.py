from abc import ABCMeta, abstractmethod
from time import time
from collections import OrderedDict, Counter

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from imblearn.under_sampling import ClusterCentroids

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis


class DataStorer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, X, y, data_dir_path=''):
        self.X = X
        self.y = y
        self.data_dir_path = data_dir_path

        self.X_2d = None
        self.X_3d = None
        self.X_surf = None
        self.y_surf = None


class ClassRebalancer(DataStorer):

    def __init__(self, balance_method='undersample_centroid', sampling_strategy=0.5, *args, **kwargs):
        self.balance_method = balance_method

        super().__init__(*args, **kwargs)

        if self.balance_method == 'undersample_centroid':
            self.balance_method = ClusterCentroids(sampling_strategy=sampling_strategy)
            self.rebalance_classes()

    def rebalance_classes(self, ):
        print(f'Changing balances from {Counter(self.y).items()}')
        self.X, self.y = self.balance_method.fit_sample(self.X, self.y)
        print(f'to {Counter(self.y).items()}')


class DecisionSurfaceGrider(ClassRebalancer):

    def __init__(self, clf_separator=None, num_points_per_dim=4, *args, **kwargs):
        self.clf_separator = clf_separator
        self.num_points_per_dim = num_points_per_dim

        super().__init__(*args, **kwargs)

        if self.clf_separator is not None:
            meshgrid = np.meshgrid(
                *[np.linspace(self.X[:, i].min() - 1, self.X[:, 1].max() + 1, self.num_points_per_dim) for i in
                  range(self.X.shape[-1])]
            )
            flatten_meshgrid = [x.flatten().reshape((-1, 1)) for x in meshgrid]
            self.grid = np.hstack(flatten_meshgrid)


class Visualizer(DataStorer):

    def __init__(self, save_rotate_gif=True, save_2d_img=True, elev=10, *args, **kwargs):

        self.save_rotate_gif = save_rotate_gif
        self.save_2d_img = save_2d_img
        self.elev = elev  # angle elevator

        super().__init__(*args, **kwargs)

    def scale_2d_data(self, X):
        return (X - X.min()) / (X.max() - X.min())

    def plot_line(self, data, x_label=None, y_label=None, title=None):
        plt.figure()
        plt.plot(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if title is not None:
            plt.title(title)
        else:
            title = 'line'
        plt.show()

        if self.save_2d_img:
            plt.savefig(os.path.join(self.data_dir_path, title + '.jpg'))

    def save_rotate_plot(self, fig, ax, title=None, elev=10):

        def init_fun_plot():
            if self.clf_separator is not None:
                ax.scatter3D(self.X_surf[:, 0], self.X_surf[:, 1], self.X_surf[:, 2], c=self.y_surf, alpha=0.05)

            ax.scatter3D(self.X_3d[:, 0], self.X_3d[:, 1], self.X_3d[:, 2], c=self.y, alpha=0.5)
            return fig,

        def animate_azimuth(i):
            # azimuth angle: 0 to 360 deg
            ax.view_init(elev=elev, azim=i * 4)
            return fig,

        ani = FuncAnimation(fig, animate_azimuth, init_func=init_fun_plot, frames=90, interval=50, blit=True)
        fn = os.path.join(self.data_dir_path, '3d', f'rotate azimuth {title}.gif')
        ani.save(fn, writer='pillow', fps=1000 / 50)

    def plot_2d_embed(self, title=None):

        fig = plt.figure()
        if self.clf_separator is not None:
            plt.scatter(self.X_surf[:, 0], self.X_surf[:, 1], c=self.y_surf, alpha=0.05, cmap='YlGn')

        plt.scatter(self.X_2d[:, 0], self.X_2d[:, 1], c=self.y, alpha=0.5)

        if title:
            plt.title(title)

        plt.show()

        if self.save_2d_img:
            plt.savefig(os.path.join(self.data_dir_path, '2d',  title + '.jpg'))

    def plot_3d_embed(self, title=None):
        fig = plt.figure()
        ax = Axes3D(fig)

        if self.save_rotate_gif:
            self.save_rotate_plot(fig, ax, title, elev=self.elev)

        if self.clf_separator is not None:
            ax.scatter3D(self.X_surf[:, 0], self.X_surf[:, 1], self.X_surf[:, 2], c=self.y_surf, alpha=0.05, cmap='YlGn')

        ax.scatter3D(self.X_3d[:, 0], self.X_3d[:, 1], self.X_3d[:, 2], c=self.y, alpha=0.5)

        if title:
            plt.title(title)

        plt.show()


class DimRedTool(DecisionSurfaceGrider, ClassRebalancer, Visualizer):

    def __init__(self, dim_red_method_name='all', seed=47, n_neighbors=10, *args, **kwargs):
        self.dim_red_method_name = dim_red_method_name
        self.n_neighbors = n_neighbors
        self.seed = seed

        self.dict_methods = OrderedDict({
            'Random Projection': SparseRandomProjection(), 'PCA': PCA(),
            'Isomap': Isomap(), 'MDS': MDS(n_init=1, max_iter=100),
            'LLE': LocallyLinearEmbedding(method='standard'), 'MLLE': LocallyLinearEmbedding(method='modified'),
            'HLLE': LocallyLinearEmbedding(method='hessian'), 'LTSA': LocallyLinearEmbedding(method='ltsa'),
            'Random Trees': (RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=self.seed),
                             TruncatedSVD()),
            'Spectral': SpectralEmbedding(eigen_solver='arpack'),
            'TSNE': TSNE(init='pca'), 'NCA': NeighborhoodComponentsAnalysis(init='random'),
            })
        self.all_methods = self.dict_methods.keys()

        if self.dim_red_method_name == 'all' or set(self.dim_red_method_name) == self.all_methods:
            self.dim_red_method_name = self.all_methods
            print('All methods of dimensionality reduction was chosen')

        elif (set(self.dim_red_method_name) - self.all_methods) >= 0:  # if the cardinality of input set is higher
            self.dim_red_method_name = set(self.dim_red_method_name).intersection(self.all_methods)  # use intersection
            print(f'Only {", ".join(self.dim_red_method_name)} methods of dimensionality reduction was chosen')

        elif (self.all_methods - set(self.dim_red_method_name)) == len(self.all_methods):
            self.dim_red_method_name = ['PCA']
            print('All method names entered incorrectly! PCA was chosen as method of dimensionality reduction ')

        else:
            self.dim_red_method_name = ['PCA']
            print('Some thing going wrong! PCA was chosen as method of dimensionality reduction')

        # for method_name in self.dim_red_method_name:
        #     self.dim_red_method[method_name] = self.dict_methods[method_name]

        super().__init__(*args, **kwargs)

    def pca_var_info(self, ):
        pca = PCA().fit(self.X)
        self.plot_line(
            data=np.cumsum(pca.explained_variance_ratio_),
            x_label='number components',
            y_label='cumulative explained variance',
            title='Variance reduction'
        )

    def separate_feature_space(self):

        self.clf_separator.fit(self.X, self.y)
        self.y_surf = self.clf_separator.predict(self.grid)
        self.X_surf = self.dim_reducer.transform(self.grid)

    def dim_reduction(self, method_name, n_components=2):
        method_params = {'random_state': self.seed, 'n_components': n_components}

        if method_name != 'Random Trees':

            if method_name in ['LLE', 'MLLE', 'HLLE', 'LTSA', 'Isomap']:
                method_params.update({'n_neighbors': self.n_neighbors})
                if method_name == 'Isomap':
                    del method_params['random_state']

            self.dim_reducer = self.dict_methods[method_name].set_params(**method_params)
            t0 = time()

            if method_name == 'NCA':
                X = self.dim_reducer.fit_transform(self.X, self.y)
            else:
                X = self.dim_reducer.fit_transform(self.X)

        else:
            hasher, self.dim_reducer = self.dict_methods[method_name]
            self.dim_reducer.set_params(**method_params)
            t0 = time()
            X_hashed = hasher.fit_transform(self.X)
            self.dim_reducer.set_params()
            X = self.dim_reducer.fit_transform(X_hashed)

        if self.clf_separator is not None:
            self.separate_feature_space()

        title = f'{method_name} {n_components}d in {round(time() - t0, 3)} s'
        return X, title

    def visualize_dim_red(self,):

        for method_name in self.dim_red_method_name:
            if self.clf_separator is not None and method_name in ['MDS', 'Random Trees', 'Spectral', 'TSNE']:
                continue

            self.X_2d, title = self.dim_reduction(method_name, n_components=2)

            self.plot_2d_embed(title)

            self.X_3d, title = self.dim_reduction(method_name, n_components=3)

            self.plot_3d_embed(title)

