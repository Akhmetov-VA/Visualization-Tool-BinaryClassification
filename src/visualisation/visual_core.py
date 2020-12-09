from abc import ABCMeta, abstractmethod
from time import time
from collections import OrderedDict, Counter

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from imblearn.under_sampling import ClusterCentroids

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis


class DimRedConfig:
    '''
    Contain configs that do not change in DimRedTool for easy control and change
    '''
    #  all dimension reduction methods that can be used in dependence of params
    dict_methods = OrderedDict({
        'Random Projection': SparseRandomProjection(), 'PCA': PCA(),
        'Isomap': Isomap(), 'MDS': MDS(n_init=1, max_iter=100),
        'LLE': LocallyLinearEmbedding(method='standard'), 'MLLE': LocallyLinearEmbedding(method='modified'),
        'HLLE': LocallyLinearEmbedding(method='hessian'), 'LTSA': LocallyLinearEmbedding(method='ltsa'),
        'Random Trees': (RandomTreesEmbedding(n_estimators=200, max_depth=5),
                         TruncatedSVD()),
        'Spectral': SpectralEmbedding(eigen_solver='arpack'),
        'TSNE': TSNE(init='pca'), 'NCA': NeighborhoodComponentsAnalysis(init='random'),
    })
    all_methods = dict_methods.keys()


class DataStorer(metaclass=ABCMeta):
    """
    Class contain all data that will be used in the workflow
    """

    @abstractmethod
    def __init__(self, X, y, data_dir_path=''):
        '''
        Get us capability to use data from any other inherited classes
        :param X:  np.array or pd.DataFrame
            Matrix with features
        :param y:  np.array or pd.Series
            Vector with targets
        :param data_dir_path: str
            Path destination for saving data/imgs/figs and so on
        '''
        self.X = X
        self.y = y
        self.data_dir_path = data_dir_path

        #  some attributes for plotting and flagging
        self.X_2d = None
        self.X_3d = None
        self.X_surf = None
        self.y_surf = None


class ClassRebalancer(DataStorer):
    '''
    Class that rebalances the classes according to the sampling_strategy and balance_method during initialization
    This will help us reduce volume of data to compute without losing information about decision boundary
    '''

    def __init__(self, balance_method_name='undersample_centroid', sampling_strategy=0.5, *args, **kwargs):
        '''
        Init balance_method_name for class that easy to check which used, rebalance classes
         and init DataStorer with data contained in *args and **kwargs

        :param balance_method_name:  str
            define the name of rebalance method (imblearn)
        :param sampling_strategy:  float or str ('auto')
            define of desire balance classes ratio after rebalancing
        :param args: tuple
            here should be X, y for init DataStorer Class
        :param kwargs:  dict
            here should be X, y for init DataStorer Class
        '''
        self.balance_method_name = balance_method_name

        super().__init__(*args, **kwargs)  # here init DataStorer for further potentially using it in rebalance_classes

        if self.balance_method_name == 'undersample_centroid':
            self.balance_method = ClusterCentroids(sampling_strategy=sampling_strategy)
            self.rebalance_classes()
        else:
            print(f'balance_method_name: {self.balance_method_name} doesnt fit. Сlasses were not rebalanced')

    def rebalance_classes(self, ):
        '''
        Just rebalances the data and displays information about changes in class balance
        '''
        print(f'Changing balances from {Counter(self.y).items()}')
        self.X, self.y = self.balance_method.fit_sample(self.X, self.y)
        print(f'to {Counter(self.y).items()}')


class DecisionSurfaceGrider(ClassRebalancer):
    '''
    Class that creates a grid of points that represent high dim feature space during initialization
    for further separate through classification of each point in grid
    This will help us understand how dim reduction change our base feature space into 3d or 2d
    '''

    def __init__(self, clf_separator_name='log_reg', num_points_per_dim=4, *args, **kwargs):
        '''
        Init balance_method_name for class that easy to check which used, create grid with defined num_points_per_dim
         and init ClassRebalancer(DataStorer) with data contained in *args and **kwargs

        :param clf_separator_name: str
            define the name of  method (imblearn)
        :param num_points_per_dim: int
            define num points per dimension in grid
        :param args:
            here should be balance_method_name, sampling_strategy for ClassRebalancer and X, y for init DataStorer Class
        :param kwargs:
            here should be balance_method_name, sampling_strategy for ClassRebalancer and X, y for init DataStorer Class
        '''
        self.clf_separator = None
        self.clf_separator_name = clf_separator_name
        self.num_points_per_dim = num_points_per_dim

        super().__init__(*args, **kwargs)

        if self.clf_separator_name == 'log_reg':
            self.clf_separator = LogisticRegression()

            #  create meshgrid with num_points_per_dim elements for each feature in X
            #  where each point in range between min and max value of this feature
            meshgrid = np.meshgrid(
                *[np.linspace(self.X[:, i].min() - 1, self.X[:, 1].max() + 1, self.num_points_per_dim) for i in
                  range(self.X.shape[-1])]
            )

            #  reformat created meshgrid in X format for further fit_transform to same dim and plot it
            flatten_meshgrid = [x.flatten().reshape((-1, 1)) for x in meshgrid]
            self.grid = np.hstack(flatten_meshgrid)
            print(f'Grid of points representing a multidimensional space is created and has dimension {self.grid.shape}')
        else:
            print(f'clf_separator_name: {self.clf_separator_name} does not fit. Grid of points representing a multidimensional space is not created')


class Visualizer(DecisionSurfaceGrider):
    '''
    Class contain params and funcs for visualization of 2d and 3d graphs and saving them as .jpg and .gif
    '''
    def __init__(self, save_rotate_gif=True, save_2d_img=True, elev=10, *args, **kwargs):
        '''
        Init flags and params for further calls

        :param save_rotate_gif: bool
            flag to save rotate 3d graph as gif
        :param save_2d_img: bool
            flag to save 2d graph as jpg
        :param elev: int
            elevator angle for 3d graph
        :param args:
            here should be clf_separator_name, num_points_per_dim for DecisionSurfaceGrider and another data
        :param kwargs:
            here should be clf_separator_name, num_points_per_dim for DecisionSurfaceGrider and another data
        '''
        self.save_rotate_gif = save_rotate_gif
        self.save_2d_img = save_2d_img
        self.elev = elev  # angle elevator

        super().__init__(*args, **kwargs)

    def scale_2d_data(self,):
        '''
        Normalize data
        '''
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

    def plot_line(self, data, x_label=None, y_label=None, title=None):
        '''
        Just plot line in accordance with params, for further analytics
        In this case used for plotting PCA variance ratio

        :param data: np.array 1d or 2d
            consecutive points to visualize dynamic of changes
        :param x_label: str
            label for x-coord
        :param y_label: str
            label for y-coord
        :param title: str
            title for all graph, used for naming .jpg file
        '''
        plt.figure()
        plt.plot(data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if title is not None:
            plt.title(title)
        else:
            title = 'some line'
        plt.show()

        if self.save_2d_img:
            plt.savefig(os.path.join(self.data_dir_path, title + '.jpg'))

    def save_rotate_plot(self, title=None):
        '''
        func save 3d graph as rotate .gif
        matplotlib.animation.FuncAnimation iteratively generate views on the 3d graph from the different viewpoints
        defined by the azimuth and elev and collect each image of views in one .gif
        :param title: str
        '''
        fig = plt.figure()
        ax = Axes3D(fig)

        def init_fun_plot():
            '''
            define what will be shown on the graph
            Plot 3d graph after dimension reduction, if used space separator aka clf plot this points too
            :return: tuple
                first element of tuple should be fig with points
            '''
            if self.clf_separator is not None:
                ax.scatter3D(self.X_surf[:, 0], self.X_surf[:, 1], self.X_surf[:, 2], c=self.y_surf, alpha=0.05, cmap='YlGn')

            ax.scatter3D(self.X_3d[:, 0], self.X_3d[:, 1], self.X_3d[:, 2], c=self.y, alpha=0.5)
            return fig,

        def animate_azimuth(i):
            '''
            defines how the view of the graph will change from frame to frame
            :param i: int
                change azimuth of view to plot (iteratively increasing <=> rotating)
            :return: tuple
                first element of tuple should be fig with points and changed view coords
            '''
            # azimuth angle: 0 to 360 deg
            ax.view_init(elev=self.elev, azim=i * 4)
            return fig,

        plt.title(title)

        #  iteratively change coords of view and saving it to .gif
        #  in func animate_azimuth azimuth increase by factor 4 it will cover 360 grad in 90 iterations (frames)
        ani = FuncAnimation(fig, animate_azimuth, init_func=init_fun_plot, frames=90, interval=50, blit=True)
        fn = os.path.join(self.data_dir_path, '3d', f'rotate azimuth {title}.gif')
        ani.save(fn, writer='pillow', fps=1000 / 50)

    def plot_2d_embed(self, title=None):
        '''
        Plot scaled 3d embedding of feature space if it is possible and needed with separate boundary
        :param title: str, title of graph
        '''
        # self.scale_2d_data()

        plt.figure()
        if self.clf_separator is not None:
            plt.scatter(self.X_surf[:, 0], self.X_surf[:, 1], c=self.y_surf, alpha=0.05, cmap='YlGn')

        plt.scatter(self.X_2d[:, 0], self.X_2d[:, 1], c=self.y, alpha=0.5)

        if title:
            plt.title(title)

        plt.show()

        if self.save_2d_img:
            plt.savefig(os.path.join(self.data_dir_path, '2d',  title + '.jpg'))

    def plot_3d_embed(self, title=None):
        '''
        Plot and save scaled 3d embedding of feature space if it is possible and needed with separate boundary
        :param title: str, title of graph
        '''

        if self.save_rotate_gif:
            self.save_rotate_plot(title)

        else:
            fig = plt.figure()
            ax = Axes3D(fig)

            if self.clf_separator is not None:
                ax.scatter3D(self.X_surf[:, 0], self.X_surf[:, 1], self.X_surf[:, 2], c=self.y_surf, alpha=0.05, cmap='YlGn')

            ax.scatter3D(self.X_3d[:, 0], self.X_3d[:, 1], self.X_3d[:, 2], c=self.y, alpha=0.5)

            if title:
                plt.title(title)

            plt.show()


class DimRedTool(Visualizer, DimRedConfig):
    '''
    Class for visualization dimension reduction of features
    if dim < 10 it is possible to plot how dim. red. change base feature space and how linear separable is data
    '''

    def __init__(self, dim_red_method_name='all', seed=47, n_neighbors=10, *args, **kwargs):
        '''
        Init which of methods will be used for visualization

        :param dim_red_method_name: str or list
            'all' or list of methods names to use on data
        :param seed: int
            random seed
        :param n_neighbors: int
            param for many dim reduction methods
        :param args: contain params for parrent classes
        :param kwargs: contain params for parrent classes
        '''
        self.dim_red_method_name = dim_red_method_name
        self.n_neighbors = n_neighbors
        self.seed = seed

        super().__init__(*args, **kwargs)

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

    def pca_var_info(self, ):
        '''
        Let us understand how many redundant features in dataset
        '''
        pca = PCA().fit(self.X)
        self.plot_line(
            data=np.cumsum(pca.explained_variance_ratio_),
            x_label='number components',
            y_label='cumulative explained variance',
            title='Variance reduction'
        )

    def separate_feature_space(self,):
        '''
        Separate feature space
        First of all fit logreg on original data, it will divide feature space on 2 field
        Second predict model for grid created in DecisionSurfaceGrider it will provide marks on all points feature space
        Third use the same dimension reductor as for original data for grid to repeat transformation
        So we have difference between grid in base and embedded feature space
        that can help us to understand how feature space was changed with reductor
        and how this linear separabilty displayed in smaller dimensions after dim reduction
        '''
        self.clf_separator.fit(self.X, self.y)
        self.y_surf = self.clf_separator.predict(self.grid)
        self.X_surf = self.dim_reducer.transform(self.grid)

    def dim_reduction(self, method_name, n_components=2):
        '''
        Based on some predefined rules provide dim reduction according to method_name and n_components
        :param method_name: str
            one of keys in dict DimRedConfig().dict_methods
        :param n_components: int
            n_components in output embedding
        :return: tuple(np.array, str)
            tuple of embedded features and title for this dimension reduction method
        '''
        #  define some params needed to insert in our methods
        method_params = {'random_state': self.seed, 'n_components': n_components}

        if method_name != 'Random Trees':

            #  for each of ['LLE', 'MLLE', 'HLLE', 'LTSA', 'Isomap'] methods we can define 'n_neighbors'
            if method_name in ['LLE', 'MLLE', 'HLLE', 'LTSA', 'Isomap']:
                method_params.update({'n_neighbors': self.n_neighbors})

                # but for Isomap we can't define 'random_state'
                if method_name == 'Isomap':
                    del method_params['random_state']

            self.dim_reducer = self.dict_methods[method_name].set_params(**method_params)
            t0 = time()

            if method_name == 'NCA':
                #  NearestNeighbor Component Analysis work only with X and y
                X = self.dim_reducer.fit_transform(self.X, self.y)
            else:
                X = self.dim_reducer.fit_transform(self.X)

        else:
            #  in Random Trees embeddings all little bit harder
            #  Random Trees is hasher of features after it Truncated SVD
            hasher, self.dim_reducer = self.dict_methods[method_name].set_params(**{'random_state': self.seed})
            self.dim_reducer.set_params(**method_params)
            t0 = time()
            X_hashed = hasher.fit_transform(self.X)
            self.dim_reducer.set_params()
            X = self.dim_reducer.fit_transform(X_hashed)

        #  after fitting dim_reducer we can reduce dim for grid if it needed
        if self.clf_separator is not None:
            self.separate_feature_space()

        title = f'{method_name} {n_components}d in {round(time() - t0, 3)} s'
        return X, title

    def visualize_dim_red(self,):
        '''
        Visualize and save 2d embeddings and 3d embeddings according to params defined in init
        '''

        for method_name in self.dim_red_method_name:
            #  for all of ['MDS', 'Random Trees', 'Spectral', 'TSNE'] we can't use the same transformation for grid
            #  so we can't see how linear separable our data on this methods
            if self.clf_separator is not None and method_name in ['MDS', 'Random Trees', 'Spectral', 'TSNE']:
                continue

            self.X_2d, title = self.dim_reduction(method_name, n_components=2)

            self.plot_2d_embed(title)

            self.X_3d, title = self.dim_reduction(method_name, n_components=3)

            self.plot_3d_embed(title)

