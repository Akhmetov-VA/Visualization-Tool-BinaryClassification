from src.visualisation.visual_core import DimRedTool
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():

    # generate dataset
    X, y = make_blobs(n_samples=1000, centers=3, n_features=4, random_state=1, cluster_std=3)
    xx_yy_zz = np.meshgrid(*[np.linspace(X[:, i].min() - 1, X[:, 1].max() + 1, 3) for i in range(X.shape[-1])])
    r1_r2_r3 = [x.flatten().reshape((-1, 1)) for x in xx_yy_zz]
    grid = np.hstack(r1_r2_r3)
    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    # make predictions for the grid
    yhat = model.predict(grid)
    # keep just the probabilities for class 0
    # yhat = yhat[:, 0] > 0.9
    # reshape the predictions back into a grid
    zz = yhat#.reshape(xx.shape)
    # use PCA for dim reduction
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    X_grid = pca.transform(grid)
    plt.scatter(X_grid[:, 0], X_grid[:, 1], c=zz, alpha=0.05,)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='YlGn')

    plt.show()

def main():
    X, y = make_blobs(n_samples=[1000, 20], centers=None, n_features=6, random_state=1, cluster_std=3)

    params = {
        'X': X, 'y': y, 'clf_separator_name': 'log_reg', 'data_dir_path': 'data',
        'balance_method_name': 'undersample_centroid', 'sampling_strategy': 0.5, 'num_points_per_dim':4,
        'save_rotate_gif': True, 'save_2d_img': True, 'elev': 10, 'seed': 47, 'n_neighbors': 10,
        'dim_red_method_name': 'all'
    }

    dim_red_tool = DimRedTool(**params)
    dim_red_tool.pca_var_info()
    dim_red_tool.visualize_dim_red()

if __name__ == '__main__':
    main()
