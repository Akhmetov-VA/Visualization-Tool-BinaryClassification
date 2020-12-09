from src.visualisation.visual_core import DimRedTool
from sklearn.datasets import make_blobs


def main():
    X, y = make_blobs(n_samples=[1000, 20], centers=None, n_features=6, random_state=1, cluster_std=3)

    params = {
        'X': X, 'y': y, 'clf_separator_name': 'log_reg', 'data_dir_path': 'data',
        'balance_method_name': 'undersample_centroid', 'sampling_strategy': 0.5, 'num_points_per_dim': 4,
        'save_rotate_gif': True, 'save_2d_img': True, 'elev': 10, 'seed': 47, 'n_neighbors': 10,
        'dim_red_method_name': 'all'
    }

    dim_red_tool = DimRedTool(**params)
    dim_red_tool.pca_var_info()
    dim_red_tool.visualize_dim_red()


if __name__ == '__main__':
    main()
