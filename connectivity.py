import numpy as np


def compute_corr_matrix(eeg_data):
    """
    使用皮尔逊相关系数计算单个 EEG 样本的功能连接矩阵。

    参数:
        eeg_data: numpy 数组, 形状为 (n_channels, n_times)

    返回:
        connectivity_matrix: numpy 数组, 形状为 (n_channels, n_channels)
    """
    # np.corrcoef 直接计算相关矩阵
    connectivity_matrix = np.corrcoef(eeg_data)
    return connectivity_matrix


def aggregate_connectivity_matrices(conn_matrices, method="mean"):
    """
    聚合多个样本的功能连接矩阵，生成一个代表性的连接矩阵。

    参数:
        conn_matrices: numpy 数组, 形状为 (n_samples, n_channels, n_channels)
        method: 聚合方法，"mean" 或 "median"

    返回:
        aggregated_matrix: numpy 数组, 形状为 (n_channels, n_channels)
    """
    if method == "mean":
        return np.mean(conn_matrices, axis=0)
    elif method == "median":
        return np.median(conn_matrices, axis=0)
    else:
        raise ValueError("Unknown aggregation method: choose 'mean' or 'median'.")


def compute_dataset_connectivity(eeg_dataset, method="mean"):
    """
    计算整个数据集的功能连接矩阵。对于每个 EEG 样本（形状: [n_channels, n_times]），
    计算其功能连接矩阵，然后根据给定的方法聚合得到一个整体矩阵。

    参数:
        eeg_dataset: numpy 数组, 形状为 (n_samples, n_channels, n_times)
        method: 聚合方法，"mean" 或 "median"

    返回:
        aggregated_connectivity: numpy 数组, 形状为 (n_channels, n_channels)
    """
    n_samples = eeg_dataset.shape[0]
    conn_matrices = []
    for i in range(n_samples):
        sample = eeg_dataset[i]  # (n_channels, n_times)
        conn_matrix = compute_corr_matrix(sample)
        conn_matrices.append(conn_matrix)
    conn_matrices = np.array(conn_matrices)
    aggregated_connectivity = aggregate_connectivity_matrices(conn_matrices, method=method)
    return aggregated_connectivity
