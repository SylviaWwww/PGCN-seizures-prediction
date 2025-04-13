import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch


def load_hdf5(file_path):
    """从 HDF5 文件中加载数据和标签"""
    with h5py.File(file_path, 'r') as hf:
        data = hf['data'][:]
        labels = hf['labels'][:]
    return data, labels


def preprocess_data(data, labels, fs=256, test_size=0.2, val_size=0.1):
    """数据预处理：特征提取 → 归一化 → 划分数据集"""
    # 提取时频特征，结果形状为 [n_samples, n_channels, n_features=5]
    features = extract_features(data, fs=fs)

    # 归一化（按样本和通道独立归一化）
    n_samples, n_channels, n_features = features.shape
    features = features.reshape(n_samples, -1)  # 展平为 [n_samples, n_channels * n_features]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # 标准化
    features = features.reshape(n_samples, n_channels, n_features)  # 恢复原形状

    # 由于窗口缩小后样本数量足够，不再进行过采样平衡
    # 若需要使用过采样，可在此处启用相关代码
    # features, labels = balance_data(features, labels, method="oversample")

    # 划分数据集 (训练集、验证集、测试集)，其中 stratify 用于保持各类别比例
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=42
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def extract_features(data, fs=256):
    """提取时频特征（例如各频段的功率谱密度），输出形状为 [n_samples, n_channels, n_features]"""
    n_samples, n_channels, n_times = data.shape
    n_features = 5  # 设定提取5个频段的 PSD 特征
    features = np.zeros((n_samples, n_channels, n_features))

    for i in range(n_samples):
        for j in range(n_channels):
            freqs, psd = welch(data[i, j, :], fs=fs, nperseg=fs * 2)
            # 计算各频段PSD的均值
            features[i, j, :] = [
                np.mean(psd[(freqs >= fmin) & (freqs < fmax)])
                for fmin, fmax in [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 40)]
            ]
    return features


# 如果未来需要使用过采样方法，可保留或修改该函数
# def balance_data(features, labels, method="oversample"):
#     """过采样增加少数类，目前不使用该方法"""
#     n_samples, n_channels, n_features = features.shape
#     features_flat = features.reshape(n_samples, -1)  # 展平为 (n_samples, n_channels * n_features)
#
#     if method == "oversample":
#         from imblearn.over_sampling import RandomOverSampler
#         ros = RandomOverSampler(random_state=42)
#         features_resampled, labels_resampled = ros.fit_resample(features_flat, labels)
#
#     return features_resampled.reshape(-1, n_channels, n_features), labels_resampled

def load_and_preprocess(hdf5_path, fs=256):
    """加载 HDF5 文件并预处理数据，返回训练集、验证集和测试集"""
    # 加载原始时间序列，数据形状为 [n_samples, n_channels, n_times]
    raw_data, labels = load_hdf5(hdf5_path)

    # 进行特征提取、归一化和数据集划分，得到形状为 [n_samples, n_channels, 5] 的特征数据
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(raw_data, labels, fs=fs)

    print("训练集形状:", x_train.shape)
    print("验证集形状:", x_val.shape)
    print("测试集形状:", x_test.shape)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# 示例用法：指定 HDF5 文件路径，并加载预处理数据
if __name__ == "__main__":
    hdf5_file_path = "patient_dataset/dataset_chb15.h5"  # 修改为实际 HDF5 文件路径
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess(hdf5_file_path)
