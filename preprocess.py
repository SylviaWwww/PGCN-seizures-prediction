import h5py
import numpy as np

def load_hdf5(file_path):
    """从 HDF5 文件中加载数据和标签"""
    with h5py.File(file_path, 'r') as hf:
        data = hf['data'][:]  # 形状: (n_samples, n_channels, n_times)
        labels = hf['labels'][:]  # 形状: (n_samples,)
    return data, labels

# 示例：加载 HDF5 文件
file_path = "full_dataset_chb07.h5"
data, labels = load_hdf5(file_path)
print("数据形状:", data.shape)  # 输出: (n_samples, n_channels, n_times)
print("标签形状:", labels.shape)  # 输出: (n_samples,)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# preprocess.py
def preprocess_data(data, labels, fs=256, test_size=0.2, val_size=0.2):
    """数据预处理：特征提取 → 归一化 → 划分数据集"""
    # 提取时频特征（形状: [n_samples, n_channels, n_features=5]）
    features = extract_features(data, fs=fs)

    # 归一化（按样本和通道独立归一化）
    n_samples, n_channels, n_features = features.shape
    features = features.reshape(n_samples, -1)  # 展平为 [n_samples, n_channels * n_features]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # 标准化
    features = features.reshape(n_samples, n_channels, n_features)  # 恢复形状

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

from scipy.signal import welch

def extract_features(data, fs=256):
    """提取时频特征（如功率谱密度）"""
    n_samples, n_channels, n_times = data.shape
    n_features = 5  # 假设提取5个频段的PSD
    features = np.zeros((n_samples, n_channels, n_features))

    for i in range(n_samples):
        for j in range(n_channels):
            freqs, psd = welch(data[i, j, :], fs=fs, nperseg=fs*2)
            features[i, j, :] = [np.mean(psd[(freqs >= fmin) & (freqs < fmax)]) for fmin, fmax in [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 40)]]
    return features


def load_and_preprocess(hdf5_path, fs=256):
    """加载 HDF5 文件并预处理数据"""
    # 加载原始时间序列（形状: [n_samples, n_channels, n_times]）
    raw_data, labels = load_hdf5(hdf5_path)

    # 提取特征并预处理（形状: [n_samples, n_channels, 5]）
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(raw_data, labels, fs=fs)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# 示例：数据预处理
(x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(data, labels)
print("训练集形状:", x_train.shape)  # 输出: (n_train_samples, n_channels, n_times)
print("验证集形状:", x_val.shape)    # 输出: (n_val_samples, n_channels, n_times)
print("测试集形状:", x_test.shape)   # 输出: (n_test_samples, n_channels, n_times)