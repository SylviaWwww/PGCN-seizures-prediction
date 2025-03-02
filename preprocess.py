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

def preprocess_data(data, labels, test_size=0.2, val_size=0.2):
    """数据预处理：归一化、划分数据集"""
    # 归一化（按通道归一化）
    n_samples, n_channels, n_times = data.shape
    data = data.reshape(n_samples, -1)  # 展平为 (n_samples, n_channels * n_times)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # 标准化
    data = data.reshape(n_samples, n_channels, n_times)  # 恢复原始形状

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=42)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# 示例：数据预处理
(x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(data, labels)
print("训练集形状:", x_train.shape)  # 输出: (n_train_samples, n_channels, n_times)
print("验证集形状:", x_val.shape)    # 输出: (n_val_samples, n_channels, n_times)
print("测试集形状:", x_test.shape)   # 输出: (n_test_samples, n_channels, n_times)