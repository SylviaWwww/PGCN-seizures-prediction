import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_PGCN import PGCN
from preprocess import load_hdf5, load_and_preprocess  # 该函数返回 (x_train, y_train), (x_val, y_val), (x_test, y_test)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn as nn
from connectivity import compute_dataset_connectivity


# 定义与训练时一致的参数
class Args:
    def __init__(self):
        self.n_class = 2           # 二分类：Interictal vs Preictal
        self.in_feature = 5        # 输入特征维度
        self.dropout = 0.5
        self.lr = 0.001
        self.module = "local meso global"

args = Args()

# 选择设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# 加载数据（使用 preprocess.py 中的 load_and_preprocess）
# load_and_preprocess 返回 (x_train, y_train), (x_val, y_val), (x_test, y_test)
file_path = "patient_dataset/dataset_chb18.h5"
(_, _), (_, _), (x_test, y_test) = load_and_preprocess(file_path)

# 构造邻接矩阵和通道坐标（建议保持与训练时一致，这里仅示例用随机矩阵）
raw_data, _ = load_hdf5(file_path)
adj = compute_dataset_connectivity(raw_data, method="mean")
adj = torch.FloatTensor(adj).to(device)
# 从合并文件中加载存储好的 coordinates
with h5py.File(file_path, "r") as hf:
    coordinates = hf["coordinates"][:]  # 形状应为 (n_channels, 2)
coordinate = torch.FloatTensor(coordinates).to(device)

# 初始化模型并加载训练好的参数
model = PGCN(args, adj, coordinate).to(device)
model.load_state_dict(torch.load("trainResults/chb18/pgcn_chb18.pth", map_location=device))
model.eval()


# 转换为 PyTorch 张量，并构造 DataLoader
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 在测试集上进行预测
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs, _, _ = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())


# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Interictal", "Preictal"])
conf_mat = confusion_matrix(all_labels, all_preds)

print("Test Accuracy: {:.4f}".format(accuracy))
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_mat)

# 保存评估报告到文件
with open("trainResults/chb18/evaluation_report.txt", "w") as f:
    f.write("Test Accuracy: {:.4f}\n".format(accuracy))
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_mat))
