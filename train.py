import argparse
import torch
import numpy as np
from model_PGCN import PGCN
from preprocess import load_hdf5, preprocess_data, load_and_preprocess
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn

# 参数定义
class Args:
    def __init__(self):
        self.n_class = 2  # 二分类：发作前期（Preictal） vs 发作间期（Interictal）
        self.in_feature = 5  # 输入特征维度（例如时频特征的维度）
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

# 邻接矩阵与坐标初始化
n_channels = 23  # 假设有23个通道
adj = np.random.rand(n_channels, n_channels)  # 替换为实际功能连接矩阵
adj = torch.FloatTensor(adj).to(device)

coordinate = np.random.rand(n_channels, 2)  # 假设使用二维坐标
coordinate = torch.FloatTensor(coordinate).to(device)

# 初始化模型
model = PGCN(args, adj, coordinate).to(device)
print(model)

# 加载数据
file_path = "full_dataset_chb01.h5"
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess(file_path)

# 转换为PyTorch张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义优化器和损失函数
optimizer = Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs, _, _ = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "pgcn_model.pth")