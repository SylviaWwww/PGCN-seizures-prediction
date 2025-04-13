import argparse
import torch
import h5py
import numpy as np
import pandas as pd
from model_PGCN import PGCN
from preprocess import load_hdf5, load_and_preprocess
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from connectivity import compute_dataset_connectivity

# 其他辅助库，用于评估和绘图
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

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

# 加载数据
file_path = "patient_dataset/dataset_chb18.h5"  # 请确保该文件路径与实际数据文件路径一致
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess(file_path)

# 邻接矩阵与坐标初始化
raw_data, _ = load_hdf5(file_path)
adj = compute_dataset_connectivity(raw_data, method="mean")  # 请替换为实际功能连接矩阵的计算函数
adj = torch.FloatTensor(adj).to(device)

# 从 HDF5 文件中加载存储好的 coordinates
with h5py.File(file_path, "r") as hf:
    coordinates = hf["coordinates"][:]  # 形状应为 (n_channels, 2)
coordinate = torch.FloatTensor(coordinates).to(device)

# 初始化模型
model = PGCN(args, adj, coordinate).to(device)
print(model)

# 转换为 PyTorch 张量，并构造 DataLoader
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义优化器和损失函数
optimizer = Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# 用来存储每个 epoch 的指标
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
best_val_acc = 0

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs, _, _ = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # 统计当前 batch
        total_loss += loss.item() * batch_x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    train_loss_epoch = total_loss / total
    train_acc_epoch = correct / total

    # 验证阶段
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_outputs, _, _ = model(val_x)
            v_loss = criterion(val_outputs, val_y)
            val_loss += v_loss.item() * val_x.size(0)

            v_preds = val_outputs.argmax(dim=1)
            val_correct += (v_preds == val_y).sum().item()
            val_total += val_x.size(0)

    val_loss_epoch = val_loss / val_total
    val_acc_epoch = val_correct / val_total

    # 存储本 epoch 数据
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss_epoch)
    history["train_acc"].append(train_acc_epoch)
    history["val_loss"].append(val_loss_epoch)
    history["val_acc"].append(val_acc_epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")

    # 保存最优模型
    if val_acc_epoch > best_val_acc:
        best_val_acc = val_acc_epoch
        checkpoint_name = "trainResults/chb18/pgcn_chb18.pth"
        torch.save(model.state_dict(), checkpoint_name)
        print(f"Saved best model to {checkpoint_name}")

# 导出训练日志到 Excel
df = pd.DataFrame(history)
excel_path = "trainResults/chb18/chb18results.xlsx"
df.to_excel(excel_path, index=False)
print(f"已将训练日志导出到 {excel_path}")

# ========= 验证集评估与结果可视化 ==========

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for val_x, val_y in val_loader:
        outputs, _, _ = model(val_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(val_y.cpu().numpy())

# 计算分类指标
report = classification_report(all_labels, all_preds, target_names=["Interictal", "Preictal"])
cm = confusion_matrix(all_labels, all_preds)
print("分类报告：")
print(report)
print("混淆矩阵：")
print(cm)

# 将分类报告保存为 Excel 文件
report_dict = classification_report(all_labels, all_preds, target_names=["Interictal", "Preictal"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_excel_path = "trainResults/chb18/chb18_classification_report.xlsx"
report_df.to_excel(report_excel_path)
print(f"已将分类报告保存到 {report_excel_path}")

# 绘制训练和验证 Loss 曲线
plt.figure()
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
loss_plot_path = "trainResults/chb18/chb18_loss_curve.png"
plt.savefig(loss_plot_path)
print(f"训练 Loss 曲线已保存到 {loss_plot_path}")
plt.close()

# 绘制训练和验证 Accuracy 曲线
plt.figure()
plt.plot(history["epoch"], history["train_acc"], label="Train Accuracy")
plt.plot(history["epoch"], history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
acc_plot_path = "trainResults/chb18/chb18_accuracy_curve.png"
plt.savefig(acc_plot_path)
print(f"训练 Accuracy 曲线已保存到 {acc_plot_path}")
plt.close()
