# train.py
import argparse
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from model_PGCN import PGCN
from preprocess import load_and_preprocess, load_hdf5
from connectivity import compute_dataset_connectivity
from utils import normalize_adj


def apply_ablation(model, ablation):
    """
    根据 ablation 参数修改模型中对应模块的 forward 函数，使其输出全零张量。
    这里假定输入 x 的形状为 [batch, n_channels, in_features]，
    并且各模块输出的形状与 model_PGCN.forward 中注释的形状一致。
    """
    if ablation == "no_local":
        # 用于 local_gcn_1：应输出 shape [batch, n_channels, 10]
        def zero_local1(x, lap_matrix, use_relu=True):
            batch, n_channels, _ = x.shape
            return torch.zeros(batch, n_channels, 10, device=x.device)

        # 用于 local_gcn_2：输出 [batch, n_channels, 15]
        def zero_local2(x, lap_matrix, use_relu=True):
            batch, n_channels, _ = x.shape
            return torch.zeros(batch, n_channels, 15, device=x.device)

        model.local_gcn_1.forward = zero_local1
        model.local_gcn_2.forward = zero_local2
        print("Ablation applied: no_local")
    elif ablation == "no_meso":
        # 对于 MesoLayer，直接返回输入不做转换（或全零输出，视情况而定）
        # 注意：meso_embed 负责线性变换，后续两层输出预期 shape 均为与 meso_embed 输出相同形状
        def zero_meso(x, coordinate):
            batch, n_channels, _ = x.shape
            # 假定 meso_layer 输出与 meso_embed 输出同形状，此处返回全零
            return torch.zeros(batch, n_channels, 30, device=x.device), None

        model.meso_layer_1.forward = zero_meso
        model.meso_layer_2.forward = zero_meso
        print("Ablation applied: no_meso")
    elif ablation == "no_global":
        # GlobalLayer 输出预期形状：[batch, n_channels*3, 40]
        def zero_global(x, lap_matrix):
            batch, channels, _ = x.shape
            # 这里假定 channels == n_channels*3
            return torch.zeros(batch, channels, 40, device=x.device)

        model.global_layer_1.forward = zero_global
        print("Ablation applied: no_global")
    else:
        print("Full model: no ablation applied")


def train(args):
    # 加载数据
    file_path = "single_patient_dataset/full_dataset_chb15.h5"  # 根据需要修改
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess(file_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 构造邻接矩阵与坐标（这里简化为随机示例，实际请替换为真实数据）
    raw_data, _ = load_hdf5(file_path)
    adj = compute_dataset_connectivity(raw_data, method="mean")  # 替换为实际功能连接矩阵
    adj = torch.FloatTensor(adj).to(device)

    # 从合并文件中加载存储好的 coordinates
    with h5py.File(file_path, "r") as hf:
        coordinates = hf["coordinates"][:]  # 形状应为 (n_channels, 2)
    coordinate = torch.FloatTensor(coordinates).to(device)

    # 初始化模型
    model = PGCN(args, adj, coordinate).to(device)

    # 根据消融策略修改模型（若为 full 则不修改）
    apply_ablation(model, args.ablation)

    # 定义优化器与损失函数
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 验证
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                outputs, _, _ = model(val_x)
                loss = criterion(outputs, val_y)
                val_loss += loss.item() * val_x.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == val_y).sum().item()
                val_samples += val_x.size(0)
        val_loss /= val_samples
        val_acc = val_correct / val_samples

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_name = f"trainResults/pgcn_model_chb15_{args.ablation}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"Saved best model to {checkpoint_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PGCN with Ablation Experiment")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--n_class", type=int, default=2, help="类别数")
    parser.add_argument("--in_feature", type=int, default=5, help="输入特征维度")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout 概率")
    # 消融策略：full, no_local, no_meso, no_global
    parser.add_argument("--ablation", type=str, default="no_global", help="消融策略")
    args = parser.parse_args()

    train(args)
