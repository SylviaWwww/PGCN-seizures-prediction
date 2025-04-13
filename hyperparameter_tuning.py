import os
import time
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score

# 导入自定义模块（注意各文件需位于同一目录或 Python 路径中）
from model_PGCN import PGCN
from preprocess import load_hdf5, load_and_preprocess  # 返回 (x_train,y_train), (x_val,y_val), (x_test,y_test)
from connectivity import compute_dataset_connectivity
from utils import normalize_adj


# 定义参数类（扩充了 num_heads 字段用于调节注意力头数）
class Args:
    def __init__(self, module="local meso global", lr=0.001, dropout=0.5, num_heads=6):
        self.n_class = 2  # 二分类：Interictal vs Preictal
        self.in_feature = 5  # 输入特征维度（例如时频特征）
        self.dropout = dropout  # Dropout 率
        self.lr = lr  # 学习率
        self.module = module  # 模块配置：可设置为 "local", "local meso", "local meso global" 等
        self.num_heads = num_heads  # 注意力头数（用于 meso 层），模型中需要使用 args.num_heads


def select_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )


# 训练与验证的统一函数
def train_and_evaluate(args, adj, coordinate, train_loader, val_loader, num_epochs=20):
    """
    根据给定的超参数构造模型，训练 num_epochs 后返回验证集最高准确率及对应 epoch。
    """
    device = select_device()
    # 初始化模型；注意 model_PGCN.py 中 meso_layer_1, meso_layer_2 要使用 args.num_heads
    model = PGCN(args, adj, coordinate).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_epoch = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # 训练阶段
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += batch_x.size(0)

        train_acc = epoch_correct / epoch_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                outputs, _, _ = model(val_x)
                loss = criterion(outputs, val_y)
                val_loss += loss.item() * val_x.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == val_y).sum().item()
                val_total += val_x.size(0)
        val_acc = val_correct / val_total

        # 更新最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        # 可以打印日志
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return best_val_acc, best_epoch


def run_experiment():
    # 固定部分参数（数据路径、训练轮数、batch_size）
    file_path = "patient_dataset/dataset_chb15.h5"
    num_epochs = 20
    batch_size = 16

    device = select_device()

    # 加载数据：使用 preprocess.py 中的 load_and_preprocess 取训练集和验证集
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess(file_path)

    # 构造 DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.long).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.long).to(device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 构造邻接矩阵和通道坐标（确保与训练时一致）
    raw_data, _ = load_hdf5(file_path)
    adj_np = compute_dataset_connectivity(raw_data, method="mean")
    adj = torch.FloatTensor(adj_np).to(device)
    with h5py.File(file_path, "r") as hf:
        coordinates_np = hf["coordinates"][:]
    coordinate = torch.FloatTensor(coordinates_np).to(device)

    # 定义默认值和需要调节的超参数范围
    default_lr = 0.001
    default_dropout = 0.5
    default_num_heads = 6

    lr_list = [0.0001, 0.001, 0.01]
    dropout_list = [0.1, 0.3, 0.5, 0.7]
    num_heads_list = [2, 4, 6, 8]

    results = {"lr": [], "val_acc": []}
    print("----- Tuning Learning Rate (with dropout=%.2f, num_heads=%d) -----" % (default_dropout, default_num_heads))
    # 固定 dropout 和 num_heads，不变，只调学习率
    for lr in lr_list:
        args = Args(module="local meso global", lr=lr, dropout=default_dropout, num_heads=default_num_heads)
        print(f"Testing LR = {lr}")
        val_acc, best_epoch = train_and_evaluate(args, adj, coordinate, train_loader, val_loader, num_epochs=num_epochs)
        results["lr"].append({"lr": lr, "val_acc": val_acc, "best_epoch": best_epoch})

    # 绘制“学习率-性能”折线图
    plt.figure()
    lr_values = [item["lr"] for item in results["lr"]]
    val_accs = [item["val_acc"] for item in results["lr"]]
    plt.plot(lr_values, val_accs, marker="o")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Rate vs Validation Accuracy")
    plt.xscale("log")
    plt.grid(True)
    plt.savefig("trainResults/chb15/hyper_lr.png")
    plt.close()
    print("Learning rate tuning plot saved to trainResults/chb15/hyper_lr.png")

    # 同理，调节 Dropout 率
    results["dropout"] = []
    print("----- Tuning Dropout (with lr=%.4f, num_heads=%d) -----" % (default_lr, default_num_heads))
    for dropout in dropout_list:
        args = Args(module="local meso global", lr=default_lr, dropout=dropout, num_heads=default_num_heads)
        print(f"Testing Dropout = {dropout}")
        val_acc, best_epoch = train_and_evaluate(args, adj, coordinate, train_loader, val_loader, num_epochs=num_epochs)
        results["dropout"].append({"dropout": dropout, "val_acc": val_acc, "best_epoch": best_epoch})

    plt.figure()
    dropout_values = [item["dropout"] for item in results["dropout"]]
    val_accs = [item["val_acc"] for item in results["dropout"]]
    plt.plot(dropout_values, val_accs, marker="o")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Validation Accuracy")
    plt.title("Dropout Rate vs Validation Accuracy")
    plt.grid(True)
    plt.savefig("trainResults/chb15/hyper_dropout.png")
    plt.close()
    print("Dropout tuning plot saved to trainResults/chb15/hyper_dropout.png")

    # 调节注意力头数
    results["num_heads"] = []
    print("----- Tuning Attention Heads (with lr=%.4f, dropout=%.2f) -----" % (default_lr, default_dropout))
    for num_heads in num_heads_list:
        args = Args(module="local meso global", lr=default_lr, dropout=default_dropout, num_heads=num_heads)
        print(f"Testing num_heads = {num_heads}")
        val_acc, best_epoch = train_and_evaluate(args, adj, coordinate, train_loader, val_loader, num_epochs=num_epochs)
        results["num_heads"].append({"num_heads": num_heads, "val_acc": val_acc, "best_epoch": best_epoch})

    plt.figure()
    nh_values = [item["num_heads"] for item in results["num_heads"]]
    val_accs = [item["val_acc"] for item in results["num_heads"]]
    plt.plot(nh_values, val_accs, marker="o")
    plt.xlabel("Number of Attention Heads")
    plt.ylabel("Validation Accuracy")
    plt.title("Attention Heads vs Validation Accuracy")
    plt.grid(True)
    plt.savefig("trainResults/chb15/hyper_num_heads.png")
    plt.close()
    print("Attention heads tuning plot saved to trainResults/chb15/hyper_num_heads.png")

    # 保存所有调参结果到 CSV 文件
    df_lr = pd.DataFrame(results["lr"])
    df_dropout = pd.DataFrame(results["dropout"])
    df_num_heads = pd.DataFrame(results["num_heads"])
    os.makedirs("trainResults/chb15", exist_ok=True)
    df_lr.to_csv("trainResults/chb15/hyper_lr_results.csv", index=False)
    df_dropout.to_csv("trainResults/chb15/hyper_dropout_results.csv", index=False)
    df_num_heads.to_csv("trainResults/chb15/hyper_num_heads_results.csv", index=False)
    print("Hyperparameter tuning results saved as CSV files.")

    # 选择各维度中表现最佳的参数值（如果需要全局最佳组合，还可进行全组合搜索）
    best_lr = max(results["lr"], key=lambda x: x["val_acc"])["lr"]
    best_dropout = max(results["dropout"], key=lambda x: x["val_acc"])["dropout"]
    best_num_heads = max(results["num_heads"], key=lambda x: x["val_acc"])["num_heads"]
    print(f"Best parameters: Learning Rate = {best_lr}, Dropout = {best_dropout}, Attention Heads = {best_num_heads}")

    # 可在此处用最佳参数构造模型，在测试集上评估并生成最终报告
    # 加载测试集
    (_, _), (_, _), (x_test, y_test) = load_and_preprocess(file_path)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.long).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_args = Args(module="local meso global", lr=best_lr, dropout=best_dropout, num_heads=best_num_heads)
    best_model = PGCN(best_args, adj, coordinate).to(device)
    # 此处建议加载最佳参数训练得到的模型 checkpoint（若保存了不同超参数下的模型），否则直接重新训练
    best_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs, _, _ = best_model(batch_x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    from sklearn.metrics import classification_report, confusion_matrix
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_report = classification_report(all_labels, all_preds, target_names=["Interictal", "Preictal"])
    test_conf_mat = confusion_matrix(all_labels, all_preds)
    print("Test Accuracy with Best Hyperparameters: {:.4f}".format(test_accuracy))
    print("Test Classification Report:")
    print(test_report)
    print("Test Confusion Matrix:")
    print(test_conf_mat)
    # 可保存测试报告到文件
    with open("trainResults/chb15/test_evaluation_best.txt", "w") as f:
        f.write(f"Best Hyperparameters: LR={best_lr}, Dropout={best_dropout}, Num_heads={best_num_heads}\n")
        f.write("Test Accuracy: {:.4f}\n".format(test_accuracy))
        f.write("Classification Report:\n")
        f.write(test_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(test_conf_mat))
    print("Final test evaluation report saved to trainResults/chb15/test_evaluation_best.txt")


if __name__ == "__main__":
    run_experiment()
