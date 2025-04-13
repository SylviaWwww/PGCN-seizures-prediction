import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn

# 自定义文件的导入
from model_PGCN import PGCN
from preprocess import load_and_preprocess, load_hdf5
from connectivity import compute_dataset_connectivity
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 参数类
class Args:
    def __init__(self):
        self.n_class = 2      # 二分类
        self.in_feature = 5   # 输入特征维度
        self.dropout = 0.5
        self.lr = 0.001
        # 消融实验开关，可修改为 "local", "meso", "global", "local meso", "local meso global", 等
        # self.module = "local meso global"
        self.module = "local meso"
def select_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

def main():
    args = Args()
    device = select_device()

    # ========== 1. 加载数据并预处理 ==========
    file_path = "../patient_dataset/dataset_chb15.h5"
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess(file_path)
    print("Train shape:", x_train.shape, "Val shape:", x_val.shape, "Test shape:", x_test.shape)

    # ========== 2. 计算功能连接矩阵与坐标 ==========
    raw_data, _ = load_hdf5(file_path)
    adj = compute_dataset_connectivity(raw_data, method="mean")
    # 转为 torch 张量
    adj = torch.FloatTensor(adj).to(device)

    # 加载坐标
    with h5py.File(file_path, "r") as hf:
        coordinates = hf["coordinates"][:]  # [n_channels, 2]
    coordinate = torch.FloatTensor(coordinates).to(device)

    # ========== 3. 初始化模型 ==========
    model = PGCN(args, adj, coordinate).to(device)
    print("PGCN model:", model)

    # ========== 4. 构造 DataLoader ==========
    batch_size = 16
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ========== 5. 训练准备 ==========
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    best_val_acc = 0
    num_epochs = 10

    # ========== 6. 训练循环 ==========
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(batch_x)  # model forward
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_loss_epoch = total_loss / total
        train_acc_epoch = correct / total

        # 验证
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

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_epoch)
        history["train_acc"].append(train_acc_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["val_acc"].append(val_acc_epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, "
              f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")

        # 保存最优模型
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            checkpoint_name = "trainResults/chb15/noGlobal/pgcn_chb15.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"Saved best model to {checkpoint_name}")

    # ========== 7. 保存训练日志到Excel ==========
    df = pd.DataFrame(history)
    excel_path = "trainResults/chb15/noGlobal/chb15results.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"训练日志已保存到 {excel_path}")

    # ========== 8. 训练过程可视化 ==========
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.savefig("trainResults/chb15/noGlobal/chb15_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.savefig("trainResults/chb15/noGlobal/chb15_acc_curve.png")
    plt.close()

    print("训练过程图像已保存到 trainResults/ 文件夹")

    # ========== 9. 验证集详细评估 ==========
    # 也可额外对测试集进行评估
    model.eval()
    all_preds, all_labels = [], []
    for val_x, val_y in val_loader:
        with torch.no_grad():
            val_outputs, _, _ = model(val_x)
            v_preds = val_outputs.argmax(dim=1)
        all_preds.extend(v_preds.cpu().numpy())
        all_labels.extend(val_y.cpu().numpy())

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds, target_names=["Interictal", "Preictal"])
    print("验证集分类报告：\n", report)
    cm = confusion_matrix(all_labels, all_preds)
    print("验证集混淆矩阵：\n", cm)

    # 保存分类报告到 Excel
    report_dict = classification_report(all_labels, all_preds, target_names=["Interictal", "Preictal"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_excel("trainResults/chb15/noGlobal/chb15_classification_report.xlsx")
    print("分类报告已保存到 trainResults/chb15/noGlobal/chb15_classification_report.xlsx")

if __name__ == "__main__":
    main()
