import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cnn_model import CNNClassifier
from preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


def train_cnn():
    # 加载数据（假设返回的 x_train, x_val, x_test 均为形状 [samples, channels, features]）
    (x_train, y_train), _, (x_test, y_test) = load_and_preprocess("single_patient_dataset/full_dataset_chb15.h5")

    # 对于 CNN 输入，需要添加一个通道维度
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_channels = x_train.shape[2]
    num_features = x_train.shape[3]

    model = CNNClassifier(num_channels, num_features, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_val = roc_auc_score(all_labels, all_probs)

    print("CNN Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_val:.4f}")


if __name__ == "__main__":
    train_cnn()
