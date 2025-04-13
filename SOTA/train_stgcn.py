import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from stgcn_model import STGCN
from preprocess import load_and_preprocess, load_hdf5
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from connectivity import compute_dataset_connectivity


def train_stgcn():
    # 加载数据：假设 x_train, x_val 均为形状 [samples, num_nodes, in_channels]
    (x_train, y_train), _, (x_test, y_test) = load_and_preprocess("single_patient_dataset/full_dataset_chb15.h5")
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

    num_nodes = x_train.shape[1]
    in_channels = x_train.shape[2]

    raw_data, _ = load_hdf5("single_patient_dataset/full_dataset_chb15.h5")
    dummy_adj = compute_dataset_connectivity(raw_data, method="mean")
    dummy_adj = torch.FloatTensor(dummy_adj).to(device)
    edge_index, _ = dense_to_sparse(dummy_adj)

    model = STGCN(in_channels=in_channels, hidden_channels=16, num_classes=2, num_nodes=num_nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x, edge_index)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
        train_acc = total_correct / total_samples

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x, edge_index)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_val = roc_auc_score(all_labels, all_probs)

    print("ST-GCN Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_val:.4f}")

if __name__ == "__main__":
    train_stgcn()
