import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_nodes):
        super(STGCN, self).__init__()
        # 两层空间图卷积
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        # 最后全连接层，输入为每个节点的特征拼接成向量
        self.fc = nn.Linear(hidden_channels * num_nodes, num_classes)

    def forward(self, x, edge_index):
        # x: [batch, num_nodes, in_channels]
        batch_size, num_nodes, _ = x.shape
        outputs = []
        # 对每个样本独立进行图卷积
        for i in range(batch_size):
            h = x[i]  # [num_nodes, in_channels]
            h = F.relu(self.gcn1(h, edge_index))
            h = F.relu(self.gcn2(h, edge_index))
            outputs.append(h.unsqueeze(0))
        h = torch.cat(outputs, dim=0)  # [batch, num_nodes, hidden_channels]
        h = h.view(batch_size, -1)
        out = self.fc(h)
        return out