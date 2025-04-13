import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LocalLayer(nn.Module):
    """
    支持 3D 输入的 Local GCN 层
    """
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.gcn = GCNConv(in_features, out_features, bias=use_bias)

    def forward(self, x, adj, use_relu=True):
        # 输入 x 形状: [batch_size, n_channels, in_features]
        batch_size, n_channels, in_features = x.shape

        x = x.to(adj.device)
        # 将 3D 输入转换为 2D（合并 batch 和 channel）
        x_2d = x.view(-1, in_features)  # 形状: [batch_size * n_channels, in_features]

        # 直接使用稠密邻接矩阵
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # 获取非零元素索引, 形状 [2, num_edges]

        x_2d = self.gcn(x_2d, edge_index)  # GCNConv
        x_3d = x_2d.view(batch_size, n_channels, -1)  # 恢复为 3D

        return F.leaky_relu(x_3d) if use_relu else x_3d

class MesoLayer(nn.Module):
    """
    中尺度子图层（多子图聚合示例）
    """
    def __init__(self, subgraph_num, num_heads, coordinate, trainable_vector):
        super().__init__()
        self.subgraph_num = subgraph_num
        self.num_heads = num_heads
        self.coordinate = coordinate
        # 这里以多头注意力作为一个简单的中尺度示例
        self.attention = nn.MultiheadAttention(trainable_vector, num_heads)

    def forward(self, x, coor):
        """
        x 形状: [batch_size, n_channels, hidden_dim]
        coor 形状: [n_channels, 2]
        """
        x = x.to(coor.device)
        batch_size = x.size(0)

        # 为了简单，这里不对 subgraph_num 做复杂拆分，仅做一个多头注意力的示例
        coor_expanded = coor.unsqueeze(0).repeat(batch_size, 1, 1)
        # 由于 multihead attention 期望的输入形状为 [seq_len, batch_size, embed_dim]
        x_transposed = x.transpose(0, 1)  # [n_channels, batch_size, hidden_dim]

        attn_output, _ = self.attention(x_transposed, x_transposed, x_transposed)
        # 还原为 [batch_size, n_channels, hidden_dim]
        attn_output = attn_output.transpose(0, 1)

        return attn_output, coor_expanded

class GlobalLayer(nn.Module):
    """
    全局图卷积层
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gcn = GCNConv(in_features, out_features)

    def forward(self, x, adj):
        # 确保输入数据和邻接矩阵位于同一设备
        x = x.to(adj.device)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        return self.gcn(x, edge_index)
