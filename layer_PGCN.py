# layer_PGCN.py
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

        # 确保输入数据和邻接矩阵位于同一设备
        x = x.to(adj.device)

        # 将 3D 输入转换为 2D（合并批次和通道）
        x_2d = x.view(-1, in_features)  # 形状: [batch_size * n_channels, in_features]

        # 直接使用稠密邻接矩阵
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # 获取非零元素的索引

        # 执行 GCN 卷积
        x_2d = self.gcn(x_2d, edge_index)

        # 恢复为 3D 输出
        x_3d = x_2d.view(batch_size, n_channels, -1)  # 形状: [batch_size, n_channels, out_features]
        return F.leaky_relu(x_3d) if use_relu else x_3d


# layer_PGCN.py
class MesoLayer(nn.Module):
    """
    中尺度子图层（多子图聚合）
    """

    def __init__(self, subgraph_num, num_heads, coordinate, trainable_vector):
        super().__init__()
        self.subgraph_num = subgraph_num
        self.num_heads = num_heads
        self.coordinate = coordinate
        self.attention = nn.MultiheadAttention(trainable_vector, num_heads)

    def forward(self, x, coor):
        x = x.to(coor.device)

        batch_size = x.size(0)
        coor_expanded = coor.unsqueeze(0).repeat(batch_size, 1, 1)
        attn_output, _ = self.attention(x, x, x)
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

        # 直接使用稠密邻接矩阵
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # 获取非零元素的索引

        # 执行 GCN 卷积
        return self.gcn(x, edge_index)