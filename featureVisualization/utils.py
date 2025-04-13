import torch
import numpy as np


def normalize_adj(adj):
    """
    归一化邻接矩阵（对称归一化）
    :param adj: 邻接矩阵（形状: [n_channels, n_channels]）
    :return: 归一化后的邻接矩阵
    """
    # 添加自环
    device = adj.device

    adj = adj + torch.eye(adj.size(0), device=device)

    # 计算度矩阵
    degree = torch.sum(adj, dim=1)
    degree_sqrt = torch.sqrt(degree)

    # 对称归一化
    degree_sqrt_inv = 1.0 / degree_sqrt
    degree_sqrt_inv = torch.diag(degree_sqrt_inv).to(device)
    adj_normalized = torch.mm(torch.mm(degree_sqrt_inv, adj), degree_sqrt_inv)

    return adj_normalized