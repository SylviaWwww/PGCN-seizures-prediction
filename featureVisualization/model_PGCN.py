# model_PGCN.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from layer_PGCN import LocalLayer, MesoLayer, GlobalLayer
from utils import normalize_adj

class PGCN(nn.Module):
    """癫痫发作预测的金字塔图卷积网络"""
    def __init__(self, args, local_adj, coordinate):
        super().__init__()
        self.args = args
        self.nclass = args.n_class
        self.dropout_rate = args.dropout
        self.l_relu = args.lr
        self.adj = local_adj           # 邻接矩阵，形状: [n_channels, n_channels]
        self.coordinate = coordinate   # 通道坐标，形状: [n_channels, 2]

        # Local GCN
        self.local_gcn_1 = LocalLayer(args.in_feature, 10, use_bias=True)
        self.local_gcn_2 = LocalLayer(10, 15, use_bias=True)

        # MesoLayer
        self.meso_embed = nn.Linear(args.in_feature, 30)  # 输出维度为 30 = 5+10+15
        self.meso_layer_1 = MesoLayer(subgraph_num=7, num_heads=6, coordinate=self.coordinate, trainable_vector=30)
        self.meso_layer_2 = MesoLayer(subgraph_num=2, num_heads=6, coordinate=self.coordinate, trainable_vector=30)
        self.meso_dropout = nn.Dropout(0.2)

        # GlobalLayer
        self.global_layer_1 = GlobalLayer(30, 40)

        # MLP分类器
        # 注意：经过各层拼接后，最终每个样本的特征数为 3 * n_channels * 70，
        # 其中 70 = 30（来自 meso 拼接）+ 40（global 层输出）.
        self.mlp0 = nn.Linear(3 * self.coordinate.shape[0] * 70, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, self.nclass)

        # 公用组件
        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        前向传播
        输入:
            x: [batch_size, n_channels, in_features]
        返回:
            out: 分类 logits
            lap_matrix: 归一化后的邻接矩阵
            feats: 字典，包含中间层的输出，便于可视化和消融实验
        """
        x = x.to(self.adj.device)
        lap_matrix = normalize_adj(self.adj)

        # 1. Local GCN 计算
        local_x1 = self.lrelu(self.local_gcn_1(x, lap_matrix, use_relu=True))  # 输出形状: [batch, n_channels, 10]
        local_x2 = self.lrelu(self.local_gcn_2(local_x1, lap_matrix, use_relu=True))  # 输出形状: [batch, n_channels, 15]
        res_local = torch.cat((x, local_x1, local_x2), dim=2)  # 拼接后形状: [batch, n_channels, 5+10+15=30]

        # 2. MesoLayer 计算
        meso_input = self.meso_embed(x)  # 形状: [batch, n_channels, 30]
        coarsen_x1, attn_weights1, coarsen_coor1 = self.meso_layer_1(meso_input, self.coordinate)  # coarsen_x1: [batch, n_channels, 30]
        coarsen_x1 = self.lrelu(coarsen_x1)
        coarsen_x2, attn_weights2, coarsen_coor2 = self.meso_layer_2(meso_input, self.coordinate)  # coarsen_x2: [batch, n_channels, 30]
        coarsen_x2 = self.lrelu(coarsen_x2)
        # 拼接 local 及两个 meso 分支：沿通道维度拼接 => [batch, n_channels*3, 30]
        res_meso = torch.cat((res_local, coarsen_x1, coarsen_x2), dim=1)

        # 3. GlobalLayer 计算
        global_x1 = self.lrelu(self.global_layer_1(res_meso, lap_matrix))  # 输出形状: [batch, n_channels*3, 40]
        # 拼接 global 输出：沿特征维度拼接 => [batch, n_channels*3, 30+40=70]
        res_global = torch.cat((res_meso, global_x1), dim=2)

        # 4. MLP 分类
        flat = res_global.view(res_global.size(0), -1)
        mlp_out = self.lrelu(self.mlp0(flat))
        mlp_out = self.dropout(mlp_out)
        mlp_out = self.lrelu(self.mlp1(mlp_out))
        mlp_out = self.bn(mlp_out)
        mlp_out = self.mlp2(mlp_out)

        # 收集中间输出，便于后续可视化和消融分析
        feats = {
            "local_x1": local_x1,
            "local_x2": local_x2,
            "attn1": attn_weights1,  # 来自 MesoLayer1 的注意力权重
            "attn2": attn_weights2,  # 来自 MesoLayer2 的注意力权重
            "global_x1": global_x1,
            "res_local": res_local,
            "res_meso": res_meso,
            "res_global": res_global
        }
        return mlp_out, lap_matrix, feats
