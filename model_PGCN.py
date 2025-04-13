import torch.nn as nn
from layer_PGCN import LocalLayer, MesoLayer, GlobalLayer
import torch
from utils import normalize_adj

class PGCN(nn.Module):
    """癫痫发作预测的金字塔图卷积网络"""
    def __init__(self, args, local_adj, coordinate):
        super().__init__()
        self.args = args
        self.nclass = args.n_class
        self.dropout_rate = args.dropout
        self.l_relu = args.lr
        self.adj = local_adj  # 邻接矩阵（形状: [n_channels, n_channels]）
        self.coordinate = coordinate  # 通道坐标（形状: [n_channels, 2]）

        # Local GCN
        self.local_gcn_1 = LocalLayer(args.in_feature, 10, use_bias=True)
        self.local_gcn_2 = LocalLayer(10, 15, use_bias=True)

        # MesoLayer
        self.meso_embed = nn.Linear(args.in_feature, 30)  # 输出维度为 30 = 5(in_feature) + 10 + 15
        self.meso_layer_1 = MesoLayer(subgraph_num=7, num_heads=6, coordinate=self.coordinate, trainable_vector=30)
        self.meso_layer_2 = MesoLayer(subgraph_num=2, num_heads=6, coordinate=self.coordinate, trainable_vector=30)
        self.meso_dropout = nn.Dropout(0.2)

        # GlobalLayer
        self.global_layer_1 = GlobalLayer(30, 40)

        # MLP分类器
        self.mlp0 = nn.Linear(3*self.coordinate.shape[0]*70, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, self.nclass)

        # 公用组件
        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """前向传播"""
        # 输入x形状: [batch_size, n_channels, n_features]
        x = x.to(self.adj.device)

        # 1. Local GCN
        lap_matrix = normalize_adj(self.adj)  # 归一化邻接矩阵
        local_x1 = self.lrelu(self.local_gcn_1(x, lap_matrix, use_relu=True))
        local_x2 = self.lrelu(self.local_gcn_2(local_x1, lap_matrix, use_relu=True))
        res_local = torch.cat((x, local_x1, local_x2), dim=2)  # 形状: [batch, n_channels, 5+10+15=30]

        # 2. MesoLayer
        meso_input = self.meso_embed(x)  # 形状: [batch, n_channels, 30]
        coarsen_x1, coarsen_coor1 = self.meso_layer_1(meso_input, self.coordinate)
        coarsen_x1 = self.lrelu(coarsen_x1)
        coarsen_x2, coarsen_coor2 = self.meso_layer_2(meso_input, self.coordinate)
        coarsen_x2 = self.lrelu(coarsen_x2)
        res_meso = torch.cat((res_local, coarsen_x1, coarsen_x2), dim=1)  # 形状: [batch, n_channels*3, 30]

        # 3. GlobalLayer
        global_x1 = self.lrelu(self.global_layer_1(res_meso, lap_matrix))
        res_global = torch.cat((res_meso, global_x1), dim=2)  # 形状: [batch, n_channels*3, 40]

        # 4. MLP分类
        x = res_global.view(res_global.size(0), -1)

        x = self.lrelu(self.mlp0(x))

        x = self.dropout(x)
        x = self.lrelu(self.mlp1(x))

        x = self.bn(x)
        x = self.mlp2(x)

        return x, lap_matrix, ""