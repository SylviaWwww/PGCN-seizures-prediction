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
        self.adj = local_adj  # [n_channels, n_channels]
        self.coordinate = coordinate  # [n_channels, 2]

        # --------------- Local GCN ---------------
        self.local_gcn_1 = LocalLayer(args.in_feature, 10, use_bias=True)
        self.local_gcn_2 = LocalLayer(10, 15, use_bias=True)

        # --------------- Meso Layer ---------------
        # 将输入特征先映射到 30 维 (5+10+15 = 30)
        self.meso_embed = nn.Linear(args.in_feature, 30)
        self.meso_layer_1 = MesoLayer(subgraph_num=7, num_heads=6, coordinate=self.coordinate, trainable_vector=30)
        self.meso_layer_2 = MesoLayer(subgraph_num=2, num_heads=6, coordinate=self.coordinate, trainable_vector=30)

        # --------------- Global Layer ---------------
        self.global_layer_1 = GlobalLayer(30, 40)

        # --------------- MLP分类器 ---------------
        # 这里演示一个简单的全连接
        # 假设最终要把 (n_channels*3) 个节点，每个节点有 (30 +10) 或者 (40) 维展开
        # 不同拼接会影响最终 view 的大小，请根据实际需要调整
        # 这里假设最终输出形状为 [batch_size, n_channels*3, 40] => flatten => [batch_size, 3*n_channels*40]
        n_channels = self.coordinate.shape[0]
        self.mlp0 = nn.Linear(self.coordinate.shape[0] * 90, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, self.nclass)

        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        lap_matrix = normalize_adj(self.adj)

        # ----- 1. Local -----
        if "local" in self.args.module:
            local_x1 = self.lrelu(self.local_gcn_1(x, lap_matrix, use_relu=True))
            local_x2 = self.lrelu(self.local_gcn_2(local_x1, lap_matrix, use_relu=True))
            res_local = torch.cat((x, local_x1, local_x2), dim=2)  # 形状: [batch, n_channels, 30]
        else:
            # 将 x 映射到 30 维，使其与 meso 分支一致
            res_local = self.meso_embed(x)  # 形状: [batch, n_channels, 30]

        # ----- 2. Meso -----
        if "meso" in self.args.module:
            meso_input = self.meso_embed(x)  # 形状: [batch, n_channels, 30]
            coarsen_x1, _ = self.meso_layer_1(meso_input, self.coordinate)
            coarsen_x1 = self.lrelu(coarsen_x1)
            coarsen_x2, _ = self.meso_layer_2(meso_input, self.coordinate)
            coarsen_x2 = self.lrelu(coarsen_x2)
            res_meso = torch.cat((res_local, coarsen_x1, coarsen_x2), dim=1)  # 拼接后形状: [batch, n_channels*3, 30]
        else:
            res_meso = res_local

        # ----- 3. Global -----
        if "global" in self.args.module:
            global_x1 = self.lrelu(self.global_layer_1(res_meso, lap_matrix))
            res_global = torch.cat((res_meso, global_x1), dim=2)  # 例如拼接后维度变化
        else:
            res_global = res_meso

        # ----- 4. MLP 分类 -----
        out = res_global.view(res_global.size(0), -1)
        out = self.lrelu(self.mlp0(out))
        out = self.dropout(out)
        out = self.lrelu(self.mlp1(out))
        out = self.bn(out)
        out = self.mlp2(out)
        return out, lap_matrix, None

