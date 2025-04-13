# feature_visualization.py

import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from model_PGCN import PGCN
from preprocess import load_and_preprocess, load_hdf5
from connectivity import compute_dataset_connectivity

# =========== 1. 选取要可视化的层或注意力权重 ===========

"""
示例：我们想要可视化
    1) Local 层输出特征 (local_x1, local_x2)
    2) Meso 层的注意力权重 (attn_weights)
    3) Global 层输出特征 (global_x1)

要获取 Meso 层多头注意力，需要在 MesoLayer 里改动:
  attn_output, attn_weights = self.attention(x_transposed, x_transposed, x_transposed, need_weights=True)

或者通过 forward_hook 的方式获取 attn_weights。
下面示例假设我们已能在前向传播中得到 attn_weights 并返回:
  return attn_output, attn_weights
则在 model_PGCN.py 中 MesoLayer forward 也需返回该 attn_weights。
在 forward 中可将其拼接或以 tuple 形式传递，便于我们提取。
"""


def visualize_attention_heatmap(attn_weights, title="Meso Attention Weights", save_path=None):
    """
    绘制多头注意力热力图
    attn_weights 形状:
      - [num_heads, n_channels, n_channels] 或
      - [batch_size, num_heads, n_channels, n_channels]
    若包含 batch_size，通常可以只可视化其中一个样本，比如 attn_weights[0]
    """
    if attn_weights.dim() == 4:
        # 只画第一个 batch 的
        attn_weights = attn_weights[0]  # => [num_heads, n_channels, n_channels]

    num_heads = attn_weights.size(0)
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]  # 保证可迭代

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attn_weights[i].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        ax.set_title(f"Head {i + 1}")
        fig.colorbar(im, ax=ax)
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def visualize_feature_map(feature_map, title="Feature Map", save_path=None):
    """
    绘制局部/全局层输出特征的示例，可视化方式多样。这里简单做一个热力图或二维可视化。
    feature_map: 形状 [batch_size, n_channels, out_dim]
    """
    # 仅拿第一个 batch 做示例
    feat = feature_map[0].detach().cpu().numpy()  # => [n_channels, out_dim]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(feat, cmap='viridis', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Channels")
    fig.colorbar(im, ax=ax)
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


# =========== 2. 将模型的中间输出和 attn_weights 暴露出来 ===========

"""
思路：在 model_PGCN.py 的 forward 中，或者局部/meso 层中，可以把我们想要的中间结果返回。
例如:
    return x, lap_matrix, {
        "local_x1": local_x1,
        "local_x2": local_x2,
        "coarsen_attn1": attn_weights1,
        "coarsen_attn2": attn_weights2,
        "global_x1": global_x1
    }

或者使用 register_forward_hook() 做法，这里仅示例我们用 forward 的返回值携带中间信息。
假设你已在 forward 中做了类似:
    ...
    coarsen_x1, attn_weights1 = self.meso_layer_1(...)
    coarsen_x2, attn_weights2 = self.meso_layer_2(...)
    ...
    global_x1 = ...
    ...
    return out, lap_matrix, {
        "local_x1": local_x1,
        "local_x2": local_x2,
        "attn1": attn_weights1,
        "attn2": attn_weights2,
        "global_x1": global_x1
    }
这样我们就能拿到中间信息。
"""


def visualize_samples(model, data_loader, device, out_dir="feature_plots"):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            # 假设 forward 返回 (logits, lap_matrix, dict_of_features)
            outputs, lap_matrix, feats = model(batch_x)

            # feats 中包含 local_x1, local_x2, attn1, attn2, global_x1 等
            if "attn1" in feats:
                # 可视化 Meso layer 1 的注意力
                attn1 = feats["attn1"]
                save_path = os.path.join(out_dir, "meso_attn1.png")
                visualize_attention_heatmap(attn1, title="Meso Layer1 Attention", save_path=save_path)
            if "attn2" in feats:
                attn2 = feats["attn2"]
                save_path = os.path.join(out_dir, "meso_attn2.png")
                visualize_attention_heatmap(attn2, title="Meso Layer2 Attention", save_path=save_path)

            if "local_x1" in feats:
                loc1 = feats["local_x1"]
                save_path = os.path.join(out_dir, "local_x1_feature.png")
                visualize_feature_map(loc1, title="Local GCN 1 Feature Map", save_path=save_path)
            if "local_x2" in feats:
                loc2 = feats["local_x2"]
                save_path = os.path.join(out_dir, "local_x2_feature.png")
                visualize_feature_map(loc2, title="Local GCN 2 Feature Map", save_path=save_path)
            if "global_x1" in feats:
                glb1 = feats["global_x1"]
                save_path = os.path.join(out_dir, "global_x1_feature.png")
                visualize_feature_map(glb1, title="Global GCN Feature Map", save_path=save_path)

            break  # 只看第一个 batch 的可视化示例
    print(f"Visualization plots saved to folder: {out_dir}")


# =========== 3. 主函数示例 ===========

def main():
    """
    示例：加载某个患者的模型，取验证集或测试集里的一个 batch 做可视化。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")

    # 1. 加载数据
    file_path = "../patient_dataset/dataset_chb15.h5"
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess(file_path)
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.long).to(device))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 2. 构造邻接矩阵与坐标
    raw_data, _ = load_hdf5(file_path)
    adj_np = compute_dataset_connectivity(raw_data, method="mean")
    adj = torch.FloatTensor(adj_np).to(device)
    with h5py.File(file_path, "r") as hf:
        coordinates_np = hf["coordinates"][:]
    coordinate = torch.FloatTensor(coordinates_np).to(device)

    # 3. 初始化模型并加载已训练的权重
    from model_PGCN import PGCN
    from utils import normalize_adj  # 如果需要

    class Args:
        def __init__(self):
            self.n_class = 2
            self.in_feature = 5
            self.dropout = 0.5
            self.lr = 0.001
            self.module = "local meso global"
            # 需要的话，可加其它字段，如 num_heads

    args = Args()
    model = PGCN(args, adj, coordinate).to(device)
    # 加载训练好的参数
    checkpoint_path = "../trainResults/chb15/pgcn_chb15.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 4. 可视化
    visualize_samples(model, val_loader, device, out_dir="feature_plots_chb15")


if __name__ == "__main__":
    main()
