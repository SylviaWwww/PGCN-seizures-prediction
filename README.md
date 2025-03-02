# PGCN-seizures-prediction

#### **第一步：数据理解与准备**

1. **下载与整理数据集**
   - 从 [CHB-MIT 数据库](https://physionet.org/content/chbmit/1.0.0/) 下载所有患者的EDF文件和标注文件（`chbXX-summary.txt`）。
   - 按患者文件夹组织数据（如 `chb01/`, `chb02/`），确保每个文件夹包含：
     - EDF文件（如 `chb01_03.edf`）
     - 对应的 `chb01-summary.txt`（标注发作时间）
2. **筛选有效数据**
   - 仅保留包含发作标注的EDF文件（通过 `Number of Seizures in File: N` 筛选）。
   - 移除无发作标注的EDF文件（如 `chb01_01.edf` 若其标注为 `Number of Seizures: 0`）。

#### **第二步：数据预处理与分段**

1. **提取目标时间段**

   - **发作前期（Preictal）**：从标注的发作开始时间（如 `2996秒`）向前截取30分钟（`2996 - 1800 = 1196秒`至`2996秒`）。
   - **发作间期（Interictal）**：
     - 从无发作的EDF文件中随机截取30分钟数据。
     - 避免从有发作文件的邻近时间段截取（如发作结束后至少1小时）。

2. **数据标准化与增强**

   - **通道统一**：强制所有EDF文件使用标准23通道（参考 `chbXX-summary.txt` 中的通道列表）。

   - **降噪处理**：应用带通滤波（如0.5-40Hz）去除高频噪声和工频干扰。

   - **滑动窗口分割**：将每个30分钟段切分为更小片段（如10秒一段，重叠50%），增加样本量。

   - ```
     def extract_segments(raw_data, seizure_start, fs=256, preictal_window=1800):
         preictal_start = max(0, seizure_start - preictal_window)
         preictal_data = raw_data[:, preictal_start*fs : seizure_start*fs]
         return preictal_data
     ```

#### **第三步：特征提取与图构建**

1. **时频特征提取**

   - **功率谱密度（PSD）**：计算各频段（Delta, Theta, Alpha, Beta, Gamma）的平均功率。

   - **微分熵（DE）**：对信号进行小波变换后计算熵值。

   - **代码示例**：

     ```
     from mne.time_frequency import psd_welch
     def compute_psd(data, fs=256):
         psds, freqs = psd_welch(data, fmin=0.5, fmax=40, n_fft=fs*2)
         return psds.mean(axis=2)  # 形状: (n_channels, n_freqs)
     ```

2. **构建脑电功能连接图**

   - **节点**：每个电极通道（如FP1-F7）。

   - **边权重**：基于通道间的Pearson相关系数或相位锁定值（PLV）。

   - **邻接矩阵**：阈值化处理（保留相关系数 >0.3 的边）。

   - ```py
     import numpy as np
     def build_adjacency(features, threshold=0.3):
         corr_matrix = np.corrcoef(features)  # 形状: (n_channels, n_channels)
         adj_matrix = (corr_matrix > threshold).astype(np.float32)
         return adj_matrix
     ```

#### **第四步：金字塔图卷积网络（Pyramidal GCN）实现**

1. **模型架构**

   - **多尺度图卷积**：通过不同感受野的GCN层捕获局部和全局特征。

   - **金字塔池化**：逐层聚合节点特征，生成多分辨率表示。

   - **代码参考**：

     ```
     import torch
     import torch.nn as nn
     class PyramidalGCN(nn.Module):
         def __init__(self, in_dim, hidden_dims=[64, 128], num_classes=2):
             super().__init__()
             self.gcn1 = GCNLayer(in_dim, hidden_dims[0])
             self.gcn2 = GCNLayer(hidden_dims[0], hidden_dims[1])
             self.pool = HierarchicalPooling()
             self.classifier = nn.Linear(hidden_dims[-1], num_classes)
         
         def forward(self, adj, features):
             x = self.gcn1(adj, features)
             x = self.pool(x)
             x = self.gcn2(adj, x)
             return self.classifier(x)
     ```

2. **训练策略**

   - **损失函数**：加权交叉熵（`class_weight = [1.0, 6.0]` 平衡正负样本）。
   - **优化器**：Adam（学习率 `1e-4`，权重衰减 `1e-5`）。
   - **早停机制**：验证集损失连续5轮不下降时终止训练。

#### **第五步：实验设计与分析**

1. **患者特异性分析**
   - **操作**：每位患者的数据独立训练和测试（如 `chb01` 训练，`chb01` 测试）。
   - **指标**：准确率、召回率、F1分数。
   - **图表**：柱状图展示各患者的性能差异。
2. **消融实验**
   - **对比项**：移除金字塔结构、替换为普通GCN或CNN。
   - **结果**：表格展示各变体的准确率下降情况。
3. **参数调整实验**
   - **调参目标**：学习率、图构建阈值、GCN层数。
   - **方法**：网格搜索或贝叶斯优化。
4. **SOTA对比实验**
   - **对比模型**：ResNet、LSTM、Transformer。
   - **指标**：ROC曲线下面积（AUC）、预测延迟（Preictal段提前30分钟的准确率）。

**代码示例（实验流程）**：

```python
def run_patient_specific_experiment(subject):
    # 加载数据
    (x_train, y_train), (x_val, y_val) = build_dataset(f"data/CHB-MIT/{subject}")
    # 训练模型
    model = PyramidalGCN()
    model.fit(x_train, y_train)
    # 评估
    accuracy = model.evaluate(x_val, y_val)
    return accuracy

# 遍历所有患者
results = {}
for subject in ["chb01", "chb02", "chb03", ...]:
    results[subject] = run_patient_specific_experiment(subject)
```

#### **第六步：论文撰写与优化**

1. **论文结构**
   - **引言**：癫痫预测的临床意义与现有方法局限性。
   - **方法**：金字塔GCN的理论优势（多尺度特征融合）。
   - **实验**：患者特异性结果、消融实验、参数敏感性分析。
   - **讨论**：模型在噪声环境下的鲁棒性、跨患者泛化能力。
2. **图表要求**
   - **图1**：金字塔GCN架构图（使用 [Draw.io](https://app.diagrams.net/) 绘制）。
   - **图2**：发作间期与发作前期的PSD对比（6个子图）。
   - **表1**：各患者特异性实验的准确率与F1分数。
3. **写作工具**
   - **Latex模板**：使用学院提供的毕设模板（参考往届优秀论文）。
   - **参考文献管理**：Zotero + BibTeX（引用CHB-MIT数据库和核心论文）。

| 实验名称           | 目标                               | 关键操作                               |
| :----------------- | :--------------------------------- | :------------------------------------- |
| **患者特异性分析** | 验证模型在不同患者上的泛化能力     | 按患者划分训练集和测试集               |
| **消融实验**       | 验证金字塔结构有效性               | 移除多尺度卷积层，对比性能变化         |
| **参数调整实验**   | 优化超参数（如学习率、图构建阈值） | 网格搜索或贝叶斯优化                   |
| **SOTA对比实验**   | 与现有模型（如 CNN、LSTM）对比性能 | 复现经典模型，使用相同数据集和评价指标 |

#### **1. 患者特异性结果分析（Patient-Specific Analysis）**

- **目的**：验证模型在不同患者上的个性化预测能力，分析模型是否能够适应个体差异。
- **操作步骤**：
  1. **数据划分**：对每位患者（如 `chb01`、`chb02`）的数据单独划分为训练集和测试集（如 80%-20%）。
  2. **独立训练与测试**：针对每位患者单独训练模型，并在其自身数据上测试。
  3. **指标计算**：统计每位患者的准确率、召回率、F1分数，分析个体差异。
- **示例结果**：
  - 患者 `chb01` 准确率：88%，召回率：85%。
  - 患者 `chb02` 准确率：72%，召回率：68%。

#### 2. **消融实验（Ablation Study）**

- **目的**：验证模型核心组件（如金字塔结构、多尺度卷积）的必要性。
- **操作步骤**：
  1. **移除关键模块**：
     - 变体1：移除金字塔结构，使用单层GCN。
     - 变体2：移除功能连接图，改用普通CNN。
  2. **对比性能**：在相同数据集上训练变体模型，与原模型对比准确率、F1分数等指标。
- **示例结果**：
  - 原模型（金字塔GCN）：准确率 85%。
  - 变体1（单层GCN）：准确率 76%（↓9%）。
  - 变体2（CNN）：准确率 70%（↓15%）。
  - **结论**：金字塔结构和功能连接图显著提升性能。

#### **3. 参数调整实验（Hyperparameter Tuning）**

- **目的**：优化模型超参数，找到最佳配置。
- **操作步骤**：
  1. **定义调参范围**：
     - 学习率：`[1e-3, 1e-4, 1e-5]`
     - 图构建阈值：`[0.2, 0.3, 0.4]`（相关系数阈值）
     - GCN层数：`[2, 3, 4]`
  2. **调参方法**：网格搜索（Grid Search）或贝叶斯优化（Bayesian Optimization）。
  3. **选择最优参数**：根据验证集性能（如准确率）确定最终配置。
- **示例结果**：
  - 最佳学习率：`1e-4`
  - 最佳图阈值：`0.3`
  - 最佳层数：`2`
  - **结论**：参数选择显著影响模型性能，需系统优化。

#### **4. SOTA对比实验（State-of-the-Art Comparison）**

- **目的**：证明所提模型（金字塔GCN）优于现有方法。
- **操作步骤**：
  1. **选择对比模型**：
     - 经典模型：SVM、随机森林。
     - 深度学习模型：CNN、LSTM、Transformer。
     - 最新癫痫预测模型：如 ST-GCN（时空图卷积网络）。
  2. **统一实验设置**：
     - 使用相同的数据集、划分方式和评价指标（如 AUC、F1）。
  3. **性能对比**：统计各模型的预测准确率和计算效率。
- **示例结果**：
  - 金字塔GCN：准确率 85%，AUC 0.92。
  - CNN：准确率 78%，AUC 0.85。
  - LSTM：准确率 72%，AUC 0.80。
  - **结论**：所提模型在时空特征融合上具有显著优势。
