#!/usr/bin/env python
import sys
import os
import torch
import scanpy as sc
import pickle

# 将项目根目录加入 Python 路径（确保包内模块正确导入）
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

# 采用相对导入
from .train import Train_SpatialGlue
from .preprocess import construct_neighbor_graph, pca

# 创建结果保存目录
results_dir = os.path.join(base_path, "results")
os.makedirs(results_dir, exist_ok=True)

# 定义数据文件路径
adata_path_omics1 = os.path.join(base_path, "data", "Dataset1_Mouse_Spleen1", "adata_ADT.h5ad")
adata_path_omics2 = os.path.join(base_path, "data", "Dataset1_Mouse_Spleen1", "adata_RNA.h5ad")

# 读取数据
print("Loading omics1 data from:", adata_path_omics1)
adata_omics1 = sc.read_h5ad(adata_path_omics1)
print("Loading omics2 data from:", adata_path_omics2)
adata_omics2 = sc.read_h5ad(adata_path_omics2)

# 修正变量名，确保唯一
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# 检查是否存在特征信息，如果缺失则利用 PCA 计算（默认提取 10 个主成分，可根据需要调整）
if 'feat' not in adata_omics1.obsm:
    print("Computing PCA features for omics1...")
    adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=10)
if 'feat' not in adata_omics2.obsm:
    print("Computing PCA features for omics2...")
    adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=10)

# 检查邻接图信息是否存在，如果缺失则构建邻接图
if 'adj_spatial' not in adata_omics1.uns or 'adj_feature' not in adata_omics1.obsm:
    print("Constructing neighbor graphs for omics data...")
    data_temp = construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3)
    adata_omics1 = data_temp['adata_omics1']
    adata_omics2 = data_temp['adata_omics2']

# 构造传递给训练器的输入数据字典
data = {"adata_omics1": adata_omics1, "adata_omics2": adata_omics2}

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 初始化训练器时，将 dim_output 设置为 128 以匹配模型中 self.mu 层的输入维度
trainer = Train_SpatialGlue(data, device=device, dim_output=128)

# 开始训练
print("Training the model...")
results = trainer.train()

# 保存训练结果到 results/embeddings.pkl
output_path = os.path.join(results_dir, "embeddings.pkl")
with open(output_path, "wb") as f:
    pickle.dump(results, f)

print("✅ Training complete, results saved to", output_path)