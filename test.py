#!/usr/bin/env python
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
import scanpy as sc
import time
import csv
import os

# 从模型和预处理模块中导入需要的类和函数
from SpatialGlue.model import Encoder_overall
from SpatialGlue.preprocess import adjacent_matrix_preprocessing, construct_neighbor_graph

# 设定文件名
result_file = 'test_results.csv'

# 创建一个 CSV 文件并写入标题
def initialize_result_file():
    if not os.path.exists(result_file):
        with open(result_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Run ID', 'Silhouette Score', 'MSE Omics1', 'MSE Omics2', 'Time'])

def load_embeddings(pkl_path):
    """加载 pickle 文件中的嵌入数据"""
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def evaluate_embedding(X, n_clusters=4):
    """
    对给定的嵌入 X 进行评测：
      - 使用 KMeans 聚类并计算 Silhouette Score；
      - 使用 TSNE 将 X 降至 2D，并绘制散点图；
      - 如果数据退化（方差太低或只有一个聚类），则给出警告并跳过相关计算。
    """
    if np.isnan(X).any():
        print("Warning: Detected NaN values in embedding, replacing NaN with 0.")
        X = np.nan_to_num(X, nan=0.0)
    
    if np.std(X) < 1e-6:
        print("Warning: Embedding variance is too low, skipping clustering and TSNE visualization.")
        return None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        print("Warning: Only one cluster found. Silhouette score cannot be computed.")
        sil_score = None
    else:
        sil_score = silhouette_score(X, labels)
        print("Silhouette Score for KMeans with {} clusters: {:.4f}".format(n_clusters, sil_score))
    
    try:
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
    except Exception as e:
        print("TSNE failed with error:", e)
        X_tsne = np.zeros((X.shape[0], 2))
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title("t-SNE Visualization of SpatialGlue Embedding\n(KMeans Clusters)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    plt.tight_layout()
    plt.savefig('output.png')
    
    return sil_score

def evaluate_reconstruction(latent_embeddings, original_data, model, adj_spatial, device):
    """
    评估重构效果，计算重构误差（MSE）。
    使用模型中的解码器对输入的 latent 表示进行重构，并与原始数据进行比较。
    """
    # 获取解码器
    decoder_omics1 = model.decoder_omics1
    decoder_omics2 = model.decoder_omics2

    # 将 latent_embeddings 转为 torch.Tensor，并转移到 device 上
    latent_tensor = torch.tensor(latent_embeddings, dtype=torch.float32).to(device)
    # 确保邻接矩阵也在相同设备上
    adj_spatial = adj_spatial.to(device)
    
    # 通过解码器重构数据
    emb_recon_omics1 = decoder_omics1(latent_tensor, adj_spatial)
    emb_recon_omics2 = decoder_omics2(latent_tensor, adj_spatial)
    
    emb_recon_omics1 = emb_recon_omics1.detach().cpu().numpy()
    emb_recon_omics2 = emb_recon_omics2.detach().cpu().numpy()
    
    print(f"emb_recon_omics1 shape: {emb_recon_omics1.shape}, original_data['omics1'] shape: {original_data['omics1'].shape}")
    print(f"emb_recon_omics2 shape: {emb_recon_omics2.shape}, original_data['omics2'] shape: {original_data['omics2'].shape}")
    
    assert emb_recon_omics1.shape == original_data['omics1'].shape, "Shape mismatch for omics1"
    assert emb_recon_omics2.shape == original_data['omics2'].shape, "Shape mismatch for omics2"
    
    mse_omics1 = mean_squared_error(original_data['omics1'], emb_recon_omics1)
    mse_omics2 = mean_squared_error(original_data['omics2'], emb_recon_omics2)
    
    print(f'MSE for Omics1 Reconstruction: {mse_omics1:.4f}')
    print(f'MSE for Omics2 Reconstruction: {mse_omics2:.4f}')
    
    return mse_omics1, mse_omics2

def save_results(run_id, sil_score):
    """将结果保存到 CSV 文件"""
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(result_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, sil_score,  current_time])

def main():
    # 初始化文件
    initialize_result_file()
    # 检查是否有 GPU 支持
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 载入嵌入文件
    pkl_path = 'results/embeddings.pkl'
    embeddings = load_embeddings(pkl_path)
    print("Embedding keys:", embeddings.keys())
    for key in embeddings:
        print("Key: {}, Shape: {}".format(key, np.array(embeddings[key]).shape))
    
    # 使用 'SpatialGlue' 嵌入进行聚类评测
    if 'SpatialGlue' not in embeddings:
        raise KeyError("The key 'SpatialGlue' is not found in the embeddings file.")
    X = np.array(embeddings['SpatialGlue'])
    print("Using SpatialGlue embedding with shape:", X.shape)
    sil_score = evaluate_embedding(X, n_clusters=3)
    '''
    # 评估重构效果：使用原始数据中的特征作为重构目标
    # 加载原始数据（请根据实际文件路径修改）
    adata_omics1 = sc.read('data/Dataset1_Mouse_Spleen1/adata_ADT.h5ad')
    adata_omics2 = sc.read('data/Dataset1_Mouse_Spleen1/adata_RNA.h5ad')
    
    # 如果 'feat' 不存在，则使用 dense X 转为dense格式
    if 'feat' not in adata_omics1.obsm:
        print("Warning: 'feat' not found in adata_omics1.obsm, using dense X from adata.X.")
        if hasattr(adata_omics1.X, "toarray"):
            adata_omics1.obsm['feat'] = adata_omics1.X.toarray()
        else:
            adata_omics1.obsm['feat'] = adata_omics1.X
    if 'feat' not in adata_omics2.obsm:
        print("Warning: 'feat' not found in adata_omics2.obsm, using dense X from adata.X.")
        if hasattr(adata_omics2.X, "toarray"):
            adata_omics2.obsm['feat'] = adata_omics2.X.toarray()
        else:
            adata_omics2.obsm['feat'] = adata_omics2.X
    
    # 使用原始数据中的 'feat' 作为重构目标
    original_data = {
        'omics1': np.array(adata_omics1.obsm['feat']),
        'omics2': np.array(adata_omics2.obsm['feat'])
    }
    
    # 如果原始数据中没有 'adj_spatial'，则构建邻接图
    if 'adj_spatial' not in adata_omics1.uns:
        print("Warning: 'adj_spatial' not found in adata_omics1.uns, constructing neighbor graph...")
        from SpatialGlue.preprocess import construct_neighbor_graph
        data_temp = construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3)
        adata_omics1 = data_temp['adata_omics1']
        adata_omics2 = data_temp['adata_omics2']
    
    # 生成邻接矩阵
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
    # 根据模型代码，共用同一空间邻接矩阵，使用 'adj_spatial_omics1'
    adj_spatial = adj['adj_spatial_omics1'].to(device)
    
    # 分别获取原始数据的维度
    dim_in1 = adata_omics1.obsm['feat'].shape[1]
    dim_in2 = adata_omics2.obsm['feat'].shape[1]
    
    # 实例化模型（确保参数设置与训练时一致）
    model = Encoder_overall(
        dim_in_omics1=dim_in1, dim_out_omics1=128,
        dim_in_omics2=dim_in2, dim_out_omics2=128
    ).to(device)
    
    # 评估重构效果
    mse_omics1, mse_omics2 = evaluate_reconstruction(X, original_data, model, adj_spatial, device)
    '''
    # 保存结果
    run_id = 0  # 你可以根据需要使用不同的ID或自增ID
    save_results(run_id, sil_score)
    
    
if __name__ == '__main__':
    main()