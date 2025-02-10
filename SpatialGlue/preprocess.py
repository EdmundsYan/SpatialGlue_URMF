import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """处理空间图和特征图的邻接矩阵，进行标准化并转化为稀疏矩阵"""
    
    # ===================== 构建空间图 =====================
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    
    # 将空间图从 DataFrame 转为稀疏矩阵
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    # 对称化并阈值处理：对稀疏矩阵使用 .data 进行处理
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1.data = np.where(adj_spatial_omics1.data > 1, 1, adj_spatial_omics1.data)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2.data = np.where(adj_spatial_omics2.data > 1, 1, adj_spatial_omics2.data)

    # 标准化空间图
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1)
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    # ===================== 构建特征图 =====================
    # 注意：此处 .toarray() 已将稀疏矩阵转换为 numpy 数组，因此可以直接使用 np.where
    adj_feature_omics1 = adata_omics1.obsm['adj_feature'].copy().toarray()
    adj_feature_omics2 = adata_omics2.obsm['adj_feature'].copy().toarray()
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

    # 标准化特征图
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1)
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
    }

    return adj

def transform_adjacent_matrix(adjacent):
    """将邻接图从 DataFrame 转换为稀疏矩阵"""
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def preprocess_graph(adj):
    """预处理邻接矩阵，标准化并转换为稀疏张量"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # 加上单位矩阵确保对角元素为1
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将稀疏矩阵转换为 PyTorch 的稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def cog_uncertainty_sample(mu_l, var_l, mu_v, var_v, sample_times=10):
    """不确定性采样"""
    l_list = []
    for _ in range(sample_times):
        l_list.append(reparameterize(mu_l, var_l))
    l_sample = torch.stack(l_list, dim=1)

    v_list = []
    for _ in range(sample_times):
        v_list.append(reparameterize(mu_v, var_v))
    v_sample = torch.stack(v_list, dim=1)
    
    return l_sample, v_sample

def cog_uncertainty_normal(cog_uncertainty_dict):
    """归一化不确定性"""
    sum_uncertainty = sum(cog_uncertainty_dict.values())
    for key in cog_uncertainty_dict:
        cog_uncertainty_dict[key] /= sum_uncertainty
    return cog_uncertainty_dict

def reparameterize(mu, logvar):
    """VAE 采样：重参数化技巧"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3): 
    """构建邻接图，包括共享的 Spatial Graph 和 Feature Graph"""
    # 设置空间图邻接参数
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors = 6  

    # 共享空间邻接图
    cell_positions = adata_omics1.obsm['spatial']
    adj_spatial = construct_graph_by_coordinate(cell_positions, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_spatial
    adata_omics2.uns['adj_spatial'] = adj_spatial  

    # 计算 Feature Graph
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    # 数据标准化
    adata_omics1 = clr_normalize_each_cell(adata_omics1)
    adata_omics2 = clr_normalize_each_cell(adata_omics2)

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    return data

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation", include_self=False):
    """构建基于基因表达的 Feature Graph"""
    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """构建基于空间坐标的 Spatial Graph"""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs):
    """LSI 降维"""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]

def tfidf(X):
    """TF-IDF 归一化"""
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   

def pca(adata, use_reps=None, n_comps=10):
    """PCA 降维"""
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if isinstance(adata.X, sp.csc_matrix) else adata.X)
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e., for each row of .X"""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    if scipy.sparse.issparse(adata.X):
        adata.X = np.apply_along_axis(seurat_clr, 1, adata.X.toarray())
    else:
        adata.X = np.apply_along_axis(seurat_clr, 1, np.array(adata.X))
    
    return adata  

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False