#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_embeddings(pkl_path):
    """加载 pickle 文件中的嵌入数据"""
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def evaluate_embedding(X, n_clusters=3):
    """
    对给定的嵌入 X 进行评测：
      - 使用 KMeans 聚类并计算 Silhouette Score；
      - 使用 TSNE 将 X 降至 2D，并绘制散点图；
      - 如果数据退化（方差太低或只有一个聚类），则给出警告并跳过相关计算。
    """
    # 检查 NaN 并处理
    if np.isnan(X).any():
        print("Warning: Detected NaN values in embedding, replacing NaN with 0.")
        X = np.nan_to_num(X, nan=0.0)
    
    # 检查数据方差是否足够（如果所有样本几乎相同，则聚类和 TSNE 都没有意义）
    if np.std(X) < 1e-6:
        print("Warning: Embedding variance is too low, skipping clustering and TSNE visualization.")
        return None
    
    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        print("Warning: Only one cluster found. Silhouette score cannot be computed.")
        sil_score = None
    else:
        sil_score = silhouette_score(X, labels)
        print("Silhouette Score for KMeans with {} clusters: {:.4f}".format(n_clusters, sil_score))
    
    # TSNE 降维可视化
    try:
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
    except Exception as e:
        print("TSNE failed with error:", e)
        # 若 TSNE 失败，则使用全零的矩阵（不推荐，但避免程序崩溃）
        X_tsne = np.zeros((X.shape[0], 2))
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title("t-SNE Visualization of SpatialGlue Embedding\n(KMeans Clusters)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    plt.tight_layout()
    plt.show()
    
    return sil_score

def main():
    # 指定嵌入文件路径
    pkl_path = 'resultsOrigin/embeddings.pkl'
    
    # 加载嵌入数据
    embeddings = load_embeddings(pkl_path)
    print("Embedding keys:", embeddings.keys())
    for key in embeddings:
        print("Key: {}, Shape: {}".format(key, np.array(embeddings[key]).shape))
    
    # 使用 'SpatialGlue' 嵌入进行评测
    if 'SpatialGlue' not in embeddings:
        raise KeyError("The key 'SpatialGlue' is not found in the embeddings file.")
    
    X = np.array(embeddings['SpatialGlue'])
    print("Using SpatialGlue embedding with shape:", X.shape)
    
    # 调用评测函数（此处聚类数可根据需要修改）
    evaluate_embedding(X, n_clusters=3)

if __name__ == '__main__':
    main()