import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .preprocess import pca

# ========== Clustering with Mclust ==========
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res.astype('int').astype('category')
    return adata

# ========== Modified Clustering (Supports EURO & VAE) ==========
def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', 
               start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20, use_vae=False):
    """
    Spatial clustering based on latent representation with EURO + VAE support.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int
        Number of clusters.
    key : string
        The key of the input representation in adata.obsm.
    method : string
        Clustering tool: 'mclust', 'leiden', 'louvain'.
    start : float
        Start value for searching resolution (for leiden/louvain).
    end : float 
        End value for searching resolution (for leiden/louvain).
    increment : float
        Step size for increasing resolution.
    use_pca : bool
        Whether to use PCA for dimension reduction.
    use_vae : bool
        Whether to use a VAE-based latent space.

    Returns
    -------
    None.
    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    rep_key = key + '_vae' if use_vae else key
    rep_key = rep_key + '_pca' if use_pca else rep_key

    if method == 'mclust':
       adata = mclust_R(adata, used_obsm=rep_key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
       
    elif method in ['leiden', 'louvain']:
       res = search_res(adata, n_clusters, use_rep=rep_key, method=method, start=start, end=end, increment=increment)
       if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           adata.obs[add_key] = adata.obs['leiden']
       elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           adata.obs[add_key] = adata.obs['louvain']

# ========== Adaptive Resolution Search (Supports VAE) ==========
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    """
    Search for optimal resolution parameter.

    Parameters
    ----------
    adata : anndata
    n_clusters : int
    method : string
    use_rep : string
    start : float
    end : float 
    increment : float

    Returns
    -------
    res : float
    """
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 

        print(f'resolution={res}, cluster number={count_unique}')
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution not found. Try a bigger range or smaller step."   
    return res     

# ========== EURO Mechanism: Adjust Weight Based on Uncertainty ==========
def apply_euro_weight(alpha, cog_uncertainty_dict):
    """
    Adjusts weight values based on the uncertainty from UDR.

    Parameters
    ----------
    alpha : np.array
        Initial weight values.
    cog_uncertainty_dict : dict
        Uncertainty values for RNA and Protein.

    Returns
    -------
    adjusted_alpha : np.array
    """
    weight_rna = 1 / (1 + cog_uncertainty_dict['protein'])
    weight_protein = 1 / (1 + cog_uncertainty_dict['rna'])

    adjusted_alpha = np.stack([alpha[:, 0] * weight_rna, alpha[:, 1] * weight_protein], axis=1)
    return adjusted_alpha / np.sum(adjusted_alpha, axis=1, keepdims=True)

# ========== Enhanced Weight Visualization (Supports EURO) ==========
def plot_weight_value(alpha, label, cog_uncertainty_dict=None, modality1='mRNA', modality2='protein'):
    """
    Plot weight values with EURO uncertainty correction.

    Parameters
    ----------
    alpha : np.array
    label : list
    cog_uncertainty_dict : dict
    modality1 : string
    modality2 : string

    Returns
    -------
    None.
    """
    if cog_uncertainty_dict:
        alpha = apply_euro_weight(alpha, cog_uncertainty_dict)

    df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
    df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
    df['label'] = label
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']

    ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                        split=True, inner="quart", linewidth=1)
    ax.set_title(modality1 + ' vs ' + modality2)

    plt.tight_layout()
    plt.show()