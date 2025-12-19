# shigw 2025-3-15
# train CellDAG function

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from anndata import AnnData

from scipy.stats import norm
from scipy.sparse import csr_matrix

from sklearn.neighbors import kneighbors_graph, NearestNeighbors

def get_neighbor(st_data, st_data_use, n_neighbors = 9, n_externs = 10, ntype="extern"):
    """
        kNNGraph_use, indices_use = get_neighbor(st_data, st_data_use, n_neighbors=n_neighbors, ntype="noextern")
        kNNGraph_use, indices_use, st_data_sel = get_neighbor(st_data, st_data, n_neighbors=n_neighbors, n_externs = 10, ntype="extern")
    """
    
    if ntype == "extern":
        all_spots_name = st_data.obs_names.tolist()
        indices = [all_spots_name.index(ci) for ci in st_data_use.obs_names]
        kNNGraph_all = kneighbors_graph(st_data.obsm["spatial"], n_neighbors=n_externs, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
        cell_use = np.unique(np.reshape(np.asarray(kNNGraph_all.col), [len(st_data), n_externs])[indices].flatten())
        st_data_sel = st_data[np.array(all_spots_name)[cell_use]]
        use_spots_name = st_data_sel.obs_names.tolist()
        kNNGraph_use = kneighbors_graph(st_data_sel.obsm["spatial"], n_neighbors=n_neighbors, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
        kNNGraph_use = np.reshape(np.asarray(kNNGraph_use.col), [len(st_data_sel), n_neighbors])
        indices_use = np.array([use_spots_name.index(ci) for ci in st_data_use.obs_names])
        return kNNGraph_use, indices_use, st_data_sel
    else:
        use_spots_name = st_data_use.obs_names.tolist()
        indices_use = np.array([use_spots_name.index(ci) for ci in st_data_use.obs_names])
        kNNGraph_use = kneighbors_graph(st_data_use.obsm["spatial"], n_neighbors=n_neighbors, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
        kNNGraph_use = np.reshape(np.asarray(kNNGraph_use.col), [len(st_data_use), n_neighbors])
        return kNNGraph_use, indices_use


def np_to_torch(img_np, device, requires_grad=False, dtype=torch.float32):
    return torch.tensor(img_np, requires_grad=requires_grad, dtype=dtype, device=device)


def normalize(v):
    return v / torch.linalg.vector_norm(v)

######################### 02 train & Trajectory Inference & Pseudo-time ########################
## stage1 loss function 
def MyLoss_pre(net_input, net_predict, emb, adj, indices_use, flag=None):
    net_input = net_input[indices_use]
    adj = adj[indices_use]

    if flag is not None:
        net_input = net_input[flag]
        net_predict = net_predict[flag]
        adj = adj[flag]

    ## reconstrcut
    MSEloss = torch.nn.MSELoss()            # Loss
    # SmoothL1loss = torch.nn.SmoothL1Loss()            # Loss
    mse_loss = MSEloss(net_input, net_predict)

    # TVloss
    emb_TV = emb[adj]       # n * k * m
    emb_TV = emb_TV - emb_TV[:, :1, :]
    emb_TV = torch.norm(emb_TV, p=1, dim=1)
    emb_TV = torch.norm(emb_TV, p=2) / adj.shape[0]
    # emb_TV = torch.norm(emb_TV, p=1, dim=1)
    # emb_TV = torch.norm(emb_TV, p=2)

    return mse_loss, emb_TV
    # return mse_loss

## stage2 loss function 
def MyLoss_post(net_input, net_predict, emb, h_val, adj, indices_use, flag=None, seed=24):
    net_input = net_input[indices_use]
    adj = adj[indices_use]

    if flag is not None:
        net_input = net_input[flag]
        net_predict = net_predict[flag]
        adj = adj[flag]

    ## reconstrcut
    MSEloss = torch.nn.MSELoss()                            # Loss
    # SmoothL1loss = torch.nn.SmoothL1Loss()                  # Loss
    mse_loss = MSEloss(net_input, net_predict)

    # TVloss
    emb_TV = emb[adj]       # n * k * m
    emb_TV = emb_TV - emb_TV[:, :1, :]
    emb_TV = torch.norm(emb_TV, p=1, dim=1)
    emb_TV = torch.norm(emb_TV, p=2) / adj.shape[0]

    return mse_loss, emb_TV
    # return mse_loss


## Trajectory Inference
"""
    Obtaining differentiation trajectories based on cell probability transition matrix,
    refer to https://github.com/yzf072/SpaTrack
"""

def get_velocity_grid(
    adata,
    P: np.ndarray,
    V: np.ndarray,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
) -> tuple:
    """
    Convert cell velocity to grid velocity for streamline display

    The visualization of vector field borrows idea from scTour: https://github.com/LiQian-XC/sctour/blob/main/sctour.

    Parameters
    ----------
    P
        The position of cells.
    V
        The velocity of cells.
    smooth
        The factor for scale in Gaussian pdf.
        (Default: 0.5)
    density
        grid density
        (Default: 1.0)
    Returns
    ----------
    tuple
        The embedding and unitary displacement vectors in grid level.
    """
    grids = []
    for dim in range(P.shape[1]):
        m, M = np.min(P[:, dim]), np.max(P[:, dim])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(grid_num * density))
        grids.append(gr)

    meshes = np.meshgrid(*grids)
    P_grid = np.vstack([i.flat for i in meshes]).T

    n_neighbors = int(P.shape[0] / grid_num)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(P)
    dists, neighs = nn.kneighbors(P_grid)

    scale = np.mean([grid[1] - grid[0] for grid in grids]) * smooth
    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]

    P_grid = np.stack(grids)
    ns = P_grid.shape[1]
    V_grid = V_grid.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid * V_grid).sum(0))
    min_mass = 1e-5
    min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
    cutoff = mass < min_mass

    V_grid[0][cutoff] = np.nan

    adata.uns["P_grid"] = P_grid
    adata.uns["V_grid"] = V_grid

    return P_grid, V_grid

def get_neigh_trans(
    adata: AnnData, basis: str, n_neigh_pos: int = 10, n_neigh_gene: int = 0
):
    """
    Get the transport neighbors from two ways, position and/or gene expression

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    basis
        The basis used in visualizing the cell position.
    n_neigh_pos
        Number of neighbors based on cell positions such as spatial or umap coordinates.
        (Default: 10)
    n_neigh_gene
        Number of neighbors based on gene expression (PCA).
        (Default: 0)

    Returns
    -------
    :class:`~scipy.sparse._csr.csr_matrix`
        A sparse matrix composed of transition probabilities of selected neighbor cells.
    """
    if n_neigh_pos == 0 and n_neigh_gene == 0:
        raise ValueError(
            "the number of position neighbors and gene neighbors cannot be zero at the same time."
        )

    if n_neigh_pos:
        nn = NearestNeighbors(n_neighbors=n_neigh_pos, n_jobs=-1)
        nn.fit(adata.obsm[basis])
        dist_pos, neigh_pos = nn.kneighbors(adata.obsm[basis])
        dist_pos = dist_pos[:, 1:]
        neigh_pos = neigh_pos[:, 1:]

        neigh_pos_list = []
        for i in range(adata.n_obs):
            idx = neigh_pos[i]  # embedding上的邻居
            idx2 = neigh_pos[idx]  # embedding上邻居的邻居
            idx2 = np.setdiff1d(idx2, i)

            neigh_pos_list.append(np.unique(np.concatenate([idx, idx2])))
            # neigh_pos_list.append(idx)

    if n_neigh_gene:
        if "X_pca" not in adata.obsm:
            print("X_pca is not in adata.obsm, automatically do PCA first.")
            sc.tl.pca(adata)
            
        sc.pp.neighbors(
            adata, use_rep="X_pca", key_added="X_pca", n_neighbors=n_neigh_gene
        )

        neigh_gene = adata.obsm["distances"].indices.reshape(
            -1, adata.uns["neighbors"]["params"]["n_neighbors"] - 1
        )

    indptr = [0]
    indices = []
    csr_data = []
    count = 0
    for i in range(adata.n_obs):
        if n_neigh_pos == 0:
            n_all = neigh_gene[i]
        elif n_neigh_gene == 0:
            n_all = neigh_pos_list[i]
        else:
            n_all = np.unique(np.concatenate([neigh_pos_list[i], neigh_gene[i]]))
        count += len(n_all)
        indptr.append(count)
        indices.extend(n_all)
        csr_data.extend(
            adata.obsm["trans"][i][n_all]
            / (adata.obsm["trans"][i][n_all].sum())  # normalize
        )

    trans_neigh_csr = csr_matrix(
        (csr_data, indices, indptr), shape=(adata.n_obs, adata.n_obs)
    )

    return trans_neigh_csr

def get_velocity(
    adata: AnnData,
    basis: str,
    n_neigh_pos: int = 10,
    n_neigh_gene: int = 0,
    grid_num=50,
    smooth=0.5,
    density=1.0,
) -> tuple:
    adata.obsm["trans_neigh_csr"] = get_neigh_trans(adata, basis, n_neigh_pos, n_neigh_gene)

    position = adata.obsm[basis]
    V = np.zeros(position.shape)  # 速度为2维

    for cell in range(adata.n_obs):  # 循环每个细胞
        cell_u = 0.0  # 初始化细胞速度
        cell_v = 0.0
        x1 = position[cell][0]  # 初始化细胞坐标
        y1 = position[cell][1]
        for neigh in adata.obsm["trans_neigh_csr"][cell].indices:  # 针对每个邻居
            p = adata.obsm["trans_neigh_csr"][cell, neigh]
            if (
                adata.obs["ptime"][neigh] < adata.obs["ptime"][cell]
            ):  # 若邻居的ptime小于当前的，则概率反向
                p = -p

            x2 = position[neigh][0]
            y2 = position[neigh][1]

            # 正交向量确定速度方向，乘上概率确定速度大小
            sub_u = p * (x2 - x1) / (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            sub_v = p * (y2 - y1) / (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            cell_u += sub_u
            cell_v += sub_v
        V[cell][0] = cell_u / adata.obsm["trans_neigh_csr"][cell].indptr[1]
        V[cell][1] = cell_v / adata.obsm["trans_neigh_csr"][cell].indptr[1]
    adata.obsm["velocity_" + basis] = V
    print(f"The velocity of cells store in 'velocity_{basis}'.")

    P_grid, V_grid = get_velocity_grid(
        adata,
        P=position,
        V=adata.obsm["velocity_" + basis],
        grid_num=grid_num,
        smooth=smooth,
        density=density,
    )
    return P_grid, V_grid

## cal Pseudo-time
def get_ptime(adata):
    select_trans = adata.obsm["trans"][adata.obs['start_cluster']==1]
    cell_tran = np.sum(select_trans, axis=0)
    adata.obs["tran"] = cell_tran
    cell_tran_sort = list(np.argsort(cell_tran))
    # cell_tran_sort = cell_tran_sort[::-1]

    ptime = pd.Series(dtype="float32", index=adata.obs.index)
    for i in range(adata.n_obs):
        ptime[cell_tran_sort[i]] = i / (adata.n_obs - 1)

    adata.obs['ptime'] = ptime.values
    if adata.obs['ptime'][adata.obs['start_cluster']==1].mean() > 0.5:
        adata.obs['ptime'] = 1 - adata.obs['ptime']
    return adata

