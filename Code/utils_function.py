import os 
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import scipy
import DAGAST as nu
from anndata import AnnData
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()    

mycolor = [
    "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
    "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
    "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
]


######################### 01 绘图 ########################
def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')

def plot_feature(xy1, xy2, value, title="feature plot", pointsize=5):
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=pointsize)
    sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o',
                    c = value, s=pointsize,  cmap='Spectral_r', legend = True)
    # plt.title(title)
    plt.axis('off')


def plot_cluster(xy1, st_data_sel, key="cluster", title="cluster plot", pointsize=5):
    mycolor = [
        "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
        "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
        "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
        "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
    ]
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=pointsize)
    for idx, ci in enumerate(st_data_sel.obs[key].unique().tolist()):
        if ci == -1:
            continue
        subda = st_data_sel[st_data_sel.obs[key] == ci, :]
        sns.scatterplot(x=subda.obsm['spatial'][:, 0], y=subda.obsm['spatial'][:, 1], marker='o', c=mycolor[idx], s=pointsize)
        plt.text(subda.obsm['spatial'][:, 0].mean(), subda.obsm['spatial'][:, 1].mean(), str(ci), fontsize=8)
    # plt.title(title)
    plt.axis('off')


def plot_spatial_complex(
    st_data, st_data_sel, mode="time", value=None, key="",
    figsize=(5, 5), title=None, pointsize=5, savename='./fig.pdf'):

    if mode=="time":
        assert value is not None, "value is None."
    elif mode=="cluster":
        assert key in st_data_sel.obs.columns, f"{key} is empty."

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    plt.subplot(1,1,1)
    if mode=="time":
        plot_feature(st_data.obsm['spatial'], st_data_sel.obsm['spatial'], value, title, pointsize)
    elif mode == "cluster":
        plot_cluster(st_data.obsm['spatial'], st_data_sel, key, title, pointsize)
    plt.tight_layout()
    plt.savefig(savename, dpi=300)

# plot_spatial_complex(st_data, st_data, mode="cluster", key="",
#     figsize=(5, 5), title=None, pointsize=5, savename='./fig.pdf')

# def plot_spatial(st_data, key="", figsize=(5, 5), title=None, pointsize=5, savename='./fig.pdf'):
#     plt.close('all')
#     fig = plt.figure(figsize=figsize)
#     plt.subplot(1, 1, 1)
#     plot_cluster(st_data.obsm['spatial'], st_data, key, title, pointsize)
#     plt.tight_layout()
#     plt.savefig(savename, dpi=300)


def plot_spatial_gene(st_data, key="", figsize=(5, 5), title=None, pointsize=5, savename='./fig.pdf'):
    plt.close('all')
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 1, 1)
    plot_feature(st_data.obsm['spatial'], st_data.obsm['spatial'], st_data.obs[key], title=title, pointsize=5)
    plt.tight_layout()
    plt.savefig(savename, dpi=300)


## 构建细胞邻近图
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


######################### 02 空间聚类与轨迹推断 ########################
# net_input, net_predict, emb, dc_predict, start, adj, flag = net_input[indices_use], net_predict, emb, dc_predict, start, model.adjmat[indices_use], flag
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

# net_input, net_predict, emb, dc_predict, start, adj, flag = net_input[indices_use], net_predict, emb, dc_predict, start, model.adjmat[indices_use], flag
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

## 拟时序计算
def runDiffusionMaps(data_df, n_components=10, knn=30, alpha=0, random_state=21):
    np.random.seed(random_state)

    # Determine the kernel
    N = data_df.shape[0]

    if(type(data_df).__module__ == np.__name__):
        data_df = pd.DataFrame(data_df)

    if not scipy.sparse.issparse(data_df):
        # print("Determing nearest neighbor graph...")
        temp = sc.AnnData(data_df)
        sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn, random_state=random_state)
        kNN = temp.obsp['distances']

        # Adaptive k
        adaptive_k = int(np.floor(knn / 3))
        adaptive_std = np.zeros(N)

        for i in np.arange(len(adaptive_std)):
            adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[adaptive_k - 1]

        x, y, dists = scipy.sparse.find(kNN)        # Kernel

        # X, y specific stds
        dists = dists / adaptive_std[x]
        W = scipy.sparse.csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

        # Diffusion components
        kernel = W + W.T
    else:
        kernel = data_df

    # Markov
    D = np.ravel(kernel.sum(axis=1))

    if alpha > 0:
        D[D != 0] = D[D != 0] ** (-alpha)       # L_alpha
        mat = scipy.sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)
        D = np.ravel(kernel.sum(axis=1))

    D[D != 0] = 1 / D[D != 0]
    T = scipy.sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

    # Eigen value dcomposition
    D, V = scipy.sparse.linalg.eigs(T, n_components, tol=1e-4, maxiter=500)
    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    # Normalize
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    # Create are results dictionary
    res = {"T": T, "EigenVectors": V, "EigenValues": D}
    res["EigenVectors"] = pd.DataFrame(res["EigenVectors"])
    if not scipy.sparse.issparse(data_df):
        res["EigenVectors"].index = data_df.index
    res["EigenValues"] = pd.Series(res["EigenValues"])
    res["kernel"] = kernel

    return res

## 计算拟时序
# def get_ptime(adata):
#     select_trans = adata.obsm["trans"][adata.obs['start_cluster']==1]
#     cell_tran = np.sum(select_trans, axis=0)
#     adata.obs["tran"] = cell_tran
#     cell_tran_sort = list(np.argsort(cell_tran))
#     cell_tran_sort = cell_tran_sort[::-1]

#     ptime = pd.Series(dtype="float32", index=adata.obs.index)
#     for i in range(adata.n_obs):
#         ptime[cell_tran_sort[i]] = i / (adata.n_obs - 1)

#     adata.obs['ptime'] = ptime.values
#     return adata

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


## 得到分化轨迹
from scipy.sparse import csr_matrix
from scipy.stats import norm
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


######################### 03 置换检验 ########################
def kl_divergence(P, Q, epsilon=None):
    """
    计算两个概率分布 P 和 Q 的 KL 散度。
    :param P: 数组，表示分布 P，必须归一化为概率分布
    :param Q: 数组，表示分布 Q，必须归一化为概率分布
    :return: KL 散度的值
    """
    # 确保输入为 numpy 数组
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    
    # 归一化为概率分布
    P /= np.sum(P)
    Q /= np.sum(Q)
    
    if epsilon != None:                 # 避免 log(0) 或除以 0，添加一个很小的数
        P = np.clip(P, epsilon, 1)
        Q = np.clip(Q, epsilon, 1)

    return np.sum(P * np.log(P / Q))    # 计算 KL 散度


# ### 对基因、基因集进行permutation ### single gene
def permutation_singlegene_celltype(
        model, st_data_use, trj_ori, gene_use=None, cell_use=None,
        n_permu=9, epsilon=1e-16, seed=42, device=None
    ):

    model.to(device)
    model.eval()

    cell_flag = st_data_use.obs_names.isin(cell_use)

    result_permu = []
    result_records = []

    with tqdm(total=len(gene_use)) as t:  
        with torch.no_grad():
            for gi, genei in enumerate(gene_use):
                # print(f"{gi} {genei}")
                ans_permu = []
                ans_dcrecord = []
                np.random.seed(seed)
                for i in range(n_permu):
                    ExpData = st_data_use.X.copy()

                    permu_data = ExpData[cell_flag, gi].copy()
                    np.random.shuffle(permu_data)
                    ExpData[cell_flag, gi] = permu_data
                    ExpData = torch.from_numpy(ExpData).to(dtype=torch.float, device=device)

                    emb, exp_predict, h_val, grad, trj = model(ExpData, iftrj=True)
                    trj = trj.detach().cpu().numpy()
                    sim1 = kl_divergence(trj_ori[cell_flag, :], trj[cell_flag, :], epsilon)

                    ans_permu.append(sim1)
                    # ans_dcrecord.append(trj)

                result_permu.append(ans_permu)
                # result_records.append(ans_dcrecord)
                t.update(1)

        result_permu = np.asarray(result_permu)
        # result_records = np.asarray(result_records)
        # return result_permu, result_records
    return result_permu


