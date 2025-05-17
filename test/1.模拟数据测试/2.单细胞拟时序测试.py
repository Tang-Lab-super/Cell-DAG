import os
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib import rcParams

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
# low dpi (dots per inch) 用于生成 small inline figures
# sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(3, 3), facecolor='white')
## --------------------------------

def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')


# anspath = "/data3/shigw/ST_project/FinalFile/datasets/Simulated_data/"
datapath = "/public3/Shigw/datasets/Simulated_data/"
anspath = f"{datapath}/single_cell_result"
check_path(anspath)

## 单细胞做分化轨迹
nstep = 100
data = pd.read_table(f"{datapath}/sim_path_count_{nstep}.txt", index_col=0).T
metadata = pd.read_table(f"{datapath}/sim_path_metadata_{nstep}.txt", index_col=0)

adata = AnnData(X=data, obs=metadata)
adata.X = adata.X.astype('float64')  # this is not required and results will be comparable without it

sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=40)
sc.tl.umap(adata)

sc.tl.leiden(adata, resolution=0.5)
print(f"{len(adata.obs['leiden'].unique())} clusters")
plt.figure(figsize=(10, 10))
sc.pl.umap(adata, color="leiden")
plt.savefig(f'{anspath}/1.umap_leiden_{nstep}.png')


sc.tl.draw_graph(adata)
# Force-directed graph绘图
plt.figure(figsize=(10, 10))
sc.pl.draw_graph(adata, color='leiden')
plt.savefig(f'{anspath}/2.scdata_draw_graph_{nstep}.png')


## paga轨迹
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_diffmap')
sc.tl.paga(adata, groups='leiden')

plt.figure(figsize=(10, 10))
sc.pl.paga(adata, color=['leiden'])
plt.savefig(f'{anspath}/3.scdata_paga_{nstep}.png')


## dpt拟时序
adata.uns['iroot'] = np.flatnonzero(adata.obs['leiden'] == '2')[0]
sc.tl.dpt(adata)
plt.figure(figsize=(10, 10))
sc.pl.draw_graph(adata, color=['leiden', 'dpt_pseudotime', 'Step'], legend_loc='on data')
plt.savefig(f'{anspath}/4.scdata_dpt_{nstep}.png')


## 计算spearman相关系数
import scipy.stats
scipy.stats.spearmanr(adata.obs['Step'], adata.obs['dpt_pseudotime'])[0]


