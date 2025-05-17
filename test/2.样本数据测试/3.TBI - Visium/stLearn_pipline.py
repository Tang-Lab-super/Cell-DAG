import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData
import stlearn as st
st.settings.set_figure_params(dpi=300)

import warnings
warnings.warn("This is a warning message.")

def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')

def setup_seed(seed=24):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

###################%% version control
sample_name = "stLearn"
root_path = "/public3/Shigw/datasets/Visium/TBI_stlearn/GSE236171_RAW/"

data_folder = f"{root_path}/VLP4_C1_Visium/"
save_folder = f"{data_folder}/results/{sample_name}"
check_path(f"{data_folder}/results/")
check_path(save_folder)

setup_seed()
######################### 1. 读取数据，进行必要的处理 ########################
st_data = sc.read_visium(data_folder)
st_data.var_names_make_unique()
st_data.obs[['imagerow', 'imagecol']] = st_data.obs[['array_row', 'array_col']]

flag = st_data.var_names.isin(['Fcrls', 'Tmem119'])
tarexp = st_data.X.A[:, flag].sum(1)
cell_flag = tarexp > 1
# st_data.obs['flag'] = cell_flag
st_data.obs['flag'] = 0
st_data.obs.loc[cell_flag, 'flag'] = 1


plt.close('all')
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(st_data, img_key="hires", color="flag")
plt.savefig(f"{save_folder}/1.spatial_target_cell.pdf")



n_genes = 500
sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.highly_variable_genes(st_data, n_top_genes=n_genes)
sc.pp.scale(st_data)
st_data = st_data[:, st_data.var['highly_variable'].values]

st_data_use = st_data[cell_flag].copy()


## 首先聚类
save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
check_path(save_folder_cluster)

st.em.run_pca(st_data_use, n_comps=50, random_state=0)
st.pp.neighbors(st_data_use, n_neighbors=30, random_state=0)
st.tl.clustering.louvain(st_data_use, random_state=0)

sc.tl.umap(st_data_use)

plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(st_data_use, color="louvain", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_cluster}/1.umap_cluster_stage1.png", dpi=600)


plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.spatial(st_data_use, img_key="hires", color=["louvain"])
plt.axis('off')
plt.savefig(f"{save_folder_cluster}/2.spatial_cluster_stage1.png", dpi=600, bbox_inches='tight')



## 然后选取起点进行拟时序
save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
check_path(save_folder_trajectory)

st_data_use.uns["iroot"] = st.spatial.trajectory.set_root(st_data_use, use_label="louvain", cluster='1')
st.spatial.trajectory.pseudotime(st_data_use, eps=50, use_rep="X_pca", use_label="louvain")

st_data_use.obs['dpt_pseudotime'] = 1 - st_data_use.obs['dpt_pseudotime']

## 绘制拟时序
sc.tl.umap(st_data_use)
plt.close('all')
fig = plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(st_data_use, color="dpt_pseudotime", color_map='Spectral_r')
plt.savefig(f"{save_folder_trajectory}/2.umap_dpt_pseudotime.pdf")

plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.spatial(st_data_use, img_key="hires", color=["dpt_pseudotime"], cmap='Spectral_r')
# sc.pl.spatial(st_data_use, img_key="hires", color=["ptime"], cmap='Spectral_r')
plt.axis('off')
plt.savefig(f"{save_folder_trajectory}/3.spatial_Pseudotime.png", dpi=600, bbox_inches='tight')


### 3.4 cluster 这里确实要进一步通过聚类分析验证准确性，同时也算一些结果
## umap - cluster
sc.tl.leiden(st_data_use, resolution=0.5)         # res = 0.1
st_data_use.obs['emb_cluster'] = st_data_use.obs['leiden'].values
print(f"{len(st_data_use.obs['leiden'].unique())} clusters")

mycolor = [
   "#E31A1C",  "#33A02C", "#6A3D9A", "#FF7F00", "#1F78B4",  "#B15928", "#A6CEE3", "#B2DF8A", 
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
    "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
    "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
]
umap_color = {str(c) : mycolor[idx] for idx, c in enumerate(st_data_use.obs['leiden'].unique())}
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
# ax = sc.pl.umap(adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
ax = sc.pl.umap(st_data_use, color="leiden", palette=umap_color, legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_trajectory}/4.embedding_umap_cluster_DAGAST.pdf")


plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.spatial(st_data_use, img_key="hires", color="emb_cluster", palette=umap_color)
plt.axis('off')
plt.savefig(f"{save_folder_trajectory}/5.spatial_cluster_DAGAST.png", dpi=600, bbox_inches='tight')



