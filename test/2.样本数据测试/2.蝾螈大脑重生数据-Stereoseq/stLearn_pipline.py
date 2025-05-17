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

import seaborn as sns
def plot_feature(xy1, xy2,  value, title):
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=5)
    sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o',
                    c = value, s=5,  cmap='Spectral_r', legend = True)
    # plt.title(title)
    plt.axis('off')

def plot_cluster(xy1, st_data_sel, title):
    mycolor = [
        "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
        "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
        "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
        "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
    ]
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=5)
    for idx, ci in enumerate(st_data_sel.obs.emb_cluster.unique().tolist()):
        subda = st_data_sel[st_data_sel.obs.emb_cluster == ci, :]
        sns.scatterplot(x=subda.obsm['spatial'][:, 0], y=subda.obsm['spatial'][:, 1], marker='o', c=mycolor[idx], s=5)
        plt.text(subda.obsm['spatial'][:, 0].mean(), subda.obsm['spatial'][:, 1].mean(), str(ci), fontsize=8)
    # plt.title(title)
    plt.axis('off')

def plot_spatial(st_data, st_data_sel, mode="time", value=None, figsize=(5, 5), title=None, savename='./fig.pdf'):

    assert value is not None, "value is None."
    assert title is not None, "title is None."

    if mode=="time":
        assert value is not None, "value is None."
    elif mode=="cluster":
        assert "emb_cluster" in st_data_sel.obs.columns, "emb_cluster is empty."

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    plt.subplot(1,1,1)
    if mode=="time":
        plot_feature(st_data.obsm['spatial'], st_data_sel.obsm['spatial'],  value, title)
    elif mode == "cluster":
        plot_cluster(st_data.obsm['spatial'], st_data_sel, title)
    plt.tight_layout()
    plt.savefig(savename)

###################%% version control
sample_name = "stLearn"
root_path = "/public3/Shigw/"
data_folder = f"{root_path}/datasets/Stereo-seq/regeneration/"
save_folder = f"{data_folder}/results/{sample_name}"
check_path(save_folder)

setup_seed()
######################### 2. 预处理，挑选候选细胞 ########################
# 数据处理 归一化和scale
st_data = sc.read(data_folder + "/st_data.h5")
st_data.obs['imagerow'] = st_data.obsm['spatial'][:, 0]
st_data.obs['imagecol'] = st_data.obsm['spatial'][:, 1]

n_genes = 500
df_data_pi = pd.read_csv(f"{data_folder}/results/PI_result.csv", index_col=0)
gene_flag = df_data_pi['PI'].nlargest(n_genes).index
st_data = st_data[:, gene_flag]

sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data)

st_data_use = st_data.copy()

## 首先聚类
save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
check_path(save_folder_cluster)

st.em.run_pca(st_data_use, n_comps=50, random_state=24)
st.pp.neighbors(st_data_use, n_neighbors=30, random_state=24)
st.tl.clustering.louvain(st_data_use, random_state=24)
st_data_use.obs['emb_cluster'] = st_data_use.obs['louvain']

sc.tl.umap(st_data_use)
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(st_data_use, color="louvain", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_cluster}/1.umap_cluster_stage1.png", dpi=600)

## 绘制类别选取起点
# plt.close('all')
# fig = plt.figure(figsize=(10, 10))
# plt.subplot(1, 1, 1)
# ax = sc.pl.embedding(st_data_use, basis="spatial", color="louvain", size=15, s=10, show=False, title='louvain')
# plt.savefig(f"{save_folder_cluster}/2.spatial_cluster.png", dpi=600)

plot_spatial(
    st_data, st_data_use, mode="cluster",
    value=st_data_use.obs['emb_cluster'], title="emb_cluster",
    savename=f"{save_folder_cluster}/2.spatial_cluster.png"
)

## 然后选取起点进行拟时序
save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
check_path(save_folder_trajectory)

st_data_use.uns["iroot"] = st.spatial.trajectory.set_root(st_data_use, use_label="louvain", cluster='2')
st.spatial.trajectory.pseudotime(st_data_use, eps=50, use_rep="X_pca", use_label="louvain")
st_data_use.obs['dpt_pseudotime'] = 1 - st_data_use.obs['dpt_pseudotime']


## 绘制拟时序
plt.close('all')
fig = plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(st_data_use, color="dpt_pseudotime", color_map='Spectral_r')
plt.savefig(f"{save_folder_trajectory}/2.umap_dpt_pseudotime.pdf")

plot_spatial(
    st_data, st_data_use, mode="time",
    value=st_data_use.obs['dpt_pseudotime'], title="ptime",
    savename=f"{save_folder_trajectory}/3.spatial_Pseudotime.pdf"
)

st_data_use.obs['ptime'] = st_data_use.obs['dpt_pseudotime']
st_data_use.obs.to_csv(f"{save_folder_trajectory}/df_obs_ptime.csv")
