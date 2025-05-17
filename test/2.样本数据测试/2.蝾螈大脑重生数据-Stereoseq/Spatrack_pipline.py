import os
import numpy as np
import pandas as pd
import scanpy as sc
import spaTrack as spt
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
plt.rcParams["figure.dpi"] = 300  # 分辨率

def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')

def setup_seed(seed=24):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

###################%% version control
sample_name = "Spatrack"
root_path = "/public3/Shigw/"
# root_path = "/data3/shigw/ST_project/FinalFile/"
# root_path = "/data01/suziyi/GWCode/"
data_folder = f"{root_path}/datasets/Stereo-seq/regeneration/"
save_folder = f"{data_folder}/results/{sample_name}"
check_path(save_folder)

setup_seed()

######################### 2. 预处理，挑选候选细胞 ########################
# 数据处理 归一化和scale
st_data = sc.read(data_folder + "/st_data.h5")

st_data.obs['imagerow'] = st_data.obsm['spatial'][:, 0]
st_data.obs['imagecol'] = st_data.obsm['spatial'][:, 1]
# sc.pp.filter_cells(st_data, min_genes=200)    
# sc.pp.filter_genes(st_data, min_cells=10)

n_genes = 500
df_data_pi = pd.read_csv(f"{data_folder}/results/PI_result.csv", index_col=0)
gene_flag = df_data_pi['PI'].nlargest(n_genes).index
st_data = st_data[:, gene_flag]

# sc.pp.filter_cells(st_data, min_genes=200)    
# sc.pp.filter_genes(st_data, min_cells=10)

sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data)


# celltypes = ['Spinal cord', 'NMP']      # permutation cells
# st_data_use = st_data[st_data.obs.celltypes.isin(celltypes), :].copy()
st_data_use = st_data.copy()


## 首先聚类
save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
check_path(save_folder_cluster)

sc.pp.neighbors(st_data_use, use_rep='X', n_neighbors=30)
sc.tl.umap(st_data_use)
sc.tl.leiden(st_data_use, resolution=1.0)         # res = 0.1
print(f"{len(st_data_use.obs['leiden'].unique())} clusters")

mycolor = [
    "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
    "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
    "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
]

umap_color = {str(c) : mycolor[idx] for idx, c in enumerate(st_data_use.obs['leiden'].unique())}
st_data_use.uns['cluster_colors'] = umap_color


plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(st_data_use, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_cluster}/1.umap_cluster_stage1.png", dpi=600)



plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.embedding(st_data_use, basis="spatial", color="leiden",size=15, s=10, show=False, title='leiden')
plt.savefig(f"{save_folder_cluster}/2.spatial_cluster.png", dpi=600)


## 绘制类别选取起点
st_data_use.obs['cluster'] = st_data_use.obs['leiden']
# st_data = spt.assess_start_cluster(st_data)
start_cells = st_data_use.obs['cluster'] == '2'

## 然后选取起点进行拟时序
save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
check_path(save_folder_trajectory)

st_data_use.obsm['X_spatial'] = st_data_use.obsm['spatial']
st_data_use.obsp["trans"] = spt.get_ot_matrix(st_data_use, data_type="spatial",alpha1=0.5,alpha2=0.5)
st_data_use.obs["ptime"] = spt.get_ptime(st_data_use, start_cells)

st_data_use.uns["E_grid"], st_data_use.uns["V_grid"] = spt.get_velocity(st_data_use, basis="spatial", n_neigh_pos=50)

## 绘制拟时序
sc.tl.umap(st_data_use)
plt.close('all')
fig = plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(st_data_use, color="ptime", color_map='Spectral_r')
plt.savefig(f"{save_folder_trajectory}/3.umap_ptime.pdf")


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

plot_spatial(
    st_data, st_data_use, mode="time",
    value=st_data_use.obs['ptime'], title="ptime",
    savename=f"{save_folder_trajectory}/4.spatial_Pseudotime.pdf"
)


## 绘制分化轨迹
plt.close('all')
fig, axs = plt.subplots(figsize=(5, 5))
sc.pl.embedding(st_data_use, basis='spatial',show=False,title=' ',color='cluster',ax=axs,frameon=False,palette='tab20b',legend_fontweight='normal',alpha=0.1,size=100)
axs.quiver(st_data_use.uns['E_grid'][0],st_data_use.uns['E_grid'][1],st_data_use.uns['V_grid'][0],st_data_use.uns['V_grid'][1],scale=0.008)
plt.savefig(f"{save_folder_trajectory}/5.trajectory_ptime.pdf")

plt.close('all')
fig,axs=plt.subplots(ncols=1,nrows=1,figsize=(6,6))
ax = sc.pl.embedding(st_data_use,  basis='spatial',show=False,title=' ',color='cluster',ax=axs,frameon=False,palette='tab20b',legend_fontweight='normal',alpha=0.8,size=100)
ax.streamplot(st_data_use.uns['E_grid'][0], st_data_use.uns['E_grid'][1], st_data_use.uns['V_grid'][0], st_data_use.uns['V_grid'][1],density=1.8,color='black',linewidth=2.5,arrowsize=1.5)
plt.savefig(f"{save_folder_trajectory}/6.streamplot.pdf")

st_data_use.obs.to_csv(f"{save_folder_trajectory}/df_obs_ptime.csv")

# ### 3.4 cluster 这里确实要进一步通过聚类分析验证准确性，同时也算一些结果
# ## umap - cluster
# sc.tl.leiden(st_data_use, resolution=0.5)         # res = 0.1
# st_data_use.obs['emb_cluster'] = st_data_use.obs['leiden'].values
# print(f"{len(st_data_use.obs['leiden'].unique())} clusters")

# mycolor = [
#    "#E31A1C",  "#33A02C", "#6A3D9A", "#FF7F00", "#1F78B4",  "#B15928", "#A6CEE3", "#B2DF8A", 
#     "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
#     "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
#     "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
# ]
# umap_color = {str(c) : mycolor[idx] for idx, c in enumerate(st_data_use.obs['leiden'].unique())}
# plt.close('all')
# fig = plt.figure(figsize=(10, 10))
# plt.subplot(1, 1, 1)
# # ax = sc.pl.umap(adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
# ax = sc.pl.umap(st_data_use, color="leiden", palette=umap_color, legend_loc='on data', legend_fontweight='normal')
# plt.savefig(f"{save_folder_trajectory}/5.embedding_umap_cluster_DAGAST.pdf")


# plt.close('all')
# plt.rcParams["figure.figsize"] = (5, 5)
# sc.pl.spatial(st_data_use, img_key="hires", color="emb_cluster", palette=umap_color)
# plt.axis('off')
# plt.savefig(f"{save_folder_trajectory}/5.spatial_cluster_DAGAST.png", dpi=600, bbox_inches='tight')


# plot_spatial(
#     st_data, st_data_use, mode="cluster",
#     value=st_data_use.obs['emb_cluster'], title="emb_cluster",
#     savename=f"{save_folder_trajectory}/5.spatial_cluster_DAGAST.png"
# )

import seaborn as sns
xy1 = st_data.obsm['spatial']
xy2 = st_data_use.obsm['spatial']
plt.close('all')
fig, axs = plt.subplots(figsize=(5, 5))
# sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1], color = (207/255, 185/255, 151/255, 1), s=5)
sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o', c = st_data_use.obs['ptime'], 
    s=20, cmap='Spectral_r', legend = False, alpha=0.25)
axs.quiver(st_data_use.uns['E_grid'][0],st_data_use.uns['E_grid'][1],st_data_use.uns['V_grid'][0],st_data_use.uns['V_grid'][1], 
    linewidths=4, headwidth=5)
plt.savefig(f"{save_folder_trajectory}/6.spatial_quiver.pdf", format='pdf',bbox_inches='tight')
