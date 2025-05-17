import torch
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import DAGAST as dt     # import DAGAST

import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()    

###################%% version
sample_name = "DAGAST"
root_path = "/public3/Shigw/"
data_folder = f"{root_path}/datasets/Stereo-seq/regeneration/"
save_folder = f"{data_folder}/results/{sample_name}"
dt.check_path(save_folder)

######################### 1. 设置超参数 ########################
SEED = 24
knn = 30
n_genes = 500
n_neighbors = 9
n_externs = 10

dt.setup_seed(SEED)
torch.cuda.empty_cache()     
device = torch.device('cuda:1')
args = {
    "num_input" : n_genes,  
    "num_emb" : 256,        # 256  512
    "dk_re" : 16,
    "nheads" : 1,               #  1    4
    "droprate" : 0.15,          #  0.25,
    "leakyalpha" : 0.15,        #  0.15,
    "resalpha" : 0.5, 
    "bntype" : "BatchNorm",     # LayerNorm BatchNorm
    "device" : device, 
    "mode" : "Train", 

    "info_type" : "linear",  # nonlinear
    "iter_type" : "SCC",
    "iter_num" : 200,

    "neighbor_type" : "noextern",
    "n_neighbors" : 9,
    "n_externs" : 10,

    "num_epoch1" : 1000, 
    "num_epoch2" : 1000, 
    "lr" : 0.001, 
    "update_interval" : 1, 
    "eps" : 1e-5,
    "scheduler" : None, 
    "SEED" : SEED,
    
    "cutof" : 0.1,
    "alpha" : 1.0,
    "beta" : 0.1,
    "theta1" : 0.1,
    "theta2" : 0.1
}

######################### 2. 预处理，挑选候选细胞 ########################
# 数据处理 归一化和scale
st_data = sc.read(data_folder + "/st_data.h5")
df_data_pi = pd.read_csv(f"{data_folder}/results/PI_result.csv", index_col=0)
gene_flag = df_data_pi['PI'].nlargest(n_genes).index
st_data = st_data[:, gene_flag]

sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data)

# sc.pp.normalize_total(st_data, target_sum=1e4) 
# sc.pp.log1p(st_data)
# sc.pp.highly_variable_genes(st_data, n_top_genes=n_genes)
# sc.pp.scale(st_data)
# st_data = st_data[:, st_data.var['highly_variable'].values]

st_data_use = st_data.copy()

######################### 2. 构建模型，推断出转移矩阵 ########################
save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
dt.check_path(save_folder_cluster)

trainer = dt.DAGAST_Trainer(args, st_data, st_data_use)     # 构建DAGAST训练器
trainer.init_train()                                        # 构建细胞邻居关系、初始化数据、构建模型
trainer.train_stage1(f"{save_folder_cluster}/model_{sample_name}_stage1.pkl")   # 预训练

######################### 选取起始区域（可单独提供）
# model = torch.load(f"{save_folder_cluster}/model_{sample_name}_stage1.pkl")
emb = trainer.model.get_emb(isall=False)

#########################
emb_adata = sc.AnnData(emb)
emb_adata.obs['celltypes'] = st_data_use.obs['celltypes'].values
sc.pp.neighbors(emb_adata, use_rep='X', n_neighbors=20)
sc.tl.umap(emb_adata)

## 3.4 cluster
sc.tl.leiden(emb_adata, resolution=1.0)         # res = 0.1
print(f"{len(emb_adata.obs['leiden'].unique())} clusters")
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(emb_adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_cluster}/2.umap_cluster_stage1.png", dpi=600)

st_data_use.obs['emb_cluster'] = emb_adata.obs['leiden'].values
plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
ax = sc.pl.embedding(st_data_use, basis="spatial", color="emb_cluster",size=15, s=10, show=False, title='clustering')
plt.axis('off')
plt.savefig(f"{save_folder_cluster}/2.spatial_cluster_stage1.png", dpi=600, bbox_inches='tight')
#########################
save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
dt.check_path(save_folder_trajectory)

flag = (st_data_use.obs['emb_cluster'].isin(['4'])).values   


trainer.set_start_region(flag)                                  # 设置起始区域
trainer.train_stage2(save_folder_trajectory, sample_name)       # 轨迹推断
trainer.get_Trajectory_Ptime(knn=knn, grid_num=50, smooth=0.5, density=1.0) # 获取空间轨迹推断、空间拟时序


# # 绘制概率矩阵的热力图
# plt.figure(figsize=(10, 10))
# plt.scatter(xy1[:, 0], xy1[:, 1], c=trj.sum(0), cmap='Oranges', s=10, label="Original Points")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Interpolated Directional Field on a Grid")
# plt.legend()
# plt.savefig(f"{save_folder_trajectory}/plot_{sample_name}_trj.png")

######################### 3. 验证结果 ########################
st_data, st_data_use = trainer.st_data, trainer.st_data_use
model = trainer.model

dt.plot_spatial_complex(
    st_data, st_data_use, mode="time",
    value=st_data_use.obs['ptime'], title="ptime", pointsize=5,
    savename=f"{save_folder_trajectory}/1.spatial_Pseudotime.pdf"
)

### 3.2 构建分化轨迹
xy1 = st_data.obsm['spatial']
xy2 = st_data_use.obsm['spatial']

plt.close('all')
fig, axs = plt.subplots(figsize=(5, 5))
# sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1], color = (207/255, 185/255, 151/255, 1), s=5)
sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o', c = st_data_use.obs['ptime'], 
    s=20, cmap='Spectral_r', legend = False, alpha=0.25)
axs.quiver(st_data_use.uns['E_grid'][0],st_data_use.uns['E_grid'][1],st_data_use.uns['V_grid'][0],st_data_use.uns['V_grid'][1])
plt.savefig(f"{save_folder_trajectory}/2.spatial_quiver.pdf", format='pdf',bbox_inches='tight')


### 3.3 umap看特征拟合情况以及拟时序
# model = torch.load(f"{save_folder}/model_{n_genes}_{date}.pkl")
model.eval()
emb = model.get_emb(isall=False)
adata = sc.AnnData(emb)
sc.pp.neighbors(adata, use_rep='X', n_neighbors=knn)
adata.obs['ptime'] = st_data_use.obs['ptime'].values
adata.obs['celltypes'] = st_data_use.obs['celltypes'].values
sc.tl.umap(adata)

plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(adata, color="ptime", color_map='Spectral_r')
plt.savefig(f"{save_folder_trajectory}/3.umap_ptime.pdf")

plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(adata, color="celltypes", color_map='Spectral_r')
plt.savefig(f"{save_folder_trajectory}/3.umap_celltypes.pdf")

# st_data_use.obs['emb_cluster'] = st_data_use.obs['celltypes']
# nu.plot_spatial_compare(
#     st_data, st_data_use, mode="cluster",
#     leftvalue=st_data_use.obs['ptime'].values, lefttitle='ptime', righttitle="celltypes",
#     savename=f"{save_folder_trajectory}/4.spatial_ptime_celltypes.pdf"
# )


st_data_use.obs.to_csv(f"{save_folder_trajectory}/df_obs_ptime.csv")


### 3.4 cluster 这里确实要进一步通过聚类分析验证准确性，同时也算一些结果
## umap - cluster
sc.tl.leiden(adata, resolution=1.0)         # res = 0.1
st_data_use.obs['emb_cluster'] = adata.obs['leiden'].astype(int).to_numpy()
print(f"{len(adata.obs['leiden'].unique())} clusters")

umap_color = {str(c) : mycolor[idx] for idx, c in enumerate(adata.obs['leiden'].unique())}
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
# ax = sc.pl.umap(adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
ax = sc.pl.umap(adata, color="leiden", palette=umap_color, legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_trajectory}/5.embedding_umap_cluster_DAGAST.pdf")

dt.plot_spatial_complex(
    st_data, st_data_use, mode="cluster", 
    key='emb_cluster', title="emb_cluster", pointsize=5,
    savename=f"{save_folder_trajectory}/5.spatial_cluster_DAGAST.pdf"
)
