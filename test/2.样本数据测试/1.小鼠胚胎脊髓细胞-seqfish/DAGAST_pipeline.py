import torch
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import DAGAST as dt     # import DAGAST

import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()    

###################%% version and path
sample_name = "DAGAST"
root_path = "/public3/Shigw/"
data_folder = f"{root_path}/datasets/SeqFISH/"
save_folder = f"{data_folder}/results/{sample_name}"
dt.check_path(save_folder)

######################### 1. 设置超参数 ########################
SEED = 24
knn = 30
n_neighbors = 9
n_externs = 10

dt.setup_seed(SEED)
torch.cuda.empty_cache()     
device = torch.device('cuda:1')
args = {
    "num_input" : 351,  
    "num_emb" : 256,        # 256  512
    "dk_re" : 16,
    "nheads" : 1,               #  1    4
    "droprate" : 0.15,          #  0.25,
    "leakyalpha" : 0.15,        #  0.15,
    "resalpha" : 0.5, 
    "bntype" : "BatchNorm",     # LayerNorm BatchNorm
    "device" : device, 
    "info_type" : "nonlinear",  # nonlinear
    "iter_type" : "SCC",
    "iter_num" : 200,

    "neighbor_type" : "extern",
    "n_neighbors" : 9,
    "n_externs" : 10,

    "num_epoch1" : 1000, 
    "num_epoch2" : 1000, 
    "lr" : 0.001, 
    "update_interval" : 1, 
    "eps" : 1e-5,
    "scheduler" : None, 
    "SEED" : 24,

    "cutof" : 0.1,
    "alpha" : 1.0,
    "beta" : 0.1,
    "theta1" : 0.1,
    "theta2" : 0.1
}

celltypes = ['Spinal cord', 'NMP']      # 目标细胞

######################### 2. 预处理，挑选候选细胞 ########################
save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
dt.check_path(save_folder_cluster)

# 数据处理 归一化和scale
st_data = sc.read_h5ad(data_folder + "/st_data.h5ad")
sc.pp.normalize_total(st_data, target_sum=1e4)          # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data)
st_data_use = st_data[st_data.obs.celltypes.isin(celltypes), :].copy()      ## 选取指定细胞 

dt.plot_spatial_complex(
    st_data, st_data_use, mode="cluster", key="celltypes",
    figsize=(5, 5), title=None, pointsize=5, 
    savename=f"{save_folder_cluster}/spatial_sel_cell.png"
)


######################### 2. 构建模型，训练模型 ########################
trainer = dt.DAGAST_Trainer(args, st_data, st_data_use)     # 构建DAGAST训练器，st_data是总数据，st_data_use是待推断的数据
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
sc.tl.leiden(emb_adata, resolution=1.0)         # res = 0.1
print(f"{len(emb_adata.obs['leiden'].unique())} clusters")

plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(emb_adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_cluster}/2.umap_cluster_stage1.pdf", dpi=600)

st_data_use.obs['emb_cluster'] = emb_adata.obs['leiden'].values
plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
ax = sc.pl.embedding(st_data_use, basis="spatial", color="emb_cluster",size=15, s=10, show=False, title='clustering')
plt.axis('off')
plt.savefig(f"{save_folder_cluster}/2.spatial_cluster_stage1.pdf", dpi=600, bbox_inches='tight')
#########################

save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
dt.check_path(save_folder_trajectory)

flag = (st_data_use.obs['emb_cluster'].isin(['3'])).values   

trainer.set_start_region(flag)                                  # 设置起始区域
trainer.train_stage2(save_folder_trajectory, sample_name)       # 轨迹推断
trainer.get_Trajectory_Ptime(knn, grid_num=50, smooth=0.5, density=0.7) # 获取空间轨迹推断、空间拟时序


######################### 3. 验证结果 ########################
st_data, st_data_use = trainer.st_data, trainer.st_data_use
model = trainer.model

## 绘制空间轨迹图和空间拟时序图
xy1 = st_data.obsm['spatial']
xy2 = st_data_use.obsm['spatial']
plt.close('all')
fig, axs = plt.subplots(figsize=(5, 5))
sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o', c = st_data_use.obs['ptime'], s=20, cmap='Spectral_r', legend = False, alpha=0.25)
axs.quiver(st_data_use.uns['E_grid'][0], st_data_use.uns['E_grid'][1], st_data_use.uns['V_grid'][0], st_data_use.uns['V_grid'][1], 
    scale=0.2, linewidths=4, headwidth=5)
plt.savefig(f"{save_folder_trajectory}/2.spatial_quiver.pdf", format='pdf',bbox_inches='tight')

dt.plot_spatial_complex(
    st_data, st_data_use, mode="time",
    value=st_data_use.obs['ptime'], title="ptime", pointsize=5,
    savename=f"{save_folder_trajectory}/1.spatial_Pseudotime.pdf"
)


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

st_data_use.obs.to_csv(f"{save_folder_trajectory}/df_obs_ptime.csv")


### 3.4 cluster 这里确实要进一步通过聚类分析验证准确性，同时也算一些结果
sc.tl.leiden(adata, resolution=0.3)         # res = 0.1
st_data_use.obs['emb_cluster'] = adata.obs['leiden'].astype(int).to_numpy()
print(f"{len(adata.obs['leiden'].unique())} clusters")

umap_color = {str(c) : dt.mycolor[idx] for idx, c in enumerate(adata.obs['leiden'].unique())}
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
# ax = sc.pl.umap(adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
ax = sc.pl.umap(adata, color="leiden", palette=umap_color, legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_trajectory}/5.embedding_umap_cluster_DAGAST.pdf")

dt.plot_spatial_complex(
    st_data, st_data_use, mode="cluster",
    value=st_data_use.obs['emb_cluster'], title="emb_cluster", pointsize=5,
    savename=f"{save_folder_trajectory}/5.spatial_cluster_DAGAST.pdf"
)
