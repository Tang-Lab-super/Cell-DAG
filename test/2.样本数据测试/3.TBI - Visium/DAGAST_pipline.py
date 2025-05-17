import torch
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import DAGAST as dt     # import DAGAST

import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()    

###################%% version control
sample_name = "DAGAST"
root_path = "/public3/Shigw/datasets/Visium/TBI_stlearn/GSE236171_RAW/"

data_folder = f"{root_path}/VLP4_C1_Visium/"
save_folder = f"{data_folder}/results/{sample_name}"
dt.check_path(f"{data_folder}/results/")
dt.check_path(save_folder)

######################### 1. 设置超参数 ########################
SEED = 24
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

######################### 1. 读取数据，进行必要的处理 ########################
st_data = sc.read_visium(data_folder)
st_data.var_names_make_unique()

## 选取细胞
flag = st_data.var_names.isin(['Fcrls', 'Tmem119'])
tarexp = st_data.X.A[:, flag].sum(1)
cell_flag = tarexp > 1
st_data.obs['flag'] = cell_flag

# plt.close('all')
# plt.rcParams["figure.figsize"] = (8, 8)
# sc.pl.spatial(st_data, img_key="hires", color="flag")
# plt.savefig(f"{save_folder}/1.spatial_target_cell.png", dpi=600)

sc.pp.normalize_total(st_data, target_sum=1e4) 
sc.pp.log1p(st_data)
sc.pp.highly_variable_genes(st_data, n_top_genes=n_genes)
sc.pp.scale(st_data)
st_data = st_data[:, st_data.var['highly_variable'].values]

st_data_use = st_data[cell_flag].copy()

######################### 2. 构建模型，训练模型 ########################
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

flag = (st_data_use.obs['emb_cluster'].isin(['0'])).values   

trainer.set_start_region(flag)                                  # 设置起始区域
trainer.train_stage2(save_folder_trajectory, sample_name)       # 轨迹推断
trainer.get_Trajectory_Ptime(knn=30, grid_num=50, smooth=0.6, density=1.0) # 获取空间轨迹推断、空间拟时序


######################### 3. 验证结果 ########################
st_data, st_data_use = trainer.st_data, trainer.st_data_use
model = trainer.model

plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.spatial(st_data_use, img_key="hires", color=["ptime"], cmap='Spectral_r')
plt.axis('off')
plt.savefig(f"{save_folder_trajectory}/2.spatial_ptime.png", dpi=600, bbox_inches='tight')


### 3.2 构建分化轨迹
xy1 = st_data.obsm['spatial']
xy2 = st_data_use.obsm['spatial']
plt.close('all')
fig, axs = plt.subplots(figsize=(5, 5))
sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o', c = st_data_use.obs['ptime'], 
s=20, cmap='Spectral_r', legend = False, alpha=0.25)
axs.quiver(st_data_use.uns['E_grid'][0],st_data_use.uns['E_grid'][1],st_data_use.uns['V_grid'][0],st_data_use.uns['V_grid'][1])
# plt.quiver(grid_x, grid_y, grid_directions_x, grid_directions_y, color='red')
plt.savefig(f"{save_folder_trajectory}/2.spatial_quiver.pdf", format='pdf',bbox_inches='tight')


### 3.3 umap看特征拟合情况以及拟时序
# model = torch.load(f"{save_folder}/model_{n_genes}_{date}.pkl")
model.eval()
emb = model.get_emb(isall=False)
adata = sc.AnnData(emb)
sc.pp.neighbors(adata, use_rep='X', n_neighbors=30)
adata.obs['ptime'] = st_data_use.obs['ptime'].values
sc.tl.umap(adata)

plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
ax = sc.pl.umap(adata, color="ptime", color_map='Spectral_r')
plt.savefig(f"{save_folder_trajectory}/3.umap_ptime.pdf")


adata.obs.to_csv(f"{save_folder_trajectory}/df_obs_ptime.csv")


### 3.4 cluster 这里确实要进一步通过聚类分析验证准确性，同时也算一些结果
## umap - cluster
sc.tl.leiden(adata, resolution=0.5)         # res = 0.1
st_data_use.obs['emb_cluster'] = adata.obs['leiden'].values
print(f"{len(adata.obs['leiden'].unique())} clusters")

mycolor = [
   "#E31A1C",  "#33A02C", "#6A3D9A", "#FF7F00", "#1F78B4",  "#B15928", "#A6CEE3", "#B2DF8A", 
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
    "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
    "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
]
umap_color = {str(c) : mycolor[idx] for idx, c in enumerate(adata.obs['leiden'].unique())}
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
# ax = sc.pl.umap(adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
ax = sc.pl.umap(adata, color="leiden", palette=umap_color, legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{save_folder_trajectory}/5.embedding_umap_cluster_DAGAST.pdf")


plt.close('all')
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.spatial(st_data_use, img_key="hires", color="emb_cluster", palette=umap_color)
plt.axis('off')
plt.savefig(f"{save_folder_trajectory}/5.spatial_cluster_DAGAST.png", dpi=600, bbox_inches='tight')





# ######################### 4. 置换检验找特征基因 ########################
# # 导入模型和迁移矩阵
# # gene_use = st_data_use.var_names.values[:18]      # 确定扰动gene
# # cell_use = st_data_use.obs_names.values
# cell_use = st_data_use.obs_names.tolist()
# gene_use = st_data_use.var_names.tolist()

# trj_ori = np.load(f"{save_folder_trajectory}/trj_{sample_name}.npy")
# model = torch.load(f"{save_folder_trajectory}/model_{sample_name}.pkl")

# nu.setup_seed(SEED)
# torch.cuda.empty_cache()     
# device = torch.device('cuda:1')
# result_permu = permutation_singlegene_celltype(      # 单个扰动
#     model, st_data_sel, trj_ori, gene_use, cell_use,
#     n_permu=30, epsilon=1e-16, seed=24, device=device
# )

# result_permu_df = pd.DataFrame(result_permu.mean(1), index=gene_use, columns=['total_sim'])
# result_permu_sorted = result_permu_df.sort_values('total_sim', ascending=False)
# result_permu_sorted.to_csv(f"{save_folder_trajectory}/5.permutation_single_gene.csv")

# # 原位展示
# result_permu_sorted = pd.read_csv(f"{save_folder_trajectory}/5.permutation_single_gene.csv", index_col=0)
# st_data_plotgene = sc.read_h5ad(data_folder + "/st_data.h5ad")
# plot_genes = result_permu_sorted.head(50).index.tolist()
# nu.plot_permutation_genes(
#     st_data_plotgene, st_data_use.obs_names.tolist(), plot_genes[0:6], 
#     f"{save_folder_trajectory}/5.spatial_permutation_singleGene1.pdf", 
#     figsize=(6, 4), prow=2, pcol=3)
# nu.plot_permutation_genes(
#     st_data_plotgene, st_data_use.obs_names.tolist(), plot_genes[6:12], 
#     f"{save_folder_trajectory}/5.spatial_permutation_singleGene2.pdf", 
#     figsize=(6, 4), prow=2, pcol=3)
# nu.plot_permutation_genes(
#     st_data_plotgene, st_data_use.obs_names.tolist(), plot_genes[12:18], 
#     f"{save_folder_trajectory}/5.spatial_permutation_singleGene3.pdf", 
#     figsize=(6, 4), prow=2, pcol=3)


# # 绘制柱状图
# result_permu_sorted.head(50)

# # 绘制条形图
# plt.close('all')
# plt.figure(figsize=(10, 6))
# plt.bar(result_permu_sorted.head(25).index, result_permu_sorted.head(25).total_sim, color='skyblue')
# plt.xticks(rotation=45)
# plt.xlabel('Gene', fontsize=12)
# plt.ylabel('Mean of Sim', fontsize=12)
# plt.savefig(f"{save_folder_trajectory}/6.barplot_gene_top50.pdf")




