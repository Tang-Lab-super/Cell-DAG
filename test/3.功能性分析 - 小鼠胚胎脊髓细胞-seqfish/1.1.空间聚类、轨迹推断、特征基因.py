import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import DAGAST as dt     # import DAGAST

import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()    

###################%% version
sample_name = "250116"
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
# 数据处理 归一化和scale
st_data = sc.read_h5ad(data_folder + "/st_data.h5ad")
sc.pp.normalize_total(st_data, target_sum=1e4)          # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data)
st_data_use = st_data[st_data.obs.celltypes.isin(celltypes), :].copy()      ## 选取指定细胞 

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

flag = (st_data_use.obs['emb_cluster'].isin(['3'])).values   

trainer.set_start_region(flag)                                  # 设置起始区域
trainer.train_stage2(save_folder_trajectory, sample_name)       # 轨迹推断
trainer.get_Trajectory_Ptime(knn, grid_num=50, smooth=0.5, density=0.7) # 获取空间轨迹推断、空间拟时序


######################### 3. 验证结果 ########################
st_data, st_data_use, st_data_sel = trainer.st_data, trainer.st_data_use, trainer.st_data_sel
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


######################### 4. 置换检验找特征基因 ########################
# 导入模型和迁移矩阵
ci = 'Spinal cord'
celltypes = ['Spinal cord', 'NMP']      # permutation cells

# gene_use = st_data_use.var_names.values[:18]      # 确定扰动gene
# cell_use = st_data_use.obs_names.values
cell_use = st_data_use.obs_names[st_data_use.obs['celltypes'].isin(celltypes)].tolist()
gene_use = st_data_use.var_names.tolist()

trj_ori = np.load(f"{save_folder_trajectory}/trj_{sample_name}.npy")
model = torch.load(f"{save_folder_trajectory}/model_{sample_name}.pkl")

dt.setup_seed(SEED)
torch.cuda.empty_cache()     
device = torch.device('cuda:1')
result_permu = dt.permutation_singlegene_celltype(      # 单个扰动
    model, st_data_sel, trj_ori, gene_use, cell_use,
    n_permu=30, epsilon=1e-16, seed=24, device=device
)

result_permu_df = pd.DataFrame(result_permu.mean(1), index=gene_use, columns=['total_sim'])
result_permu_sorted = result_permu_df.sort_values('total_sim', ascending=False)
result_permu_sorted.to_csv(f"{save_folder_trajectory}/5.permutation_single_gene_{ci}.csv")
result_permu_sorted.head(50)

# 绘制条形图
plt.close('all')
plt.figure(figsize=(10, 6))
plt.bar(result_permu_sorted.head(20).index, result_permu_sorted.head(20).total_sim, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Gene', fontsize=12)
plt.ylabel('Mean of KL divergence', fontsize=12)
plt.savefig(f"{save_folder_trajectory}/6.barplot_gene_top20.pdf")

save_folder_trajectory_permugene = f"{save_folder_trajectory}/permutation_genes/"
dt.check_path(save_folder_trajectory_permugene)

st_data_plotgene = sc.read_h5ad(data_folder + "/st_data.h5ad")
plot_genes = result_permu_sorted.head(50).index.tolist()
dt.plot_permutation_genes(
    st_data_plotgene, st_data_use.obs_names.tolist(), plot_genes, 
    f"{save_folder_trajectory_permugene}/spatial_", 
    figsize=(5, 5))


['Hoxb9', 'Foxd4', 'Hoxd4', 'Nid1', 'Ets1', 'Hoxb5', 'Hoxc6', 'Hoxb1', 'Hoxa1', 'Apob', 'Hoxb3', 'Eng', 'Six1', 'Atp1b1', 'Plvap', 'Etv4', 'Nkx1-2', 'Foxh1', 'Gypa', 'Runx1']

# Hoxb9, Hoxd4, Hoxb5, Hoxc6, Hoxb1, Hoxa1, Hoxb3, 
######################################## 验证特征基因
## 检查基因1 - hox
result_permu_sorted['number'] = np.arange(1, 352)
df_plot = result_permu_sorted.head(300).loc[result_permu_sorted.head(300).index.str.startswith('Hox')]

plt.close('all')
plt.figure(figsize=(10, 6))
plt.bar(df_plot.index, df_plot.total_sim, color='skyblue')
plt.xticks(rotation=45)
plt.xlabel('Gene', fontsize=12)
plt.ylabel('Mean of KL divergence', fontsize=12)
plt.savefig(f"{save_folder_trajectory}/6.barplot_hoxgene.pdf")


## 检查基因2 - spinal通路上的基因
# spinal_path_gene = ["Cdx1", "Nkx1-2", "Hoxa1", "Hes3", "Cdx2", "Hoxa1", "Mis18bp1", "Cenpa", "Nkx2-9", "Foxa2", "Nkx6-1", "Phox2b", "Olig2"]
spinal_path_gene = ["Cdx1", "Nkx1-2", "Hoxa1", "Hes3", "Cdx2", "Foxa2"]
result_permu_sorted.loc[spinal_path_gene]

### 3.6 沿拟时序的特征基因热图
# st_data_use_ci = st_data_use[st_data_use.obs['celltypes'] == ci]
exp_ci = pd.DataFrame(st_data_use.X, index=st_data_use.obs_names.values, columns=st_data_use.var_names.values)
exp_ci['ptime'] = st_data_use.obs['ptime']
exp_ci = exp_ci.sort_values('ptime')
exp_ci.to_csv(f"{save_folder_trajectory}/6.heatmap_ci.csv")

# 筛选基因
"""/public1/yuchen/software/miniconda3/envs/R4.2_yyc/bin/R

library(monocle)

adata = read.csv("/public3/Shigw//datasets/SeqFISH//results/250116/3.spatial_trajectory/6.heatmap_ci.csv", row.names=1)
ptime = adata[, 352]
adata = adata[, -352]

# ann_cell = adata['ptime']
# ann_gene = data.frame(gene_short_name=rownames(adata))
# rownames(ann_gene) = ann_gene$gene_short_name

# 创建细胞元信息
ann_cell <- data.frame(
  cell_id = rownames(adata),
  cell_type = sample(c("Type1"), nrow(adata), replace = TRUE),
  ptime,
  row.names = rownames(adata)
)

# 创建基因元信息
ann_gene <- data.frame(
  gene_id = colnames(adata),
  gene_short_name = colnames(adata),
  row.names = colnames(adata)
)

exp = data.matrix(adata)
pd <- new("AnnotatedDataFrame", data=ann_cell)
fd <- new("AnnotatedDataFrame", data=ann_gene)
cds <- newCellDataSet(t(exp), phenoData = pd, featureData = fd)

cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds)

diff_test_res <- differentialGeneTest(cds, fullModelFormulaStr = "~sm.ns(ptime)")

write.csv(diff_test_res, "/public3/Shigw//datasets/SeqFISH//results/250116/3.spatial_trajectory/6.diff_test_res.csv")
"""

diff_test_res = pd.read_csv(f"{save_folder_trajectory}/6.diff_test_res.csv", index_col=0)
diff_test_res.index = st_data_use.var_names.values

flag = diff_test_res.index[diff_test_res.status == 'OK'].tolist()
# exp_ci1 = exp_ci[flag]

diff_test_res_ok = diff_test_res[diff_test_res.status == 'OK']
diff_test_res_ok = diff_test_res_ok.sort_values('qval')
diff_test_res_ok['number'] = np.arange(len(diff_test_res_ok))+1
diff_test_res_ok.loc[diff_test_res_ok.index.isin(spinal_path_gene), ['status', 'pval', 'number']]


### 计算拟时序的相关性

# genelists = ["Hoxaas3", "Hoxb5os", "Hoxb9", "Hoxb7", "Tlx2", "Foxa3", "Hoxd3", "Hoxa2", "Rfx4", "Hoxd4"]
genelists = ["Hoxb9", "Hoxd4"]
st_data_use.obs.to_csv(f"{save_folder_trajectory}/df_obs_ptime.csv")


