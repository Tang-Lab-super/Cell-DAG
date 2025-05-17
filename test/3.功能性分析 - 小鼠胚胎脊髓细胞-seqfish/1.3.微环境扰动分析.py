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
sample_name = "250109"
root_path = "/public3/Shigw/"
data_folder = f"{root_path}/datasets/SeqFISH/"
save_folder = f"{data_folder}/results/{sample_name}"
dt.check_path(save_folder)
SEED = 42

######################### 2. 预处理，挑选候选细胞 ########################
save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
dt.check_path(save_folder_trajectory)
save_folder_permu = f"{save_folder}/6.permu_env/"
dt.check_path(save_folder_permu)


# 数据处理 归一化和scale
st_data = sc.read_h5ad(data_folder + "/st_data.h5ad")

# sc.pp.filter_cells(st_data, min_genes=200)    
# sc.pp.filter_genes(st_data, min_cells=10)

sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data, max_value=5)

celltypes = ['Spinal cord', 'NMP']      # permutation cells
st_data_use = st_data[st_data.obs.celltypes.isin(celltypes), :].copy()

n_neighbors = 9
n_externs = 10
kNNGraph_use, indices_use, st_data_sel = dt.get_neighbor(st_data, st_data_use, n_neighbors=n_neighbors, n_externs=n_externs, ntype="extern")

######################### 3. 在特定细胞中对指定基因进行置换检验 ########################
# 导入模型和迁移矩阵
save_folder_attention_gene = f"{save_folder}/4.gene_attention/"
save_folder_attention_gene_ci = f"{save_folder_attention_gene}/Spinal cord/"
adata_geneatt = sc.read_h5ad(f"{save_folder_attention_gene_ci}/adata_geneatt.h5")
adata_geneatt_patten = sc.read_h5ad(f"{save_folder_attention_gene_ci}/adata_geneatt_patten.h5")

patteni = '3'
clusteri = '6'
save_folder_permu_sub = f"{save_folder_permu}/cluster{clusteri}_patten{patteni}/"
dt.check_path(save_folder_permu_sub)


patten3_genesets = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs.leiden == patteni].tolist()
gset1 = st_data_use.var_names[st_data_use.var_names.str.startswith('Notch')].tolist()  # Notch1
gset2 = st_data_use.var_names[st_data_use.var_names.str.startswith('Wnt')].tolist()  # Notch1
gene_cytoscopy = gset1 + gset2
gene_use = patten3_genesets

index_sel = np.array(st_data_sel.obs_names.tolist())
index_nb = index_sel[kNNGraph_use][indices_use]
celltypes_list = np.array(st_data_use.obs.celltypes.tolist())
flag_cell_patten = np.unique(index_nb[celltypes_list == ci, :][adata_geneatt.obs.leiden.values==clusteri].flatten())
cell_use = st_data_sel.obs.index.isin(flag_cell_patten)

trj_ori = np.load(f"{save_folder_trajectory}/trj_{sample_name}.npy")
model = torch.load(f"{save_folder_trajectory}/model_{sample_name}.pkl")

dt.setup_seed(SEED)
torch.cuda.empty_cache()     
device = torch.device('cuda:1')
df_permu_all, result_permu_all = dt.permutation_singlegene_celltype_env(      # 单个扰动
    model, st_data_sel, trj_ori, gene_use, cell_use,
    n_permu=30, epsilon=1e-16, seed=42, device=device
)

df_permu_all.to_csv(f"{save_folder_permu_sub}/1.permutation_singlegene_env.csv")

import pickle
with open(f"{save_folder_permu_sub}/1.permutation_singlegene_env_all.pkl", 'wb') as f:
    pickle.dump(result_permu_all, f)


## 验证结果是否显著
for gi in gene_use:     # gi = 'Dll1'
    gene_idx = df_permu_all.index.tolist().index(gi)
    x_sets, y_sets = [], []
    for ct in df_permu_all.columns:
        if ct in celltypes:
            continue
        x_sets.extend([ct] * len(result_permu_all[ct][gene_idx]))
        y_sets.extend(result_permu_all[ct][gene_idx])

    df_data = pd.DataFrame({'celltype' : x_sets, 'gene' : y_sets})
    mean_values = df_data.groupby('celltype')['gene'].mean().sort_values(ascending=False)
    sorted_types = mean_values.index
    df_data['celltype'] = pd.Categorical(df_data['celltype'], categories=sorted_types, ordered=True)
    cbox = [dt.color_box[ct] for ct in sorted_types]

    # 绘制箱型图
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='celltype', y='gene', data=df_data, order=sorted_types, palette=cbox)
    for idx, type_ in enumerate(sorted_types):      # 添加平均值标记
        mean_value = mean_values[type_]
        plt.scatter(idx, mean_value, color='red', s=20, marker='*', zorder=5)

    plt.xticks(rotation=90)
    plt.title(f'Env permutation of {gi} in cluster {clusteri}', fontsize=16)
    plt.xlabel('celltype', fontsize=14)
    plt.ylabel('genes', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_folder_permu_sub}/boxplot_permu_env_{gi}.pdf")