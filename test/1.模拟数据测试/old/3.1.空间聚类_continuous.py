import os 
import torch
import PROST 
import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from anndata import AnnData

import warnings 
warnings.filterwarnings("ignore") 

torch.cuda.empty_cache()     

## 要不要把PI拿过来做？
######################### 1. 挑选候选基因 ########################
SEED = 24

PROST.setup_seed(SEED)

## 1. continuous
sample_name = "continuous"

datapath = "/data3/shigw/ST_project/FinalFile/datasets/Simulated_data/"
anspath = f"/data3/shigw/ST_project/FinalFile/datasets/Simulated_data/{sample_name}/"

data = pd.read_table(f"{datapath}/sim_path_count.txt", index_col=0).T
metadata = pd.read_table(f"{anspath}/sim_path_metadata_{sample_name}.txt", index_col=0)
st_data = AnnData(X=data, obs=metadata)
st_data.X = st_data.X.astype('float64')  # this is not required and results will be comparable without it
st_data.obs['x'] = st_data.obs['x'] - st_data.obs['x'].min()
st_data.obs['y'] = st_data.obs['y'] - st_data.obs['y'].min()
st_data.obsm['spatial'] = st_data.obs[['x', 'y']]
st_data_use = st_data

# 数据处理 归一化和scale
n_neighbors = 9
n_genes = 3000

st_data_use.raw = st_data_use
sc.pp.normalize_total(st_data_use, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data_use)

# st_data_use = PROST.feature_selection(st_data_use, by="scanpy", n_top_genes=n_genes)

# Preprocessing
sc.pp.normalize_total(st_data_use)
sc.pp.log1p(st_data_use)
# Set the number of clusters

PROST.run_PNN(
    st_data_use, 
    adj_mode='neighbour', 
    k_neighbors = n_neighbors,
    init="leiden", 
    res = 1.0, 
    lr = 0.001,
    SEED=SEED, 
    post_processing = True)


# Plot clustering results
plt.rcParams["figure.figsize"] = (5,5)
ax = sc.pl.embedding(st_data_use, basis="spatial", color="pp_clustering",size=7,s=6, show=False, title='clustering')
ax.invert_yaxis()
plt.axis('off')
plt.savefig(anspath+f"/clustering_{sample_name}.png", dpi=600, bbox_inches='tight')


# 确认中心类别数
flag = (st_data_use.obs['pp_clustering'] == 49).values
st_data_use.obs['start_cluster'] = 0
st_data_use.obs.loc[flag, 'start_cluster'] = 1
plt.rcParams["figure.figsize"] = (5,5)
ax = sc.pl.embedding(st_data_use, basis="spatial", color="start_cluster",size=7,s=6, show=False, title='start_cluster')
ax.invert_yaxis()
plt.axis('off')
plt.savefig(anspath+f"/start_cluster_{sample_name}.png", dpi=600, bbox_inches='tight')


# Save clustering results
st_data_use.write_h5ad(anspath + f"/PNN_result_{sample_name}.h5")