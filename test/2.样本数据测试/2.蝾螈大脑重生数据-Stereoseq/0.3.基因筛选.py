import os 
import torch
import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from anndata import AnnData

import PROST 

import warnings 
warnings.filterwarnings("ignore") 

torch.cuda.empty_cache()     

def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')

## 要不要把PI拿过来做？
######################### 1. 挑选候选基因 ########################
SEED = 24
PROST.setup_seed(SEED)

###################%% version
date = "241202"
root_path = "/public3/Shigw/"
# root_path = "/data3/shigw/ST_project/FinalFile/"
# root_path = "/data01/suziyi/GWCode/"
data_folder = f"{root_path}/datasets/Stereo-seq/regeneration/"
save_folder = f"{data_folder}/results/{date}"
check_path(save_folder)

###################%% 01 cal COVET, create trajectory
st_data = sc.read(f"{data_folder}/st_data.h5")
st_data.var_names_make_unique()

# # 不做筛选
# sc.pp.filter_cells(st_data, min_genes=200)    
# sc.pp.filter_genes(st_data, min_cells=3)

# 1. Calculate PI
n_neighbors = 9
st_data_use = st_data.copy()
# st_data_use, kNNGraph_use, indices_use = select_data(st_data, st_data_sel, n_neighbors)
st_data_use = PROST.prepare_for_PI(st_data_use, platform="stereo-seq")
st_data_use = PROST.cal_PI(st_data_use, platform="stereo-seq", multiprocess=True)


df_gene_metadata = st_data_use.var
df_gene_metadata.to_csv(f"{save_folder}/PI_result.csv")
# st_data_use.write_h5ad(save_folder+"/PI_result.h5")


