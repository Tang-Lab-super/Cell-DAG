__author__ = " Guowei Shi "
__version__ = " 1.0.0 "

from .utils import *
from .plot import *
from .model import *


"""
# DAGAST使用简介
######################### 1. 设置超参数 ########################
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

######################### 3. 构建模型，训练模型 ########################
trainer = dt.DAGAST_Trainer(args, st_data, st_data_use)     # 构建DAGAST训练器
trainer.init_train()                                        # 构建细胞邻居关系、初始化数据、构建模型
trainer.train_stage1(f"{save_folder_cluster}/model_{sample_name}_stage1.pkl")   # 预训练

######################### 通过聚类选取起始区域（可单独提供）
    xxxx    
#########################

flag = (st_data_use.obs['emb_cluster'].isin(['3'])).values   
trainer.set_start_region(flag)                                  # 设置起始区域
trainer.train_stage2(save_folder_trajectory, sample_name)       # 轨迹推断
trainer.get_Trajectory_Ptime(knn, grid_num=50, smooth=0.5, density=0.7) # 获取空间轨迹推断、空间拟时序

######################### 4. 验证结果 ########################
st_data, st_data_use = trainer.st_data, trainer.st_data_use
model = trainer.model
"""