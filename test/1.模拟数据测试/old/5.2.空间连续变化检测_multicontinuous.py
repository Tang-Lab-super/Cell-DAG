import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams

import DAGAST_test as nu
from anndata import AnnData
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()     

## 要不要把PI拿过来做？
######################### 1. 挑选候选基因 ########################
SEED = 21
n_neighbors = 21
anspath = "/data3/shigw/ST_project/FinalFile/datasets/Simulated_data/"

## 3. discrete
sample_name = "discrete"

anspath = f"/data3/shigw/ST_project/FinalFile/datasets/Simulated_data/{sample_name}"
st_data = sc.read(f"{anspath}/PNN_result_{sample_name}.h5")

# 数据处理 归一化和scale
n_genes = 300
st_data_use = st_data
sc.pp.normalize_total(st_data_use, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data_use)
sc.pp.highly_variable_genes(st_data_use, n_top_genes=n_genes)
sc.pp.scale(st_data_use, max_value=5)
st_data_use_hvg = st_data_use[:, st_data_use.var['highly_variable'].values]

exp_imput = st_data_use_hvg.X
start_flag = st_data_use_hvg.obs['start_cluster'].values

# sc.tl.pca(st_data_use_hvg, svd_solver='arpack')
# exp_imput = st_data_use_hvg.obsm['X_pca']

use_spots_name = st_data_use.obs_names.tolist()
indices_use = np.array([use_spots_name.index(ci) for ci in st_data_use.obs_names])
kNNGraph_use = kneighbors_graph(st_data_use.obsm["spatial"], n_neighbors=n_neighbors, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
kNNGraph_use = np.reshape(np.asarray(kNNGraph_use.col), [len(st_data_use), n_neighbors])


# net_input, net_predict, emb, dc_predict, start, adj, flag = net_input[indices_use], net_predict, emb, dc_predict, start, model.adjmat[indices_use], flag
def MyLoss_pre(net_input, net_predict, emb, adj, indices_use, flag=None):
    net_input = net_input[indices_use]
    adj = adj[indices_use]

    if flag is not None:
        net_input = net_input[flag]
        net_predict = net_predict[flag]
        adj = adj[flag]

    ## reconstrcut
    MSEloss = torch.nn.MSELoss()            # Loss
    SmoothL1loss = torch.nn.SmoothL1Loss()            # Loss
    mse_loss = MSEloss(net_input, net_predict)

    # TVloss
    emb_TV = emb[adj]       # n * k * m
    emb_TV = emb_TV - emb_TV[:, :1, :]
    emb_TV = torch.norm(emb_TV, p=1, dim=1)
    emb_TV = torch.norm(emb_TV, p=2) / adj.shape[0]

    return mse_loss, emb_TV
    # return mse_loss

# net_input, net_predict, emb, dc_predict, start, adj, flag = net_input[indices_use], net_predict, emb, dc_predict, start, model.adjmat[indices_use], flag
def MyLoss_post(net_input, net_predict, emb, h_val, adj, indices_use, flag=None, seed=24):
    net_input = net_input[indices_use]
    adj = adj[indices_use]

    if flag is not None:
        net_input = net_input[flag]
        net_predict = net_predict[flag]
        adj = adj[flag]

    ## reconstrcut
    MSEloss = torch.nn.MSELoss()                            # Loss
    SmoothL1loss = torch.nn.SmoothL1Loss()                  # Loss
    mse_loss = MSEloss(net_input, net_predict)

    # TVloss
    emb_TV = emb[adj]       # n * k * m
    emb_TV = emb_TV - emb_TV[:, :1, :]
    emb_TV = torch.norm(emb_TV, p=1, dim=1)
    emb_TV = torch.norm(emb_TV, p=2) / adj.shape[0]

    return mse_loss, emb_TV
    # return mse_loss


nu.setup_seed(SEED)
torch.cuda.empty_cache()     
device = torch.device('cuda:1')
args = {
    "num_input" : exp_imput.shape[1],  
    "num_emb" : n_genes,        # 256  512
    "nheads" : 1,               #  1    4
    "droprate" : 0.05,          #  0.25,
    "leakyalpha" : 0.15,        #  0.15,
    "resalpha" : 0.5, 
    "bntype" : "BatchNorm",     # LayerNorm BatchNorm
    "device" : device, 
    "mode" : "Train", 
    "iter_type" : "nSCC",
    "iter_num" : 200,

    "num_epoch1" : 2000, 
    "num_epoch2" : 600, 
    "lr" : 0.001, 
    "batch_size" : 0,
    "update_interval" : 1, 
    "eps" : 1e-5,
    "scheduler" : None, 
    "SEED" : SEED
}
# start = -np.argmin(st_data_sel.obsm['DC_COVET'][:, 0])       ## 指定轨迹方向
# model, dc_predict = trajectory_get_twostage(exp_imput, kNNGraph_use, indices_use, args, start, alpha=0.1, beta=0.1)

covet_mat, adj, indices_use, args = exp_imput, kNNGraph_use, indices_use, args

device = args["device"]
num_epoch1 = args["num_epoch1"]
num_epoch2 = args["num_epoch2"]
update_interval = args["update_interval"]
lr = args["lr"]
scheduler = args["scheduler"]
batch_size = args["batch_size"]

model = nu.DAGAST(args, adj, indices_use)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

if scheduler is not None:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler, gamma=0.1)

exp_input = nu.np_to_torch(covet_mat, device, requires_grad=False)
indices_torch = nu.np_to_torch(indices_use, device, dtype=torch.long, requires_grad=False)
startflag_torch = nu.np_to_torch(start_flag, device, dtype=torch.long, requires_grad=False)

model.to(device)

cutof = 0.1
alpha, beta = 1.0, 1.0
theta1, theta2 = 0.01, 0.01
# 第一阶段就是首先完成了样本的图神经网络的自回归训练
print('stage1 training on:', device)  
with tqdm(total=num_epoch1) as t:  
    model.train()
    for i in range(num_epoch1):
        emb, exp_predict = model(exp_input)
        optimizer.zero_grad()
        mse_loss, emb_TV = MyLoss_pre(exp_input, exp_predict, emb, model.adjmat, indices_torch)
        total_loss = mse_loss * alpha + emb_TV * beta
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()    
        if i % update_interval == 0:
            t.update(update_interval)
            t.set_postfix({'MSEloss' : mse_loss.item(), 'emb_TV' : emb_TV.item()})
        if mse_loss * cutof > emb_TV:
            break


# 然后第二阶段在这个基础上对其进行了特征约束的微调训练
# 这里需要设置参数是否可训练
old_loss = 0.0
optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=1e-4)
print('stage2 training on:', device)  
with tqdm(total=num_epoch2) as t:  
    model.train()
    for i in range(num_epoch2):
        emb, exp_predict, h_val, grad, trj = model(exp_input, iftrj=True)
        optimizer.zero_grad()

        mse_loss, emb_TV = MyLoss_post(exp_input, exp_predict, emb, h_val, model.adjmat, indices_torch)
        total_loss = mse_loss * alpha + emb_TV * beta + h_val * theta1

        if startflag_torch is not None:
            startloss = startflag_torch.sum() - torch.norm(trj[:, startflag_torch==1].mean(0), p=1)
            total_loss = total_loss + startloss * theta2

        total_loss.backward()
        optimizer.step()
        # delta_loss = abs(total_loss - old_loss)
        if scheduler is not None:
            scheduler.step()    
        if i % update_interval == 0:
            t.update(update_interval)
            # t.set_postfix({'MSEloss' : mse_loss.item(), 'h_val' : h_val.item()})
            t.set_postfix({'MSEloss' : mse_loss.item(), 'emb_TV' : emb_TV.item(), 'h_val' : h_val.item(), 's_val' : startloss.item()})
        # if h_val.item() < startloss.item():
        #     break
        old_loss = total_loss.item()

model.eval()
emb, exp_predict, h_val, grad, trj = model(exp_input, iftrj=True)
trj = trj.detach().cpu().numpy()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# 计算方向向量
# direction_vector = end_point - start_point
# unit_vector = direction_vector / np.linalg.norm(direction_vector)
kNN_use = kNNGraph_use[indices_use]
xy_use = st_data_use.obsm['spatial']
xy1 = st_data_use.obsm['spatial'].values

n = kNN_use.shape[0]  # 细胞数量
dirs = []
xys = []

# 构建 NearestNeighbors 对象
nbrs = NearestNeighbors(n_neighbors=len(st_data_use), algorithm='auto').fit(st_data_use.obsm["spatial"])
distances, indices = nbrs.kneighbors(st_data_use.obsm["spatial"])

## 考虑距离衰减
for j in range(n):
    sub_dis = distances[j]
    sub_idx = indices[j]

    sub_dis = np.exp(sub_dis - sub_dis[n_neighbors])
    sub_dis[:n_neighbors] = 1.0

    sub_trj = trj[j]
    sub_trj[sub_idx] = sub_trj[sub_idx] / sub_dis

    flag = sub_trj != 0
    # flag = sub_idx[:n_neighbors*4]
    if(flag.sum() <= 0): 
        continue
    
    # print(flag.sum())
    # 取最长的
    # tmp_idx = np.argmax(sub_trj)
    # tmp = xy1[j, :] - xy1[tmp_idx, :]

    # 取平均
    tmp_all = xy1[j, :] - xy1[flag, :]
    z_all = sub_trj[flag]
    tmp_all[:, 0] *= z_all
    tmp_all[:, 1] *= z_all
    lin_norm = np.linalg.norm(tmp_all)
    if np.linalg.norm(tmp_all) != 0:
        tmp_all = tmp_all / lin_norm
    tmp = tmp_all.mean(0)
    # tmp = tmp_all[np.argmax(z_all)]

    dirs.append(tmp)
    xys.append(xy1[j])

dirs = np.array(dirs)
xys = np.array(xys)

max_xy = np.max(xy1, 0)
grid_x, grid_y = np.mgrid[0:int(max_xy[0]+1):50j, 0:int(max_xy[1]+1):50j]  # 生成100x100的等格网
# 分别对dx和dy方向进行插值
grid_directions_x = griddata(xys, dirs[:, 0], (grid_x, grid_y), method='cubic', fill_value=0.0)
grid_directions_y = griddata(xys, dirs[:, 1], (grid_x, grid_y), method='cubic', fill_value=0.0)

# 创建掩膜，仅保留有数据的区域
# mask = np.isnan(grid_directions_x) | np.isnan(grid_directions_y)
# grid_directions_x = np.ma.array(grid_directions_x, mask=mask)
# grid_directions_y = np.ma.array(grid_directions_y, mask=mask)


### 绘图
# 绘制原始散点
plt.figure(figsize=(10, 10))
plt.scatter(xy1[:, 0], xy1[:, 1], color='k', s=1, label="Original Points")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Interpolated Directional Field on a Grid")
plt.legend()
plt.savefig(f"{anspath}/plot0_{sample_name}.png")


# 绘制方向矢量
plt.figure(figsize=(10, 10))
plt.scatter(xys[:, 0], xys[:, 1], color='k', s=1, label="Original Points")
plt.quiver(grid_x, grid_y, grid_directions_x, grid_directions_y, color='red')
plt.quiver(xys[:, 0], xys[:, 1], dirs[:, 0], dirs[:, 1], color='black')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Interpolated Directional Field on a Grid")
plt.legend()
plt.savefig(f"{anspath}/plot1_{sample_name}.png")


# 绘制热力图
plt.figure(figsize=(10, 10))
plt.scatter(xy1[:, 0], xy1[:, 1], c=trj.sum(0), cmap='Oranges', s=1, label="Original Points")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Interpolated Directional Field on a Grid")
plt.legend()
plt.savefig(f"{anspath}/plot_{sample_name}_trj.png")

