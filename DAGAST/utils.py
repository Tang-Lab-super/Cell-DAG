# shigw 2025-3-15
# train utils functions

import os 
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')


######################### 03 Permutation test ########################
def kl_divergence(P, Q, epsilon=None):
    # 确保输入为 numpy 数组
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    
    # 归一化为概率分布
    P /= np.sum(P)
    Q /= np.sum(Q)
    
    if epsilon != None:                 # 避免 log(0) 或除以 0，添加一个很小的数
        P = np.clip(P, epsilon, 1)
        Q = np.clip(Q, epsilon, 1)

    return np.sum(P * np.log(P / Q))    # 计算 KL 散度

### Perform perturbation analysis on target cells
def permutation_singlegene_celltype(
        model, st_data_use, trj_ori, gene_use=None, cell_use=None,
        n_permu=9, epsilon=1e-16, seed=42, device=None
    ):

    model.to(device)
    model.eval()

    cell_flag = st_data_use.obs_names.isin(cell_use)

    result_permu = []
    result_records = []

    with tqdm(total=len(gene_use)) as t:  
        with torch.no_grad():
            for gi, genei in enumerate(gene_use):
                # print(f"{gi} {genei}")
                ans_permu = []
                ans_dcrecord = []
                np.random.seed(seed)
                for i in range(n_permu):
                    ExpData = st_data_use.X.copy()

                    permu_data = ExpData[cell_flag, gi].copy()
                    np.random.shuffle(permu_data)
                    ExpData[cell_flag, gi] = permu_data
                    ExpData = torch.from_numpy(ExpData).to(dtype=torch.float, device=device)

                    emb, exp_predict, h_val, grad, trj = model(ExpData, iftrj=True)
                    trj = trj.detach().cpu().numpy()
                    sim1 = kl_divergence(trj_ori, trj, epsilon)

                    ans_permu.append(sim1)
                    # ans_dcrecord.append(trj)

                result_permu.append(ans_permu)
                # result_records.append(ans_dcrecord)
                t.update(1)

        result_permu = np.asarray(result_permu)
        # result_records = np.asarray(result_records)
        # return result_permu, result_records
    return result_permu

### Perturbation analysis of the target cell’s surrounding cells
def permutation_singlegene_celltype_env(
        model, st_data_sel, trj_ori, gene_use=None, cell_use=None,
        n_permu=9, epsilon=1e-16, seed=42, device=None
    ):

    model.to(device)
    model.eval()

    gene_list = st_data_sel.var_names.tolist()
    cell_types = st_data_sel.obs.celltypes.unique().tolist()

    df_permu_all = pd.DataFrame([], index=gene_use)
    result_permu_all = {}

    with tqdm(total=len(cell_types)) as t:  
        with torch.no_grad():
            for ct in cell_types:
                cell_flag = (st_data_sel.obs.celltypes == ct).values & cell_use

                if cell_flag.sum() == 0:
                    continue

                result_permu = []
                for genei in gene_use:
                    gi = gene_list.index(genei)

                    ans_permu = []
                    np.random.seed(seed)
                    for i in range(n_permu):
                        ExpData = st_data_sel.X.copy()

                        permu_data = ExpData[cell_flag, gi].copy()
                        np.random.shuffle(permu_data)
                        ExpData[cell_flag, gi] = permu_data
                        ExpData = torch.from_numpy(ExpData).to(dtype=torch.float, device=device)

                        emb, exp_predict, h_val, grad, trj = model(ExpData, iftrj=True)
                        trj = trj.detach().cpu().numpy()

                        sim1 = kl_divergence(trj_ori, trj, epsilon)
                        ans_permu.append(sim1)

                    result_permu.append(ans_permu)
                    # result_records.append(ans_dcrecord)

                result_permu_all[ct] = np.asarray(result_permu)
                df_permu = pd.DataFrame(np.asarray(result_permu), index=gene_use)
                df_permu_all[ct] = df_permu.mean(1).values
                t.update(1)

    return df_permu_all, result_permu_all

