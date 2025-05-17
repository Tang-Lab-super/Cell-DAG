import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
plt.rcParams["figure.dpi"] = 300  # 分辨率

def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f'mkdir {path}')

def setup_seed(seed=24):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def cal_stindex(seq1, seq2, coords, data_flag=None, k=9, cutrate=0.01, cutoff=None):

    from scipy.stats import spearmanr, rankdata
    from sklearn.neighbors import kneighbors_graph
    seq1 = rankdata(seq1)
    seq2 = rankdata(seq2)

    if data_flag is None:
        data_flag = np.array([True] * len(seq1))

    ## 1. spearman相关性
    sim1, p1 = spearmanr(seq1[data_flag], seq2[data_flag])
    ## 判断是否为负
    if sim1 < 0:
        vmax = max(seq2)
        seq2 = rankdata(vmax-seq2)
        sim1, p1 = spearmanr(seq1[data_flag], seq2[data_flag])

    ## 2. 时空连续性
    # 2.1 空间上，认为邻近状态是相似的
    adj = kneighbors_graph(coords, n_neighbors=k, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
    kNNGraphIndex = np.reshape(np.asarray(adj.col), [len(seq1), k])
    kNNGraphIndex_sel = kNNGraphIndex[data_flag]

    # 2.2 时间上，认为邻近状态是相似的
    if cutoff is None:
        cutoff = cutrate * len(kNNGraphIndex)

    sim2_list = []
    for nibor in kNNGraphIndex_sel:
        seq1_k = np.sort(seq1[nibor])
        seq2_k = np.sort(seq2[nibor])
        sim2_list.append(np.sum(np.abs(seq1_k - seq2_k) <= cutoff) / k)

    # sim2_list = [np.sum(np.abs(seq1[nibor] - seq2[nibor]) <= cutoff) / k for nibor in kNNGraphIndex_sel]
    sim2 = np.mean(sim2_list)
    total_sim = (abs(sim1) + abs(sim2)) / 2

    print(f'spearmanr = {sim1:.6f} | stindex = {sim2:.6f} | total sim = {total_sim:.6f}.')
    return total_sim

import seaborn as sns
def plot_feature(xy1, xy2,  value, title):
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=20)
    sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o',
                    c = value, s=20,  cmap='Spectral_r', legend = True)
    # plt.title(title)
    plt.axis('off')

def plot_cluster(xy1, st_data_sel, title):
    mycolor = [
        "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
        "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
        "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
        "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
    ]
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=20)
    for idx, ci in enumerate(st_data_sel.obs.emb_cluster.unique().tolist()):
        subda = st_data_sel[st_data_sel.obs.emb_cluster == ci, :]
        sns.scatterplot(x=subda.obsm['spatial'][:, 0], y=subda.obsm['spatial'][:, 1], marker='o', c=mycolor[idx], s=20)
        plt.text(subda.obsm['spatial'][:, 0].mean(), subda.obsm['spatial'][:, 1].mean(), str(ci), fontsize=8)
    # plt.title(title)
    plt.axis('off')

def plot_spatial(st_data, st_data_sel, mode="time", value=None, figsize=(5, 5), title=None, savename='./fig.pdf'):

    assert value is not None, "value is None."
    assert title is not None, "title is None."

    if mode=="time":
        assert value is not None, "value is None."
    elif mode=="cluster":
        assert "emb_cluster" in st_data_sel.obs.columns, "emb_cluster is empty."

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    plt.subplot(1,1,1)
    if mode=="time":
        plot_feature(st_data.obsm['spatial'], st_data_sel.obsm['spatial'],  value, title)
    elif mode == "cluster":
        plot_cluster(st_data.obsm['spatial'], st_data_sel, title)
    plt.tight_layout()
    plt.savefig(savename)

###################%% version control
sample_name = "Slingshot"
root_path = "/public3/Shigw/datasets/Simulated_data/"

data_folder = f"{root_path}/discrete/"
save_folder = f"{data_folder}/results/{sample_name}"
check_path(f"{data_folder}/results/")
check_path(save_folder)

######################### 1. 读取数据，进行必要的处理 ########################
all_expdata = pd.read_table(f"{data_folder}/sim_path_count.txt", index_col=0).T
all_metadata = pd.read_table(f"{data_folder}/sim_path_metadata.txt", index_col=0)

df_ans = pd.read_table(f"{save_folder}/df_pseudo.txt", index_col=0)
df_ans.index = all_metadata.index
df_ans = df_ans.fillna(0)
all_batch = all_metadata.Batch.unique()
sel_batch = all_batch[0]

ans_list = []
df_ptime = all_metadata[['Step']]
df_ptime['Ptime'] = 0.0

for sel_batch in all_batch:
    print(sel_batch)
    setup_seed(seed=24)
    
    ## 获取一个新的
    metadata = all_metadata.loc[all_metadata.Batch == sel_batch]
    expdata = all_expdata.loc[metadata.index]

    st_data = AnnData(X=expdata, obs=metadata)
    st_data.X = st_data.X.astype('float64')  # this is not required and results will be comparable without it
    # st_data.obs['x'] = st_data.obs['x'] - st_data.obs['x'].min()
    # st_data.obs['y'] = st_data.obs['y'] - st_data.obs['y'].min()
    st_data.obsm['spatial'] = st_data.obs[['x', 'y']].values
    st_data.obs['imagerow'] = st_data.obs['x']
    st_data.obs['imagecol'] = st_data.obs['y']

    # 数据处理 归一化和scale

    n_genes = 300
    st_data.layers['counts'] = st_data.X
    sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
    sc.pp.log1p(st_data)
    st_data.raw = st_data
    sc.pp.highly_variable_genes(st_data, n_top_genes=n_genes)
    sc.pp.scale(st_data)
    st_data = st_data[:, st_data.var['highly_variable'].values]

    ## 首先聚类
    save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
    check_path(save_folder_cluster)

    sc.pp.neighbors(st_data, use_rep='X', n_neighbors=30)
    sc.tl.umap(st_data)

    ## 然后选取起点进行拟时序
    save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
    check_path(save_folder_trajectory)

    st_data.obsm['X_spatial'] = st_data.obsm['spatial']
    st_data.obs["ptime"] = df_ans.loc[metadata.index, 'pseudo']
    st_data.obs['ptime'] = st_data.obs['ptime'] / st_data.obs['ptime'].max()

    ## 绘制拟时序
    sc.tl.umap(st_data)
    plt.close('all')
    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    ax = sc.pl.umap(st_data, color="ptime", color_map='Spectral_r')
    plt.savefig(f"{save_folder_trajectory}/2.umap_ptime_{sel_batch}.pdf")

    # plt.close('all')
    # fig = plt.figure(figsize=(5, 5))
    # plt.subplot(1, 1, 1)
    # sc.pl.embedding(st_data, basis="spatial", color="ptime",size=15, s=20, show=False, title='ptime')
    # plt.savefig(f"{save_folder_trajectory}/3.spatial_ptime_{sel_batch}.pdf")

    plot_spatial(
        st_data, st_data, mode="time",
        value=st_data.obs['ptime'], title="DC1",
        savename=f"{save_folder_trajectory}/3.spatial_ptime_{sel_batch}.pdf"
    )

    ans = cal_stindex(st_data.obs['Step'], st_data.obs['ptime'], st_data.obsm['spatial'], k=30, cutrate=0.01)

    df_ptime.loc[st_data.obs_names, 'Ptime'] = st_data.obs['ptime']
    ans_list.append(ans)

df_ptime.to_csv(f"{save_folder}/df_ptime.csv")
ans_list

# ans_list = 
[0.57543626057848,
 0.5725169444615635,
 0.5660982157765365,
 0.5275373185302765,
 0.5470939413262413,
 0.5637367647854593,
 0.5724368193739463,
 0.5448538999620094,
 0.5333793800012653,
 0.545107971107628]




