from utils_function import *

###################%% version control
sample_name = "DAGAST"
root_path = "/public3/Shigw/datasets/Simulated_data/"

data_folder = f"{root_path}/discrete/"
save_folder = f"{data_folder}/results/{sample_name}"
check_path(f"{data_folder}/results/")
check_path(save_folder)

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

## 构建细胞邻近图
def get_neighbor(st_data, st_data_use, n_neighbors = 9, n_externs = 10, ntype="extern"):
    if ntype == "extern":
        all_spots_name = st_data.obs_names.tolist()
        indices = [all_spots_name.index(ci) for ci in st_data_use.obs_names]
        kNNGraph_all = kneighbors_graph(st_data.obsm["spatial"], n_neighbors=n_externs, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
        cell_use = np.unique(np.reshape(np.asarray(kNNGraph_all.col), [len(st_data), n_externs])[indices].flatten())
        st_data_sel = st_data[np.array(all_spots_name)[cell_use]]
        use_spots_name = st_data_sel.obs_names.tolist()
        kNNGraph_use = kneighbors_graph(st_data_sel.obsm["spatial"], n_neighbors=n_neighbors, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
        kNNGraph_use = np.reshape(np.asarray(kNNGraph_use.col), [len(st_data_sel), n_neighbors])
        indices_use = np.array([use_spots_name.index(ci) for ci in st_data_use.obs_names])
    else:
        use_spots_name = st_data_use.obs_names.tolist()
        indices_use = np.array([use_spots_name.index(ci) for ci in st_data_use.obs_names])
        kNNGraph_use = kneighbors_graph(st_data_use.obsm["spatial"], n_neighbors=n_neighbors, include_self=True, mode="connectivity", n_jobs=-1).tocoo()
        kNNGraph_use = np.reshape(np.asarray(kNNGraph_use.col), [len(st_data_use), n_neighbors])

    return kNNGraph_use, indices_use


def run(st_data, st_data_use, args, knn = 30, cutof = 0.1, alpha = 1.0, beta = 0.5, theta1 = 0.1, theta2 = 0.1):
    torch.cuda.empty_cache()

    SEED = args["SEED"]
    device = args["device"]
    num_epoch1 = args["num_epoch1"]
    num_epoch2 = args["num_epoch2"]
    lr = args["lr"]
    scheduler = args["scheduler"]
    update_interval = args["update_interval"]
    batch_size = args["batch_size"]
    n_neighbors = args["n_neighbors"]

    nu.setup_seed(SEED)

    ######################### 1. 预处理，挑选候选细胞，构建细胞邻近图 ########################
    save_folder_cluster = f"{save_folder}/2.spatial_cluster/"
    nu.check_path(save_folder_cluster)

    kNNGraph_use, indices_use = get_neighbor(st_data, st_data_use, n_neighbors=n_neighbors, ntype="noextern")
    indices_torch = nu.np_to_torch(indices_use, device, dtype=torch.long, requires_grad=False)
    
    exp_imput = st_data_use.X
    exp_input = nu.np_to_torch(exp_imput, device, requires_grad=False)

    ######################### 2. 构建模型，推断出转移矩阵 ########################
    model = nu.DAGAST(args, kNNGraph_use, indices_use)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    ###################%% stage 01
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
            # if emb_TV < 0.01 or mse_loss * cutof > emb_TV:
            #     break

    torch.save(model, f"{save_folder_cluster}/model_{sel_batch}_stage1.pkl")

    model.eval()
    emb = model.get_emb(isall=False)
    emb_adata = sc.AnnData(emb)
    emb_adata.obs['Step'] = st_data_use.obs['Step'].values
    sc.pp.neighbors(emb_adata, use_rep='X', n_neighbors=knn)
    sc.tl.umap(emb_adata)

    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    ax = sc.pl.umap(emb_adata, color="Step", color_map='Spectral_r')
    plt.savefig(f"{save_folder_cluster}/1.umap_Step_{sel_batch}.pdf")

    ## 3.4 cluster
    sc.tl.leiden(emb_adata, resolution=1.0)         # res = 0.1
    print(f"{len(emb_adata.obs['leiden'].unique())} clusters")
    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    ax = sc.pl.umap(emb_adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
    plt.savefig(f"{save_folder_cluster}/2.umap_cluster_stage1_{sel_batch}.pdf")

    st_data_use.obs['emb_cluster'] = emb_adata.obs['leiden'].values.tolist()
    plt.close('all')
    plt.rcParams["figure.figsize"] = (5, 5)
    ax = sc.pl.embedding(st_data_use, basis="spatial", color="emb_cluster",size=15, s=10, show=False, title='clustering')
    plt.axis('off')
    plt.savefig(f"{save_folder_cluster}/2.spatial_cluster_stage1_{sel_batch}.pdf", bbox_inches='tight')


    # 计算每个点与目标点的欧氏距离
    target_point = np.array([0, 0])
    distances = np.linalg.norm(st_data_use.obs[['x', 'y']].values - target_point, axis=1)
    closest_index = np.argmin(distances)
    sel_cluster = st_data_use.obs.iloc[closest_index]['emb_cluster']

    flag = (st_data_use.obs['emb_cluster'].isin([sel_cluster])).values   
    st_data_use.obs['start_cluster'] = 0
    st_data_use.obs.loc[flag, 'start_cluster'] = 1
    start_flag = st_data_use.obs['start_cluster'].values    # 轨迹起点
    startflag_torch = nu.np_to_torch(start_flag, device, dtype=torch.long, requires_grad=False)

    ###################%% stage 02
    save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
    nu.check_path(save_folder_trajectory)

    torch.cuda.empty_cache()     
    theta1, theta2 = 0.1, 0.1

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
                startloss = torch.norm(trj[:, startflag_torch==1].sum(0), p=2)
                total_loss = total_loss + startloss * theta2

            total_loss.backward()
            optimizer.step()  
            if i % update_interval == 0:
                t.update(update_interval)
                t.set_postfix({'MSEloss' : mse_loss.item(), 'emb_TV' : emb_TV.item(), 'h_val' : h_val.item(), 's_val' : startloss.item()})

            old_loss = total_loss.item()

    model.eval()
    emb, exp_predict_stage2, h_val, grad, trj = model(exp_input, iftrj=True)
    trj = trj.detach().cpu().numpy()
    emb = emb[indices_torch].detach().cpu().numpy()
    st_data_use.obsm['emb'] = emb
    st_data_use.obsm['trans'] = trj

    kNN_use = kNNGraph_use[indices_use]
    xy_use = st_data_use.obsm['spatial']
    xy1 = st_data_use.obsm['spatial']

    # 绘制概率矩阵的热力图
    plt.figure(figsize=(10, 10))
    plt.scatter(xy1[:, 0], xy1[:, 1], c=trj.sum(0), cmap='Oranges', s=10, label="Original Points")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Interpolated Directional Field on a Grid")
    plt.legend()
    plt.savefig(f"{save_folder_trajectory}/plot_trj_{sel_batch}.png")

    # 保存模型
    torch.save(model, f"{save_folder_trajectory}/model_{sel_batch}.pkl")
    np.save(f"{save_folder_trajectory}/trj_{sel_batch}.npy", trj)
    np.save(f"{save_folder_trajectory}/emb_{sel_batch}.npy", emb)


    ######################### 3. UMAP、计算拟时序、得出分化轨迹 ########################
    ### 3.1 计算拟时序
    st_data_use = get_ptime(st_data_use)

    plot_spatial_complex(
        st_data, st_data_use, mode="time",
        value=st_data_use.obs['ptime'], title="ptime", pointsize=20,
        savename=f"{save_folder_trajectory}/1.spatial_Pseudotime2_{sel_batch}.pdf"
    )

    plot_spatial_complex(
        st_data, st_data_use, mode="time",
        value=st_data_use.obs['Step'] / st_data_use.obs['Step'].max(), title="Step", pointsize=20,
        savename=f"{save_folder_trajectory}/1.spatial_Pseudotime2_{sel_batch}.pdf"
    )


    ### 3.2 构建分化轨迹
    st_data_use.uns["E_grid"], st_data_use.uns["V_grid"] = get_velocity(
        st_data_use, basis="spatial", n_neigh_pos=knn, grid_num=50, smooth=0.5, density=0.7)

    xy1 = st_data.obsm['spatial']
    xy2 = st_data_use.obsm['spatial']

    plt.close('all')
    fig, axs = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o', c = st_data_use.obs['ptime'], 
        s=20, cmap='Spectral_r', legend = False, alpha=0.25)
    axs.quiver(st_data_use.uns['E_grid'][0],st_data_use.uns['E_grid'][1],st_data_use.uns['V_grid'][0],st_data_use.uns['V_grid'][1], 
        scale=0.2, linewidths=4, headwidth=5)
    plt.savefig(f"{save_folder_trajectory}/2.spatial_quiver2_{sel_batch}.pdf", format='pdf',bbox_inches='tight')


    ### 3.3 umap看特征拟合情况以及拟时序
    # model = torch.load(f"{save_folder}/model_{n_genes}_{date}.pkl")
    model.eval()
    emb = model.get_emb(isall=False)
    adata = sc.AnnData(emb)
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=knn)
    adata.obs['ptime'] = st_data_use.obs['ptime'].values
    adata.obs['Step'] = st_data_use.obs['Step'].values
    sc.tl.umap(adata)

    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    ax = sc.pl.umap(adata, color="ptime", color_map='Spectral_r')
    plt.savefig(f"{save_folder_trajectory}/3.umap_ptime_{sel_batch}.pdf")

    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    ax = sc.pl.umap(adata, color="Step", color_map='Spectral_r')
    plt.savefig(f"{save_folder_trajectory}/3.umap_Step_{sel_batch}.pdf")

    st_data_use.obs['emb_cluster'] = st_data_use.obs['Step']

    ### 3.4 cluster 这里确实要进一步通过聚类分析验证准确性，同时也算一些结果
    ## umap - cluster
    sc.tl.leiden(adata, resolution=0.5)         # res = 0.1
    st_data_use.obs['emb_cluster'] = adata.obs['leiden'].astype(int).to_numpy()
    print(f"{len(adata.obs['leiden'].unique())} clusters")

    umap_color = {str(c) : mycolor[idx] for idx, c in enumerate(adata.obs['leiden'].unique())}
    plt.close('all')
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    # ax = sc.pl.umap(adata, color="leiden", color_map='Spectral_r', legend_loc='on data', legend_fontweight='normal')
    ax = sc.pl.umap(adata, color="leiden", palette=umap_color, legend_loc='on data', legend_fontweight='normal')
    plt.savefig(f"{save_folder_trajectory}/5.embedding_umap_cluster_DAGAST_{sel_batch}.pdf")

    plot_spatial_complex(
        st_data, st_data_use, mode="cluster",
        value=st_data_use.obs['emb_cluster'], key="emb_cluster", title="emb_cluster",
        savename=f"{save_folder_trajectory}/5.spatial_cluster_DAGAST_{sel_batch}.pdf"
    )

    ans = cal_stindex(st_data_use.obs['Step'], st_data_use.obs['ptime'], st_data_use.obsm['spatial'], k=knn, cutrate=0.01)
    return ans


######################### 1. 读取数据，进行必要的处理 ########################
all_expdata = pd.read_table(f"{data_folder}/sim_path_count.txt", index_col=0).T
all_metadata = pd.read_table(f"{data_folder}/sim_path_metadata.txt", index_col=0)

all_batch = all_metadata.Batch.unique()
ans_list = []
sel_batch = all_batch[9]

df_ptime = all_metadata[['Step']]
df_ptime['Ptime'] = 0.0

for sel_batch in all_batch:
    print(sel_batch)

    metadata = all_metadata.loc[all_metadata.Batch == sel_batch]
    expdata = all_expdata.loc[metadata.index]

    ## 获取一个新的
    st_data = AnnData(X=expdata, obs=metadata)
    st_data.X = st_data.X.astype('float64')  # this is not required and results will be comparable without it
    st_data.obsm['spatial'] = st_data.obs[['x', 'y']].values
    st_data_use = st_data

    # 数据处理 归一化和scale
    n_genes = 300
    sc.pp.normalize_total(st_data_use, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
    sc.pp.log1p(st_data_use)
    sc.pp.highly_variable_genes(st_data_use, n_top_genes=n_genes)
    sc.pp.scale(st_data_use)
    st_data_use_hvg = st_data_use[:, st_data_use.var['highly_variable'].values]
    st_data_use = st_data_use[:, st_data_use.var['highly_variable'].values]

    args = {
        "num_input" : n_genes,  
        "num_emb" : 256,        # 256  512
        "dk_re" : 16,
        "nheads" : 1,               #  1    4
        "droprate" : 0.15,          #  0.25,
        "leakyalpha" : 0.15,        #  0.15,
        "resalpha" : 0.5, 
        "bntype" : "BatchNorm",     # LayerNorm BatchNorm
        "device" : torch.device('cuda:1'), 
        "mode" : "Train", 

        "info_type" : "linear",
        "iter_type" : "SCC",
        "iter_num" : 200,

        "num_epoch1" : 1000, 
        "num_epoch2" : 1000, 
        "lr" : 0.001, 
        "batch_size" : 0,
        "update_interval" : 1, 
        "eps" : 1e-5,
        "scheduler" : None, 
        "SEED" : 24,
        "n_neighbors" : 9
    }

    cutof = 0.1
    alpha, beta = 1.0, 0.5
    theta1, theta2 = 0.1, 0.1
    knn = 30
    sim = run(st_data, st_data_use, args, knn, cutof, alpha, beta, theta1, theta2)

    df_ptime.loc[st_data_use.obs_names, 'Ptime'] = st_data_use.obs['ptime']
    ans_list.append(sim)

df_ptime.to_csv(f"{save_folder}/df_ptime.csv")
ans_list

# ans_list = 
[0.6986010847854444,
 0.7125082651060013,
 0.6936650757067873,
 0.7033144619298685,
 0.7131730025148231,
 0.6945092774016415,
 0.7112438173159369,
 0.7065188575231229,
 0.7065033773029153,
 0.718893746195292]


