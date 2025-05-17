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
sample_name = "250113"
root_path = "/public3/Shigw/"
data_folder = f"{root_path}/datasets/SeqFISH/"
save_folder = f"{data_folder}/results/{sample_name}"
dt.check_path(save_folder)

######################### 准备数据和模型 ########################
save_folder_trajectory = f"{save_folder}/3.spatial_trajectory/"
dt.check_path(save_folder_trajectory)

##### 导入数据
st_data = sc.read_h5ad(data_folder + "/st_data.h5ad")
sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
sc.pp.scale(st_data)

celltypes = ['Spinal cord', 'NMP']      # permutation cells
st_data_use = st_data[st_data.obs.celltypes.isin(celltypes), :].copy()

n_neighbors = 9
n_externs = 10
kNNGraph_use, indices_use, st_data_sel = dt.get_neighbor(st_data, st_data_use, n_neighbors=n_neighbors, n_externs=n_externs, ntype="extern")

##### 导入模型和迁移矩阵
SEED = 42
dt.setup_seed(SEED)
trj_ori = np.load(f"{save_folder_trajectory}/trj_{sample_name}.npy")
model = torch.load(f"{save_folder_trajectory}/model_{sample_name}.pkl")

model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

att_gene_re = att_gene_re_all[indices_use]
att_gene_cc = att_gene_cc_all[indices_use]
att_cell = att_cell_all[indices_use]

celltypes_list = np.array(st_data_use.obs.celltypes.tolist())

# ######################### 4. gene_attention可解释性分析 ########################
save_folder_attention_gene = f"{save_folder}/4.gene_attention/"
dt.check_path(save_folder_attention_gene)

### 4.1 细胞内外gene attention对比
ci = 'Spinal cord'
cell_use = st_data_use.obs_names[st_data_use.obs['celltypes'].isin(celltypes_list)].tolist()
gene_use = st_data_use.var_names.tolist()

save_folder_attention_gene_ci = f"{save_folder_attention_gene}/{ci}/"
dt.check_path(save_folder_attention_gene_ci)
saveFolder_geneAtt_01module = f"{save_folder_attention_gene_ci}/1.module/"
dt.check_path(saveFolder_geneAtt_01module)

flag = (celltypes_list == ci)
att_gene_cc_sel = att_gene_cc[flag]
att_gene_cc_sel_df = pd.DataFrame(np.mean(att_gene_cc_sel, 0), index=gene_use, columns=gene_use)

## 细胞内gene attention
att_gene_re_sel = att_gene_re[flag]
att_gene_re_sel_df = pd.DataFrame(np.mean(att_gene_re_sel, 0), index=gene_use, columns=gene_use)

plt.close('all')
clustermap = sns.clustermap(att_gene_re_sel_df, square=True, row_cluster=True, col_cluster=True, z_score=1, vmin=-1, vmax=1, cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('att gene')
plt.savefig(f"{saveFolder_geneAtt_01module}/1.re_clustermap_{ci}.png", dpi=300)

row_order = clustermap.dendrogram_row.reordered_ind     # 聚类并提取顺序（按行）
col_order = clustermap.dendrogram_col.reordered_ind
sorted_labels_r = [gene_use[i] for i in row_order]
sorted_labels_c = [gene_use[i] for i in col_order]
# df_sorted = att_gene_cc_sel_df.loc[sorted_labels_r, sorted_labels_c]

# 对比细胞外
plt.close('all')
fig = plt.figure(figsize=(15, 15))
plt.subplot(1, 1, 1)
df_sorted = att_gene_cc_sel_df.loc[sorted_labels_r, sorted_labels_c]
sns.clustermap(df_sorted, square=True, row_cluster=False, col_cluster=False, z_score=1, vmin=-1, vmax=1, cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('att gene')
plt.savefig(f"{saveFolder_geneAtt_01module}/1.cc_comp_re_clustermap_{ci}.png", dpi=300)

## 细胞外gene attention
plt.close('all')
fig = plt.figure(figsize=(15, 15))
plt.subplot(1, 1, 1)
clustermap = sns.clustermap(att_gene_cc_sel_df, square=True, row_cluster=True, col_cluster=True, z_score=1, vmin=-1, vmax=1, cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('att gene')
plt.savefig(f"{saveFolder_geneAtt_01module}/1.cc_clustermap_{ci}.png", dpi=300)
print(f"celltype {ci}")


### 4.2 对指定细胞类型的细胞外attention进行聚类分析，然后再空间上对其进行细胞内外的对比分析，看这种互作是否有空间特异性
# 聚类并提取顺序（按行）
saveFolder_geneAtt_02spatialModule = f"{save_folder_attention_gene_ci}/2.spatialModule/"
dt.check_path(saveFolder_geneAtt_02spatialModule)

## 分类别
dt.setup_seed(SEED)
nc, ng, ng = att_gene_cc_sel.shape
att_gene_cc_sel_reshape = att_gene_cc_sel.reshape(nc, ng * ng)

## 表达模式
adata_geneatt = sc.AnnData(att_gene_cc_sel_reshape)
# adata_geneatt.obsm['spatial'] = st_data_use[flag, :].obsm['spatial']
sc.pp.pca(adata_geneatt)
sc.pp.neighbors(adata_geneatt, n_neighbors=30, n_pcs=50)
# sc.pp.neighbors(adata_geneatt, use_rep='X', n_neighbors=30)
sc.tl.umap(adata_geneatt)
sc.tl.leiden(adata_geneatt, resolution=0.3)         # res = 0.5
# sc.tl.leiden(adata_geneatt, resolution=0.1)         # res = 0.1
# adata_geneatt.obs.leiden = adata_geneatt.obs['leiden'].astype(int).to_numpy()       # 对比分析
print(f"{len(adata_geneatt.obs['leiden'].unique())} clusters")
adata_geneatt.write_h5ad(f"{saveFolder_geneAtt_02spatialModule}/adata_geneatt.h5")


## 绘制一下UMAP的类别标签
plt.close('all')
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
# ax = sc.pl.umap(adata_geneatt, color="leiden", color_map='Spectral', legend_loc='on data', legend_fontweight='normal')
ax = sc.pl.umap(
    adata_geneatt, color="leiden", 
    palette={ci : dt.mycolor[int(idx)] for idx, ci in enumerate(adata_geneatt.obs['leiden'].unique())}, 
    legend_loc='on data', legend_fontweight='normal')
plt.savefig(f"{saveFolder_geneAtt_02spatialModule}/2.umap_cluster_cc1.pdf")

## 看是否有空间上连续性 
st_data_use_sel = st_data_use[flag, :]
st_data_use_sel.obs['emb_cluster'] = adata_geneatt.obs.leiden.values
dt.plot_spatial(
    st_data, st_data_use_sel, mode="cluster",
    value=st_data_use_sel.obs['emb_cluster'], title="subcluster",
    savename=f"{saveFolder_geneAtt_02spatialModule}/2.spatial_{ci}_subcluster1.pdf"
)

## 绘制各类的gene attention层次聚类图
for idx, cj in enumerate(adata_geneatt.obs['leiden'].unique().tolist()):
    flag_ci = adata_geneatt.obs.leiden == cj
    att_gene_re_cj = pd.DataFrame(np.mean(att_gene_re_sel[flag_ci], 0), index=gene_use, columns=gene_use)
    att_gene_cc_cj = pd.DataFrame(np.mean(att_gene_cc_sel[flag_ci], 0), index=gene_use, columns=gene_use)

    plt.close('all')
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(1, 1, 1)
    clustermap = sns.clustermap(
        att_gene_re_cj, square=True, row_cluster=True, col_cluster=True, 
        z_score=1, vmin=-1, vmax=1, cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
    plt.title(f'celltype {ci}, subcluster {cj}')
    plt.savefig(f"{saveFolder_geneAtt_02spatialModule}/2.re_subClustermap_{cj}.png", dpi=300)

    # row_order = clustermap.dendrogram_row.reordered_ind     # 聚类并提取顺序（按行）
    # col_order = clustermap.dendrogram_col.reordered_ind
    # sorted_labels_r = [gene_use[i] for i in row_order]
    # sorted_labels_c = [gene_use[i] for i in col_order]
    # df_sorted = att_gene_cc_sel_df.loc[sorted_labels_r, sorted_labels_c]

    plt.close('all')
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(1, 1, 1)
    # df_sorted = att_gene_cc_sel_df.loc[sorted_labels_r, sorted_labels_c]
    # sns.clustermap(
    #     df_sorted, square=True, row_cluster=False, col_cluster=False, 
    #     z_score=1, vmin=-1, vmax=1, cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))

    sns.clustermap(att_gene_cc_cj, square=True, row_cluster=True, col_cluster=True, z_score=1, vmin=-1, vmax=1, cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
    plt.title(f'celltype {ci}, subcluster {cj}')
    plt.savefig(f"{saveFolder_geneAtt_02spatialModule}/2.cc_subClustermap_{cj}.pdf")

    print(f"subcluster {cj}")


def add_module_score(adata, gene_sets, ctrl_size=100, n_bins=24, use_raw=False):
    """
    模拟 Seurat 的 AddModuleScore 功能，计算模块分数。
    
    参数：
    - adata: AnnData 对象
    - gene_sets: 包含基因模块的列表，例如 [['GeneA', 'GeneB'], ['GeneC', 'GeneD']]
    - ctrl_size: 匹配的背景基因数量
    - n_bins: 基因分组数量，用于选择背景基因
    - use_raw: 是否使用 raw 数据计算分数（如果 raw 数据可用）
    
    返回：
    - adata: 增加模块分数的 AnnData 对象
    """
    # 确保输入基因模块在数据中存在
    gene_lists = []
    for gene_set in gene_sets:
        valid_genes = [gene for gene in gene_set if gene in adata.var_names]
        gene_lists.append(valid_genes)
    
    # 使用 raw 数据或当前数据层
    expression_data = adata.raw.X if use_raw and adata.raw is not None else adata.X
    
    # 平均表达量分组
    avg_exp = np.array(expression_data.mean(axis=0)).flatten()
    bins = pd.qcut(avg_exp, n_bins, labels=False)
    
    # 计算模块分数
    for idx, gene_set in enumerate(gene_lists):
        # 目标基因模块的表达
        target_indices = [adata.var_names.get_loc(gene) for gene in gene_set]
        module_expr = expression_data[:, target_indices].mean(axis=1)
        
        # 匹配背景基因
        ctrl_indices = []
        for gene in gene_set:
            bin_idx = bins[adata.var_names.get_loc(gene)]
            candidates = np.where(bins == bin_idx)[0]
            np.random.shuffle(candidates)
            ctrl_indices.extend(candidates[:ctrl_size])
        
        # 背景基因的表达
        ctrl_expr = expression_data[:, ctrl_indices].mean(axis=1)
        
        # 计算模块分数（目标模块表达 - 背景模块表达）
        module_score = module_expr - ctrl_expr
        adata.obs[f'module_score_{idx}'] = module_score.A1 if hasattr(module_score, 'A1') else module_score
    
    return adata

## 分析各类别最受影响的基因，并计算其addmodulescore
geneatt_cluster_set = []      # ['0', '1', '2', '3', '4', '5', '6', '7']
for idx, cj in enumerate(['0', '1', '2', '3', '4', '5', '6']):
    flag_ci = adata_geneatt.obs.leiden.values == cj
    att_gene_sel_cj = pd.DataFrame(np.mean(att_gene_cc_sel[flag_ci], 0), index=gene_use, columns=gene_use)
    geneatt_cluster_set += [att_gene_sel_cj.sum().sort_values(ascending=False).index[:10]]     # 取出top10

st_data_geneatt_modulescore = add_module_score(st_data.copy(), geneatt_cluster_set, ctrl_size=100, n_bins=24)

for idx, cj in enumerate(['0', '1', '2', '3', '4', '5', '6']):
    st_data_geneatt_modulescore.obs['plot'] = st_data_geneatt_modulescore.obs[f"module_score_{cj}"]
    dt.plot_spatial_gene(
        st_data_geneatt_modulescore, 'plot', figsize=(5, 5), title=f"module_score_{cj}", 
        pointsize=10, savename=f"{saveFolder_geneAtt_02spatialModule}/spatial_module_score_{cj}.png")



## 表达模式聚类分析 - 基因聚类
geneatts = []
# adata_geneatt_label = ['6', '0', '5', '1', '2', '4', '3']
adata_geneatt_label = ['5', '0', '4', '1', '2', '3', '6']
df_gene_attention = pd.DataFrame(index=gene_use)
for cj in adata_geneatt_label:
    flag_ci = adata_geneatt.obs.leiden == cj
    att_gene_sel_cj = pd.DataFrame(np.mean(att_gene_cc_sel[flag_ci], 0), index=gene_use, columns=gene_use)

    df_gene_attention[cj] = att_gene_sel_cj.sum(0)
    geneatts.append(att_gene_cc_sel[flag_ci])

ci = 'Spinal cord'
result_permu_sorted = pd.read_csv(f"{save_folder_trajectory}/5.permutation_single_gene_{ci}.csv", index_col=0)
# flag = result_permu_sorted.index[result_permu_sorted.total_sim > 0.005]
flag = result_permu_sorted.index[:300]
df_gene_attention = df_gene_attention.loc[flag, :]

row_mean = df_gene_attention.mean(axis=1)
row_std = df_gene_attention.std(axis=1)
df_gene_attention = df_gene_attention.sub(row_mean, axis=0).div(row_std, axis=0)

dt.setup_seed(SEED)
adata_geneatt_patten = sc.AnnData(df_gene_attention)
adata_geneatt_patten.var['index'] = [x for x in range(len(adata_geneatt_patten.var_names))]
sc.pp.neighbors(adata_geneatt_patten, use_rep='X', n_neighbors=30)
sc.tl.leiden(adata_geneatt_patten, resolution=1.0)         # res = 0.1
print(f"{len(adata_geneatt_patten.obs['leiden'].unique())} clusters")

for ci in adata_geneatt_patten.obs['leiden'].unique():
    sub_adata_geneatt = adata_geneatt_patten[adata_geneatt_patten.obs.leiden == ci].copy()
    sub_adata_geneatt = sub_adata_geneatt.T

    plt.close('all')
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(1, 1, 1)

    for cj, gene in enumerate(sub_adata_geneatt.var_names):
        sub_adata_geneatt.obs['plot'] = sub_adata_geneatt[:, gene].X.T[0].tolist()
        plt.plot(sub_adata_geneatt.obs['index'], sub_adata_geneatt.obs['plot'], marker='o', label=gene)

    plt.xticks(np.arange(len(sub_adata_geneatt)), sub_adata_geneatt.obs_names.tolist())
    plt.title(f'gene Patten {ci}')
    plt.xlabel('class') 
    plt.ylabel('Att.')
    # plt.legend()
    plt.savefig(f"{saveFolder_geneAtt_02spatialModule}/cc_patten_{ci}.pdf")

    print(ci)


adata_geneatt_patten.write_h5ad(f"{saveFolder_geneAtt_02spatialModule}/adata_geneatt_patten.h5")


### 4.3 指定特征基因在细胞内外进行分析,可能要参考一些信号通路情况
# 首先看看基因表达
st_data_plot = sc.read_h5ad(data_folder + "/st_data.h5ad")
sc.pp.normalize_total(st_data_plot)
sc.pp.log1p(st_data_plot)
sc.pp.scale(st_data_plot)

patten_genesets_all = {}

for pi in adata_geneatt_patten.obs.leiden.unique():
    print(f"patten {pi}")
    save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{pi}/"
    dt.check_path(save_folder_attention_gene_ci_patten)
    dt.check_path(f"{save_folder_attention_gene_ci_patten}/gene_spatial_plot/")
    patten_genesets = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs.leiden == pi].tolist()

    patten_genesets_all[pi] = patten_genesets

    for plotgene in patten_genesets:
        st_data_plot.obs['plot'] = st_data_plot[:, plotgene].X.T[0].tolist()
        dt.plot_spatial_gene(st_data_plot, 'plot', figsize=(5, 5), title=plotgene, pointsize=10, savename=f"{save_folder_attention_gene_ci_patten}/gene_spatial_plot/spatial_gene_{plotgene}.png")


import json
json_geneset = json.dumps(patten_genesets_all)
with open(f'{saveFolder_geneAtt_02spatialModule}/patten_genesets_all.json', 'w') as json_file:
    json_file.write(json_geneset)

with open(f'{saveFolder_geneAtt_02spatialModule}/patten_genesets_all.json', 'r') as fcc_file:
    json_file = json.load(fcc_file)
    print(json_file)


## 对每个类别的基因集进行通路富集分析go 、 KEGG
''' /public1/yuchen/software/miniconda3/envs/R4.2_yyc/bin/R

library(clusterProfiler)
# library(org.Hs.eg.db)  # 人类的注释数据库（如果是小鼠，则使用 org.Mm.eg.db）
library(org.Mm.eg.db)
library(enrichplot)
library(ggplot2)
library(glue)

# R语言中进行通路富集分析
library(jsonlite)
json_data <- fromJSON("/public3/Shigw//datasets/SeqFISH//results/250116/4.gene_attention//Spinal cord//2.spatialModule/patten_genesets_all.json")

root_path = "/public3/Shigw//datasets/SeqFISH//results/250116/4.gene_attention//Spinal cord//2.spatialModule/"

for(ci in names(json_data)) {       # "4" "1" "0" "3" "5" "2"
    # ci = '4'
    save_path = glue("{root_path}/Patten_{ci}")

    symbol_genes <- as.character(json_data[[ci]])
    entrez_genes <- bitr(symbol_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db, drop = TRUE)
    entrez_genes = as.data.frame(entrez_genes)
    write.table(entrez_genes, glue('{save_path}/entrezid.csv'), quote=F, sep=',')

    ## go富集分析
    go_enrichment <- enrichGO(
        gene          = entrez_genes$ENTREZID,  # 输入基因 ID
        OrgDb         = org.Mm.eg.db,           # 使用人类注释数据库
        ont           = "BP",                   # GO 分类：BP, MF, CC
        pvalueCutoff  = 0.05,                   # P 值阈值
        qvalueCutoff  = 0.2,                    # Q 值阈值
        readable      = TRUE                    # 将 Entrez ID 转换为 Symbol
    )
    df_go_enrichment = as.data.frame(go_enrichment)
    head(df_go_enrichment)

    # 可视化 GO 结果（点图）
    pdf(file=glue("{save_path}/go_dotplot.pdf"), width=5, height=5)
    print(dotplot(go_enrichment, showCategory = 10) + ggtitle("GO Enrichment - Biological Process"))
    dev.off()

    # 可视化 GO 结果（柱状图）
    pdf(file=glue("{save_path}/go_barplot.pdf"), width=5, height=5)
    print(barplot(go_enrichment, showCategory = 10, title = "GO Enrichment - BP"))
    dev.off()

    write.table(df_go_enrichment, glue('{save_path}/df_go_enrichment.csv'), quote=F, sep=',')

    ## KEGG富集分析
    kegg_enrichment <- enrichKEGG(
        gene          = entrez_genes$ENTREZID,  # 输入基因 ID
        organism      = 'mmu',                  # 物种：人类为 'hsa'，小鼠为 'mmu'
        keyType = "kegg", 
        pvalueCutoff  = 0.05,                   # P 值阈值
        qvalueCutoff  = 0.2                     # Q 值阈值
    )

    df_kegg_enrichment = as.data.frame(kegg_enrichment)
    head(df_kegg_enrichment)

    if(nrow(df_kegg_enrichment) != 0) {
        # 可视化 KEGG 结果（点图）
        pdf(file=glue("{save_path}/kegg_dotplot.pdf"), width=5, height=5)
        print(dotplot(kegg_enrichment, showCategory = 10) + ggtitle("KEGG Pathway Enrichment"))
        dev.off()

        # 可视化 KEGG 通路富集（网络图）
        pdf(file=glue("{save_path}/kegg_cnetplot.pdf"), width=5, height=5)
        print(cnetplot(kegg_enrichment, showCategory = 5, foldChange = NULL))
        dev.off()  
    }
    write.table(df_kegg_enrichment, glue('{save_path}/df_kegg_enrichment.csv'), quote=F, sep=',')

    print(ci)
}



# genelist = c()
# for(ci in names(json_data)) {       # "4" "1" "0" "3" "5" "2"
#     symbol_genes <- as.character(json_data[[ci]])
#     entrez_genes <- bitr(symbol_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db, drop = TRUE)
#     genelist = c(genelist, entrez_genes$ENTREZID)
# }

# ## go富集分析
# go_enrichment <- enrichGO(
#     gene          = entrez_genes$ENTREZID,  # 输入基因 ID
#     OrgDb         = org.Mm.eg.db,          # 使用人类注释数据库
#     ont           = "CC",                  # GO 分类：BP, MF, CC
#     pvalueCutoff  = 0.05,                  # P 值阈值
#     qvalueCutoff  = 0.2,                   # Q 值阈值
#     readable      = TRUE                   # 将 Entrez ID 转换为 Symbol
# )
# df_go_enrichment = as.data.frame(go_enrichment)
# head(df_go_enrichment)

# # 可视化 GO 结果（点图）
# pdf(file=glue("{root_path}/all_go_dotplot_cc.pdf"), width=5, height=5)
# print(dotplot(go_enrichment, showCategory = 10) + ggtitle("GO Enrichment - Biological Process"))
# dev.off()

# # 可视化 GO 结果（柱状图）
# pdf(file=glue("{root_path}/all_go_barplot_cc.pdf"), width=5, height=5)
# print(barplot(go_enrichment, showCategory = 10, title = "GO Enrichment - BP"))
# dev.off()

# write.table(df_go_enrichment, glue('{root_path}/df_all_go_enrichment_cc.csv'), quote=F, sep=',')

'''
df_gos = []
for pi in adata_geneatt_patten.obs.leiden.unique():
    tmppath = f"{saveFolder_geneAtt_02spatialModule}/Patten_{pi}/"
    df_tmp = pd.read_csv(f"{tmppath}/df_go_enrichment.csv")
    df_gos.append(df_tmp)

### 然后在每个patten中，从全局的角度出发，看互作对，并进一步分析
''' /public1/yuchen/software/miniconda3/envs/R4.2_yyc/bin/R

library(clusterProfiler)
library(org.Mm.eg.db)  # 小鼠物种

library(glue)
library(jsonlite)
library(CellChat)

options(timeout = 300) # 设置超时时间为 300 秒

json_data <- fromJSON("/public3/Shigw//datasets/SeqFISH//results/250116/4.gene_attention//Spinal cord//2.spatialModule/patten_genesets_all.json")
root_path = "/public3/Shigw//datasets/SeqFISH//results/250116/4.gene_attention//Spinal cord//2.spatialModule/"


for(ci in names(json_data)) {       # "4" "1" "0" "3" "5" "2"
    ci = '2'
    symbol_genes <- as.character(json_data[[ci]])


    # 加载配体-受体数据库
    CellChatDB.use <- CellChatDB.mouse  # 筛选基因相关对

    # 查找作为配体的基因
    ligand_pairs <- CellChatDB.use$interaction[CellChatDB.use$interaction$ligand %in% symbol_genes, ]
    receptor_pairs <- CellChatDB.use$interaction[CellChatDB.use$interaction$receptor %in% symbol_genes, ]

    # 查看结果
    print(ligand_pairs)
    print(receptor_pairs)


# 查看互作对
head(interactions)

CellChatDB.mouse$interaction
CellChatDB.mouse$cofactor
CellChatDB.mouse$complex
CellChatDB.mouse$geneInfo

df_lrpairs = as.data.frame(CellChatDB.mouse$interaction)
write.table(df_lrpairs, glue('{root_path}/df_lrpairs.csv'), quote=F, sep=',')

'''

### 使用t-test对系数进行检验
def attention_t_test(index_row, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.01, cutrate = 1):
    from scipy.stats import ttest_rel
    # from scipy.stats import wilcoxon
    
    df_ans = pd.DataFrame(0, index=index_row, columns=index_row)    # df_ans中，行表示receptor，列表示ligand
    df_Pvalue = pd.DataFrame(0.0, index=index_row, columns=index_row)

    with tqdm(total=len(index_row)) as t:  
        for ni, gi in enumerate(index_row):         # receptor
            for nj, gj in enumerate(index_row):     # ligand
                list_re = att_gene_re_sel_df_patten[:, ni, nj]      # attentionMat中，行表示receptor，列表示ligand
                list_cc = att_gene_cc_sel_df_patten[:, ni, nj]
                # stat, p_value = wilcoxon(list_re, list_cc)
                # t_stat, p_value = ttest_rel(list_re, list_cc)
                # t_stat, p_value_g = ttest_rel(list_re, list_cc, alternative='greater')
                # t_stat, p_value_l = ttest_rel(list_re, list_cc, alternative='less')

                # sum(sum(sum(att_gene_re_sel_df_patten > 0.0001))) / (7299 * 351 * 351)
                # cut = 1 / len(index_row) / 10
                cut = cutrate / len(index_row)
                f1 = list_re >= cut
                f2 = list_cc >= cut
                ff = f1 & f2

                t_stat, p_value = ttest_rel(list_re[ff], list_cc[ff])
                # print(t_stat)
                if p_value < alpha:
                    df_Pvalue.loc[gi, gj] = p_value
                    # mean_re = np.mean(list_re)
                    # mean_cc = np.mean(list_cc)
                    # if mean_re > mean_cc:

                    if np.mean(list_re[ff]) > np.mean(list_cc[ff]):
                        df_ans.loc[gi, gj] = -1
                    else:
                        df_ans.loc[gi, gj] = 1
                else:
                    df_ans.loc[gi, gj] = 0
                    df_Pvalue.loc[gi, gj] = 0.1

                # 判断显著性
                # if p_value < alpha:
                #     if t_stat > 0:
                #         print("单个数值显著高于样本均值")
                #     else:
                #         print("单个数值显著低于样本均值")
                # else:
                #     print("没有显著差异")

            t.update(1)    
    return df_ans, df_Pvalue

def scaleData_axis(matrix, axis=0):
    mean = np.mean(matrix, axis=axis, keepdims=True)  # 保持维度一致
    std = np.std(matrix, axis=axis, keepdims=True)   # 保持维度一致
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix

## 这里应该使用箱型图来展示
def boxplot_geneAtt(gene_leg, gene_rec, att_re_patten, att_cc_patten, star_type='mean', sorted_types=['re', 'cc'], cutrate = 1):
    from scipy.stats import ttest_rel

    gi = gene_use.index(gene_rec)
    gj = gene_use.index(gene_leg)

    list_re = att_re_patten[:, gi, gj]
    list_cc = att_cc_patten[:, gi, gj]

    cut = cutrate / 351
    f1 = list_re >= cut
    f2 = list_cc >= cut
    ff = f1 & f2

    df_data_re = pd.DataFrame({'att' : list_re[ff], 'label' : 're'})
    df_data_cc = pd.DataFrame({'att' : list_cc[ff], 'label' : 'cc'})
    df_data = pd.concat([df_data_re, df_data_cc], axis=0)

    if star_type == 'mean':
        mean_values = df_data.groupby('label')['att'].mean().sort_values(ascending=False)
        # sorted_types = mean_values.index
    else:
        median_values = df_data.groupby('label')['att'].median().sort_values(ascending=False)
        # sorted_types = median_values.index

    df_data['label'] = pd.Categorical(df_data['label'], categories=sorted_types, ordered=True)
    # cbox = [mycolor[idx] for idx, ct in enumerate(sorted_types)]

    # 绘制箱型图
    plt.figure(figsize=(4, 5))
    sns.boxplot(x='label', y='att', data=df_data, order=sorted_types, palette='Set2')
    # sns.boxplot(x='label', y='att', data=df_data, order=sorted_types, palette=cbox)

    for idx, type_ in enumerate(sorted_types):      # 添加平均值标记
        mean_value = mean_values[type_]
        plt.scatter(idx, mean_value, color='red', s=20, marker='*', zorder=5)

    # 添加显著性标记
    _, p_value = ttest_rel(list_re[ff], list_cc[ff])  # 独立样本 t 检验

    # 设置显著性标记
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'

    # 在箱型图上添加标记
    x1, x2 = 0, 1  # 两组的索引
    y, h = df_data['att'].max() + 0.003, 0.003  # 显著性标记的高度和间距
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color='black')
    plt.text((x1 + x2) * 0.5, y + h, sig, ha='center', va='bottom', color='black', fontsize=10)

    plt.xticks(rotation=90)
    plt.title(f'{gene_leg} - {gene_rec}', fontsize=12)
    plt.xlabel('celltype', fontsize=10)
    plt.ylabel('att.', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_folder_attention_gene_ci_patten}/boxplot_{gene_leg}_{gene_rec}.pdf")

def extend_lr(df_lrpair):
    df_lrpair_el = []
    for _, row in df_lrpair.iterrows():
        subrow = row['ligand'].split('_')  # 按逗号分割'Genes'列的值
        for gene in subrow:
            df_lrpair_el.append({'ligand': gene.capitalize(), 'receptor': row['receptor'].capitalize(), 'annotation':row['annotation'], 'pathway_name':row['pathway_name']})  # 将每个分割后的基因作为新行添加
    df_lrpair_el = pd.DataFrame(df_lrpair_el)

    df_lrpair_el_er = []
    for _, row in df_lrpair_el.iterrows():
        subrow = row['receptor'].split('_')  # 按逗号分割'Genes'列的值
        for gene in subrow:
            df_lrpair_el_er.append({'ligand': row['ligand'].capitalize(), 'receptor': gene.capitalize(), 'annotation':row['annotation'], 'pathway_name':row['pathway_name']})  # 将每个分割后的基因作为新行添加
    df_lrpair_el_er = pd.DataFrame(df_lrpair_el_er)
    return df_lrpair_el_er.drop_duplicates(ignore_index=True)

def cal_consistance_lr(df_lr_patten, gene_use):
    lig_use, rec_use = [], []
    cnt_all, cnt_vaild = 0, 0
    for _, row in df_lr_patten.iterrows():
        gene_lig, gene_rec = row[['ligand', 'receptor']]
        if (gene_lig in gene_use) and (gene_rec in gene_use):
            lig_use.append(gene_lig)
            rec_use.append(gene_rec)
            cnt_all += 1
            if df_ans.loc[gene_rec, gene_lig] == 1:
                cnt_vaild += 1
    print(f"consistance rate : {cnt_vaild / cnt_all}, lr number : {len(lig_use)}")
    return lig_use, rec_use

##### 对指定patten进行深入分析
#########################################################################
########################## pattern 0 cluster 0 ##########################
#########################################################################
dt.setup_seed(SEED)
patteni = '0'
clusteri = ['0']
save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{patteni}/"
dt.check_path(save_folder_attention_gene_ci_patten)
## patten3
patten_genes = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs['leiden'] == patteni]
# att_gene_cc_sel_df_patten = pd.DataFrame(np.mean(att_gene_cc_sel[adata_geneatt.obs.leiden.values==pi], 0), index=gene_use, columns=gene_use)
patten_genes_sorted = np.sort(patten_genes)


model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

# att_gene_re = [indices_use]
# att_gene_cc = att_gene_cc_all[indices_use]
# att_cell = att_cell_all[indices_use]

flag = (celltypes_list == 'Spinal cord')
att_gene_index_use = indices_use[flag]
flag_cluster = att_gene_index_use[adata_geneatt.obs.leiden.isin(clusteri).values]
flag_att = np.unique(kNNGraph_use[flag_cluster].flatten())
# flag_att = kNNGraph_use[flag_cluster].flatten()
att_gene_re_sel_df_patten = att_gene_re_all[flag_att].copy()
att_gene_cc_sel_df_patten = att_gene_cc_all[flag_att].copy()
# att_gene_re_scale = scaleData_axis(att_gene_re_sel_df_patten, 1)
# att_gene_cc_scale = scaleData_axis(att_gene_cc_sel_df_patten, 1)

df_ans, df_Pvalue = attention_t_test(gene_use, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.05, cutrate=1)


boxplot_geneAtt(
    'Dll1', 'Notch1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=2.0
)

boxplot_geneAtt(
    'Dll3', 'Notch1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=2.0
)

boxplot_geneAtt(
    'Notch1', 'Lfng',
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=2.0
)

boxplot_geneAtt(
    'Lfng', 'Notch1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=2.0
)

# boxplot_geneAtt(
#     'Wnt2', 'Hes1', 
#     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
#     star_type='mean', cutrate=2.0
# )

# boxplot_geneAtt(
#     'Wnt2', 'Hes5', 
#     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
#     star_type='mean', cutrate=2.0
# )




# # 设置图形尺寸
# plt.close('all')
# plt.figure(figsize=(5, 5))
# sns.clustermap(
#     df_ans, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
#     cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
# plt.title('Gene Interaction Heatmap')
# plt.xlabel('Genes')
# plt.ylabel('Genes')
# # plt.tight_layout()
# plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_cluster{clusteri}.png", dpi=300)

# # 设置图形尺寸
# plt.close('all')
# plt.figure(figsize=(5, 5))
# sns.clustermap(
#     -np.log(df_Pvalue), square=True, row_cluster=True, col_cluster=True, vmin=0, vmax=50, 
#     cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
# plt.title('Gene Interaction Heatmap')
# plt.xlabel('Genes')
# plt.ylabel('Genes')
# # plt.tight_layout()
# plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.p-values_mat_patten{patteni}_cluster{clusteri}.png", dpi=300)

##### 互作对分析
df_lrpair = pd.read_csv(f"{saveFolder_geneAtt_02spatialModule}/df_lrpairs.csv", index_col=0, error_bad_lines=False)
df_lrpair.index = np.arange(len(df_lrpair))
df_lrpair_el_er = extend_lr(df_lrpair[['ligand', 'receptor', 'annotation', 'pathway_name']])

# ['interaction_name', 'pathway_name', 'ligand', 'receptor', 'agonist',
#  'antagonist', 'co_A_receptor', 'co_I_receptor', 'evidence',
#  'annotation', 'interaction_name_2']

## ligand
flag = df_lrpair_el_er.ligand.isin(patten_genes) | df_lrpair_el_er.receptor.isin(patten_genes)
df_lr_patten = df_lrpair_el_er.loc[flag, ['ligand', 'receptor', 'annotation', 'pathway_name']]

lig_list = df_lr_patten.ligand.unique()
lig_list = list(set(lig_list).intersection(set(gene_use)))
rec_list = df_lr_patten.receptor.unique()
rec_list = list(set(rec_list).intersection(set(gene_use)))

plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans.loc[rec_list, lig_list].T, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_mat_{patteni}_lig_rec.png", dpi=300)

## 补一个整体的统计量
cal_consistance_lr(df_lr_patten, gene_use)
# consistance rate : 0.7575757575757576

['Bmp2', 'Bmp4', 'Bmp5', 'Bmp7', 'Cd34', 'Cdh2', 'Col4a1', 'Dlk1',
       'Dll1', 'Dll3', 'Efna5', 'Fgf10', 'Fgf15', 'Fgf17', 'Fgf3', 'Fgf5',
       'Gdf3', 'Nodal', 'Pecam1', 'Ptn', 'Wnt11', 'Wnt2', 'Wnt2b', 'Wnt3',
       'Wnt3a', 'Wnt5a', 'Wnt5b', 'Wnt8a']

['Acvr2a', 'Cd34', 'Cdh2', 'Epha5', 'Fgfr1', 'Fgfr2', 'Fgfr3',
       'Fgfr4', 'Fzd2', 'Itga3', 'Notch1', 'Pecam1', 'Podxl']

df_lr_patten.loc[df_lr_patten.receptor == 'Epha5']

boxplot_geneAtt(
    'Bmp2', 'Acvr2a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=1.0
)

boxplot_geneAtt(
    'Efna5', 'Epha5', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=1.0
)

boxplot_geneAtt(
    'Dll1', 'Notch1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=2.0
)

boxplot_geneAtt(
    'Dll3', 'Notch1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=2.0
)

boxplot_geneAtt(
    'Fgf15', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=1.0
)

boxplot_geneAtt(
    'Fgf15', 'Fgfr2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=1.0
)

boxplot_geneAtt(
    'Fgf15', 'Fgfr4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate=1.0
)


##### 基因调控网络分析，使用kegg进行信号通路检测
df_kegg = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/df_kegg_enrichment.csv", index_col=0, error_bad_lines=False)
entrezid = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/entrezid.csv", index_col=0, error_bad_lines=False)
entrezid.index = [str(ei) for ei in entrezid.ENTREZID.values]
geneNameSet = {}
for ci in df_kegg.index:
    ci_geneid = df_kegg.loc[ci, 'geneID']
    ci_geneid = ci_geneid.split('/')
    ci_geneid = [entrezid.loc[ei, 'SYMBOL'] for ei in ci_geneid]
    geneNameSet[ci] = ci_geneid


for ci in df_kegg.index:
    plt.close('all')
    plt.figure(figsize=(5, 5))
    sns.clustermap(
        df_ans.loc[geneNameSet[ci], geneNameSet[ci]], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
        cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
    plt.title('Gene Interaction Heatmap')
    plt.xlabel('Genes')
    plt.ylabel('Genes')
    # plt.tight_layout()
    plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_kegg_{ci}.png", dpi=300)


{'mmu04550': ['Sox2', 'Pcgf2', 'Pcgf3', 'Jarid2', 'Axin2', 'Pcgf6', 'Fgfr1', 'Fzd2', 'Pax6', 'Isl1', 'Acvr2a', 'Fgfr4', 'Fgfr3', 'Wnt2'],
 'mmu05224': ['Dll3', 'Notch1', 'Axin2', 'Fgf15', 'Dll1', 'Fgfr1', 'Fzd2', 'Tcf7l1', 'Bak1', 'Wnt2'],
 'mmu00310': ['Kmt2b', 'Aldh2', 'Ezh2', 'Setd2', 'Setd1b', 'Setd1a', 'Kmt2d'],
 'mmu03083': ['Pcgf2', 'Pcgf3', 'Jarid2', 'Ezh2', 'Pcgf6', 'Sfmbt2', 'Eed'],
 'mmu05217': ['Axin2', 'Fzd2', 'Tcf7l1', 'Bak1', 'Wnt2'],
 'mmu05206': ['Marcks', 'Sox4', 'Notch1', 'Ezh2', 'Mcl1', 'Dnmt3a', 'Bak1', 'Efna5', 'Fgfr3'],
 'mmu05226': ['Axin2', 'Fgf15', 'Fzd2', 'Tcf7l1', 'Bak1', 'Wnt2'],
 'mmu04390': ['Sox2', 'Axin2', 'Fzd2', 'Tcf7l1', 'Tead2', 'Wnt2'],
 'mmu04330': ['Dll3', 'Notch1', 'Dll1', 'Lfng'],
 'mmu04310': ['Sfrp2', 'Axin2', 'Fzd2', 'Tcf7l1', 'Sfrp1', 'Wnt2'],
 'mmu05165': ['Notch1', 'Col4a1', 'Axin2', 'Fzd2', 'Tcf7l1', 'Bak1', 'Lfng', 'Wnt2'],
 'mmu04934': ['Axin2', 'Fzd2', 'Tcf7l1', 'Kmt2d', 'Wnt2'],
 'mmu04014': ['Fgf15', 'Rgl1', 'Fgfr1', 'Efna5', 'Fgfr4', 'Fgfr3'],
 'mmu05225': ['Axin2', 'Fzd2', 'Tcf7l1', 'Bak1', 'Wnt2'],
 'mmu05213': ['Axin2', 'Tcf7l1', 'Bak1']}



boxplot_geneAtt(
    'Notch1', 'Lfng', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Dll1', 'Lfng', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Pax6', 'Sox2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Isl1', 'Sox2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Pcgf2', 'Jarid2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Pcgf3', 'Jarid2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Pcgf6', 'Jarid2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)


boxplot_geneAtt(
    'Lfng', 'Notch1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)

boxplot_geneAtt(
    'Lfng', 'Dll3', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean'
)


#########################################################################
########################## pattern 2 cluster 4 ##########################
#########################################################################
dt.setup_seed(SEED)
patteni = '2'
clusteri = ['4']
save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{patteni}/"
dt.check_path(save_folder_attention_gene_ci_patten)
## patten3
patten_genes = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs['leiden'] == patteni]

model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

flag = (celltypes_list == 'Spinal cord')
att_gene_index_use = indices_use[flag]
flag_cluster = att_gene_index_use[adata_geneatt.obs.leiden.isin(clusteri).values]
flag_att = np.unique(kNNGraph_use[flag_cluster].flatten())
att_gene_re_sel_df_patten = att_gene_re_all[flag_att].copy()
att_gene_cc_sel_df_patten = att_gene_cc_all[flag_att].copy()

df_ans, df_Pvalue = attention_t_test(gene_use, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.05, cutrate=1)

# # 设置图形尺寸
# plt.close('all')
# plt.figure(figsize=(5, 5))
# sns.clustermap(
#     df_ans, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
#     cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
# plt.title('Gene Interaction Heatmap')
# plt.xlabel('Genes')
# plt.ylabel('Genes')
# # plt.tight_layout()
# plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_cluster{clusteri}.png", dpi=300)

# # 设置图形尺寸
# plt.close('all')
# plt.figure(figsize=(5, 5))
# sns.clustermap(
#     -np.log(df_Pvalue), square=True, row_cluster=True, col_cluster=True, vmin=0, vmax=50, 
#     cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
# plt.title('Gene Interaction Heatmap')
# plt.xlabel('Genes')
# plt.ylabel('Genes')
# # plt.tight_layout()
# plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.p-values_mat_patten{patteni}_cluster{clusteri}.png", dpi=300)

##### 互作对分析
df_lrpair = pd.read_csv(f"{saveFolder_geneAtt_02spatialModule}/df_lrpairs.csv", index_col=0, error_bad_lines=False)

df_lrpair.index = np.arange(len(df_lrpair))
df_lrpair_el_er = extend_lr(df_lrpair[['ligand', 'receptor', 'annotation', 'pathway_name']])
flag = df_lrpair_el_er.ligand.isin(patten_genes) | df_lrpair_el_er.receptor.isin(patten_genes)
df_lr_patten = df_lrpair_el_er.loc[flag, ['ligand', 'receptor', 'annotation']]

lig_list = df_lr_patten.ligand.unique()
lig_list = list(set(lig_list).intersection(set(gene_use)))
rec_list = df_lr_patten.receptor.unique()
rec_list = list(set(rec_list).intersection(set(gene_use)))

## 补一个整体的统计量
cal_consistance_lr(df_lr_patten, gene_use)
# consistance rate : 0.0, lr number : 1

# plt.close('all')
# plt.figure(figsize=(5, 5))
# sns.clustermap(
#     df_ans.loc[rec_lists, lig_list], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
#     cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
# plt.title('Gene Interaction Heatmap')
# plt.xlabel('Genes')
# plt.ylabel('Genes')
# # plt.tight_layout()
# plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_mat_{patteni}_lig_rec.png", dpi=300)


# ##### 基因调控网络分析，使用kegg进行信号通路检测
# df_kegg = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/df_kegg_enrichment.csv", index_col=0, error_bad_lines=False)
# entrezid = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/entrezid.csv", index_col=0, error_bad_lines=False)
# entrezid.index = [str(ei) for ei in entrezid.ENTREZID.values]


# geneNameSet = {}
# for ci in df_kegg.index:
#     ci_geneid = df_kegg.loc[ci, 'geneID']
#     ci_geneid = ci_geneid.split('/')
#     sub_entrezid = entrezid.loc[entrezid.index.isin(ci_geneid)]
#     ci_geneid = [sub_entrezid.loc[ei, 'SYMBOL'] for ei in ci_geneid]
#     geneNameSet[ci] = ci_geneid

# for ci in df_kegg.index:
#     plt.close('all')
#     plt.figure(figsize=(5, 5))
#     sns.clustermap(
#         df_ans.loc[geneNameSet[ci], geneNameSet[ci]], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
#         cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
#     plt.title('Gene Interaction Heatmap')
#     plt.xlabel('Genes')
#     plt.ylabel('Genes')
#     # plt.tight_layout()
#     plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_kegg_{ci}.png", dpi=300)

# boxplot_geneAtt(
#     'Spi1', 'Lyl1',
#     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
#     star_type='mean'
# )

# boxplot_geneAtt(
#     'Hoxa11', 'Bmi1',
#     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
#     star_type='mean'
# )


########################################################################
######################### pattern 1 cluster 1 ##########################
########################################################################
dt.setup_seed(SEED)
patteni = '1'
clusteri = ['1']
save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{patteni}/"
dt.check_path(save_folder_attention_gene_ci_patten)
## patten3
patten_genes = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs['leiden'] == patteni]
# att_gene_cc_sel_df_patten = pd.DataFrame(np.mean(att_gene_cc_sel[adata_geneatt.obs.leiden.values==pi], 0), index=gene_use, columns=gene_use)

model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

# att_gene_re = [indices_use]
# att_gene_cc = att_gene_cc_all[indices_use]
# att_cell = att_cell_all[indices_use]

flag = (celltypes_list == 'Spinal cord')
att_gene_index_use = indices_use[flag]
flag_cluster = att_gene_index_use[adata_geneatt.obs.leiden.isin(clusteri).values]
flag_att = np.unique(kNNGraph_use[flag_cluster].flatten())
# flag_att = kNNGraph_use[flag_cluster].flatten()
att_gene_re_sel_df_patten = att_gene_re_all[flag_att].copy()
att_gene_cc_sel_df_patten = att_gene_cc_all[flag_att].copy()
# att_gene_re_scale = scaleData_axis(att_gene_re_sel_df_patten, 1)
# att_gene_cc_scale = scaleData_axis(att_gene_cc_sel_df_patten, 1)

df_ans, df_Pvalue = attention_t_test(gene_use, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.05, cutrate=1)


boxplot_geneAtt(
    'Fgf5', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf5', 'Fgfr2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf5', 'Fgfr3', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf5', 'Fgfr4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)


boxplot_geneAtt(
    'Fgf17', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf17', 'Fgfr2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf17', 'Fgfr3', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf17', 'Fgfr4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)


boxplot_geneAtt(
    'Fgfr1', 'Cdx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgfr1', 'Cdx2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgfr1', 'Cdx4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)





# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_cluster{clusteri}.png", dpi=300)

# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    -np.log(df_Pvalue), square=True, row_cluster=True, col_cluster=True, vmin=0, vmax=50, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.p-values_mat_patten{patteni}_cluster{clusteri}.png", dpi=300)

##### 互作对分析
df_lrpair = pd.read_csv(f"{saveFolder_geneAtt_02spatialModule}/df_lrpairs.csv", index_col=0, error_bad_lines=False)
df_lrpair.index = np.arange(len(df_lrpair))
df_lrpair_el_er = extend_lr(df_lrpair[['ligand', 'receptor', 'annotation', 'pathway_name']])
flag = df_lrpair_el_er.ligand.isin(patten_genes) | df_lrpair_el_er.receptor.isin(patten_genes)
df_lr_patten = df_lrpair_el_er.loc[flag, ['ligand', 'receptor', 'annotation']]

lig_list = df_lr_patten.ligand.unique()
lig_list = list(set(lig_list).intersection(set(gene_use)))
rec_list = df_lr_patten.receptor.unique()
rec_list = list(set(rec_list).intersection(set(gene_use)))

plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans.loc[rec_list, lig_list].T, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_mat_{patteni}_lig_rec.png", dpi=300)

## 补一个整体的统计量
cal_consistance_lr(df_lr_patten, gene_use)
# consistance rate : 0.9285714285714286, lr number : 14

['Bmp5', 'Nodal', 'Wnt8a', 'Fgf5', 'Fgf17', 'Kitl', 'Esam']
['Acvr1', 'Acvr2a', 'Fgfr2', 'Esam', 'Fgfr1', 'Fgfr3', 'Fgfr4', 'Fzd2']

boxplot_geneAtt(
    'Fgf17', 'Fgfr4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(
    'Fgf17', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(
    'Fgf5', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)


#########################################################################
########################## pattern 4 cluster 2 ##########################
#########################################################################
dt.setup_seed(SEED)
patteni = '4'
clusteri = ['2']
save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{patteni}/"
dt.check_path(save_folder_attention_gene_ci_patten)
## patten3
patten_genes = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs['leiden'] == patteni]
# att_gene_cc_sel_df_patten = pd.DataFrame(np.mean(att_gene_cc_sel[adata_geneatt.obs.leiden.values==pi], 0), index=gene_use, columns=gene_use)

model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

# att_gene_re = [indices_use]
# att_gene_cc = att_gene_cc_all[indices_use]
# att_cell = att_cell_all[indices_use]

flag = (celltypes_list == 'Spinal cord')
att_gene_index_use = indices_use[flag]
flag_cluster = att_gene_index_use[adata_geneatt.obs.leiden.isin(clusteri).values]
flag_att = np.unique(kNNGraph_use[flag_cluster].flatten())
# flag_att = kNNGraph_use[flag_cluster].flatten()
att_gene_re_sel_df_patten = att_gene_re_all[flag_att].copy()
att_gene_cc_sel_df_patten = att_gene_cc_all[flag_att].copy()
# att_gene_re_scale = scaleData_axis(att_gene_re_sel_df_patten, 1)
# att_gene_cc_scale = scaleData_axis(att_gene_cc_sel_df_patten, 1)

df_ans, df_Pvalue = attention_t_test(gene_use, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.05, cutrate=1)


boxplot_geneAtt(
    'Bmp2', 'Acvr2a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Bmp7', 'Acvr2a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

boxplot_geneAtt(
    'Acvr2a', 'Gata4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)
boxplot_geneAtt(
    'Acvr2a', 'Msx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)
boxplot_geneAtt(
    'Acvr2a', 'Cdx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)
boxplot_geneAtt(
    'Acvr2a', 'Cdx2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)
boxplot_geneAtt(
    'Acvr2a', 'Tbx5', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)


boxplot_geneAtt(
    'Bmp2', 'Msx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1 
)
boxplot_geneAtt(
    'Bmp2', 'Cdx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)



boxplot_geneAtt(
    'Wnt2b', 'Cdx2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Wnt3a', 'Cdx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Wnt3a', 'Cdx2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Acvr2a', 'Msx1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)



# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_cluster{clusteri}.png", dpi=300)

# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    -np.log(df_Pvalue), square=True, row_cluster=True, col_cluster=True, vmin=0, vmax=50, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.p-values_mat_patten{patteni}_cluster{clusteri}.png", dpi=300)

##### 互作对分析
df_lrpair = pd.read_csv(f"{saveFolder_geneAtt_02spatialModule}/df_lrpairs.csv", index_col=0, error_bad_lines=False)
df_lrpair.index = np.arange(len(df_lrpair))
df_lrpair_el_er = extend_lr(df_lrpair[['ligand', 'receptor', 'annotation', 'pathway_name']])

# ['interaction_name', 'pathway_name', 'ligand', 'receptor', 'agonist',
#  'antagonist', 'co_A_receptor', 'co_I_receptor', 'evidence',
#  'annotation', 'interaction_name_2']

## ligand
flag = df_lrpair_el_er.ligand.isin(patten_genes) | df_lrpair_el_er.receptor.isin(patten_genes)
df_lr_patten = df_lrpair_el_er.loc[flag, ['ligand', 'receptor', 'annotation', 'pathway_name']]

lig_list = df_lr_patten.ligand.unique()
lig_list = list(set(lig_list).intersection(set(gene_use)))
rec_list = df_lr_patten.receptor.unique()
rec_list = list(set(rec_list).intersection(set(gene_use)))

plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans.loc[rec_list, lig_list].T, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_mat_{patteni}_lig_rec.png", dpi=300)

## 补一个整体的统计量
cal_consistance_lr(df_lr_patten, gene_use)
# consistance rate : 0.625, lr number : 8
# (['Bmp2', 'Bmp7', 'Bmp7', 'Wnt2b', 'Wnt3a', 'Fgf10', 'Fgf10', 'Itga4'],
#  ['Acvr2a', 'Acvr1', 'Acvr2a', 'Fzd2', 'Fzd2', 'Fgfr1', 'Fgfr2', 'Vcam1'])

['Wnt3a', 'Ptprc', 'Bmp2', 'Wnt2b', 'Itga4', 'Bmp7', 'Fgf10']
['Ptprc',  'Adora2b',  'Acvr1',  'Vcam1',  'Itga4',  'Acvr2a',  'Fgfr2',  'Fgfr1',  'Epor',  'Fzd2']

df_lr_patten.loc[df_lr_patten.receptor == 'Fzd2']


boxplot_geneAtt(
    'Itga4', 'Vcam1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

boxplot_geneAtt(
    'Bmp2', 'Acvr2a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

boxplot_geneAtt(
    'Fgf10', 'Fgfr2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf10', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

boxplot_geneAtt(
    'Wnt2b', 'Fzd2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Wnt3a', 'Fzd2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1.1
)


##### 基因调控网络分析，使用kegg进行信号通路检测
df_kegg = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/df_kegg_enrichment.csv", index_col=0, error_bad_lines=False)
entrezid = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/entrezid.csv", index_col=0, error_bad_lines=False)
entrezid.index = [str(ei) for ei in entrezid.ENTREZID.values]
geneNameSet = {}
for ci in df_kegg.index:
    ci_geneid = df_kegg.loc[ci, 'geneID']
    ci_geneid = ci_geneid.split('/')
    ci_geneid = [entrezid.loc[ei, 'SYMBOL'] for ei in ci_geneid]
    geneNameSet[ci] = ci_geneid


for ci in df_kegg.index:
    plt.close('all')
    plt.figure(figsize=(5, 5))
    sns.clustermap(
        df_ans.loc[geneNameSet[ci], geneNameSet[ci]], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
        cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
    plt.title('Gene Interaction Heatmap')
    plt.xlabel('Genes')
    plt.ylabel('Genes')
    # plt.tight_layout()
    plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_kegg_{ci}.png", dpi=300)

# {'mmu04550': ['Hoxb1', 'Hoxa1', 'Wnt3a', 'Pcgf5', 'Wnt2b'],
#  'mmu05226': ['Cdx2', 'Wnt3a', 'Fgf10', 'Wnt2b'],
#  'mmu05217': ['Wnt3a', 'Bmp2', 'Wnt2b'],
#  'mmu04390': ['Bmp7', 'Wnt3a', 'Bmp2', 'Wnt2b'],
#  'mmu04640': ['Gypa', 'Epor', 'Itga4']}
# mmu04550 mmu05226 是小鼠的 "Intestinal immune network for IgA production" 通路，涉及肠道免疫反应和 IgA（免疫球蛋白A） 生成。
boxplot_geneAtt(
    'Hoxa1', 'Hoxb1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(
    'Hoxb1', 'Pcgf5', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(
    'Pcgf5', 'Wnt3a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(
    'Hoxa1', 'Wnt3a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Hoxb1', 'Wnt3a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
# mmu04390 mmu04390 是小鼠的 "Hippo signaling pathway" 通路，涉及细胞增殖、分化、凋亡及组织修复等多个方面。
boxplot_geneAtt(
    'Bmp7', 'Wnt3a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

# mmu04640 mmu04640 是小鼠的 "Hematopoietic cell lineage" 通路，涉及血液细胞发育、增殖和分化等过程。
boxplot_geneAtt(
    'Gypa', 'Epor', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)
boxplot_geneAtt(
    'Epor', 'Itga4', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)


#########################################################################
########################## pattern 5 cluster 3 ##########################
#########################################################################
dt.setup_seed(SEED)
patteni = '5'
clusteri = ['3']
save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{patteni}/"
dt.check_path(save_folder_attention_gene_ci_patten)
## patten3
patten_genes = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs['leiden'] == patteni]
# att_gene_cc_sel_df_patten = pd.DataFrame(np.mean(att_gene_cc_sel[adata_geneatt.obs.leiden.values==pi], 0), index=gene_use, columns=gene_use)

model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

# att_gene_re = [indices_use]
# att_gene_cc = att_gene_cc_all[indices_use]
# att_cell = att_cell_all[indices_use]

flag = (celltypes_list == 'Spinal cord')
att_gene_index_use = indices_use[flag]
flag_cluster = att_gene_index_use[adata_geneatt.obs.leiden.isin(clusteri).values]
flag_att = np.unique(kNNGraph_use[flag_cluster].flatten())
# flag_att = kNNGraph_use[flag_cluster].flatten()
att_gene_re_sel_df_patten = att_gene_re_all[flag_att].copy()
att_gene_cc_sel_df_patten = att_gene_cc_all[flag_att].copy()
# att_gene_re_scale = scaleData_axis(att_gene_re_sel_df_patten, 1)
# att_gene_cc_scale = scaleData_axis(att_gene_cc_sel_df_patten, 1)

df_ans, df_Pvalue = attention_t_test(gene_use, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.05, cutrate=1)

# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_cluster{clusteri}.png", dpi=300)

# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    -np.log(df_Pvalue), square=True, row_cluster=True, col_cluster=True, vmin=0, vmax=50, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.p-values_mat_patten{patteni}_cluster{clusteri}.png", dpi=300)

##### 互作对分析
df_lrpair = pd.read_csv(f"{saveFolder_geneAtt_02spatialModule}/df_lrpairs.csv", index_col=0, error_bad_lines=False)
df_lrpair.index = np.arange(len(df_lrpair))
df_lrpair_el_er = extend_lr(df_lrpair[['ligand', 'receptor', 'annotation', 'pathway_name']])

# ['interaction_name', 'pathway_name', 'ligand', 'receptor', 'agonist',
#  'antagonist', 'co_A_receptor', 'co_I_receptor', 'evidence',
#  'annotation', 'interaction_name_2']

## ligand
flag = df_lrpair_el_er.ligand.isin(patten_genes) | df_lrpair_el_er.receptor.isin(patten_genes)
df_lr_patten = df_lrpair_el_er.loc[flag, ['ligand', 'receptor', 'annotation', 'pathway_name']]

lig_list = df_lr_patten.ligand.unique()
lig_list = list(set(lig_list).intersection(set(gene_use)))
rec_list = df_lr_patten.receptor.unique()
rec_list = list(set(rec_list).intersection(set(gene_use)))

# df_ans中，行表示receptor，列表示ligand
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans.loc[rec_list, lig_list].T, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_mat_{patteni}_lig_rec.png", dpi=300)

## 补一个整体的统计量
cal_consistance_lr(df_lr_patten, gene_use)
# consistance rate : 0.5555555555555556, lr number : 9

['Gdf3', 'Bmp5', 'Cdh1', 'Fgf3', 'Bmp7', 'Apln', 'Cdh5', 'Icam2', 'Wnt11']
['Cdh1', 'Acvr1', 'Acvr2a', 'Fgfr2', 'Fgfr1', 'Cdh5', 'Icam2', 'Aplnr', 'Fzd2']

df_lr_patten.loc[df_lr_patten.receptor == 'Fzd2']

boxplot_geneAtt(
    'Gdf3', 'Acvr2a', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(
    'Fgf3', 'Fgfr1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)
boxplot_geneAtt(
    'Fgf3', 'Fgfr2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

boxplot_geneAtt(
    'Apln', 'Aplnr', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 0
)

boxplot_geneAtt(
    'Wnt11', 'Fzd2', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

##### 基因调控网络分析，使用kegg进行信号通路检测
df_kegg = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/df_kegg_enrichment.csv", index_col=0, error_bad_lines=False)
entrezid = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/entrezid.csv", index_col=0, error_bad_lines=False)
entrezid.index = [str(ei) for ei in entrezid.ENTREZID.values]
geneNameSet = {}
for ci in df_kegg.index:
    ci_geneid = df_kegg.loc[ci, 'geneID']
    ci_geneid = ci_geneid.split('/')
    ci_geneid = [entrezid.loc[ei, 'SYMBOL'] for ei in ci_geneid]
    geneNameSet[ci] = ci_geneid


for ci in df_kegg.index:
    plt.close('all')
    plt.figure(figsize=(5, 5))
    sns.clustermap(
        df_ans.loc[geneNameSet[ci], geneNameSet[ci]], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
        cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
    plt.title('Gene Interaction Heatmap')
    plt.xlabel('Genes')
    plt.ylabel('Genes')
    # plt.tight_layout()
    plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_kegg_{ci}.png", dpi=300)

# {'mmu04371': ['Apln', 'Mef2c', 'Cdh1'],
#  'mmu05418': ['Cdh5', 'Acvr1', 'Mef2c'],
#  'mmu05226': ['Fgf3', 'Cdh1', 'Wnt11'],
#  'mmu04514': ['Cdh5', 'Cdh1', 'Icam2']}

boxplot_geneAtt(        ## 在细胞外发挥作用
    'Apln', 'Mef2c', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 0
)

boxplot_geneAtt(        ## 在细胞外发挥作用
    'Mef2c', 'Cdh1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 1
)

boxplot_geneAtt(        
    'Acvr1', 'Mef2c', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 2
)

boxplot_geneAtt(        
    'Wnt11', 'Cdh1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 0
)


boxplot_geneAtt(        
    'Cdh5', 'Cdh1', 
    att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
    star_type='mean', cutrate = 0
)

#########################################################################
########################## pattern 3 cluster 6 ##########################
#########################################################################
dt.setup_seed(SEED)
patteni = '3'
clusteri = ['6']
save_folder_attention_gene_ci_patten = f"{saveFolder_geneAtt_02spatialModule}/Patten_{patteni}/"
dt.check_path(save_folder_attention_gene_ci_patten)
## patten3
patten_genes = adata_geneatt_patten.obs_names[adata_geneatt_patten.obs['leiden'] == patteni]
# att_gene_cc_sel_df_patten = pd.DataFrame(np.mean(att_gene_cc_sel[adata_geneatt.obs.leiden.values==pi], 0), index=gene_use, columns=gene_use)

model.eval()
att_gene_re_all, att_gene_cc_all, att_cell_all = model.get_encoder_attention()

# att_gene_re = [indices_use]
# att_gene_cc = att_gene_cc_all[indices_use]
# att_cell = att_cell_all[indices_use]

flag = (celltypes_list == 'Spinal cord')
att_gene_index_use = indices_use[flag]
flag_cluster = att_gene_index_use[adata_geneatt.obs.leiden.isin(clusteri).values]
flag_att = np.unique(kNNGraph_use[flag_cluster].flatten())
# flag_att = kNNGraph_use[flag_cluster].flatten()
att_gene_re_sel_df_patten = att_gene_re_all[flag_att].copy()
att_gene_cc_sel_df_patten = att_gene_cc_all[flag_att].copy()
# att_gene_re_scale = scaleData_axis(att_gene_re_sel_df_patten, 1)
# att_gene_cc_scale = scaleData_axis(att_gene_cc_sel_df_patten, 1)

df_ans, df_Pvalue = attention_t_test(gene_use, att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, alpha = 0.05, cutrate=1)

# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    df_ans, square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_cluster{clusteri}.png", dpi=300)

# 设置图形尺寸
plt.close('all')
plt.figure(figsize=(5, 5))
sns.clustermap(
    -np.log(df_Pvalue), square=True, row_cluster=True, col_cluster=True, vmin=0, vmax=50, 
    cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
plt.title('Gene Interaction Heatmap')
plt.xlabel('Genes')
plt.ylabel('Genes')
# plt.tight_layout()
plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.p-values_mat_patten{patteni}_cluster{clusteri}.png", dpi=300)

##### 互作对分析
df_lrpair = pd.read_csv(f"{saveFolder_geneAtt_02spatialModule}/df_lrpairs.csv", index_col=0, error_bad_lines=False)
df_lrpair.index = np.arange(len(df_lrpair))
df_lrpair_el_er = extend_lr(df_lrpair[['ligand', 'receptor', 'annotation', 'pathway_name']])

# ['interaction_name', 'pathway_name', 'ligand', 'receptor', 'agonist',
#  'antagonist', 'co_A_receptor', 'co_I_receptor', 'evidence',
#  'annotation', 'interaction_name_2']

## ligand
flag = df_lrpair_el_er.ligand.isin(patten_genes) | df_lrpair_el_er.receptor.isin(patten_genes)
df_lr_patten = df_lrpair_el_er.loc[flag, ['ligand', 'receptor', 'annotation', 'pathway_name']]

lig_list = df_lr_patten.ligand.unique()
lig_list = list(set(lig_list).intersection(set(gene_use)))
rec_list = df_lr_patten.receptor.unique()
rec_list = list(set(rec_list).intersection(set(gene_use)))

# plt.close('all')
# plt.figure(figsize=(5, 5))
# sns.clustermap(
#     df_ans.loc[lig_list, rec_list], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
#     cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
# plt.title('Gene Interaction Heatmap')
# plt.xlabel('Genes')
# plt.ylabel('Genes')
# # plt.tight_layout()
# plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_mat_{patteni}_lig_rec.png", dpi=300)

## 补一个整体的统计量
cal_consistance_lr(df_lr_patten, gene_use)
# consistance rate : 0.5714285714285714, lr number : 7

# ['Col1a2', 'Thbs1', 'Col1a1', 'Pdgfa', 'Cxcl12', 'Dlk1', 'Col4a1', 'Bmp4']
# ['Pdgfra', 'Acvr2a', 'Itga3', 'Nrp1', 'Notch1']

# df_lr_patten.loc[df_lr_patten.ligand == 'Bmp4']

# boxplot_geneAtt(
#     'Col4a1', 'Itga3', 
#     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
#     star_type='mean', cutrate = 1
# )

# # boxplot_geneAtt(
# #     'Bmp4', 'Acvr2a', 
# #     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
# #     star_type='mean', cutrate = 1.5
# # )

# ##### 基因调控网络分析，使用kegg进行信号通路检测
# df_kegg = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/df_kegg_enrichment.csv", index_col=0, error_bad_lines=False)
# entrezid = pd.read_csv(f"{save_folder_attention_gene_ci_patten}/entrezid.csv", index_col=0, error_bad_lines=False)
# entrezid.index = [str(ei) for ei in entrezid.ENTREZID.values]
# geneNameSet = {}
# for ci in df_kegg.index:
#     ci_geneid = df_kegg.loc[ci, 'geneID']
#     ci_geneid = ci_geneid.split('/')
#     ci_geneid = [entrezid.loc[ei, 'SYMBOL'] for ei in ci_geneid]
#     geneNameSet[ci] = ci_geneid


# for ci in df_kegg.index:
#     plt.close('all')
#     plt.figure(figsize=(5, 5))
#     sns.clustermap(
#         df_ans.loc[geneNameSet[ci], geneNameSet[ci]], square=True, row_cluster=True, col_cluster=True, vmin=-1, vmax=1, 
#         cmap='seismic', cbar_pos=(0.1, 0.85, 0.05, 0.10))
#     plt.title('Gene Interaction Heatmap')
#     plt.xlabel('Genes')
#     plt.ylabel('Genes')
#     # plt.tight_layout()
#     plt.savefig(f"{save_folder_attention_gene_ci_patten}/4.t-test-pair_patten{patteni}_kegg_{ci}.png", dpi=300)

# # {'mmu04510': ['Col1a2', 'Col1a1', 'Itga3', 'Pdgfra', 'Bcl2'],
# #  'mmu04974': ['Col1a2', 'Col26a1', 'Col1a1', 'Fxyd2'],
# #  'mmu04820': ['Nid1', 'Col1a2', 'Col1a1', 'Nebl', 'Itga3'],
# #  'mmu04512': ['Col1a2', 'Col1a1', 'Itga3'],
# #  'mmu04933': ['Col1a2', 'Col1a1', 'Bcl2'],
# #  'mmu04151': ['Col1a2', 'Col1a1', 'Itga3', 'Pdgfra', 'Bcl2'],
# #  'mmu05202': ['Six1', 'Etv4', 'Dusp6', 'Pax8']}


# boxplot_geneAtt(
#     'Wnt11', 'Cdh1', 
#     att_gene_re_sel_df_patten, att_gene_cc_sel_df_patten, 
#     star_type='mean', cutrate = 2
# )
