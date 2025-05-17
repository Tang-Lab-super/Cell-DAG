###################%% version
date = "241202"
root_path = "/public3/Shigw/"
# root_path = "/data3/shigw/ST_project/FinalFile/"
# root_path = "/data01/suziyi/GWCode/"
data_folder = f"{root_path}/datasets/Stereo-seq/regeneration/"
save_folder = f"{data_folder}/results/{date}"
check_path(save_folder)

###################%% 01 数据读取 %%###################
st_data = sc.read(f"{data_folder}/D15_4.ST.exp.tsv")
df_annot = pd.read_table(f"{data_folder}/D15_4.ST.annot.tsv")
st_data.obs["celltypes"] = df_annot["cluster"].values
st_data.obsm["spatial"] = df_annot[["x", "y"]].values
st_data.layers["counts"] = st_data.X
st_data.var_names_make_unique()

# 对基因做筛选
amex_genes = [gene for gene in st_data.var_names if gene.startswith('AMEX')]       # 剔除蝾螈未确定名称基因
st_data = st_data[:, ~st_data.var_names.isin(amex_genes)].copy()     
loc_genes = [gene for gene in st_data.var_names if gene.startswith('LOC')]            # 剔除为完全标注的基因
st_data = st_data[:, ~st_data.var_names.isin(loc_genes)].copy()
mitochondrial_genes = [gene for gene in st_data.var_names if gene.startswith('MT-')]        # 剔除线粒体基因
st_data = st_data[:, ~st_data.var_names.isin(mitochondrial_genes)].copy()                   
ribosomal_genes = [gene for gene in st_data.var_names if gene.startswith(('RPL', 'RPS'))]   # 剔除核糖体基因
st_data = st_data[:, ~st_data.var_names.isin(ribosomal_genes)].copy()

print(f"找到的蝾螈未确定基因数量: {len(amex_genes)}")
print(f"找到的未标注基因数量: {len(loc_genes)}")
print(f"找到的线粒体基因数量: {len(mitochondrial_genes)}")
print(f"找到的核糖体基因数量: {len(ribosomal_genes)}")

# sc.pp.filter_cells(st_data, min_genes=200)    
# sc.pp.filter_genes(st_data, min_cells=3)
st_data.write_h5ad(data_folder + f"/st_data.h5")


# st_data = sc.read(data_folder + "/st_data.h5")
celltypes = st_data.obs['celltypes'].unique().tolist()
# ['MSN',
#  'sfrpEGC',
#  'npyIN',
#  'VLMC',
#  'IMN',
#  'CP',
#  'wntEGC',
#  'mpEX',
#  'reaEGC',
#  'sstIN',
#  'nptxEX',
#  'rIPC1',
#  'scgnIN',
#  'dpEX',
#  'rIPC2']


save_folder_celltype = f"{save_folder}/1.spatial_celltypes/"
check_path(save_folder_celltype)
for ct in celltypes:
    flag = st_data.obs['celltypes'] == ct
    st_data.obs['plot'] = -1
    st_data.obs.loc[flag, 'plot'] = 0

    if ct == 'Forebrain/Midbrain/Hindbrain':
        ct = 'Forebrain-Midbrain-Hindbrain'
    plot_spatial(st_data, 'plot', title=ct, pointsize=10, savename=f"{save_folder_celltype}/celltype_{ct}.png")
    print(ct)

