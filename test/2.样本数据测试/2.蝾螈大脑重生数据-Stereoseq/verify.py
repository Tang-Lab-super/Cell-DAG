import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

###################%% version
sample_name = "Spatrack"
root_path = "/public3/Shigw/"
data_folder = f"{root_path}/datasets/Stereo-seq/regeneration/"

######################### 2. 预处理，挑选候选细胞 ########################
# 数据处理 归一化和scale
st_data = sc.read(data_folder + "/st_data.h5")

st_data.obs['imagerow'] = st_data.obsm['spatial'][:, 0]
st_data.obs['imagecol'] = st_data.obsm['spatial'][:, 1]

sc.pp.normalize_total(st_data, target_sum=1e4) # 不要和log顺序搞反了 ，这个是去文库的
sc.pp.log1p(st_data)
# sc.pp.scale(st_data)

sample_name_list = ['DAGAST', 'Spatrack', 'stLearn']

# downgenes = ["Vim", "Nes", "Emx1"]
# upgenes = ["Cdkn1a", "Cdkn1c", "Ankrd1", "Nefm", "Scg3"]
# genelists = ["Vim", "Nes", "Emx1", "Cdkn1a", "Cdkn1c", "Ankrd1", "Nefm", "Scg3"]
# genelists = [g.upper() for g in genelists]
# genelists = ['VIM', 'NES', 'S100A10', 'CDKN1A', 'ANKRD1', 'NEFM', 'SCG3']
genelists = ['VIM', 'NES', 'S100A10', 'CDKN1A', 'ANKRD1', 'SCG3']

st_data.var_names[st_data.var_names.isin(genelists)]

df_ans = st_data.obs
df_ans[genelists] = st_data[:, genelists].X.tolist()

for md in sample_name_list:
    df_tmp = pd.read_csv(f"{data_folder}/results/{md}/3.spatial_trajectory/df_obs_ptime.csv", index_col=0)
    df_ans[md] = df_tmp['ptime']

df_ans = df_ans.loc[df_ans.celltypes.isin(['IMN', 'reaEGC', 'nptxEX','rIPC1'])]

## 计算相关性，然后绘制双边柱状图
r_lists = []
alpha_lists = []
for md in sample_name_list:
    tmp_r = []
    tmp_a = []
    for gene in genelists:
        pc = pearsonr(df_ans[md], df_ans[gene])
        tmp_r.append(pc[0])
        tmp_a.append(pc[1])
    r_lists.append(tmp_r)
    alpha_lists.append(tmp_a)

df_r = pd.DataFrame(r_lists, index=['DAGAST', 'Spatrack', 'stLearn'], columns=genelists)
df_alpha = pd.DataFrame(alpha_lists, index=['DAGAST', 'Spatrack', 'stLearn'], columns=genelists)



import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
labels = ['DAGAST', 'Spatrack', 'stLearn']
# correlation_set1 = np.array([0.61858708, 0.53606456, 0.01757088, 0, 0, 0])  # 正相关
# correlation_set2 = np.array([0, 0, 0, -0.20398919, -0.19665504,  0.05316599])  # 负相关
colorlist = ["#E31A1C", "#1F78B4", "#33A02C"]
# 设置 Y 轴刻度（特征）
y_pos = np.arange(len(labels))

# 绘制图形
for idx, ge in enumerate(genelists):
    correlation_set1 = df_r.iloc[:, idx].values

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    bar1 = ax.barh(y_pos, correlation_set1[::-1], color=colorlist[::-1])
    # bar2 = ax.barh(y_pos, correlation_set2[::-1], color=colorlist[::-1])
    # ax.bar_label(bar1, fmt="%.2f", padding=3, fontsize=10, color='black')
    # ax.bar_label(bar2, fmt="%.2f", padding=3, fontsize=10, color='black')
    # ax.set_yticks([1, 4])
    # ax.set_yticklabels(["Hoxd4", "Hoxb9"])
    ax.set_xticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    ax.set_xlabel("Correlation Coefficient")
    # ax.set_title("Bidirectional Horizontal Bar Chart of Correlations")
    ax.axvline(0, color='black', linewidth=1)  # 在 x=0 处添加分界线
    plt.savefig(f"{data_folder}/results/pearsonr_{ge}.pdf")

