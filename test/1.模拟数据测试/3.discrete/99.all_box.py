import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

DAGAST = [0.6986010847854444,
 0.7125082651060013,
 0.6936650757067873,
 0.7033144619298685,
 0.7131730025148231,
 0.6945092774016415,
 0.7112438173159369,
 0.7065188575231229,
 0.7065033773029153,
 0.718893746195292]


Spatrack = [0.556757237318257,
 0.6522788380829156,
 0.5382399776770538,
 0.601951864515212,
 0.6603678148719272,
 0.6059836008618305,
 0.5631145560115236,
 0.6278183644379308,
 0.5908165609201361,
 0.6012467434548341]

stLearn = [0.5979670105432163,
 0.5991627011584464,
 0.4640148992974427,
 0.4736641698611921,
 0.5782528345361654,
 0.6156398720133491,
 0.6339744986983257,
 0.6249620264086994,
 0.6057574584381046,
 0.4505168853081186]

Slingshot = [0.57543626057848,
 0.5725169444615635,
 0.5660982157765365,
 0.5275373185302765,
 0.5470939413262413,
 0.5637367647854593,
 0.5724368193739463,
 0.5448538999620094,
 0.5333793800012653,
 0.545107971107628]

Spaceflow = [0.6267020654138832,
 0.6608026080577948,
 0.647369178634774,
 0.640131378157999,
 0.2118759849274141,
 0.07571845582099564,
 0.6363911838829324,
 0.6444696944638255,
 0.3386565383855902,
 0.6680922067815768]



root_path = "/public3/Shigw/datasets/Simulated_data/"
data_folder = f"{root_path}/discrete/"

df_ans = pd.DataFrame({'DAGAST1':DAGAST, 'Spatrack1':Spatrack, 'stLearn1':stLearn, 'SpaceFlow1':Spaceflow, 'Slingshot1':Slingshot})
df_ans = df_ans.T
df_ans['group'] = df_ans.index

df_melted = pd.melt(df_ans, id_vars=['group'], value_vars=[i for i in range(10)],
                    var_name='Measurement', value_name='Value')

## 图片美化
# 绘制 Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='group', y='Value', data=df_melted, palette='Set2')

mean_values = df_melted.groupby('group')['Value'].mean().sort_values(ascending=False)
for idx, type_ in enumerate(df_ans.index):      # 添加平均值标记
    mean_value = mean_values[type_]
    plt.scatter(idx, mean_value, color='red', s=20, marker='*', zorder=5)

plt.xlabel('Methods')
plt.ylabel('Space-time similarity')
plt.savefig(f"{data_folder}/boxplot_compare.pdf")

# 进行配对 t 检验：比较每两个组之间
# result_AB = ttest_rel(DAGAST, Spatrack)
# result_AC = ttest_rel(DAGAST, stLearn)
# result_AD = ttest_rel(DAGAST, Slingshot)
# result_AE = ttest_rel(DAGAST, Spaceflow)

result_AB = wilcoxon(DAGAST, Spatrack)
result_AC = wilcoxon(DAGAST, stLearn)
result_AD = wilcoxon(DAGAST, Slingshot)
result_AE = wilcoxon(DAGAST, Spaceflow)
# 打印每一对的 t 检验结果
print(f"DAGAST vs Spatrack: t-statistic = {result_AB.statistic}, p-value = {result_AB.pvalue}")
print(f"DAGAST vs stLearn: t-statistic = {result_AC.statistic}, p-value = {result_AC.pvalue}")
print(f"DAGAST vs Slingshot: t-statistic = {result_AD.statistic}, p-value = {result_AD.pvalue}")
print(f"DAGAST vs SpaceFlow: t-statistic = {result_AE.statistic}, p-value = {result_AE.pvalue}")

# DAGAST vs Spatrack: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs stLearn: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs Slingshot: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs SpaceFlow: t-statistic = 0.0, p-value = 0.001953125

# DAGAST vs Spatrack: t-statistic = 2.0, p-value = 0.005859375  2
# DAGAST vs stLearn: t-statistic = 0.0, p-value = 0.001953125   2
# DAGAST vs Slingshot: t-statistic = 0.0, p-value = 0.001953125  2

## 尝试复现Spatrack方法的一致性计算方式
methods = ["DAGAST", "Spatrack", "stLearn", "SpaceFlow", "Slingshot"]
df_all = pd.read_table(f"{data_folder}/sim_path_metadata.txt", index_col=0)
df_all = df_all[['Batch', 'Step']]

for md in methods:
    df_md = pd.read_csv(f"{data_folder}/results/{md}/df_ptime.csv", index_col=0)
    df_all[md] = df_md['Ptime'].values

consistance = []
df_all['Step'] = df_all['Step'] / df_all['Step'].max()
for batch in df_all.Batch.unique():
    subconsist = []
    subda = df_all.loc[df_all.Batch == batch]
    for md in methods:
        tmp = (subda[md] - subda['Step']).abs()
        subconsist.append((tmp < 0.2).sum() / len(tmp))
    consistance.append(subconsist)

consistance = pd.DataFrame(consistance, index=df_all.Batch.unique(), columns=methods)
consistance.mean()

df_melted = consistance.copy()
df_melted = df_melted.T
df_melted['group'] = df_melted.index

df_melted = pd.melt(df_melted, id_vars=['group'], value_vars=df_all.Batch.unique(),
                    var_name='Measurement', value_name='Value')

plt.figure(figsize=(8, 5))
sns.boxplot(x='group', y='Value', data=df_melted, palette='Set2')

mean_values = df_melted.groupby('group')['Value'].mean().sort_values(ascending=False)
for idx, type_ in enumerate(methods):      # 添加平均值标记
    mean_value = mean_values[type_]
    plt.scatter(idx, mean_value, color='red', s=20, marker='*', zorder=5)

plt.xlabel('Methods')
plt.ylabel('Consistence')
plt.savefig(f"{data_folder}/boxplot_compare_consistance.pdf")

# # 进行配对 t 检验：比较每两个组之间
# result_AB = ttest_rel(consistance['DAGAST'], consistance['Spatrack'])
# result_AC = ttest_rel(consistance['DAGAST'], consistance['stLearn'])
# result_AD = ttest_rel(consistance['DAGAST'], consistance['Slingshot'])

# # 打印每一对的 t 检验结果
# print(f"DAGAST vs Spatrack: t-statistic = {result_AB.statistic}, p-value = {result_AB.pvalue}")
# print(f"DAGAST vs stLearn: t-statistic = {result_AC.statistic}, p-value = {result_AC.pvalue}")
# print(f"DAGAST vs Slingshot: t-statistic = {result_AD.statistic}, p-value = {result_AD.pvalue}")

# DAGAST vs Spatrack: t-statistic = 6.452701298238018, p-value = 0.00011776916421165118
# DAGAST vs stLearn: t-statistic = 2.2555934562875697, p-value = 0.05053923873354081
# DAGAST vs Slingshot: t-statistic = 91.50462321379871, p-value = 1.1271321274623478e-14

# 进行Wilcoxon检验
statistic_AB, p_value_AB = wilcoxon(consistance['DAGAST'], consistance['Spatrack'])
statistic_AC, p_value_AC = wilcoxon(consistance['DAGAST'], consistance['stLearn'])
statistic_AD, p_value_AD = wilcoxon(consistance['DAGAST'], consistance['Slingshot'])
statistic_AE, p_value_AE = wilcoxon(consistance['DAGAST'], consistance['SpaceFlow'])

print(f"DAGAST vs Spatrack: t-statistic = {statistic_AB}, p-value = {p_value_AB}")
print(f"DAGAST vs stLearn: t-statistic = {statistic_AC}, p-value = {p_value_AC}")
print(f"DAGAST vs Slingshot: t-statistic = {statistic_AD}, p-value = {p_value_AD}")
print(f"DAGAST vs SpaceFlow: t-statistic = {statistic_AE}, p-value = {p_value_AE}")

# DAGAST vs Spatrack: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs stLearn: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs Slingshot: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs SpaceFlow: t-statistic = 1.0, p-value = 0.00390625
