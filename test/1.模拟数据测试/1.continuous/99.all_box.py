import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

DAGAST = [0.7077535506780133,
 0.7172767969890788,
 0.7270842457297765,
 0.71393403251562,
 0.7158939428785949,
 0.718178306429203,
 0.7092628432512378,
 0.7164898383592717,
 0.721813087569487,
 0.7248652150644025]

Spatrack = [0.646337435991121,
 0.6773573291987011,
 0.6823867297258099,
 0.686097920320917,
 0.6625855315018421,
 0.61147595118897,
 0.7212712131035001,
 0.7073361067210289,
 0.6776933845065738,
 0.6924168570367873]

stLearn = [0.5590974091430964,
 0.5994607007020761,
 0.6180370625841098,
 0.5965663154029419,
 0.6134272220810657,
 0.5199944115458144,
 0.6044592963911851,
 0.6080629671125092,
 0.5822417073773534,
 0.513224805067858]


Slingshot = [0.5426309447279716,
 0.53170865637275,
 0.5534318345197282,
 0.5630886915266882,
 0.5261660616762047,
 0.5123656502622563,
 0.5129655087814426,
 0.5610380208472997,
 0.5645528668222697,
 0.5633356442200651]

Spaceflow = [0.6418957299140141,
 0.6052331294324044,
 0.41998324018233546,
 0.6345471785221133,
 0.6390780984084399,
 0.285074509506998,
 0.6102242181095856,
 0.6602131288895146,
 0.6234830163062994,
 0.04661450032139454]



root_path = "/public3/Shigw/datasets/Simulated_data/"
data_folder = f"{root_path}/continuous/"


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

# DAGAST vs Spatrack: t-statistic = 2.0, p-value = 0.005859375    2
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
# DAGAST vs stLearn: t-statistic = 2.0, p-value = 0.005859375
# DAGAST vs Slingshot: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs SpaceFlow: t-statistic = 4.0, p-value = 0.013671875
