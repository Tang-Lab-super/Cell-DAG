import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

DAGAST = [0.7244090760321319,
 0.7341910751737479,
 0.7178911149644409,
 0.7373502336338039,
 0.7404582847317389,
 0.7230603035864526,
 0.725380884079978,
 0.7346063376227772,
 0.7394593155790788,
 0.7395733159241656]

Spatrack = [0.622801559065225,
 0.6280520728154446,
 0.6288625671076195,
 0.6729529079843681,
 0.6381655199157302,
 0.6752813175394776,
 0.7175492691242837,
 0.31003314945356986,
 0.6529687478031456,
 0.7188433947015933]

stLearn = [0.5466140758097631,
 0.579438478479854,
 0.5923926181396654,
 0.5778440931807197,
 0.5918605554143991,
 0.5090166337680366,
 0.5834092963911852,
 0.5842185226680647,
 0.5672417073773534,
 0.5052859161789691]


Slingshot = [0.5252031669501938,
 0.5178586563727501,
 0.5408318345197282,
 0.5481942470822437,
 0.5123216172317603,
 0.5017656502622563,
 0.5017710643369983,
 0.5429046875139664,
 0.547636200155603,
 0.5472411997756207]

Spaceflow = [0.6270423023207357,
 0.6263736423474779,
 0.6319110007321095,
 0.6653955746173503,
 0.3778067830751407,
 0.6612685865990429,
 0.6426286004602996,
 0.6458412314012848,
 0.1689348392728021,
 0.660540319622908]


root_path = "/public3/Shigw/datasets/Simulated_data/"
data_folder = f"{root_path}/branch/"

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

# DAGAST vs Spatrack: t-statistic = 2.0, p-value = 0.005859375
# DAGAST vs stLearn: t-statistic = 2.0, p-value = 0.005859375
# DAGAST vs Slingshot: t-statistic = 0.0, p-value = 0.001953125
# DAGAST vs SpaceFlow: t-statistic = 3.0, p-value = 0.009765625

