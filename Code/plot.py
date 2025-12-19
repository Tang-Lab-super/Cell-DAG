# shigw 2025-3-15
# plot function

import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

mycolor = [
    "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
    "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
    "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
    "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
]

def plot_feature(xy1, xy2, value, title="feature plot", pointsize=5):
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=pointsize)
    sns.scatterplot(x = xy2[:, 0], y = xy2[:, 1], marker = 'o',
                    c = value, s=pointsize,  cmap='Spectral_r', legend = True)
    # plt.title(title)
    plt.axis('off')

def plot_cluster(xy1, st_data_sel, key="cluster", title="cluster plot", pointsize=5):
    mycolor = [
        "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A", "#B15928", "#A6CEE3", "#B2DF8A", 
        "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#1F77B4", "#AEC7E8", "#98DF8A", "#FF9896",
        "#C5B0D5", "#C49C94", "#F7B6D2", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
        "#D62728", "#9467BD", "#2CA02C", "#FFBB78", "#C7C7C7", "#8C6D31"
    ]
    sns.scatterplot(x = xy1[:, 0], y = xy1[:, 1],  color = (207/255,185/255,151/255, 1), s=pointsize)
    for idx, ci in enumerate(st_data_sel.obs[key].unique().tolist()):
        if ci == -1:
            continue
        subda = st_data_sel[st_data_sel.obs[key] == ci, :]
        sns.scatterplot(x=subda.obsm['spatial'][:, 0], y=subda.obsm['spatial'][:, 1], marker='o', c=mycolor[idx], s=pointsize)
        # plt.text(subda.obsm['spatial'][:, 0].mean(), subda.obsm['spatial'][:, 1].mean(), str(ci), fontsize=8)
    # plt.title(title)
    plt.axis('off')

def plot_spatial_complex(
    st_data, st_data_sel, mode="time", value=None, key="",
    figsize=(5, 5), title=None, pointsize=5, savename='./fig.pdf'):

    if mode=="time":
        assert value is not None, "value is None."
    elif mode=="cluster":
        assert key in st_data_sel.obs.columns, f"{key} is empty."

    plt.close('all')
    fig = plt.figure(figsize=figsize)
    plt.subplot(1,1,1)
    if mode=="time":
        plot_feature(st_data.obsm['spatial'], st_data_sel.obsm['spatial'], value, title, pointsize)
    elif mode == "cluster":
        plot_cluster(st_data.obsm['spatial'], st_data_sel, key, title, pointsize)
    plt.tight_layout()
    plt.savefig(savename, dpi=300)

def plot_spatial_gene(st_data, key="", figsize=(5, 5), title=None, pointsize=5, savename='./fig.pdf'):
    plt.close('all')
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 1, 1)
    plot_feature(st_data.obsm['spatial'], st_data.obsm['spatial'], st_data.obs[key], title=title, pointsize=5)
    plt.tight_layout()
    plt.savefig(savename, dpi=300)

def plot_permutation_genes(st_data, plot_cells, plot_genes, savename, figsize=(15, 10), pp=True):
    st_data_plot = st_data.copy()
    if pp:
        sc.pp.normalize_total(st_data_plot)
        sc.pp.log1p(st_data_plot)
        sc.pp.scale(st_data_plot)
    st_data_plot = st_data_plot[plot_cells, :]

    for idx, gi in enumerate(plot_genes):
        plt.close('all')
        fig = plt.figure(figsize=figsize)
        sns.scatterplot(x=st_data.obsm['spatial'][:, 0], y = st_data.obsm['spatial'][:, 1],  color = (207/255,185/255,151/255, 1), s=5)
        sns.scatterplot(x=st_data_plot.obsm['spatial'][:, 0], y = st_data_plot.obsm['spatial'][:, 1], marker = 'o',
                        c=st_data_plot[:,gi].X.T[0], s=5,  cmap='Spectral_r', legend = True)
        plt.axis('off')
        # plt.title(gi)
        plt.savefig(f"{savename}_{gi}.pdf")
        print(gi)
