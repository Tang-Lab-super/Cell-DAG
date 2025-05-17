# /public1/yuchen/software/miniconda3/envs/R4.2_yyc/bin/R
library(Seurat)
library(cowplot)
library(ggplot2)
library(slingshot)

###################%% version control
sample_name <- "Slingshot"
root_path <- "/public3/Shigw/datasets/Simulated_data/"

data_folder <- file.path(root_path, "branch")
save_folder <- file.path(data_folder, "results", sample_name)

# 创建目录（如果不存在）
dir.create(file.path(data_folder, "results"), showWarnings = FALSE, recursive = TRUE)
dir.create(save_folder, showWarnings = FALSE, recursive = TRUE)

######################### 1. 读取数据，进行必要的处理 ########################
all_expdata <- read.table(file.path(data_folder, "sim_path_count.txt"), row.names = 1)
all_metadata <- read.table(file.path(data_folder, "sim_path_metadata.txt"), row.names = 1)

R_list = c()
sep_list = c()
all_batch <- unique(all_metadata$Batch)
# sel_batch <- all_batch[1]
for(sel_batch in all_batch) {
    print(sel_batch)
    # 筛选特定批次数据
    metadata <- all_metadata[all_metadata$Batch == sel_batch, ]
    expdata <- all_expdata[, rownames(metadata)]
    # metadata$x <- metadata$x - min(metadata$x)
    # metadata$y <- metadata$y - min(metadata$y)

    # Data analysis with Seurat pipeline
    data <- CreateSeuratObject(counts = as.matrix(expdata), meta.data = metadata)
    data <- NormalizeData(data)
    data <- FindVariableFeatures(data, nfeatures = 300)
    data <- ScaleData(data, features = VariableFeatures(data))
    data <- RunPCA(data, features = VariableFeatures(data))

    data <- FindNeighbors(data, features = VariableFeatures(data))
    data <- FindClusters(data, resolution = 1.0, features = VariableFeatures(data))
    data <- RunUMAP(data, n.neighbors = 30, dims = 1:50, spread = 2, min.dist = 0.3)

    ## 绘制聚类图，并进一步选出起始点
    # Plot the clusters
    save_folder_cluster = file.path(save_folder, "2.spatial_cluster")
    dir.create(save_folder_cluster, showWarnings = FALSE, recursive = TRUE)

    pdf(file=file.path(save_folder_cluster, "2.UMAP_cluster.pdf"), width=5, height=5)
    print(DimPlot(data, group.by = "RNA_snn_res.1"))
    dev.off()

    pdf(file=file.path(save_folder_cluster, "2.spatial_cluster.pdf"), width=5, height=5)
    print(ggplot(data@meta.data, aes(x = x, y = y, color = seurat_clusters)) +
    geom_point() + theme_minimal())
    dev.off()

    # Save the objects as separate matrices for input in slingshot
    dimred <- data@reductions$umap@cell.embeddings
    clustering <- data$seurat_clusters
    counts <- as.matrix(data@assays$RNA@counts[data@assays$RNA@var.features, ])

    # 获取 x 和 y 坐标
    x_vals <- metadata$x
    y_vals <- metadata$y
    distances <- sqrt((x_vals - 0)^2 + (y_vals - 0)^2)
    nearest_index <- which.min(distances)
    start_point <- data@meta.data$seurat_clusters[nearest_index]

    save_folder_trajectory = file.path(save_folder, "3.spatial_trajectory")
    dir.create(save_folder_trajectory, showWarnings = FALSE, recursive = TRUE)


    set.seed(1)
    lineages <- getLineages(data = dimred,
                            clusterLabels = clustering,
                            start.clus = start_point) #define where to start the trajectories

    # # Plot the lineages
    # par(mfrow = c(1, 2))
    # plot(dimred[, 1:2], col = pal[clustering], cex = 0.5, pch = 16)
    # for (i in levels(clustering)) {
    #     text(mean(dimred[clustering == i, 1]), mean(dimred[clustering == i, 2]), labels = i, font = 2)
    # }
    # plot(dimred[, 1:2], col = pal[clustering], cex = 0.5, pch = 16)
    # lines(lineages, lwd = 3, col = "black")
    # #Plot the lineages
    # par(mfrow=c(1,2))
    # plot(dimred[,1:2], col = pal[clustering],  cex=.5,pch = 16)
    # for(i in levels(clustering)){ 
    #   text( mean(dimred[clustering==i,1]),
    #         mean(dimred[clustering==i,2]), labels = i,font = 2) }
    # plot(dimred, col = pal[clustering],  pch = 16)
    # lines(lineages, lwd = 3, col = 'black')

    curves <- getCurves(lineages, approx_points = 3000, thresh = 0.01, stretch = 0.8, allow.breaks = FALSE, shrink = 0.99)
    pseudo = as.numeric(as.character(curves@assays@data$pseudotime[, 1]))
    target_order = metadata$Step

    r = cor.test(pseudo, target_order, method="spearman")$estimate[[1]]
    print(r)

    sep_list = c(sep_list, c(pseudo))

    R_list = c(R_list, r)
}


df_res = data.frame(ori=all_metadata$Step, pseudo=sep_list)
write.table(df_res, file.path(save_folder, "df_pseudo.txt"), quote=F, sep='\t')

# R_list = c(0.9452369, 0.9562694, 0.9608358, 0.9444693, 0.9565799, 0.9540634, 0.9609866, 0.9590831, 0.9343447, 0.9543782)
