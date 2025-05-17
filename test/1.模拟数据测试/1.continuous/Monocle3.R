# /public1/yuchen/software/miniconda3/envs/R4.2_yyc/bin/R
library(monocle3)
library(Seurat)
library(ggplot2)

###################%% version control
sample_name <- "Monocle3"
root_path <- "/public3/Shigw/datasets/Simulated_data/"

data_folder <- file.path(root_path, "continuous")
save_folder <- file.path(data_folder, "results", sample_name)

# 创建目录（如果不存在）
dir.create(file.path(data_folder, "results"), showWarnings = FALSE, recursive = TRUE)
dir.create(save_folder, showWarnings = FALSE, recursive = TRUE)

######################### 1. 读取数据，进行必要的处理 ########################
all_expdata <- read.table(file.path(data_folder, "sim_path_count.txt"), row.names = 1)
all_metadata <- read.table(file.path(data_folder, "sim_path_metadata.txt"), row.names = 1)

all_batch <- unique(all_metadata$Batch)
sel_batch <- all_batch[1]

# 筛选特定批次数据
metadata <- all_metadata[all_metadata$Batch == sel_batch, ]
expdata <- all_expdata[, rownames(metadata)]
metadata$x <- metadata$x - min(metadata$x)
metadata$y <- metadata$y - min(metadata$y)

# Data analysis with Seurat pipeline
data <- CreateSeuratObject(counts = as.matrix(expdata), meta.data = metadata)
data <- NormalizeData(data)
data <- FindVariableFeatures(data, nfeatures = 2000)
data <- ScaleData(data)
data <- RunPCA(data)

data <- FindNeighbors(data)
data <- FindClusters(data, resolution = 1)
data <- RunUMAP(data, n.neighbors = 10, dims = 1:50, spread = 2, min.dist = 0.3)

counts <- GetAssayData(seurat_obj, assay = "RNA", slot = "counts")
cell_metadata <- seurat_obj@meta.data
gene_metadata <- data.frame(gene_short_name = rownames(counts), row.names = rownames(counts))

cds <- new_cell_data_set(
    counts,
    cell_metadata = cell_metadata,
    gene_metadata = gene_metadata)

save_folder_cluster = file.path(save_folder, "2.spatial_cluster")
dir.create(save_folder_cluster, showWarnings = FALSE, recursive = TRUE)

cds <- preprocess_cds(cds, method='PCA', num_dim=50)
cds <- reduce_dimension(cds, reduction_method='UMAP')

cds <- cluster_cells(cds = cds, reduction_method = "UMAP", k=20, cluster_method="leiden", resolution=0.001, num_iter=2)

pdf(file.path(save_folder_cluster, "2.spatial_cluster.pdf"))
p1 <- plot_cells(cds, show_trajectory_graph = FALSE)
print(p1)
dev.off()


# cds <- cluster_cells(cds)
cds <- cluster_cells(cds, cluster_method="louvain")
colData(cds)['louvain'] = clusters(cds, reduction_method = "UMAP")
cds <- learn_graph(cds)

pdf(file=file.path(save_folder_cluster, "2.spatial_cluster.pdf"), width=5, height=5)
print(ggplot(as.data.frame(colData(cds)), aes(x = x, y = y, color = louvain)) +
  geom_point() + theme_minimal())
dev.off()


# 获取 x 和 y 坐标
x_vals <- colData(cds)$x
y_vals <- colData(cds)$y
distances <- sqrt((x_vals - 100)^2 + (y_vals - 100)^2)
nearest_index <- which.min(distances)
nearest_point <- colData(cds)[nearest_index, ]

start_cell = rownames(colData(cds))[colData(cds)$louvain == nearest_point$louvain]
cds <- order_cells(cds, root_cells=rownames(nearest_point))

# 获取每个细胞的伪时间
pseudotime_values <- colData(cds)$pseudotime
head(pseudotime_values)
