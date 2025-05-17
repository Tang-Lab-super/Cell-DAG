# conda activate r-4.1.3
library(glue)
library(ggplot2)
suppressPackageStartupMessages({
    library(splatter)
    library(scater)
})

# 1. continuous
anspath = "/public3/Shigw/datasets/Simulated_data/continuous/"

set.seed(21)
nstep = 100
batcell = c(3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000)
params <- newSplatParams()
# Set multiple parameters at once (using a list)
params <- setParams(params, update = list(
    nGenes = 3000, batchCells=batcell, mean.shape=0.6, mean.rate = 0.3, 
    bcv.common=0.2, dropout.mid=0, dropout.shape=-1, out.prob=0.05, de.prob=1,
    path.nSteps=nstep, path.skew=0.2, path.nonlinearProb=0.1))
getParams(params, c("nGenes", "mean.rate", "mean.shape"))
sim <- splatSimulate(params, method="paths")
sim_counts <- counts(sim)
write.table(sim_counts, glue('{anspath}sim_path_count.txt'), quote=F, sep='\t')


## 使用PCA展示生成的单细胞状态
sim_path <- logNormCounts(sim)
sim_path <- runPCA(sim_path)
pdf(glue('{anspath}PCA_sim_path.pdf'), w=6, h=4)
# plotPCA(sim_path, colour_by = "Step")
p <- plotPCA(sim_path, colour_by = "Step") +
    #  theme(axis.title.x = element_blank(), axis.title.y = element_blank())
     xlab("PC1") + 
     ylab("PC2")
print(p)
dev.off()

## 添加空间位置
steps = sim_path@colData$Step

set.seed(21)
n <- 30000
# radii <- sqrt(steps) - 1
radii <- steps
angles <- runif(n, 0, 2*pi)
x <- radii * cos(angles) + rnorm(length(angles), mean = 0, sd = 1)
y <- radii * sin(angles) + rnorm(length(angles), mean = 0, sd = 1)
data <- data.frame(x, y, group = as.factor(radii))
data$batch = sim_path@colData$Batch
rownames(data) = rownames(sim_path@colData)

pdf(glue('{anspath}/spatial_sim_path.pdf'), w=6, h=4)
p <- ggplot(data[data$batch == 'Batch1',], aes(x, y, color = group)) +
  geom_point() +
  coord_fixed() +
  theme_minimal() +
  theme(legend.position = "none")
print(p)
dev.off()

metadata = sim_path@colData
metadata$x = data$x
metadata$y = data$y
write.table(metadata, glue('{anspath}/sim_path_metadata.txt'), quote=F, sep='\t')


# 4. branch
anspath = "/public3/Shigw/datasets/Simulated_data/branch/"

set.seed(21)
nstep = 100
batcell = c(3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000)
params <- newSplatParams()
# Set multiple parameters at once (using a list)
params <- setParams(params, update = list(
    nGenes = 3000, batchCells=batcell, mean.shape=0.6, mean.rate = 0.3, 
    bcv.common=0.2, dropout.mid=0, dropout.shape=-1, out.prob=0.05, de.prob=1,
    path.nSteps=nstep, path.skew=0.2, path.nonlinearProb=0.1))
getParams(params, c("nGenes", "mean.rate", "mean.shape"))
sim <- splatSimulate(params, method="paths")
sim_counts <- counts(sim)
write.table(sim_counts, glue('{anspath}sim_path_count.txt'), quote=F, sep='\t')

## 使用PCA展示生成的单细胞状态
sim_path <- logNormCounts(sim)
sim_path <- runPCA(sim_path)

pdf(glue('{anspath}PCA_sim_path.pdf'), w=6, h=4)
# plotPCA(sim_path, colour_by = "Step")
p <- plotPCA(sim_path, colour_by = "Step") +
    #  theme(axis.title.x = element_blank(), axis.title.y = element_blank())
     xlab("PC1") + ylab("PC2")
print(p)
dev.off()

## 添加空间位置
steps = sim_path@colData$Step

set.seed(21)
n <- 30000
branch <- sample(1:2, n, replace = TRUE)
angle_offset <- c(0, pi/2, -pi/2)
angles <- runif(n, -pi/6, pi/6) + angle_offset[branch]
radii <- steps
# radii <- sqrt(steps)
x <- radii * cos(angles) + rnorm(length(angles), mean = 0, sd = 1)
y <- radii * sin(angles) + rnorm(length(angles), mean = 0, sd = 1)
data <- data.frame(x, y, group = as.factor(radii))
data$batch = sim_path@colData$Batch
rownames(data) = rownames(sim_path@colData)

pdf(glue('{anspath}/spatial_sim_path.pdf'), w=6, h=4)
p <- ggplot(data[data$batch == 'Batch1',], aes(x, y, color = group)) +
  geom_point() +
  coord_fixed() +
  theme_minimal() +
  theme(legend.position = "none")
print(p)
dev.off()

metadata = sim_path@colData
metadata$x = data$x
metadata$y = data$y
write.table(metadata, glue('{anspath}/sim_path_metadata.txt'), quote=F, sep='\t')


# 5. discrete
anspath = "/public3/Shigw/datasets/Simulated_data/discrete/"

set.seed(21)
nstep = 100
batcell = c(3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000)
params <- newSplatParams()
# Set multiple parameters at once (using a list)
params <- setParams(params, update = list(
    nGenes = 3000, batchCells=batcell, mean.shape=0.6, mean.rate = 0.3, 
    bcv.common=0.2, dropout.mid=0, dropout.shape=-1, out.prob=0.05, de.prob=1,
    path.nSteps=nstep, path.skew=0.3, path.nonlinearProb=0.1))
getParams(params, c("nGenes", "mean.rate", "mean.shape"))
sim <- splatSimulate(params, method="paths")
sim_counts <- counts(sim)
write.table(sim_counts, glue('{anspath}sim_path_count.txt'), quote=F, sep='\t')


## 使用PCA展示生成的单细胞状态
sim_path <- logNormCounts(sim)
sim_path <- runPCA(sim_path)

pdf(glue('{anspath}PCA_sim_path.pdf'), w=6, h=4)
p <- plotPCA(sim_path, colour_by = "Step") +
     xlab("PC1") + 
     ylab("PC2")
print(p)
dev.off()

## 添加空间位置
steps = sim_path@colData$Step

set.seed(21)
n <- 30000
radii <- steps
angles <- runif(n, 0, 2*pi)
x <- radii * cos(angles) + rnorm(length(angles), mean = 0, sd = 1)
y <- radii * sin(angles) + rnorm(length(angles), mean = 0, sd = 1)
data <- data.frame(x, y, group = as.factor(radii))
data$batch = sim_path@colData$Batch
rownames(data) = rownames(sim_path@colData)

# 随机选择一半的点并对其偏移
selected_indices <- sample(1:n, n / 2)
# data$group[selected_indices] <- "Group2"  # 为选中的点打标签
data$x[selected_indices] <- data$x[selected_indices] + 200  # 对x轴进行偏移
data$y[selected_indices] <- data$y[selected_indices] + 200  # 对y轴进行偏移

pdf(glue('{anspath}/spatial_sim_path.pdf'), w=6, h=4)
p <- ggplot(data[data$batch == 'Batch2',], aes(x, y, color = group)) +
  geom_point() +
  coord_fixed() +
  theme_minimal() +
  theme(legend.position = "none")
print(p)
dev.off()

metadata = sim_path@colData
metadata$x = data$x
metadata$y = data$y
write.table(metadata, glue('{anspath}/sim_path_metadata.txt'), quote=F, sep='\t')



# # 2. fluctuant
# set.seed(21)
# n <- 3000
# radii <- sqrt(steps)
# # radii <- steps
# angles <- runif(n, 0, 2*pi)
# x <- radii * cos(angles) + rnorm(length(angles), mean = 0, sd = 1)
# y <- radii * sin(angles) + rnorm(length(angles), mean = 0, sd = 1)
# data <- data.frame(x, y, group = as.factor(radii))

# pdf(glue('{anspath}/fluctuant/sim_path_{nstep}.pdf'), w=6, h=4)
# p <- ggplot(data, aes(x, y, color = group)) +
#   geom_point() +
#   coord_fixed() +
#   theme_minimal() +
#   theme(legend.position = "none")
# print(p)
# dev.off()

# metadata = sim_path@colData
# metadata$x = data$x
# metadata$y = data$y
# write.table(metadata, glue('{anspath}/fluctuant/sim_path_metadata_{nstep}.txt'), quote=F, sep='\t')


# # 3. uneven(在使用的时候需要将表达量进行uneven再处理，否则这个和1没有区别)
# set.seed(21)
# n <- 3000
# radii <- steps
# angles <- runif(n, 0, 2*pi)
# x <- radii * cos(angles) + rnorm(length(angles), mean = 0, sd = 1)
# y <- radii * sin(angles) + rnorm(length(angles), mean = 0, sd = 1)
# group = radii
# group[group < 60] = 1
# group[group > 60] = group[group > 60] - 59
# data <- data.frame(x, y, group = as.factor(group))

# pdf(glue('{anspath}/uneven/sim_path_{nstep}.pdf'), w=6, h=4)
# p <- ggplot(data, aes(x, y, color = group)) +
#   geom_point() +
#   coord_fixed() +
#   theme_minimal() +
#   theme(legend.position = "none")
# print(p)
# dev.off()

# metadata = sim_path@colData
# metadata$x = data$x
# metadata$y = data$y
# write.table(metadata, glue('{anspath}/uneven/sim_path_metadata_{nstep}.txt'), quote=F, sep='\t')
