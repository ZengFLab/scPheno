library(dplyr)
library(Matrix)
library(Seurat)
library(SeuratDisk)


seurat <- LoadH5Seurat('lee_2020_processed.HDF5')

seurat <- RunPCA(seurat)
seurat <- RunUMAP(seurat, dims = 1:20, metric = 'cosine')

DimPlot(seurat, reduction = 'umap', group.by='disease_status_standard')
DimPlot(seurat, reduction = 'umap', group.by = 'predicted.celltype.l1')

covids <- WhichCells(seurat, expression = disease_status_standard=='COVID-19')
DimPlot(seurat, reduction = 'umap', group.by='disease_severity_standard', cells = covids)
DimPlot(seurat, reduction = 'umap', group.by='predicted.celltype.l1', cells = covids)

save(seurat, file = 'lee_seurat.Robj')

hvgs <- VariableFeatures(seurat)

X <- GetAssayData(seurat, 'data')
X <- t(X[hvgs, ])
X[X<0] <- 0

cells <- colnames(seurat)

cat('write data\n')
spMat <- X[cells, ]
cat(nrow(spMat), ',', ncol(spMat), '\n')
writeMM(spMat, file = paste('lee_dataset.mtx',sep=''))

# save label (text)
write.table(seurat@meta.data[cells, 'predicted.celltype.l1'] %>% as.character,
            file = paste('lee_dataset_label_text.txt',sep=''),
            sep = '\n', row.names = F, col.names = F, quote = T)

write.table(seurat@meta.data[cells, 'predicted.celltype.l2'] %>% as.character,
            file = paste('lee_dataset_sublabel_text.txt',sep=''),
            sep = '\n', row.names = F, col.names = F, quote = T)


# save phenotype (text)

write.table(seurat@meta.data[cells, 'disease_status_standard'] %>% as.character,
            file = paste('lee_dataset_disease_text.txt',sep=''),
            sep = '\n', row.names = F, col.names = F, quote = T)




########## COVID

cat('write data\n')
spMat <- X[covids, ]
cat(nrow(spMat), ',', ncol(spMat), '\n')
writeMM(spMat, file = paste('lee_covid_dataset.mtx',sep=''))

# save label (text)
write.table(seurat@meta.data[covids, 'predicted.celltype.l1'] %>% as.character,
            file = paste('lee_covid_dataset_label_text.txt',sep=''),
            sep = '\n', row.names = F, col.names = F, quote = T)

write.table(seurat@meta.data[covids, 'predicted.celltype.l2'] %>% as.character,
            file = paste('lee_covid_dataset_sublabel_text.txt',sep=''),
            sep = '\n', row.names = F, col.names = F, quote = T)


# save phenotype (text)

write.table(seurat@meta.data[covids, 'disease_status_standard'] %>% as.character,
            file = paste('lee_covid_dataset_disease_text.txt',sep=''),
            sep = '\n', row.names = F, col.names = F, quote = T)

