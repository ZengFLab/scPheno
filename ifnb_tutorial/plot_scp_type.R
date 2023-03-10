library(Seurat)
library(SeuratObject)
library(SeuratDisk)
library(dplyr)
library(useful)
library(Matrix)
library(matrixStats)
library(data.table)
library(ggplot2)
library(cowplot)
library(ggsci)

# seurat <- LoadH5Seurat('Mouse_olfa.h5seurat')

data <- fread('ifnb_celltype_denoised_expression.txt',data.table = FALSE)
rownames(data) <- data[,1]
data <- data[,-1]
data <- t(data)

counts <- expm1(data)


recon <- CreateSeuratObject(counts, meta.data = ifnb@meta.data[colnames(data), ])
recon <- SetAssayData(recon, slot = 'data', data)
recon <- FindVariableFeatures(recon, nfeatures = 5000)
recon <- ScaleData(recon)

recon <- RunPCA(recon, verbose=FALSE)
recon <- RunUMAP(recon, dims=1:30)


DimPlot(recon, reduction = 'umap', group.by = 'cell_type') +
  ggtitle('Factor combination 1') +
  theme(plot.title = element_text(size=24),
        axis.title = element_text(size=18),
        legend.text = element_text(size=16))



DimPlot(recon, reduction = 'umap', group.by = 'cell_subtype') +
  ggtitle('Factor combination 1') +
  theme(plot.title = element_text(size=24),
        axis.title = element_text(size=18),
        legend.text = element_text(size=16))



