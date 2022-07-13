library(dplyr)
library(Matrix)
library(Seurat)
library(SeuratObject)


load('lee_seurat.Robj')

scp.umap <- read.csv('lee_dataset_umap.txt', header = FALSE)
colnames(scp.umap) <- c('SCP_1','SCP_2')
rownames(scp.umap) <- Cells(seurat)


seurat[['scp']] <- CreateDimReducObject(embeddings = as.matrix(scp.umap),
                                        key = 'SCP_',
                                        assay = 'RNA')

DimPlot(seurat, reduction = 'scp', group.by='disease_status_standard')
DimPlot(seurat, reduction = 'scp', group.by = 'predicted.celltype.l1')
DimPlot(seurat, reduction = 'scp', group.by = 'predicted.celltype.l2')


############ covid
covids <- WhichCells(seurat, expression = disease_status_standard=='COVID-19')
seurat.covid <- seurat[,covids]

scp.covid.umap <- read.csv('lee_covid_dataset_umap.txt', header = FALSE)
colnames(scp.covid.umap) <- c('SCP_1','SCP_2')
rownames(scp.covid.umap) <- Cells(seurat.covid)


seurat.covid[['scp']] <- CreateDimReducObject(embeddings = as.matrix(scp.covid.umap),
                                              key = 'SCP_',
                                              assay = 'RNA')
DimPlot(seurat.covid, reduction = 'scp', group.by='disease_severity_standard')
DimPlot(seurat.covid, reduction = 'scp', group.by = 'predicted.celltype.l1')



DimPlot(seurat, reduction = 'scp', group.by='disease_severity_standard', cells = covids)
DimPlot(seurat, reduction = 'scp', group.by = 'predicted.celltype.l1', cells = covids)
