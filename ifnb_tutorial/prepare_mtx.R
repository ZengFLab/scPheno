library(Seurat)
library(SeuratData)
library(dplyr)
library(useful)
library(Matrix)
library(data.table)
library(reticulate)


# Specify python environment. Please change to your own environment.
use_python('/home/zengbio/anaconda3/envs/pyro/bin/python')
skl <- import('sklearn')


# Load data
LoadData('ifnb')

# Preprocessing
ifnb <- NormalizeData(ifnb)
ifnb <- FindVariableFeatures(ifnb, nfeatures = 5000)

ifnb$cell_type <- plyr::mapvalues(ifnb$seurat_annotations,
                                  from = c('CD8 T', 'CD4 Memory T', 'T activated', 'CD4 Naive T', 'B', 'B Activated', 'CD14 Mono', 'CD16 Mono', 'pDC'),
                                  to = c('T', 'T', 'T', 'T', 'B', 'B', 'Mono', 'Mono', 'DC'))

ifnb$cell_subtype <- ifnb$seurat_annotations

##################################################
# Prepare data
cells <- colnames(ifnb)
hvgs <- VariableFeatures(ifnb)

X <- GetAssayData(ifnb, 'data')
X <- t(X[hvgs, ] %>% as.matrix)
X[X<0] <- 0

cat('write data\n')
fwrite(X, file = paste('ifnb.txt',sep=''))



###################################
# Prepare phenotype data for scPheno
# Cell type
enc <- skl$preprocessing$OneHotEncoder(sparse=FALSE)$fit(ifnb@meta.data[cells,c('cell_type'),drop=FALSE])
onehot <- enc$transform(ifnb@meta.data[,c('cell_type'),drop=FALSE]) %>% as.data.frame
colnames(onehot) <- unlist(enc$categories_)
write.table(onehot, 
            file = 'ifnb_celltype.txt', 
            sep = ',', row.names = F, col.names = T, quote = T)


# Cell subtype
enc <- skl$preprocessing$OneHotEncoder(sparse=FALSE)$fit(ifnb@meta.data[cells,c('cell_subtype'),drop=FALSE])
onehot <- enc$transform(ifnb@meta.data[,c('cell_subtype'),drop=FALSE]) %>% as.data.frame
colnames(onehot) <- unlist(enc$categories_)
write.table(onehot, 
            file = 'ifnb_cellsubtype.txt', 
            sep = ',', row.names = F, col.names = T, quote = T)


# Stimulation status
enc <- skl$preprocessing$OneHotEncoder(sparse=FALSE)$fit(ifnb@meta.data[cells,c('stim'),drop=FALSE])
onehot <- enc$transform(ifnb@meta.data[,c('stim'),drop=FALSE]) %>% as.data.frame
colnames(onehot) <- unlist(enc$categories_)
write.table(onehot, 
            file = 'ifnb_condition.txt', 
            sep = ',', row.names = F, col.names = T, quote = T)


write.table(hvgs, file = 'ifnb_genes.txt', row.names = F, col.names = F)
write.table(cells, file = 'ifnb_cells.txt', row.names = F, col.names = F)