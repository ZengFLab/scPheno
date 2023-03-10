# scPheno
 scPheno is to extract variations on gene expression associated with phenotypes, which can be categorical, compositional, normal, and discrete covariates.

## Citation
Feng Zeng, Xuwen Kong, Fan Yang, Ting Chen, Jiahuai Han. Extraction of biological signals by factorization enables the reliable analysis of single-cell transcriptomics. Submission. 2023

## Installation
1. Install [pytorch](https://pytorch.org/get-started/locally/) according to your computational platform
2. Install dependencies:
    - `pip3 install numpy scipy pandas scikit-learn pyro-ppl matplotlib`

    - `pip install datatable`


## Tutorial

### Example 1: IFNB factor combination 1
For more information, please refer to the [IFNB](https://github.com/ZengFLab/scPheno/blob/main/ifnb.ipynb) example.

### Example 2: Multi-omics

### Example 3: Mouse embryo development


## Usage
```
usage: scPheno.py [-h] [--cuda] [--jit] [-n NUM_EPOCHS] [-bs BATCH_SIZE] [-lr LEARNING_RATE]
                  [--sup-data-file SUP_DATA_FILE] 
                  [--sup-label-file SUP_LABEL_FILE] 
                  [--sup-condition-file SUP_CONDITION_FILE]
                  [--sup-condition2-file SUP_CONDITION2_FILE]
                  [--zero-inflation] [--likelihood LIKELIHOOD] [--use-dirichlet]
                  [--label-type LABEL_TYPE]
                  [--condition-type CONDITION_TYPE]
                  [--condition2-type CONDITION2_TYPE]
                  [--save-model SAVE_MODEL]

scPheno example run: python scPheno.py --sup-data-file <sup_data_file> --sup-label-file <sup_label_file> --sup-condition-file <sup_condition_file> --sup-condition2-file <sup_condition2_file> --cuda -lr 0.0001 -bs 1000 -n 50 --save-model best_model.pth

```


## Types of phenotype variable

Users can specify the type of phenotype variable by setting ```--label-type```, ```--condition-type```, ```--condition2-type```.

scPheno supports various types of phenotype variables, including
1. categorical: Categorical variable
2. onehot: One-hot encoding variable
3. compositional: relative abundance, whose sum equals to one
4. rate: ratios, whose sum could not equal to one
5. real: normal variable


## Likelihood function
scPheno supports the following distributions for observed expression data:
1. negbinomial: Negative-binomial
2. poisson: Poisson
3. multinomial: Multinomial
