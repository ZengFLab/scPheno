# scPheno
 scPheno is a deep generative learning method for the joint analysis of scRNA-seq and phenotypes. scPheno can be broadly applied to scRNA-seq data or databases in mapping the transcriptional variations to phenotypes at single-cell level to uncover the functional insight of cell-to-cell variability.

## Citation
Feng Zeng, Xuwen Kong, Fan Yang, Ting Chen, Jiahuai Han. scPheno-XMBD: A deep generative model to integrate scRNA-seq with phenotypes and its application in COVID-19. Submission. 2022

## Installation
1. Install [pytorch](https://pytorch.org/get-started/locally/) according to your computational platform
2. Install dependencies:
    `pip3 install numpy scipy pandas scikit-learn pyro-ppl matplotlib`


## Tutorial
For more information, please refer to the [COVID](https://github.com/ZengFLab/scPheno/blob/main/PBMC_predictor_LEE_type.ipynb) example.

## Usage
```
usage: scPheno.py [-h] [--cuda] [--jit] [-n NUM_EPOCHS] [--aux-loss] [-alm AUX_LOSS_MULTIPLIER] [-enum ENUM_DISCRETE] [--sup-data-file SUP_DATA_FILE] [--sup-label-file SUP_LABEL_FILE] [--sup-condition-file SUP_CONDITION_FILE] [--unsup-data-file UNSUP_DATA_FILE]
                  [--unsup-label-file UNSUP_LABEL_FILE] [--unsup-condition-file UNSUP_CONDITION_FILE] [-64] [-lt] [-cv VALIDATION_FOLD] [-zd Z_DIM] [-hl HIDDEN_LAYERS [HIDDEN_LAYERS ...]] [-lr LEARNING_RATE] [-dr DECAY_RATE] [-de DECAY_EPOCHS] [-b1 BETA_1] [-bs BATCH_SIZE] [-rt]
                  [-log LOGFILE] [--seed SEED] [--save-model SAVE_MODEL] [-ba]

scPheno example run: python scPheno.py --sup-data-file <sup_data_file> --sup-label-file <sup_label_file> --sup-condition-file <sup_condition_file> --cuda --aux-loss -lr 0.0001 -bs 1000 -n 50 -ba --validation-fold 10 --save-model best_model.pth

```

