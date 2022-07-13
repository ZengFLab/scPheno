import MyUMAP
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def DPCluster(Z, n_components=10, max_iter=1000, weight_cutoff=0.05, weight_concentration_prior=1e5,
              weight_concentration_prior_type='dirichlet_process', use_pca = False, pca_dim = 50, random_state=42):
    if use_pca:
        Z = PCA(n_components=pca_dim).fit_transform(Z)

    bgm = BayesianGaussianMixture(n_components=n_components, 
                                  weight_concentration_prior_type=weight_concentration_prior_type, 
                                  weight_concentration_prior=weight_concentration_prior, 
                                  max_iter=max_iter, 
                                  random_state=random_state).fit(Z)
    is_kept = bgm.weights_ > weight_cutoff

    gm = GaussianMixture(n_components=sum(is_kept), covariance_type=bgm.covariance_type)
    gm.weights_ = bgm.weights_[is_kept] / np.sum(bgm.weights_[is_kept])
    gm.means_ = bgm.means_[is_kept]
    gm.covariances_ = bgm.covariances_[is_kept]
    gm.precisions_ = bgm.precisions_[is_kept]
    gm.precisions_cholesky_ = bgm.precisions_cholesky_[is_kept]

    return gm.predict(Z)
