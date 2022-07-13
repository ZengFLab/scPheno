from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.preprocessing import StandardScaler
import umap

def umap_embed(Z, metric='cosine', n_components=2, n_neighbors=15, min_dist=0.1, scale=False):
    if scale:
        Z2 = StandardScaler().fit_transform(Z)
    else:
        Z2 = Z
    return umap.UMAP(metric=metric, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(Z2)
