from tkinter import Label
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.preprocessing import LabelEncoder

from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

from sklearn.linear_model import Lasso, LassoCV, SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV

def FeatureImportance(X,y,n_features_to_select,max_depth=5):
    #estimator = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=max_depth)
    estimator = LogisticRegression(n_jobs=-1)
    selector = RFE(estimator, step=1, n_features_to_select=n_features_to_select)
    selector = selector.fit(X, y)
    return selector.support_,selector.ranking_


def Cond_type_scores(scPheno_model, data_loader, use_cuda=True):
    # predict conditions
    cond_scores = []
    type_scores = []

    # use the appropriate data loader
    for xs,ys,ks in data_loader:
        # use classification function to compute all predictions for each batch
        if use_cuda:
            xs = xs.cuda()

        _, kscores = scPheno_model.predicted_condition(xs)
        _, yscores = scPheno_model.classifier_with_probability(xs)

        if use_cuda:
            kscores = kscores.cpu().detach().numpy()
            yscores = yscores.cpu().detach().numpy()
        else:
            kscores = kscores.detach().numpy()
            yscores = yscores.detach().numpy()

        cond_scores.append(kscores)
        type_scores.append(yscores)

    cond_scores = np.concatenate(cond_scores, axis=0)
    type_scores = np.concatenate(type_scores, axis=0)
    
    return cond_scores, type_scores

def DiseaseFeature(scPheno_model, data_loader, sample_source, use_weight=False, weights=None, middle=0.05, use_cuda = True):

    cond_scores, type_scores = Cond_type_scores(scPheno_model, data_loader, use_cuda)

    # type importance
    type_weights=None
    if use_weight:
        if weights is None:
            type_weights = FeatureImportance(type_scores, sample_source['Disease'])
        else:
            type_weights = weights
        #type_weights = softmax(type_weights - middle)
        type_scores = type_scores * type_weights[None,...]

    cond_df = pd.DataFrame(cond_scores)
    type_df = pd.DataFrame(type_scores)

    type_cond_df = pd.DataFrame((cond_df.values[..., None] * type_df.values[:, None]).reshape(cond_df.shape[0],-1))

    #cond_df = pd.concat([cond_df, sample_source], axis=1)
    #cond_df = cond_df.groupby(['Sample', 'Disease']).mean()
    #cond_df_ = cond_df.reset_index(level='Disease')
    #cond_df = cond_df.values
    #
    #type_df = pd.concat([np.log1p(type_df), sample_source], axis=1)
    #type_df = type_df.groupby(['Sample', 'Disease']).sum()
    #type_df = softmax(type_df.values, axis=1)
    #
    type_cond_df = pd.concat([type_cond_df, sample_source], axis=1)
    type_cond_df_ = type_cond_df.groupby(['Sample', 'Disease']).mean()
    type_cond_df = type_cond_df_.values
    #type_cond_df = normalize(type_cond_df, axis=1, norm='l1')
    
    data = type_cond_df

    type_cond_df_ = type_cond_df_.reset_index(level='Disease')
    label = type_cond_df_['Disease'].values.tolist()

    return data, label, type_weights



def DiseasePredictor(scPheno_model, train_data_loader, train_sample_source, use_weight=False, use_cuda=True, scaling=False):
    print('Extract features\n')
    train_data, train_label, train_weights = DiseaseFeature(scPheno_model, train_data_loader, train_sample_source, use_weight=use_weight, use_cuda=use_cuda)

    if scaling:
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
    else:
        scaler = None

    print('Train model\n')
    model = SVC(kernel='linear',probability=True)
    #model = MLPClassifier(hidden_layer_sizes=[1000,500,300])
    model.fit(train_data, train_label)

    print('Validation\n')
    y = model.predict(train_data)
    print(sum(y == train_label) / train_data.shape[0])

    return {'model':model, 'scaler':scaler, 'weights':train_weights}


def DiseasePredict(predictor, scPheno_model, test_data_loader, test_sample_source, use_weight=True, use_cuda=True):

    test_data, test_label, _ = DiseaseFeature(scPheno_model,test_data_loader,test_sample_source,use_weight=use_weight,weights=predictor['weights'],use_cuda=use_cuda)

    if predictor['scaler'] is not None:
        test_data = predictor['scaler'].transform(test_data)

    result = predictor['model'].predict(test_data)

    return result, test_label


def DiseasePredictScore(predictor, scPheno_model, test_data_loader, test_sample_source, use_cuda = True):

    test_data, test_label = DiseaseFeature(scPheno_model,test_data_loader,test_sample_source,use_cuda)

    if predictor['scaler'] is not None:
        test_data = predictor['scaler'].transform(test_data)

    result = predictor['model'].predict_proba(test_data)

    pred_label = np.argmax(result, axis=1)
    pred_label = predictor['model'].classes_[pred_label]

    result = pd.DataFrame(result, columns=predictor['model'].classes_)
    
    return result, pred_label, test_label


