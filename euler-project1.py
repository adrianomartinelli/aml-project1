#%%
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GridSearchCV, ParameterGrid
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, SelectKBest, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor


#%% Data

## Training
x_train = pd.read_csv('./task1/X_train.csv')
x_train = x_train.iloc[:, 1:]

y_train = pd.read_csv('./task1/Y_train.csv')
y_train = y_train.iloc[:,1]

## Test
x_test = pd.read_csv('./task1/X_test.csv')
id = x_test.iloc[:,0]
x_test = x_test.iloc[:,1:]

#%%
pipe = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', RFE(Lasso())),
    ('estimator', SVR())])

param_grid = [{
    'impute_value__strategy': ['median', 'mean'],
    'scale__quantile_range': [(25.0,75.0), (1.0,99.0)],
    'feature_selection__estimator__alpha': [0.1,1,10],
    'feature_selection__n_features_to_select': [100,150],
    'estimator__C': np.array([0.1, 1, 10]),
    'estimator__degree': np.array([3]),
    'estimator__kernel': ['rbf']
}]
#%%
pipe2 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', RFE(Lasso())),
    ('estimator', RandomForestRegressor())]) 

param_grid2 = [
    {'impute_value__strategy': ['median', 'mean'],
    'scale__quantile_range': [(20.0,80.0), (10.0,90.0), (1.0, 99.0)],
    'feature_selection__estimator__alpha': [0.1,1,10],
    'feature_selection__n_features_to_select': [50,100,150],
    'estimator__n_estimators': [10, 50, 100],
    'estimator__bootstrap': [True],
    'estimator__max_depth': [None, 30, 50]}
]

#%%
from functools import partial
pipe3 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', SelectKBest()),
    ('estimator', RandomForestRegressor())])

param_grid3 = [{
    'impute_value__strategy': ['median', 'mean'],
    'scale__quantile_range': [(20.0,80.0), (10.0,90.0)],
    'feature_selection__score_func': [
        partial(mutual_info_regression, n_neighbors = 5),
        partial(mutual_info_regression, n_neighbors = 10),
        partial(mutual_info_regression, n_neighbors = 50)],
    'feature_selection__k': np.array([50, 100, 150, 200]),
    'estimator__n_estimators': [10, 50, 100],
    'estimator__bootstrap': [True],
    'estimator__max_depth': [None, 30, 50]}
]

#%%
pipe4 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', RFE(Lasso())),
    ('estimator', RandomForestRegressor())]) 

param_grid4 = [
    {'impute_value__strategy': ['median', 'mean'],
    'scale__quantile_range': [(20.0,80.0), (10.0,90.0), (1.0, 99.0)],
    'feature_selection__estimator__alpha': [0.1,1,10],
    'feature_selection__n_features_to_select': [50,100,150],
    'estimator__n_estimators': [10, 50, 100],
    'estimator__bootstrap': [True],
    'estimator__max_depth': [None, 30, 50]}
]
#%%
clf = GridSearchCV(pipe, param_grid, cv=5, scoring = 'r2', n_jobs= -1)
res_cv = clf.fit(x_train, y_train)

#%%
res = res_cv.best_estimator_.predict(x_test)

#%% Write results
id = id.values.reshape(-1,1)
res = res.reshape(-1,1)
res = np.concatenate((id,res), axis = 1)
np.savetxt('result.csv', res, fmt='%10.15f',delimiter=',', header='id,y', comments= '')

pd.DataFrame(res_cv.cv_results_).to_csv('cv_results.csv')
pd.DataFrame(res_cv.best_params_).to_csv('best_params.csv')
np.savetxt('best_score.csv', np.array([res_cv.best_score_]))



