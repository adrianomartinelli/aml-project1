#%% Modules
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GridSearchCV
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
x = SimpleImputer().fit_transform(x_train)

#%%Cosumte Transfomrer: Outlier Detection
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierDetection( BaseEstimator, TransformerMixin ):

    #Class Constructor 
    def __init__( self, method):
        self.method = method
        #self.params = params
    
    #Return self nothing else to do here    
    def fit( self, X, y = None):
        self.idx = self.method.fit_predict(X)
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None):
        #Outlier Detection
        x = X[self.idx == 1,:]
        return x

    def transform_y(self, y):
        return y[self.idx == 1]

#%%
od = OutlierDetection(method= LocalOutlierFactor())

pipe4 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('outlier_detection', od),
    ('feature_selection', RFE(Lasso())),
    ('estimator', RandomForestRegressor())]) 

# param_grid4 = [
#     {'impute_value__strategy': ['median', 'mean'],
#     'scale__quantile_range': [(20.0,80.0), (10.0,90.0), (1.0, 99.0)],
#     'feature_selection__estimator__alpha': [0.1,1,10],
#     'feature_selection__n_features_to_select': [50,100,150],
#     'estimator__n_estimators': [10,50, 100],
#     'estimator__bootstrap': [True],
#     'estimator__max_depth': [None, 30]}
# ]

param_grid4 = [
    {'impute_value__strategy': ['median'],
    'scale__quantile_range': [(20.0,80.0)],
    'feature_selection__estimator__alpha': [0.1],
    'feature_selection__n_features_to_select': [150],
    'estimator__n_estimators': [10],
    'estimator__bootstrap': [True],
    'estimator__max_depth': [30]}
]
#%%

clf = GridSearchCV(pipe4, param_grid4, cv=5, scoring = 'r2', n_jobs= 1)
res_cv = clf.fit(x_train, y_train)

#%%
res_cv.best_estimator_
#%% Predict
res = res_cv.best_estimator_.predict(x_test)

#%% Write results
id = id.values.reshape(-1,1)
res = res.reshape(-1,1)
res = np.concatenate((id,res), axis = 1)
np.savetxt('result.csv', res, fmt='%10.15f',delimiter=',', header='id,y', comments= '')



# #%%
# class RandomForestRegressor_own(RandomForestRegressor):
#     def __init__(self, n_estimators='warn',
#                  criterion="mse",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False):
#         super().__init__(n_estimators='warn',
#                  criterion="mse",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False)

#     #Return self nothing else to do here    
#     def fit(self, X, y):
#         return super().fit(X,y)
            
#     #Method that describes what we need this transformer to do
#     def predict( self, X, y = None):
#         #Outlier Detection
#         return super().predict(X)
# # %%
