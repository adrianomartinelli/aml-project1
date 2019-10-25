# #%% Isolation tree
# from sklearn.ensemble import IsolationForest
# isofor = IsolationForest()
# isofor_pred = isofor.fit_predict(x_train_scale)
# idx = isofor_pred == 1

# x_filter = x_train_scale[idx,:]
# y_filter = y_train[idx,:]

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

#%% Pipeline

# clf = Pipeline([
#     ('impute_value', SimpleImputer(strategy='mean')),
#     ('scale', RobustScaler(quantile_range=(25.0,75.0))),
#     ('feature_selection', PCA(n_components = 250)),
#     ('estimator', SVR(kernel='linear', degree=1))])

pipe = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', SelectKBest()),
    ('estimator', SVR())])

pipe2 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', RFE(SVR())),
    ('estimator', SVR())])   

pipe3 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', RFE(Lasso())),
    ('estimator', SVR())])   

pipe4 = Pipeline([
    ('impute_value', SimpleImputer()),
    ('scale', RobustScaler()),
    ('feature_selection', RFE(Lasso())),
    ('estimator', RandomForestRegressor())])   

#%% Parameter Grid
from functools import partial
param_grid = [
    {'impute_value__strategy': ['median', 'mean']},
    {'scale__quantile_range': [(25.0,75.0), (10.0,90.0)]},
    {'feature_selection__score_func': [5, 30]},
    {'feature_selection__k': np.array([50, 100, 200])},
    {'estimator__C': np.array([0.1, 1, 10])},
    {'estimator__degree': np.array([3])},
    {'estimator__kernel': ['linear', 'rbf']}
]

param_grid1 = [
    {'impute_value__strategy': ['mean']},
    {'scale__quantile_range': [(20.0,80.0), (30.0,70.0)]},
    {'feature_selection__score_func': [5, 10]},
    {'feature_selection__k': np.array([150, 200])},
    {'estimator__C': np.array([10, 100])},
    {'estimator__degree': np.array([3,4])},
    {'estimator__kernel': ['rbf']}
]

param_grid2 = [
    {'impute_value__strategy': ['median', 'mean']},
    {'scale__quantile_range': [(20.0,80.0), (30.0, 70.0)]},
    {'feature_selection__estimator__C': np.array([0.1, 1, 10])},
    {'feature_selection__n_features_to_select': np.array([50, 100, 150])},
    {'feature_selection__estimator__gamma': np.array(['scale'])},
    {'feature_selection__estimator__C': np.array([0.1, 1, 10])},
    {'estimator__gamma': np.array(['scale'])},
    {'estimator__epsilon': np.array([0.1, 1])}
]

param_grid3 = [
    {'impute_value__strategy': ['median', 'mean']},
    {'scale__quantile_range': [(20.0,80.0), (30.0, 70.0)]},
    {'feature_selection__estimator__alpha': np.array([0.1, 1, 10])},
    {'feature_selection__n_features_to_select': np.array([50, 100, 150])},
    {'feature_selection__estimator__alpha': np.array([0.1, 1, 10])},
    {'estimator__gamma': np.array(['scale'])},
    {'estimator__epsilon': np.array([0.1, 1])}
]

param_grid4 = [
    {'impute_value__strategy': ['median', 'mean']},
    {'scale__quantile_range': [(20.0,80.0), (30.0, 70.0)]},
    {'feature_selection__estimator__alpha': np.array([0.1, 1, 10])},
    {'feature_selection__n_features_to_select': np.array([50, 100, 150])},
    {'feature_selection__estimator__alpha': np.array([0.1, 1, 10])},
    {'estimator__n_estimators': np.array([10, 50, 100])},
    {'estimator__bootstrap': [True, False]},
    {'estimator__gamma': np.array(['scale'])},
    {'estimator__epsilon': np.array([0.1, 1])}
]




#%% Grid search
cv_results = []
best_score = 0
best_param = []
for a in param_grid2[0]['impute_value__strategy']:
    for b in param_grid2[1]['scale__quantile_range']:
        for c in param_grid2[2]['feature_selection__score_func']:
            for d in param_grid2[3]['feature_selection__k']:
                for e in param_grid2[4]['estimator__C']:
                    for f in param_grid2[5]['estimator__degree']:
                        for g in param_grid2[6]['estimator__kernel']:
                            #Impute
                            x = SimpleImputer(strategy = a).fit_transform(x_train)
                            #Scale
                            x = RobustScaler(quantile_range = b).fit_transform(x)
                            #Feature selection
                            def fun(X,y):
                                return mutual_info_regression(X,y, n_neighbors = c)
                            
                            x = SelectKBest(score_func = fun, k = d).fit_transform(x,y_train)
                            #Estimator_C
                            res = cross_validate(SVR(kernel=g, degree=f, C=e, gamma='scale'), x, y_train, cv=5)
                            test_score = np.mean(res['test_score'])
                            train_score = np.mean(res['train_score'])
                            if(best_score < train_score):
                                best_score = train_score
                                best_param = (a,b,c,d,e,f,g)
                            cv_results.append((res, test_score, train_score))


#%% Fit best model
a = best_param[0]
b = best_param[1]
c = best_param[2]
d = best_param[3]
e = best_param[4]
f = best_param[5]
g = best_param[6]

#Impute
imputer = SimpleImputer(strategy = a)
x = imputer.fit_transform(x_train)
#Scale
scaler = RobustScaler(quantile_range = b)
x = scaler.fit_transform(x)
#Feature selection
selector = SelectKBest(score_func = partial(mutual_info_regression,n_neighbors = c), k = d)
x = selector.fit_transform(x,y_train)
#Estimator
est = SVR(kernel=g, degree=f, C=e, gamma='scale')
x = est.fit(x, y_train)

#%% Predict
#Impute
xx = imputer.transform(x_test)
#Scale
xx = scaler.transform(xx)
#Feature selection
xx = selector.transform(xx)
#Estimator
res = est.predict(xx)

#%% Write results
iid = id.values.reshape(-1,1)
res = res.reshape(-1,1)
res = np.concatenate((iid,res), axis = 1)
np.savetxt('result.csv', res, fmt='%10.15f',delimiter=',', header='id,y', comments= '')

#%%
cv_results = []
best_score = 0
best_param = []

pipe4 = Pipeline([
    ('feature_selection', RFE(Lasso())),
    ('estimator', RandomForestRegressor())])   

param_grid5 = [
    {'impute_value__strategy': ['mean']},
    {'scale__quantile_range': [(20.0,80.0), (30.0, 70.0)]},
    {'outlier_detection': [[20,30], [10, 15], [80, 30]]},
    {'feature_selection__estimator__alpha': np.array([0.1, 1, 10])},
    {'feature_selection__n_features_to_select': np.array([50, 100, 150])},
    {'feature_selection__estimator__alpha': np.array([0.1, 1, 10])},
    {'estimator__n_estimators': np.array([10, 50, 100])},
    {'estimator__max_depth': [None, 30]},
    {'estimator__bootstrap': [True]}
]

for a in param_grid2[0]['impute_value__strategy']:
    for b in param_grid2[1]['scale__quantile_range']:
        for c in param_grid2[3]['outlier_detection']:
            #Impute
            x = SimpleImputer(strategy = a).fit_transform(x_train)
            #Scale
            x = RobustScaler(quantile_range = b).fit_transform(x)
            #Outlier Detection
            idx = LocalOutlierFactor(n_neighbors= c[0], leaf_size=c[1], contamination = 'auto', p=2, n_jobs= -1).fit_predict(x)
            x = x[idx == 1]
            y = y_train[idx == 1]

            clf = GridSearchCV(pipe4, param_grid4, cv=5, scoring = 'r2', n_jobs= -1)
            res_cv = clf.fit(x, y)


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
    'estimator__n_estimators': [10,50, 100],
    'estimator__bootstrap': [True],
    'estimator__max_depth': [None, 30]}
]
#%%

clf = GridSearchCV(pipe4, param_grid4, cv=5, scoring = 'r2', n_jobs= -1)
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

#%% Estimation
#res_cv = cross_validate(clf, x_train, y_train, cv=5, scoring='r2')
#print(res_cv)
