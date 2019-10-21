#%%
import numpy as np
import pandas as pd

#%%
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#%% Import data
x_train = pd.read_csv('./task1/X_train.csv')
x_train = x_train.iloc[:, 1:]

y_train = pd.read_csv('./task1/Y_train.csv')


#%% Find missing values
# Every row has has missing value with at least 37 up to 88.
idx = x_train.isnull().any(1)
nb_missing_v = x_train.isnull().sum(1)
print(min(nb_missing_v))
print(max(nb_missing_v))

#%% Impute missing values
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
x_train_imp = imp_mean.fit_transform(x_train)

#%% Comput distances
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
rScaler = RobustScaler(quantile_range=(5.0,95.0))

x_train_scale = rScaler.fit_transform(x_train_imp)
dist = pairwise_distances(x_train_scale, metric = 'euclidean')

#%%
kNN = []
for i in range(dist.shape[1]):
    kNN.append(np.sort(dist[:,i])[0:10].max())

#%% Plot
import matplotlib.pyplot as plt
plt.plot(np.sort(kNN))

#%% DBSCAN
clustering = DBSCAN(eps=12.5, min_samples=800).fit(x_train_scale)


#%% Isolation tree
from sklearn.ensemble import IsolationForest
isofor = IsolationForest()
isofor_pred = isofor.fit_predict(x_train_scale)
idx = isofor_pred == 1

x_filter = x_train_scale[idx,:]
y_filter = y_train[idx,:]

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
x_pca = pca.fit_transform(x_train_scale)

#%%
import seaborn as sns; sns.set()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#%%
pca = PCA(n_components = np.min(np.where(np.cumsum(pca.explained_variance_ratio_)>0.8)))
x_pca = pca.fit_transform(x_filter)


#%% from the plot it seems like x_pca[0,0] is an outlier
plt.scatter(x_pca[0,:], x_pca[1,:])

#%% Regression fit
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate

linearsvr = SVR(kernel='linear', degree=3)
res = cross_validate(linearsvr, x_pca, y_train.iloc[:,1], cv=5, scoring='r2')

#%%
rbfsvr = SVR(kernel='rbf', degree=3)
res = cross_validate(rbfsvr, x_pca, y_filter, cv=5, scoring='r2')

#%%
polysvr = SVR(kernel='poly', degree=3)
res = cross_validate(polysvr, x_pca, y_filter, cv=5, scoring='r2')

#%% Predict
estimator = linearsvr.fit(x_pca, y_filter)

x_predict = pd.read_csv('./task1/X_test.csv')
id = x_predict.iloc[:,0]
x_predict = x_predict.iloc[:,1:]
x_predict = imp_mean.transform(x_predict)
x_predict = rScaler.transform(x_predict)
x_predict = pca.transform(x_predict)
res = linearsvr.predict(x_predict)


#%%
id = id.values.reshape(-1,1)
res = res.reshape(-1,1)
res = np.concatenate((id,res), axis = 1)
np.savetxt('result.csv', res, fmt='%10.15f',delimiter=',', header='id,y', comments= '')

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
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, f_regression, SelectKBest

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

param_grid2 = [
    {'impute_value__strategy': ['mean']},
    {'scale__quantile_range': [(20.0,80.0), (30.0,70.0)]},
    {'feature_selection__score_func': [5, 10]},
    {'feature_selection__k': np.array([150, 200])},
    {'estimator__C': np.array([10, 100])},
    {'estimator__degree': np.array([3,4])},
    {'estimator__kernel': ['rbf']}
]

param_grid1 = [
    {'impute_value__strategy': ['median', 'mean']},
    {'scale__quantile_range': [(20.0,80.0), (10.0,90.0)]},
    {'feature_selection__score_func': [
        partial(mutual_info_regression, n_neighbors = 5),
        partial(mutual_info_regression, n_neighbors = 10),
        partial(mutual_info_regression, n_neighbors = 50)]},
    {'feature_selection__k': np.array([50, 100, 150, 200])},
    {'estimator__C': np.array([0.1, 1,10])},
    {'estimator__degree': np.array([3])},
    {'estimator__kernel': ['linear', 'rbf']}
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
clf = GridSearchCV(pipe, param_grid1, cv=5, scoring = 'r2')
res_cv = clf.fit(x_train, y_train)

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
