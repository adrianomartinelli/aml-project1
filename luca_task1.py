#%% Modules
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, f_regression, SelectKBest



#%% Data

## Training
x_train = pd.read_csv('C:/Users/lucaw/Google Drive/Studium/CBB Semester 3/AML/task1/task1/X_train.csv')
x_train = x_train.iloc[:, 1:]

y_train = pd.read_csv('C:/Users/lucaw/Google Drive/Studium/CBB Semester 3/AML/task1/task1/Y_train.csv')
y_train = y_train.iloc[:,1]

## Test
x_test = pd.read_csv('C:/Users/lucaw/Google Drive/Studium/CBB Semester 3/AML/task1/task1/X_test.csv')
id = x_test.iloc[:,0]
x_test = x_test.iloc[:,1:]


#median data impuation is suitable for continuous data with outliers
print("data  imputation")
imp = SimpleImputer(missing_values=np.nan, strategy='median')
x_imp=imp.fit_transform(x_train)
x_test_imp=imp.fit_transform(x_test) #Todo check


#Scale Data
scaler=RobustScaler()
x_scale=scaler.fit_transform(x_imp)
x_train=pd.DataFrame(x_scale, columns=x_train.columns)
x_test_scaled=scaler.transform(x_test_imp)
x_test=pd.DataFrame(x_test_scaled, columns=x_test.columns)

#outlier detection
print("outlier detection")
clf = IsolationForest(behaviour='new')
clf.fit(x_train, y_train)
pred = clf.decision_function(x_train) #outliers have negative score
print ("outliers: ",sum(pred <=0))

#remove outliers
x_train=x_train[pred > 0]
y_train=y_train[pred > 0]

"""
print("running lasso regression feature selection")

#feature selection with Lasso CV
lasso_cv = LassoCV().fit(x_train,y_train)
coefs=pd.Series(lasso_cv.coef_, index=x_train.columns)

print("kept {} features".format(sum(coefs.values > 0)))

print("before: ", x_train.shape, x_test.shape)

x_train.drop(columns=coefs.index[coefs.values < 10e-5])
x_test.drop(columns=coefs.index[coefs.values < 10e-5])
print("before: ", x_train.shape, x_test.shape)

"""
reg = RandomForestRegressor()
reg.fit(x_train ,y_train)

y_pred=reg.predict(x_test)

f=open("output.csv", mode="w")
f.write("id,y\n")
for i in range(len(id)):
    f.write("{},{}\n".format(i, y_pred[i]))
f.close()