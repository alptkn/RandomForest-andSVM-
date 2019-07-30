# -*- coding: utf-8 -*-
"""
Created on Mon May 27 03:06:40 2019

@author: 90553
"""


# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:03:30 2019

@author: 90553
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

data = np.asarray(pd.read_csv("train.csv",skiprows = 0))
data_test = np.asarray(pd.read_csv("test.csv",skiprows = 0))
X = data[:,:595]
Y = data[:,595]

test = SelectKBest(score_func = f_classif,k = 400)
X_train_clean = test.fit_transform(X,Y)
X_test_clean  = test.transform(data_test)

regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
regressor.fit(X_train_clean,Y)
y_predict = regressor.predict(X_test_clean)    
y_predict.shape = (np.size(y_predict),1)
temp = np.ones((80,1),dtype = float)
for i in range(0,80):
    temp[i] = i + 1;
Y_csv = np.concatenate((temp,y_predict),1)

np.savetxt('goal1.csv',Y_csv,delimiter = ",",fmt = '%.0f',header = "ID,Predicted" )


