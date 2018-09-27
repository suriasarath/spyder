# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 00:44:16 2018

@author: AMD_suria
"""

import numpy as np
from sklearn import datasets
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

diabetes = datasets.load_diabetes()

d_data = diabetes.data[:,np.newaxis,2]
d_target = diabetes.target[:]


tr_data = d_data[:-20] 
ts_data = d_data[-20:]

tr_target = d_target[:-20]
ts_target = d_target[-20:]


regr = linear_model.LinearRegression()

regr.fit(tr_data,tr_target)

result = regr.predict(ts_data)

plt.scatter(ts_data,ts_target,color='black',marker='o')

plt.scatter(ts_data,result,color='blue')

plt.show()



