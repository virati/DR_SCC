#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:24:26 2018

@author: virati

Do a simple synthetic regression to make sure our approaches are working

"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

M_known = np.array([20,2,3.4,-1]).reshape(-1,1)
X_base = np.random.multivariate_normal([5,3,2,1],5 * np.identity(4),500)

Y_known = np.dot(X_base,M_known) #
Y_meas = Y_known + np.random.normal(0,10,(500,1))

#%%
from sklearn.linear_model import LinearRegression, RANSACRegressor


regmodel = LinearRegression(copy_X=True,normalize=True,fit_intercept=True)

regmodel.fit(X_base,Y_meas)

#%%
#TESTING PHASE
test_obs = 1500
xnoiz = 1
X_test = np.random.multivariate_normal([5,3,2,1],5 * np.ones((4,4)),test_obs)
X_noise = X_test # + np.random.normal(0,xnoiz,(test_obs,1))
ynoiz = 10

Y_true = (np.dot(X_noise,M_known) + np.random.normal(0,ynoiz,(test_obs,1)))

Y_test = regmodel.predict(X_noise)
#%%
plt.figure()
plt.subplot(211)
plt.scatter(Y_true,Y_test)
plt.subplot(212)

featvect = np.arange(4)
mcoefs = regmodel.coef_.T
#plt.bar(featvect,mcoefs,0.4)
plt.plot(featvect,mcoefs,label='Model Coefficients')
plt.plot(featvect,M_known,label='True Coefficients')
plt.legend()

#plt.figure()
#plt.scatter(np.ones_like(Y_known),Y_known,alpha=0.1)
#plt.title('')

plt.show()