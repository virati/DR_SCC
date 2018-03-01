#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:53:58 2018

@author: virati
VERY VERY FUCKING SIMPLE REGRESSION
"""

from sklearn import linear_model
import numpy as np

reg = linear_model.LinearRegression()
X = np.random.multivariate_normal(loc=5,scale=2,size=(5000,1))
Y = np.array()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(reg.coef_)

