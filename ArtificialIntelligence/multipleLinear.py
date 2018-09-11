# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:26:35 2018

@author: y510p
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df=pd.read_csv("veri2.csv",sep=",")

x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

#%%
multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0:",multiple_linear_regression.intercept_)
print("b1,b2:",multiple_linear_regression.coef_) 
