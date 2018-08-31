# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 00:52:27 2018

@author: y510p
"""
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("veri.csv",sep=",")

plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% sklearn --> machine learning algoritmaları içerir

from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()

x=df.deneyim.values.reshape(-1,1)
y=df.deneyim.values.reshape(-1,1)

linear_reg.fit(x,y)
#%% prediction
import numpy as np

b0=linear_reg.predict(0)
print("b0:",b0)

b0_ =linear_reg.intercept_
print("b0:",b0_)  #y eksenini kestigi nokta intercept

b1=linear_reg.coef_
print("b1:",b1) #egim slope

#maas =1663+1138*deneyim

maas_yeni=1663+1138*11
print(maas_yeni)

print(linear_reg.predict(11))

array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)


y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="red")
linear_reg.predict(100)
