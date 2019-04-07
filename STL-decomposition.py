# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:29:16 2018

@author: Md Shafayet Chy
"""

# load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


sales=pd.read_csv("sales_train.csv")

ts=pd.read_csv("flights.csv")


df=ts['count']


df=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
df.astype('float')
plt.figure(figsize=(12,6))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(df);

import statsmodels.api as sm
# multiplicative
res = sm.tsa.seasonal_decompose(df.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
trnd = pd.DataFrame(res.trend)
season = pd.DataFrame(res.seasonal)
residu = pd.DataFrame(res.resid)

# Additive model
res = sm.tsa.seasonal_decompose(df.values,freq=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()




trend = pd.DataFrame(res.trend)
seasonal = pd.DataFrame(res.seasonal)
residual = pd.DataFrame(res.resid)
#trnd.to_csv("trend.csv");
#season.to_csv("seasonal.csv");
#residu.to_csv("residuals.csv");


# adding the dates to the Time-series as index
df=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
df.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
df=df.reset_index()
df.head()

 
# create Series object
y = df['item_cnt_day']

y=trnd

y=ts['count']

# split into training and test sets
y_train = y[:22]
y_test = y[22:]

import itertools

# define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
 
# generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
 
# generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


import sys

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
tmp_model = None
best_mdl = None
 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            tmp_mdl = sm.tsa.statespace.SARIMAX(y_train,
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
            res = tmp_mdl.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_mdl = tmp_mdl
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


# define SARIMAX model and fit it to the data
mdl = sm.tsa.statespace.SARIMAX(y_train,
                                order=(1, 1, 0),
                                seasonal_order=(0, 0, 0, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
res = mdl.fit()

# print statistics
print(res.aic)
print(res.summary())


 
#print(res)
# in-sample-prediction and confidence bounds
pred = res.get_prediction(start=22, 
                          end=33,
                          dynamic=True)
pred_ci = pred.conf_int()
 
# plot in-sample-prediction
ax = y[0:].plot(label='Observed',color='#006699');
pred.predicted_mean.plot(ax=ax, label='One-step Ahead Prediction', alpha=.7, color='#ff0066');
 
# draw confidence bound (gray)
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);
 

y_hat = pred.predicted_mean
print(y_hat)
plt.plot(y_hat)
y_true = y[22:]
#print(y_true)
plt.plot(y_true)

import math
from math import *

# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))













