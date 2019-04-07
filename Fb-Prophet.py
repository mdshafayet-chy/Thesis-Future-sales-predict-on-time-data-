# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 02:04:54 2018

@author: JEWEL
"""

# load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

df1 = pd.read_csv('sales_train.csv')


# adding the dates to the Time-series as index
df=df1.groupby(["date_block_num"])["item_cnt_day"].sum()
df.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
df=df.reset_index()
df.head()
 
# create new coumns, specific headers needed for Prophet
df['ds'] = df['index']
df['y'] = pd.DataFrame(df['item_cnt_day'])
df.pop('index')
df.pop('item_cnt_day')


# log transform data
df['y'] = pd.DataFrame(np.log(df['y']))


 
# plot data
ax = df.set_index('ds').plot(color='#006699');
ax.set_ylabel('item_cnt_day');
ax.set_xlabel('Date');
#plt.savefig('./img/log_transformed_passenger.png')
plt.show()


# train test split
df_train = df[:22]
df_test = df[22:]



# instantiate the Prophet class

model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(df_train)

 
# define future time frame
future = model.make_future_dataframe(periods=12, freq='MS')





# generate the forecast
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



model.plot(forecast);
plt.show()

# plot time series components
model.plot_components(forecast[22:])
plt.show()


import math
from math import *

y_hat = np.exp(forecast['yhat'][22:])
y_true = np.exp(df_test['y'])
print(y_hat)
print(y_true)

 
# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))





