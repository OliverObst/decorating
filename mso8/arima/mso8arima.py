#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 00:21:36 2022

@author: Oliver Obst
"""

import numpy as np
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.metrics.metrics import mse, rmse
import matplotlib.pyplot as plt

N = 1
results = np.zeros((N,5))

def plot_predicted_actual(data, prediction, name = None, filename = None):
    title = 'MSO8'
    if isinstance(name, str):
        title += ': ' + name
    plt.title(title)
    plt.ylabel('y')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    plt.plot(data)
    plt.plot(prediction)
            
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    return plt


for n in range(N):
    filename = f"../data/signal{n+1:02d}.csv"
    print("Using " + filename)
    data = np.genfromtxt(filename, delimiter=',')
    data = data.reshape((len(data), 1))

    series = TimeSeries.from_values(data)
    series, _ = series.split_before(300)

    train_val, test = series.split_before(250)
    train, val = train_val.split_before(200)

    model_arima = AutoARIMA()
    model_arima.fit(train_val)
    yhat = model_arima.predict(len(test))

    p = plot_predicted_actual(test.values(copy=True), yhat.values(copy=True),
                              name = None, filename = f"signal{n+1:02d}.png")
    p.close()
    
    results[n,0] = n+1
    results[n,1] = train_val.mean(axis=0).values()[0][0]
    results[n,2] = test.mean(axis=0).values()[0][0]
    results[n,3] = mse(test, yhat, intersect = True)
    results[n,4] = rmse(test, yhat, intersect = True)

#np.savetxt('results-mso8-arima.csv', results, delimiter=',', fmt='%i,%.8f,%.8f,%.8f,%.8f')
