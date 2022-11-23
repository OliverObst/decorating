#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 00:21:36 2022

@author: Oliver Obst
"""

import numpy as np
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.metrics.metrics import mse
import matplotlib.pyplot as plt

N = 20
results = np.zeros((N,2))

for n in range(N):
    filename = f"../data/signal{n+1:02d}.csv"
    print("Using " + filename)
    data = np.genfromtxt(filename, delimiter=',')
    data = data.reshape((len(data), 1))

    series = TimeSeries.from_values(data)
    series, _ = series.split_before(300)

    train, test = series.split_before(250)

    model_arima = AutoARIMA()
    model_arima.fit(train)
    prediction_arima = model_arima.predict(len(test))

    #series.plot(label='actual')
    #prediction_arima.plot(label='forecast', lw=3)
    #plt.legend()
    results[n,0] = n+1
    results[n,1] = np.sqrt(mse(series, prediction_arima, intersect = True))

np.savetxt('results.csv', results, delimiter=',')

