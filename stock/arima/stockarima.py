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
from datetime import datetime
import time
import os


N = 20
results = np.zeros((N,5))

def plot_predicted_actual(data, prediction, name = None, filename = None):
    title = 'Stocks (DAX)'
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


def get_acronyms(datadir):
    acronyms = {}
    with open(datadir + '/40DAX.txt', 'r') as f:
        for line in f:
            name = line.split('=', 1)
            if len(name) < 2:
                continue
            name[0] = name[0].strip()
            name[1] = name[1].split('[', 1)[0].strip()
            acronyms[name[0]] = name[1]

    return acronyms


# Data sizes
n_train = 200
n_val = 50
n_test = 50
N = n_train + n_val + n_test

# Data location
datadir = '../data'
# Create new results directory
dir_created = False
while not dir_created:
    # Format the date and time as a string
    now = datetime.now()
    resultsdir = now.strftime("./RESULTS-%Y%m%d%H%M%S")
    if os.path.exists(resultsdir):
        time.sleep(1)
    else:
        os.makedirs(resultsdir)
        dir_created = True

acronyms = get_acronyms(datadir)
results = np.zeros((39,5))

with open(datadir + '/files.txt', 'r') as f:
    for i, filename in enumerate(f):

        filename = filename.strip()
        name = filename.split('.', 1)[0]
        plotfile = resultsdir + '/' + name + '.png'
        predfile = resultsdir + '/' + name + 'prediction.csv'
        paramfile = resultsdir + '/' + name + 'params.txt'
        file = datadir + '/' + filename

        print('Using file: ' + filename)
        data = np.genfromtxt(datadir + '/' + filename, delimiter=',', skip_header=1, usecols=4)
        data = data.reshape((len(data), 1))
        data = data[-N:,:]

        series = TimeSeries.from_values(data)
        train_val, test = series.split_before(n_train + n_val)
        train, val = train_val.split_before(n_train)

        model_arima = AutoARIMA()
        model_arima.fit(train_val)
 
        yhat = model_arima.predict(len(test))

        p = plot_predicted_actual(test.values(copy=True), 
                                  yhat.values(copy=True),
                                  name = None, 
                                  filename = f"stock_{name}.png")
        p.close()
    
        results[i,0] = i+1
        results[i,1] = train_val.mean(axis=0).values()[0][0]
        results[i,2] = test.mean(axis=0).values()[0][0]
        results[i,3] = mse(test, yhat, intersect = True)
        results[i,4] = rmse(test, yhat, intersect = True)

np.savetxt(resultsdir + 'results-mso8-arima.csv', results, 
           delimiter=',', fmt='%i,%.8f,%.8f,%.8f,%.8f')
