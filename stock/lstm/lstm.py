#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:39:54 2022

@author: 30045063
"""

import sys
sys.path.insert(1, '../../python')
from timeseriesloader import TimeSeriesLoader

import numpy as np
import csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import maya
import pandas as pd
import matplotlib.pyplot as plt
import json

def create_samples(sequence, steps, index = 0, out_index = None):
    if out_index is None and isinstance(index, int):
        out_index = index
    if sequence.ndim > 1 and not isinstance(out_index, int):
        error = f"you must specify out_index as integer (0..{sequence.shape[1]-1})"
        raise Exception(error)
        
    X = []
    y = []

    for j0 in range(len(sequence) - steps):
        j1 = j0 + steps 
        if sequence.ndim == 1:
            sx, sy = sequence[j0:j1], sequence[j1]
        else:
            sx, sy = sequence[j0:j1, index], sequence[j1, out_index]
        X.append(sx)
        y.append(sy)
    return np.array(X), np.array(y)

def plot_predicted_actual(data, prediction, x0 = None, x1 = None, name = None, filename = None):
    title = 'Stock value'
    if isinstance(name, str):
        title += ': ' + name
    plt.title(title)
    plt.ylabel('Total value')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    if x0 is None:    
        plt.plot(data)
        plt.plot(prediction)
    else:
        if isinstance(x0[0], maya.MayaDT):
            x0 = [pd.to_datetime(str(s.date)) for s in x0]
            plt.plot(x0, data)
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(x0, data)
        
        if x1 is None:
            x1 = x0.copy()
        if isinstance(x1[0], maya.MayaDT):
            x1 = [pd.to_datetime(str(s.date)) for s in x1]
        plt.plot(x1, prediction)
            
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    return plt

def prediction(filename, steps = 5, hidden = 150, epochs = 200, validation = False):
    d = TimeSeriesLoader()
    d.load_data(filename)

    # for univariate input, use a single index. For multivariate, use an index list
    X,y = create_samples(d.training(), steps, index=4)
    tme = d.training(time=True)[:-steps]
    # univariate input
    features = 1
    X = X.reshape((X.shape[0], X.shape[1], features))

    # create an LSTM model
    model = Sequential()
    model.add(LSTM(hidden, activation='relu', input_shape=(steps, features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # training
    model.fit(X, y, epochs=epochs, verbose=0)

    # predict on training data
    yhat = model.predict(X, verbose=0)
    rmse = d.rmse(y, yhat, mode=-1)
    
    # predict on validation / test data
    if validation:
        X1, y1 = create_samples(d.validation(), steps, index=4)
        tme1 = d.validation(time=True)[:-steps]
    else:
        X1, y1 = create_samples(d.testing(), steps, index=4)
        tme1 = d.testing(time=True)[:-steps]
        
    X1 = X1.reshape((X1.shape[0], X1.shape[1], features))
    y1hat = model.predict(X1, verbose=0)
    
    rmse1 = d.rmse(y1, y1hat, mode=-1)

    return y1, y1hat, tme1, rmse1, y, yhat, tme, rmse


datadir = '../data'
resultsdir = './RESULTS'

acronyms = {}
with open(datadir + '/40DAX.txt', 'r') as f:
    for line in f:
        name = line.split('=', 1)
        if len(name) < 2:
            continue
        name[0] = name[0].strip()
        name[1] = name[1].split('[', 1)[0].strip()
        acronyms[name[0]] = name[1]

with open(datadir + '/files.txt', 'r') as f:
    N = len(f.readlines())
        
hyper_steps = [3,5,7]
hyper_units = [50,100,150]
results = np.zeros((N, len(hyper_steps), len(hyper_units)))
hyperparms = []

with open(datadir + '/files.txt', 'r') as f:
    for i, filename in enumerate(f):
        filename = filename.strip()
        name = filename.split('.', 1)[0]
        plotfile = resultsdir + '/' + name + '.png'
        predfile = resultsdir + '/' + name + 'prediction.csv'
        paramfile = resultsdir + '/' + name + 'params.txt'
        print('File: ' + filename)
        file = datadir + '/' + filename
        
        for k0,steps in enumerate(hyper_steps):
            for k1,hidden in enumerate(hyper_units):
                _,_,_,rmse1, _,_,_,_ = prediction(file)
                print(f"RMSE for {steps} steps and {hidden} units: {rmse1}")
                results[i,k0,k1] = rmse1
        
                k0, k1 = np.unravel_index(np.argmin(results[i,:,:]), results[i,:,:].shape)
        hyperparms.append((hyper_steps[k0], hyper_units[k1])) 
        
        y1, y1hat, d1, rmse1, _, _, _, _ = prediction(datadir + '/' + filename,
                                                      steps = hyper_steps[k0],
                                                      hidden = hyper_units[k1])

        y1 = y1.reshape(-1,1)
        y = np.concatenate((y1, y1hat), axis=1)
        
        output = { 'RMSE' : rmse1[0], 
                  'steps' : hyper_steps[k0], 
                  'units' : hyper_units[k1],
                  'RMSE.val' : results[i,k0,k1]}
        
        with open(paramfile, 'w') as f:
            f.write(json.dumps(output))
        np.savetxt(predfile, y)
        p = plot_predicted_actual(y1, y1hat, d1, name=acronyms[name], filename=plotfile)
        p.close()
        print(f"Test NRMSE ({hyper_steps[k0]}, {hyper_units[k1]}): {rmse1[0]}")
