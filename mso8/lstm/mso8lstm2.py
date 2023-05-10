#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 00:21:36 2022

@author: Oliver Obst
"""

import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.metrics.metrics import mse, rmse, mape
import matplotlib.pyplot as plt


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


N = 21
results = np.zeros((N,8))

for n in range(0,1):
    filename = f"../data/signal{n:02d}.csv"
    print("Using " + filename)
    data = np.genfromtxt(filename, delimiter=',')
    data = data.reshape((len(data), 1))

    series = TimeSeries.from_values(data)
    series, _ = series.split_before(300)

    train_val, test = series.split_before(250)
    train, val = train_val.split_before(200)
    
    # Normalise based on the training data
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    train_val_transformed = transformer.transform(train_val)
    val_transformed = transformer.transform(val)
    test_transformed = transformer.transform(test)
    series_transformed = transformer.transform(series)

    model_rnn = RNNModel(
        model="LSTM",
        hidden_dim=4,
        n_epochs=500,
        model_name="MSO8_LSTM",
        random_state=None,
        input_chunk_length=4,
        output_chunk_length=1,       
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": [0],
            },
        )    
    
    hyper = {'model': ['LSTM'], 'force_reset': [True], 
             'optimizer_kwargs': [{"lr": 1e-3},{"lr": 5e-4},{"lr": 5e-3}], 
             'n_epochs': [500], 
             'hidden_dim': [4,8,16], 'input_chunk_length': [4,8,16,32]}
#             'pl_trainer_kwargs': [{'accelerator': 'gpu', 'devices': [0]}],

    model, parms, _ = model_rnn.gridsearch(hyper, series=train, 
                                           val_series=val_transformed,
                                           metric=mape, 
                                           n_random_samples=36, 
                                           n_jobs=18, verbose = 0)
    model.fit(train_val_transformed, verbose=False)
    yhat = model.predict(len(test),
                         series=val_transformed[-parms['input_chunk_length']:])
    yh = transformer.inverse_transform(yhat)

    p = plot_predicted_actual(test.values(copy=True), yh.values(copy=True),
                              name = 'LSTM', filename = f"signal{n+1:02d}.png")
    p.close()
    
    outputs = np.zeros((len(test),2))
    outputs[:,0] = test.values(copy=True)[:,0]
    outputs[:,1] = yh.values(copy=True)[:,0]
    fname = f"RESULTS-20230425/outputs-{n:02d}.csv"
    np.savetxt(fname, outputs, fmt='%.8f', delimiter=',', header='true,predicted')
    
    results[n,0] = n+1
    results[n,1] = train_transformed.mean(axis=0).values(0)[0][0]
    results[n,2] = val_transformed.mean(axis=0).values(0)[0][0]
    results[n,3] = test_transformed.mean(axis=0).values(0)[0][0]
    results[n,4] = parms['hidden_dim']
    results[n,5] = parms['input_chunk_length']
    results[n,6] = parms['optimizer_kwargs']['lr']
    results[n,7] = rmse(test_transformed, yhat, intersect = True)

np.savetxt('results-mso8-lstm-0.csv', results, delimiter=',', fmt='%i,%.8f,%.8f,%.8f,%i,%i,%.8f,%.8f')
