#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:11:05 2023

@author: oliver
"""
from stockdataloader import StockDataLoader

import numpy as np
import json
from darts.models import RNNModel
from darts.metrics import mse, rmse
import matplotlib.pyplot as plt


def load_data(filename, trainlen = 200, vallen = 50, testlen = 50):
    d = StockDataLoader()
    d.load_data(filename, trainlen, vallen, testlen, columns = [3])
    d.init_transform(minval = 0.0, maxval = 1.0)            
    return d
    
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
    # else:
    #     if isinstance(x0[0], maya.MayaDT):
    #         x0 = [pd.to_datetime(str(s.date)) for s in x0]
    #         plt.plot(x0, data)
    #         plt.gcf().autofmt_xdate()
    #     else:
    #         plt.plot(x0, data)
        
    #     if x1 is None:
    #         x1 = x0.copy()
    #     if isinstance(x1[0], maya.MayaDT):
    #         x1 = [pd.to_datetime(str(s.date)) for s in x1]
    #     plt.plot(x1, prediction)
            
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.close()


input_chunk_lengths = [10, 20, 30, 40, 50]
hidden_sizes = [16, 32, 64, 128]

datadir = '../data'
resultsdir = './LSTMRESULTS'

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

results = np.zeros((N, len(input_chunk_lengths), len(hidden_sizes)))

with open(datadir + '/files.txt', 'r') as f:
    for i, filename in enumerate(f):
        filename = filename.strip()
        name = filename.split('.', 1)[0]
        plotfile = resultsdir + '/' + name + '.png'
        predfile = resultsdir + '/' + name + 'prediction.csv'
        paramfile = resultsdir + '/' + name + 'params.txt'
        print('File: ' + filename)
        stockfile = datadir + '/' + filename


        d = load_data(stockfile)
        train_series_scaled = d.transform(d.training())
        val_series_scaled = d.transform(d.validation())
        test_series_scaled = d.transform(d.testing())

        best_model = None
        best_input_chunk_length = None
        best_hidden_size = None
        best_performance = float('inf')

        for input_chunk_length in input_chunk_lengths:
            for hidden_size in hidden_sizes:

                # Create an LSTM model
                model = RNNModel(
                    model='LSTM',
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=1,
                    hidden_dim=hidden_size,
                    n_epochs=500,
#                    random_state=42
                )

                # Train the model on the training set
                model.fit(train_series_scaled, verbose=True)

                # Forecast for validation set
                forecast_validation = model.predict(n=50, 
                                                    series=train_series_scaled[-input_chunk_length:])

                # compute performance
                performance = mse(val_series_scaled, forecast_validation)
    
                # update best model
                if performance < best_performance:
                    print(f"updated new best: {input_chunk_length} input chunk size "
                          f" and {hidden_size} hidden units.")
                    best_performance = performance
                    best_model = model
                    best_input_chunk_length = input_chunk_length
                    best_hidden_size = hidden_size
        
        model = best_model    
        input_chunk_length = best_input_chunk_length
        j = input_chunk_lengths.index(input_chunk_length)
    
        hidden_size = best_hidden_size
        k = hidden_sizes.index(hidden_size)
        
        full_training_scaled = train_series_scaled.append(val_series_scaled)
        model.fit(full_training_scaled, verbose=True)

        forecast_testing_scaled = model.predict(n=50,
                                                series=val_series_scaled[-input_chunk_length:])
        forecast_testing = d.inverse_transform(forecast_testing_scaled)

        # Evaluate the forecast 
        error = mse(d.testing(), forecast_testing)
        print(f"MSE: {error:.2f} with input chunk "
              f"length of {best_input_chunk_length} "
              f"and {best_hidden_size} hidden units")

        results[i, j, k] = rmse(test_series_scaled, forecast_testing_scaled)
        
        predictions = forecast_testing.stack(forecast_testing_scaled)
        predictions = predictions.stack(test_series_scaled)
        np.savetxt(predfile, predictions.values())
        plot_predicted_actual(d.testing().values(), forecast_testing.values(), name=name, filename=plotfile)
        output = {'RMSE' : rmse(test_series_scaled, forecast_testing_scaled), 
                  'input_chunk_length' : input_chunk_length, 
                  'units' : hidden_size }
        
        with open(paramfile, 'w') as f:
            f.write(json.dumps(output))
 