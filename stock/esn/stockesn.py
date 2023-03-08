#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:50:28 2023

@author: oliver obst
"""


""" Echo State Network demo with ReservoirPy onstock data.

"""

import matplotlib.pyplot as plt
import numpy as np

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import mse, nrmse, rmse
from reservoirpy.observables import spectral_radius
from itertools import product
import time

def plot_predicted_actual(data, prediction, name = None, filename = None):
    title = 'Stock value'
    if isinstance(name, str):
        title += ': ' + name
    plt.title(title)
    plt.ylabel('Total value')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    plt.plot(data)
    plt.plot(prediction)
            
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    return plt


datadir = '../data'
resultsdir = './RESULTS'

trainlen = 200
evallen = 50
testlen = 50
tvlen = trainlen + evallen

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

train_size = 200
val_size = 50
test_size = 50
horizon = 1  # horizon p of the forecast (predict X[t+p] from X[t])
runs = 30

allresults = np.zeros((N,runs,8))
test_data = np.zeros((test_size,N))
predictions = np.zeros((runs,test_size,N))
results = np.zeros((runs,N,8), dtype=object)

VERBOSE = False
normalize = True
rpy.verbosity(0)

SEED = int(time.time()*10000000) & 0xFFFFFFFF
rpy.set_seed(SEED)
rng = np.random.default_rng(SEED)

with open(datadir + '/files.txt', 'r') as f:
    for n, filename in enumerate(f):
        filename = filename.strip()
        name = filename.split('.', 1)[0]
        plotfile = resultsdir + '/' + name + '.png'
        predfile = resultsdir + '/' + name + 'prediction.csv'
        paramfile = resultsdir + '/' + name + 'params.txt'

        print('Using ' + filename)
        file = datadir + '/' + filename

        data = np.genfromtxt(file, delimiter=',', usecols=5, skip_header=1)
        data = data[-(tvlen+testlen+horizon):]
        data = data.reshape((len(data), 1))

        start = 0
        val_start = start + train_size
        test_start = val_start + val_size
    
        # Normalize data
        if normalize:
            # don't leak info into test set
            trainmin = data[start:train_size + start].min()
            trainmax = data[start:train_size + start].max()
            data = (data - trainmin) / (trainmax - trainmin)
        
            if VERBOSE:
                print("Normalization...")
                print("train max:", trainmax, "min:", trainmin)
                print("new (all) data mean:", data.mean(), "std:", data.std())    
    
        X = data[start:train_size + start]
        y = data[start + horizon : start + train_size + horizon]
    
        X_val = data[val_start : val_start + val_size]
        y_val = data[val_start + horizon : val_start + val_size + horizon]

        X_test = data[test_start : test_start + test_size]
        y_test = data[test_start + horizon : test_start + test_size + horizon]
        test_data[:,n,np.newaxis] = y_test 

        if VERBOSE:
            print("X, y dimensions", X.shape, y.shape)
            print("X_val, y_val dimensions", X_val.shape, y_val.shape)
            print("X_test, y_test dimensions", X_test.shape, y_test.shape)
       
        if VERBOSE:
            print("Data dimensions", data.shape)
            print("max:", data.max(), "min:", data.min())
            print("mean:", data.mean(), "std:", data.std())

        for r in range(runs):
            # Input dimension
            input_bias = True  # add a constant input to 1
            n_inputs = 1  # input dimension (optional, can be infered at runtime)
            n_outputs = 1  # output dimension (optional, can be infered at runtime)

            # Reservoir parameter
            units = 30  # number of recurrent units
            leak_rate = 0.3  # leaking rate (=1/time_constant_of_neurons)
            rho = 1.2  # Scaling of recurrent matrix
            input_scaling = 1.0  # Scaling of input matrix

            # Connectivity
            # Connectivity defines the probability that two neurons in the
            # reservoir are being connected
            # (the two neurons can be a neuron and itself)
            rc_connectivity = 0.2  # Connectivity of recurrent matrix W
            input_connectivity = 1.0  # Connectivity of input matrix
            fb_connectivity = 1.0  # Connectivity of feedback matrix

            # Readout parameters
            regularization_coef = 1e-8
            warmup = 10

            # Enable feedback
            feedback = False

            # ---- Generating random weight matrices with custom method ----
            # (this is also optional)
            W = rng.random((units, units)) - 0.5
            Win = rng.random((units, n_inputs + int(input_bias))) - 0.5
            Wfb = rng.random((units, n_outputs)) - 0.5

            # Delete the fraction of connections given the connectivity
            # (i.e. proba of non-zero connections in the reservoir):
            mask = rng.random((units, units))  # create a mask Uniform[0;1]
            W[mask > rc_connectivity] = 0  # set to zero some connections

            mask = rng.random((units, Win.shape[1]))
            Win[mask > input_connectivity] = 0

            mask = rng.random((units, Wfb.shape[1]))
            Wfb[mask > fb_connectivity] = 0

            # Scaling of matrices

            # Scaling of input matrix
            Win = Win * input_scaling

            # Scaling of recurrent matrix using a specific spectral radius
            # First compute the spectral radius of these weights:
            original_spectral_radius = spectral_radius(W)

            # Rescale them to reach the requested spectral radius:
            W = W * (rho / original_spectral_radius)

            # ---- Create an Echo State Network ----

            # Create a reservoir
            reservoir = Reservoir(
                units,
                W = W,
                Win = Win,
                Wfb = Wfb,
                lr=leak_rate,
                sr=rho,
                input_bias=input_bias,
                input_scaling=input_scaling,
                rc_connectivity=rc_connectivity,
                input_connectivity=input_connectivity,
                fb_connectivity=fb_connectivity,
            )

            # create a readout layer equiped with an offline learning rule
            readout = Ridge(ridge=regularization_coef)

            if feedback:
                reservoir = reservoir << readout

            esn = reservoir >> readout

            # Reset the reservoir state: we want to start training from scratch
            reservoir.reset()

            # ---- Train the ESN ----
            
            esn = esn.fit(X, y, warmup=warmup)

            # ---- Evaluate the ESN ----
            wy = esn.run(X[:-warmup], reset=True)
            x = wy[-1].reshape(1,-1)
    
            y_pred = np.zeros_like(X_test)
            for i in range(y_pred.shape[0]):    
                x = esn(x)
                y_pred[i] = x

            predictions[r,:,n,np.newaxis] = y_pred

            # Mean Squared Error
            mse_score = mse(y_test, y_pred)
            # Root Mean Squared Error
            rmse_score = rmse(y_test, y_pred)
            # Normalised RMSE (based on mean of training data)
            nmrse_mean = nrmse(y_test, y_pred, norm_value=y.mean())

            # set results
            results[r,n,0] = n+1
            results[r,n,1] = mse_score
            results[r,n,2] = rmse_score
            results[r,n,3] = nmrse_mean
            results[r,n,4] = units
            results[r,n,5] = leak_rate    
            results[r,n,6] = rho
            results[r,n,7] = rc_connectivity
    
            print("\n********************")
            print(f"Errors computed over {test_size} time steps")
            print("\nMean Squared error (MSE):\t%.4e" % mse_score)
            print("Root Mean Squared error (RMSE):\t%.4e" % rmse_score)
            print("Normalized RMSE (based on mean):\t%.4e" % nmrse_mean)
            print("********************")

            if r == 0:

                p = plot_predicted_actual(y_test, y_pred, 
                                          name = 'ESN', 
                                          filename = f"stocks{n+1:02d}.png")

                p.close()

meanresults = np.mean(results, axis=0)
stdresults = np.std(results.astype(float), axis=0)
allresults = np.zeros((N,11))
idx=np.argmin(results[:,:,1],axis=0)   

for n in range(N):
    p = plot_predicted_actual(test_data[:,n], predictions[idx[n], :, n],
                              name = 'best ESN', 
                              filename=f"best_stocks{n+1:02d}.png")
    p.close()
    allresults[n,0] = n+1
    allresults[n,1] = meanresults[n,1]
    allresults[n,2] = stdresults[n,1]
    allresults[n,3] = meanresults[n,2]
    allresults[n,4] = stdresults[n,2]
    allresults[n,5] = meanresults[n,3]
    allresults[n,6] = stdresults[n,3]
    allresults[n,7] = meanresults[n,4]
    allresults[n,8] = meanresults[n,5]
    allresults[n,9] = meanresults[n,6]
    allresults[n,10] = meanresults[n,7]
    
np.savetxt('results-stocks-esn.csv', allresults, delimiter=',', 
           fmt='%i,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.1f,%.2f,%.3f,%.2f')

best_results = results[idx, np.arange(N),:]
np.savetxt('best-stocks-esn.csv', best_results, delimiter=',', 
           fmt='%i,%.8f,%.8f,%.8f,%i,%.2f,%.3f,%.2f')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    