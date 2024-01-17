#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:38:55 2024

@author: oliver
"""

"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
from https://mantas.info/code/simple_esn/
(c) 2012-2020 Mantas LukoÅ¡eviÄius
Distributed under MIT license https://opensource.org/licenses/MIT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
# numpy.linalg is also an option for even fewer dependencies

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.metrics.metrics import mse, rmse, mape
import time

import os

N = 21
M = 1
results = np.zeros((M,N))

total_time = 0

for m in range(0,M):
    for n in range(0,N):
        filename = f"../data/signal{n:02d}.csv"
        print("Using " + filename)
        data = np.genfromtxt(filename, delimiter=',')
        data = data.reshape((len(data), 1))

        start_time = time.time()

        series = TimeSeries.from_values(data)
        series, _ = series.split_before(300)

        train, test = series.split_before(250)
#    train, val = train_val.split_before(200)
    
    # Normalise based on the training data
        transformer = Scaler()
        train_transformed = transformer.fit_transform(train)
        test_transformed = transformer.transform(test)
        series_transformed = transformer.transform(series)
    
        # load the data
        trainLen = 250
        testLen = 50
        initLen = 10

        if False:
            # plot some of it
            plt.figure(10).clear()
            plt.plot(train_transformed.values())
            plt.title('Transformed training data')

        # generate the ESN reservoir
        inSize = outSize = 1
        resSize = 1000
        a = 0.3 # leaking rate
        np.random.seed(42+m)
        Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1
        W = np.random.rand(resSize,resSize) - 0.5 
        # normalizing and setting spectral radius (correct, slow):
        print('Computing spectral radius...')
        rhoW = max(abs(linalg.eig(W)[0]))
        print('done.')
        W *= 1.25 / rhoW

        # allocated memory for the design (collected states) matrix
        X = np.zeros((1+inSize+resSize,trainLen-initLen))
        # set the corresponding target matrix directly
        Yt = train_transformed.values()[initLen:trainLen+1,:] 

        # run the reservoir with the data and collect X
        x = np.zeros((resSize,1))
        for t in range(trainLen):
            u = train_transformed.values()[t]
            x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
            if t >= initLen:
                X[:,t-initLen] = np.vstack((1,u,x))[:,0]
    
        # train the output by ridge regression
        reg = 1e-8  # regularization coefficient
        # direct equations from texts:
        #X_T = X.T
        #Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
            #    reg*np.eye(1+inSize+resSize) ) )
            # using scipy.linalg.solve:
        Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(1+inSize+resSize), np.dot(X,Yt) ).T

        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        Y = np.zeros((outSize,testLen))
        u = train_transformed.values()[trainLen-1]
        for t in range(testLen):
            x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
            y = np.dot( Wout, np.vstack((1,u,x)) )
            Y[:,t] = y
            # generative mode:
            u = y
            ## this would be a predictive mode:
            #u = data[trainLen+t+1] 

        # compute MSE for the first errorLen time steps
        errorLen = 50
        mymse = sum( np.square( test_transformed.values()[0:errorLen,0] - Y[0,0:errorLen] ) ) / errorLen
        rmse = np.sqrt(mymse)
        print('RMSE = ' + str( rmse ))
    
        outputs = np.zeros((testLen,2))
        outputs[:,0] = test_transformed.values(copy=True)[:,0]
        outputs[:,1] = Y[:,0]

        end_time = time.time()
        total_time = total_time + (end_time - start_time)
        
        parentdir = "RESULTS-20240116/"
        dir_str = str(m).zfill(2)
        
        os.makedirs(parentdir + dir_str, exist_ok = True)
        
        fname = parentdir + dir_str + f"/outputs-{n:02d}.csv"
        np.savetxt(fname, outputs, fmt='%.8f', delimiter=',', header='true,predicted')
        
        results[m,n] = rmse
 
        if False:
            # plot some signals
            plt.figure(1).clear()
            plt.plot( test_transformed.values()[0:50,0], 'g' )
            plt.plot( Y.T, 'b' )
            plt.title('Target and generated signals $y(n)$ starting at $n=0$')
            plt.legend(['Target signal', 'Free-running predicted signal'])
        
            plt.show()

print(f"Average execution time: {total_time / (M*N)} seconds")            
np.savetxt(parentdir+'results-mso8-esn-all.csv', results, delimiter=',', fmt='%.8f')
np.savetxt(parentdir+'results-mso8-esn-avg.csv', np.mean(results, axis=0), delimiter=',', fmt='%.8f')
    
