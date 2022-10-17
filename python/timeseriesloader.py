#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022 for the decorating project.

@author: Oliver Obst
"""

import matplotlib.pyplot as plt
import numpy as np
import datetime
import maya
import csv
import pandas as pd

class TimeSeriesLoader:

    def __init__(self):
        self.length = 0
        self.dim = 6
        self.data = np.zeros((self.length, self.dim))
        self.timestamps = np.zeros((self.length))
        self.names = {}
        header = ['Open','High','Low','Close','Adj Close','Volume']
        for i,name in enumerate(header):
            self.names[i] = name
        self.trainstart = 0
        self.trainlen = 0
        self.valstart = 0
        self.vallen =0
        self.teststart = 0
        self.testlen = 0
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
        
    # data format: Date,Open,High,Low,Close,Adj Close,Volume
    # example: 2018-01-02, 83.29, 83.339, 82.26, 82.82, 76.685104, 96995
    def randomize(self, length, mag = 100.0, 
                  start = datetime.datetime(2018,1,2), interval = 24,
                  dim = 6, 
                  header = ['Open','High','Low','Close','Adj Close','Volume']):
        self.length = length
        self.dim = dim

        if not isinstance(mag, float):
            if mag.ndim != 1 or mag.shape[0] != dim:
                error = f"mag should be float or 1d array size ({dim})"
                raise Exception(error)

        self.data = mag * np.random.rand(length, dim)

        timestamps = [maya.MayaDT.from_datetime(start)] * length
        self.timestamps = np.array(timestamps)
        for i in range(1,length):
            newday = self.timestamps[i-1].add(hours=interval)
            while newday.weekday < 1 or newday.weekday > 5:
                newday = newday.add(hours=interval)
            self.timestamps[i] = newday

        self.names = {}
        for i,name in enumerate(header):
            self.names[i] = name

        self.trainstart = 0
        self.trainlen = int(0.6 * length)
        self.valstart = self.trainlen
        self.vallen = int(0.2 * length)
        self.teststart = self.valstart + self.vallen
        self.testlen = length - self.trainlen - self.vallen
                
    def load_data(self, filename, trainlen = 200, vallen = 50, testlen = 50):
        newdata = []
        timestamps = []        
        names = {}
        tz = 'Europe/Berlin'
        with open(filename, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            first = True
            for row in csv_reader:
                if first:
                    first = False
                    for j, name in enumerate(row[1:]):
                        names[j] = name
                    continue
                
                for j, val in enumerate(row):
                    if j == 0:
                        timestamps.append(maya.when(val, tz))
                        newdata.append([])
                    else:
                        newdata[-1].append(float(val))

        self.data = np.array(newdata)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1,1)
        self.timestamps = np.array(timestamps)
        self.length, self.dim = self.data.shape
        self.names = names
        self.split(trainlen, vallen, testlen)
        
    def split(self, trainlen, vallen, testlen):
        self.trainstart = self.length - trainlen - vallen - testlen
        self.trainlen = trainlen
        self.valstart = self.trainstart + trainlen
        self.vallen = vallen
        self.teststart = self.valstart + vallen
        self.testlen = testlen
        return
    
    def training(self, time = False):
        if time:
            return self.timestamps[self.trainstart:self.trainstart+self.trainlen]
        return self.data[self.trainstart:self.trainstart+self.trainlen,:]
    
    def validation(self, time = False):
        if time:
            return self.timestamps[self.valstart:self.valstart+self.vallen]
        return self.data[self.valstart:self.valstart+self.vallen,:]
    
    def testing(self, time = False):
        if time:
            return self.timestamps[self.teststart:self.teststart+self.testlen]
        return self.data[self.teststart:self.teststart+self.testlen,:]
    
    def plot(self, index = 4):
        # create a plot, for the Adj Close val by default
        days = [str(s.date) for s in self.timestamps]
        days = pd.to_datetime(days)
        plt.plot(days, self.data[:,index])
        plt.gcf().autofmt_xdate()
        plt.show()
        
    def rmse(self, y, yhat, mode = 1, flag = 1):
        if y.ndim == 1:
            y = y.reshape(-1,1)
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1,1)
            
        result = (y-yhat)**2
        
        if flag == 1:
            result = result.sum(axis=0)
        elif flag == 2:
            result = result.mean(axis=0)
        
        if mode == 0:
            result = result.sum(axis=1)
        elif mode == 1:
            result = result.mean(axis=1)
        elif mode == 2:
            result = result.mean(axis=1) / y.var(axis=1, ddof=1)
        
        return np.sqrt(result)
            
    def normalise(self):
        # normalisation scales all columns, so that the training portion
        # of the data of each column is in the range between 0 and 1.0.
        # After this operation, the full data set may still contain values 
        # outside the range, because we only use the training data to estimate
        # the limits.
        minima = np.nanmin(self.training(), 0)
        maxima = np.nanmax(self.training(), 0)
        ranges = maxima - minima
        self.data = (self.data - minima) / ranges
        
