#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022 for the decorating project.

@author: Oliver Obst
"""

import pandas as pd
import numpy as np
import maya
import csv
from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers.invertible_data_transformer import InvertibleDataTransformer

class NullScaler(InvertibleDataTransformer):
    def __init__(
            self,
            name: str = "NullTransformer",
            n_jobs: int = 1,
            verbose: bool = False,
        ):
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        
    def fit(self, series: TimeSeries):
        pass
    
    def transform(self, series: TimeSeries):
        return series

    def inverse_transform(self, series: TimeSeries):
        return series    
        
    @staticmethod
    def ts_inverse_transform(series: TimeSeries) -> TimeSeries:
        return series

    @staticmethod
    def ts_transform(series: TimeSeries) -> TimeSeries:
        return series
    

class StockDataLoader:

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
        self.scaler = NullScaler()        
        self.transformer = Scaler(self.scaler)
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
        
                
    def load_data(self, filename, 
                  trainlen = 200, vallen = 50, testlen = 50, columns = [3]):
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
                    elif j-1 in columns:
                        newdata[-1].append(float(val))

        self.data = pd.DataFrame(newdata)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1,1)
        self.timestamps = np.array(timestamps)
        self.length, self.dim = self.data.shape
        self.names = names
        self.split(trainlen, vallen, testlen)
        
    def split(self, trainlen, vallen, testlen):
        self.trainstart = t0 = self.length - trainlen - vallen - testlen
        self.trainlen = trainlen
        self.valstart = t1 = self.trainstart + trainlen
        self.vallen = vallen
        self.teststart = t2 = self.valstart + vallen
        self.testlen = testlen
        t3 = t2 + testlen
        self._training = TimeSeries.from_dataframe(self.data.iloc[t0:t1,:])
        self._validation = TimeSeries.from_dataframe(self.data.iloc[t1:t2,:])
        self._testing = TimeSeries.from_dataframe(self.data.iloc[t2:t3,:])
        
    def init_transform(self, minval = 0.0, maxval = 1.0):
        self.scaler = MinMaxScaler(feature_range=(minval, maxval))
        self.transformer = Scaler(self.scaler)                
        self.transformer.fit(self._training)
        
    def transform(self, x):
        return self.transformer.transform(x)
        
    def inverse_transform(self, y):
        return self.transformer.inverse_transform(y)

    def training(self, time = False):
        if time:
            return self.timestamps[self.trainstart:self.trainstart+self.trainlen]
        return self._training
    
    def validation(self, time = False):
        if time:
            return self.timestamps[self.valstart:self.valstart+self.vallen]
        return self._validation
    
    def testing(self, time = False):
        if time:
            return self.timestamps[self.teststart:self.teststart+self.testlen]
        return self._testing
