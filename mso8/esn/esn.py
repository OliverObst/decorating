#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:26:16 2023

@author: oliver
"""

from reservoirpy.nodes import Reservoir, Ridge
import numpy as np
import matplotlib.pyplot as plt

filename = "../data/signal00.csv"
print("Using " + filename)
data = np.genfromtxt(filename, delimiter=',')
data = data.reshape((len(data), 1))

X_train = data[0:249,:]
Y_train = data[1:250,:]
X_test = data[250:300,:]
Y_pred = np.zeros((50,1))
Y_pred[0,0] = data[300,0]

reservoir = Reservoir(100, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=10)


for i in range(49):
    Y_pred[i+1] = esn_model.run(X_test[i,0])

plt.figure(figsize=(10, 3))
plt.title("MSO8 prediction")
plt.plot(Y_pred, label="Predicted", color="blue")
plt.plot(X_test, label="Real", color="red")
plt.legend()
plt.show()

results = np.hstack((X_test, Y_pred))
np.savetxt("esnresults00.csv", results, delimiter=',', header="# True, Predicted", comments="")
