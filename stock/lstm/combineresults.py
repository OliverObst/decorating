#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:28:35 2022

@author: 30045063
"""

import json
import numpy as np

results = np.zeros((39,4))
with open('./RESULTS/files.csv', 'r') as g:
    for i, filename in enumerate(g):
        filename = './RESULTS/' + filename.rstrip('\n')
        with open(filename, 'r') as f:
            # {"RMSE": 12.55299415061254, "steps": 7, "units": 50, "RMSE.val": 6.806260676293108}
            entry = json.load(f)
            
            results[i, 0] = entry['RMSE']
            results[i, 1] = entry['steps']
            results[i, 2] = entry['units']
            results[i, 3] = entry['RMSE.val']
            
np.savetxt('results.csv', results, delimiter=',')
