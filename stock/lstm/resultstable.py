#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:46:39 2023

@author: oliver
"""
import os
import json
import csv

# Change this to the path of your subdirectory
subdir = './NEWRESULTS/'

# Find all [NAME]params.txt files in the subdirectory
params_files = [f for f in os.listdir(subdir) if f.endswith('params.txt')]

data_list = []

# Loop over each [NAME]params.txt file
for params_file in params_files:
    # Extract the NAME from the file name
    name = params_file[:-10]

    # Read the JSON content from the file
    with open(os.path.join(subdir, params_file), 'r') as f:
        data = json.load(f)

    # Extract the required values
    rmse = data['RMSE']
    input_chunk_length = data['input_chunk_length']
    units = data['units']

    # Append the values to the data_list
    data_list.append([name, rmse, input_chunk_length, units])

# Sort the data_list by NAME
data_list.sort(key=lambda x: x[0])

# Open a new CSV file for writing the output
with open(subdir+'results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(['NAME', 'RMSE', 'input_chunk_length', 'units'])

    # Write the sorted data to the CSV file
    for row in data_list:
        csv_writer.writerow(row)
