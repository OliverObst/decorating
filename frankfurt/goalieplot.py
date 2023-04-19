#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:29:39 2021

@author: oliver
"""

import matplotlib.pyplot as plt
import numpy as np
import csv

with open('d_in1.csv', newline='') as csvfile:
    r01 = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
    d01 = []
    for row in r01:
        d01.append(row[:])
    d01 = d01[0]

with open('d_in2.csv', newline='') as csvfile:
    r02 = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
    d02 = []
    for row in r02:
        d02.append(row[:])
    d02 = d02[0]
    
with open('d_out11.csv', newline='') as csvfile:
    r11 = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
    d11 = []
    for row in r11:
        d11.append(row[:])
    d11 = d11[0]

with open('d_out12.csv', newline='') as csvfile:
    r12 = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
    d12 = []
    for row in r12:
        d12.append(row[:])
    d12 = d12[0]
    
with open('d_out21.csv', newline='') as csvfile:
    r21 = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
    d21 = []
    for row in r21:
        d21.append(row[:])
    d21 = d21[0]

with open('d_out22.csv', newline='') as csvfile:
    r22 = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
    d22 = []
    for row in r22:
        d22.append(row[:])
    d22 = d22[0]

N0 = 40
N1 = 69
grid = np.zeros((N0,N1))
a0 = min(d01)
z0 = max(d01)
a1 = min(d02)
z1 = max(d02)
r0 = (z0 - a0) / N0
r1 = (z1 - a1) / N1

for m0, m1 in zip(d01, d02):
    x = min(N0-1, int(np.floor((m0 - a0) / r0)))
    y = min(N1-1, int(np.floor((m1 - a1) / r1)))
    grid[x,y] = 1 + grid[x,y]

fig, ax = plt.subplots()
ymin = -22
ymax = 22
plt.ylim((ymin, ymax))
plt.xlim((-57.5, -27))
plt.gca().set_aspect('equal')
ax.plot(d01,d02, 'k:', linewidth=0.5, dashes=(2,5), label='original trajectory')
ax.plot(d11,d12, 'b-', linewidth=0.5, label='PrNN prediction')
ax.plot(d21,d22, 'r:', linewidth=0.5, dashes=(5,5), label='reduced PrNN prediction')
plt.xlabel("x position [m]")
plt.ylabel("y position [m]")
plt.legend(loc="upper right", fontsize='x-small', framealpha=1)

ax.axvspan(-57.1, -55, 0.5-7.32/(ymax-ymin), 0.5+7.32/(ymax-ymin), alpha=0.6, color='black')
ax.axvspan(-55.01, -55, 0, 1.0, alpha=0.5, color='black')
ax.axvspan(-55, -38.5, 0.5+20.16/(ymax-ymin), 0.5+20.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-55, -38.5, 0.5-20.16/(ymax-ymin), 0.5-20.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-38.5, -38.49, 0.5-20.16/(ymax-ymin), 0.5+20.20/(ymax-ymin), alpha=0.9, color='black')

ax.axvspan(-55, -49.5, 0.5+9.16/(ymax-ymin), 0.5+9.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-55, -49.5, 0.5-9.16/(ymax-ymin), 0.5-9.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-49.5, -49.49, 0.5-9.16/(ymax-ymin), 0.5+9.17/(ymax-ymin), alpha=0.9, color='black')

#circle = plt.Circle((0, 0), 9.15, color='black', fill=False, alpha=0.9)
#ax.add_patch(circle)

# ax.axvspan(8, 14, alpha=0.5, color='red')

plt.savefig('../../Publications/RecurrentPrediction/fig/goalie.png', dpi=1200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ymin = -10
ymax = 10
plt.ylim((ymin, ymax))
plt.xlim((-57.5, -32))
plt.gca().set_aspect('equal')
plt.xlabel("x position [m]")
plt.ylabel("y position [m]")

xN = np.arange(N0)
yN = np.arange(N1)
xV = xN * r0 + r0/2 + a0
yV = yN * r1 + r1/2 + a1
x, y = np.meshgrid(xN, yN) 
gg = grid.copy()
gg[grid<3] = 0
gg[gg>0] *= 3 

ax.scatter(xV[x], yV[y], s=gg[x,y], edgecolors='none')

ax.axvspan(-57.1, -55, 0.5-7.32/(ymax-ymin), 0.5+7.32/(ymax-ymin), alpha=0.6, color='black')
ax.axvspan(-55.01, -55, 0, 1.0, alpha=0.5, color='black')
ax.axvspan(-55, -38.5, 0.5+20.16/(ymax-ymin), 0.5+20.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-55, -38.5, 0.5-20.16/(ymax-ymin), 0.5-20.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-38.5, -38.49, 0.5-20.16/(ymax-ymin), 0.5+20.20/(ymax-ymin), alpha=0.9, color='black')

ax.axvspan(-55, -49.5, 0.5+9.16/(ymax-ymin), 0.5+9.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-55, -49.5, 0.5-9.16/(ymax-ymin), 0.5-9.17/(ymax-ymin), alpha=0.9, color='black')
ax.axvspan(-49.5, -49.49, 0.5-9.16/(ymax-ymin), 0.5+9.17/(ymax-ymin), alpha=0.9, color='black')


plt.savefig('../../Publications/RecurrentPrediction/fig/goalievisits.png', dpi=1200, bbox_inches='tight')
plt.show()

