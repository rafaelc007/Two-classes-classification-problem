# -*- coding: utf-8 -*-
"""
Created on Thu May 10 08:19:35 2018

@author: rafael

Generating two classes non-linear separable data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## size of the data
n = 1000
n_valid = int(n/2)

## train data
train_data = np.random.rand(n,2)*2-1
test_data = np.random.rand(n_valid,2)*2-1

## test data
train_labels = np.ones(n).reshape((n, 1))
test_labels = np.ones(n_valid).reshape((n_valid, 1))

## making the labeling
for i in range(n):
    if (train_data[i,0]-1)**2 + train_data[i,1]**2 < 1 or (train_data[i,0]+1)**2 + train_data[i,1]**2 < 1 :
        if train_data[i,0]**2 + (train_data[i,1]-1)**2 < 1 or train_data[i,0]**2 + (train_data[i,1]+1)**2 < 1:
            train_labels[i] = -1
            
for i in range(n_valid):
    if (test_data[i,0]-1)**2 + test_data[i,1]**2 < 1 or (test_data[i,0]+1)**2 + test_data[i,1]**2 < 1 :
        if test_data[i,0]**2 + (test_data[i,1]-1)**2 < 1 or test_data[i,0]**2 + (test_data[i,1]+1)**2 < 1:
            test_labels[i] = -1
            
## graph plot
sns.set()
sns.set_style("white")

fig = plt.figure()
ax1 = plt.subplot(111)

ax1.scatter(train_data[:,0], train_data[:,1], c=train_labels.reshape(n), cmap=plt.cm.Set3, edgecolors ='black')
ax1.set_title("training data")
ax1.axis([-1,1,-1,1])

fig2 = plt.figure()
ax2 = plt.subplot(111)

ax2.scatter(test_data[:,0], test_data[:,1], c=test_labels.reshape(n_valid), cmap=plt.cm.Set3, edgecolors ='black')
ax2.set_title("test data")
ax2.axis([-1,1,-1,1])

np.savez('generated_data.npz', train_data = train_data, train_labels = train_labels, test_data = test_data, test_labels = test_labels)

plt.show()