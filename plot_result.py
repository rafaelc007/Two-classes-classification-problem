# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:31:00 2018

@author: rafael

plot data generated with train_MLP.py
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## import data
dados = np.load('plot_data.npz')

## extract training data
train_data = dados['train_data']
train_labels = dados['train_labels']

## extract validation data
test_data = dados['test_data']
test_labels = dados['test_labels']

## data sizes
num_examples = train_data.shape[0]
num_test = test_data.shape[0]

## import results
res = np.load('test_data_Q04a.npz')

Z = res['Z']
CM = res['CM']

## free memory
del dados, res

print("Accuracy:")
print(1-(CM[0,1]+CM[1,0])/(CM[0,0]+CM[1,1]))
print("Confusion matrix")
print(CM)

sns.set()
sns.set_style("white")
plt.figure()

x_min, x_max = [-1,1]
y_min, y_max = [-1,1]
h = abs((x_max / x_min)/100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h, dtype = 'float32'), np.arange(y_min, y_max, h, dtype = 'float32'))

plt.contourf(xx, yy, Z, cmap=plt.cm.Set3, alpha=0.8)
    
plt.scatter(train_data[:,0], train_data[:,1], c=train_labels.reshape(num_examples), cmap=plt.cm.Set3, edgecolors ='black')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xx.min(), xx.max())
plt.title('MLP (train data)')
plt.axis([-1,1,-1,1])

plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Set3, alpha=0.8)
plt.scatter(test_data[:,0], test_data[:,1], c=test_labels.reshape(num_test), cmap=plt.cm.Set3, edgecolors ='black')

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xx.min(), xx.max())
plt.title('MLP (test data)')
plt.axis([-1,1,-1,1])

plt.show()