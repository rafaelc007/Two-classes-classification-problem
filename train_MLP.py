# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:37:12 2018

@author: rafael

Solving the two classes classification problem using a MLP
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## import generated data
dados = np.load('generated_data.npz')

## extract training data
train_data = dados['train_data']
train_labels = dados['train_labels']

## extract validation data
test_data = dados['test_data']
test_labels = dados['test_labels']

## free some memory
del dados

## data sizes
num_examples = train_data.shape[0]
num_test = test_data.shape[0]

## training parameters
learning_rate = 0.003   # gradient learning rate
training_epochs = 10000 # maximum number of epochs
batch_size = 10     # batch size
display_step = 100  # printing period

## L2 Regularization rate
## This is a good beta value to start with
beta = 0.005 #0.01

## network parameters
n_hidden_1 = 30 # number of neurons in the first layer
n_hidden_2 = int(n_hidden_1/2) # number of neurons in the second layer
n_input = train_data.shape[1] # number of inputs
n_classes = 1 # number of classes in the output

## input and output tensors
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

## MLP model
def multilayer_perceptron(x, weights, biases):
    ## hidden 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.tanh(layer_1)
    ## hidden 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.tanh(layer_2)
    ## output layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.tanh(out_layer)
    return out_layer

## weights and bias definitions
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

## Training model construction
pred = multilayer_perceptron(x, weights, biases)

## cost function
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = pred))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

## regularization function and loss
regularizer = tf.nn.l2_loss(weights['h1'])+tf.nn.l2_loss(weights['h2'])
loss = tf.reduce_mean(cost + beta * regularizer)
## optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# variable initialization
init = tf.global_variables_initializer()

avg_cost = 2

## session start
with tf.Session() as sess:
    sess.run(init)
    print('Start training')
    ## training epoch
    for epoch in range(training_epochs):
        if(avg_cost < 0.11):
            break
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        ## batch loop
        for i in range(total_batch):
            batch_x, batch_y = [train_data[i*batch_size:(i + 1)*batch_size], train_labels[i*batch_size:(i + 1)*batch_size]]
            ## run backprop. and loss function
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            ## compute average loss
            avg_cost += c / total_batch
        ## display results
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.6f}".format(avg_cost))
    print("Finish!")

    # Test model
    correct_prediction = tf.equal(tf.sign(pred), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), axis = 0)
    print("Accuracy:", accuracy.eval({x: test_data, y: test_labels}))
    ############################################################################
    # create a mesh to plot in
    x_min, x_max = [-1,1]
    y_min, y_max = [-1,1]
    h = abs((x_max / x_min)/100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h, dtype = 'float32'),
    np.arange(y_min, y_max, h, dtype = 'float32'))
    
    ## predict results
    Z = np.sign(pred.eval({x : np.c_[xx.ravel(), yy.ravel()]}))
    Z = Z.reshape(xx.shape)
    
    ## confusion matrix
    CM = confusion_matrix(test_labels, np.sign(pred.eval({x : test_data})))
    print("Confusion Matrix: ")
    print(CM)
    ## save data to file for future plots
    np.savez('plot_data.npz', CM = CM, Z = Z)

## graph plotting
fig = plt.figure()    

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
plt.scatter(train_data[:,0], train_data[:,1], c=train_labels.reshape(num_examples), cmap=plt.cm.Paired, edgecolors ='black')
#plt.scatter(test_data[:,0], test_data[:,1], c=test_labels.reshape(num_test), cmap=plt.cm.Paired, edgecolors ='red')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xx.min(), xx.max())
plt.title('MLP ')
plt.axis([-1,1,-1,1])

plt.show()