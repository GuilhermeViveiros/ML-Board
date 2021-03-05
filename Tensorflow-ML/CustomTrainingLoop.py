#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 23:19:06 2021

@author: guilhermeviveiros


Train a model using a custom training loop to tackle the Fashion Mnist dataset (chapter 10)

a) Display the epoch, iteration, mean training loss, and mean accuracy over each epoch(uptaded at each iteration)
as well as the validation loss and accuracy at the end of each epoch.

b) Try using a different optimizer with a different learning rate for the upper layers and the lower layers
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


def plot_info(plot=False):

    idx = random.sample(range(x_train.shape[0]),3*3)

    for i in range(0,3*3):
        plt.subplot(3,3,i+1)
        plt.title(y_train[idx[i]])
        plt.imshow(x_train[idx[i]])
        

def random_batch(X,Y,batch_size):
    idx = random.sample(range(len(X)),k=batch_size)
    return X[idx],Y[idx]


def print_loss(step,total_steps,epoch,loss,metrics):
    
    metrics = "".join([" {} : {:.4f} ".format(m.name,m.result()) for m in (loss) + (metrics or [])])
    end = "" if step < total_steps else "\n"
    print("\r{}/{} - ".format(step,total_steps) + metrics , end =  end)

    

def build_model(input_dim,output):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_dim),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax'),
    ])
    

    return model


x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

epochs = 10
batch_size = 64
steps = len(x_train) // batch_size
learning_rate = 0.001


model = build_model(x_train.shape[1:],10)

optimizer1 = tf.keras.optimizers.Adam(lr = learning_rate)
optimizer2 = tf.keras.optimizers.SGD(lr = learning_rate,momentum=0.9)

scc = tf.keras.losses.sparse_categorical_crossentropy

l = tf.keras.metrics.Mean()
mean_loss = [tf.keras.metrics.Mean("loss"),tf.keras.metrics.Mean("val_loss")]
metrics = [tf.keras.metrics.Mean("accuracy"),tf.keras.metrics.Mean(name="val_accuracy")]

for epoch in range(epochs):
    
    for step in range(steps):
        
        X_batch,y_batch = random_batch(x_train,y_train,batch_size)
        X_val_batch,y_val_batch = random_batch(x_val,y_val,batch_size)
        
        with tf.GradientTape(persistent=True) as tape:
        
            y_predicted = model(X_batch) 
            y_val_predict = model(X_val_batch)
    
            loss = scc(y_batch,y_predicted)
            
            val_loss = scc(y_batch,y_predicted)
            
            
            #gradients of the respective feature extraction and classification variables 
            grads1 = tape.gradient(loss,model.trainable_variables[:3])
            grads2 = tape.gradient(loss,model.trainable_variables[3:])
            del tape
            
            #optimization of with respect too two optimizers
            optimizer1.apply_gradients(zip(grads1,model.trainable_variables[:3]))
            optimizer2.apply_gradients(zip(grads2,model.trainable_variables[3:]))
        
        

        metrics[0](tf.equal(tf.argmax(y_predicted,axis=1),y_batch))
        metrics[1](tf.equal(tf.argmax(y_predicted,axis=1),y_batch))
        mean_loss[0](loss)
        mean_loss[1](val_loss)
        
    
        print_loss(step,steps,epoch,mean_loss,metrics)
        


