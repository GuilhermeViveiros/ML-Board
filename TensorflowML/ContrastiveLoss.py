#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:48:49 2021

@author: guilhermeviveiros
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train = x_train/255
x_test = x_train/255

def create_data_pairs(X,y):
    
    
    pairs,labels = [],[]
    
    indices = [np.where(y_train == i)[0] for i in range(0,10)]
    
    #minumum number of samples between evey class
    minimum = np.min([len(i) for i in indices]) - 1
    
    #for ecery class
    for i in range(0,10):
        # for every sample in the sample
        for n in range(minimum):
            
            z1,z2 = indices[i][n],indices[i][n+1]
            pairs += [[X[z1],X[z2]]]
            
            random_label = (np.random.choice(range(1,10)) + i) % 10         
            z1,z2 = indices[i][n],indices[random_label][n]
            pairs += [[X[z1],X[z2]]]
            
            labels += [1.0,0.0]
            
    return np.asarray(pairs),np.asarray(labels)


def encoder():
    
    inputs = tf.keras.Input(shape=(28,28,1))

    x = tf.keras.layers.Conv2D(32,3,activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32,3,activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    
    return tf.keras.models.Model(inputs=inputs,outputs=x)

enc = encoder()

def plot_pairs(pair,label):
    plt.figure(figsize=(10,5))
    
    same = pair[0]
    plt.subplot(2,2,1)
    plt.title("Same representation = " + str(label[0]))
    plt.imshow(same[0])
    
    plt.subplot(2,2,2)
    plt.imshow(same[1])
    
    different = pair[1]
    plt.subplot(2,2,3)
    plt.title("Different representation = " + str(label[0]))
    plt.title(label[1])
    plt.imshow(different[0])
    
    plt.subplot(2,2,4)
    plt.imshow(different[1])
    
    plt.show()

pairs_train, labels_train = create_data_pairs(x_train,y_train)
plot_pairs(pairs_train[0:2],labels_train[0:2])


#siamese network that outputs a dissimilarity scalar between two images
#siamese in the sence that we have two identy encoder that outputs one embedding of one parituclar image
#with this two embeddings one can calculate the dissimilarity between then and minizime use contrastive loss
#to learn how to perform a better encoding
import tensorflow.keras.backend as K
tf.keras.backend.clear_session()

def euclidianDistance(x):
    x1,x2 = x
    #ensure that we don't get the ssquare-root of 0
    return K.sqrt(K.maximum(K.sum(K.square(x1-x2)), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def siamese_network():

    inputs_left = tf.keras.Input(shape=(28,28,))
    inputs_right = tf.keras.Input(shape=(28,28,))
    vect_output_a = encoder()(inputs_left)   
    vect_output_b = encoder()(inputs_right)   
    output = tf.keras.layers.Lambda(euclidianDistance,
                    name="output_layer",
                    output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])
    
    return tf.keras.models.Model(inputs=[inputs_left,inputs_right],outputs=output)

model = siamese_network()
model.summary()
#tf.keras.utils.plot_model(model)


def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer='adam')
history = model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train, epochs=20, batch_size=128)




