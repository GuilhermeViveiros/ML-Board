#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:23:54 2021

@author: guilhermeviveiros
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import numpy as np
import IPython.display as display

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


def plot_images(img1,img2):
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.show()

def read_images(content_path=content_path,style_path=style_path):
    img = tf.io.read_file(content_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)#/255
    
    style = tf.io.read_file(style_path)
    style = tf.image.decode_image(style, channels=3)
    style = tf.image.convert_image_dtype(style,tf.float32)#/255
    
    return img,style
    
    
# I have an image and a style image, main goal its to optimize an image that contains both the statistics of
#content and style
content,style = read_images()

#plot the content and style that we want to achieve together
plot_images(content,style)


#preprocessing step for next operations
content = content[np.newaxis]
style = style[np.newaxis]

#I will use the intermediate layers of VGG19 to caputure style representation as well as content
#Load a VGG19 and test run it on our image to ensure it's used correctly


#next choose some layers to represent the content and the style, I already now the names cause I have inspected
#the vgg architecture

#1 layer for the content, typically a higher one since this is a classification network
#higher levels have high-feature representations
content_layer = [
    'block5_conv2'
]
num_content_layers = len(content_layer)

#intermidiate layers to represent the style layers
style_layers = [
    'block2_conv1',
    'block3_conv1',
    'block3_conv3',
    'block4_conv1'
]
num_style_layers = len(style_layers)

#given a set of layers, output the according vgg matched layers
def vgg_layers(layer_names):
    
    vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(layer).output for layer in layer_names]
    
    model = tf.keras.models.Model(vgg.inputs, outputs)
    
    return model


'''
gram matrix for the style feature maps
Turns out, the style of an image can be described by the means and 
correlations across the different feature maps.
'''
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


#given an input it outputs both the style layers and content layer
class StyleContentModel(tf.keras.models.Model):
    def __init__(self,style_layers,content_layers,**kargs):
        super().__init__(**kargs)
        
        self.vgg = vgg_layers(style_layers+content_layers)      
        self.style_layers = style_layers
        self.content_layers = content_layers
        
        
    def call(self,inputs):
        inputs = tf.image.resize(inputs, (224, 224))
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs *  255)
        
        outputs = self.vgg(inputs)        
        
        style_layer_out = outputs[:len(self.style_layers)]
        content_layer_out = outputs[len(self.style_layers):]
        
        style_content = {name:out for name,out in zip(self.style_layers,style_layer_out)}
        dict_content = {name:out for name,out in zip(self.content_layers,content_layer_out)}
        
        return {'content':dict_content, 'style':style_content}
        


#fixed
#This targets are fixed, since we dont use VGG to train.
#We only want to build an image that closely matches the content and style through a VGG non trainable network
#that outputs content and style representations given an input
#we will optimie this input so that the content and style closely align with content_targets and style_targets
extractor = StyleContentModel(style_layers,content_layer)
content_targets = extractor(content)['content']
style_targets = extractor(style)['style']

style_weight=1e-2 #we have a lower error to style since we want to preverse more about the context
content_weight=1e4


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

#image to optimze, the gradients will be computed based on this input only
#Since the loss is affected only by the image and not the model itself define this as a Variable
#We initialize the variable with the pixels of the content to a faster convergence
image = tf.Variable(content)

#Since this is a float image, define a function to keep the pixel values between 0 and 1:
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.keras.optimizers.Adam(learning_rate=0.03)

@tf.function
def train_step(image):
    
    with tf.GradientTape() as tape:
        results = extractor(image)
        loss = style_content_loss(results)
    
    gradients = tape.gradient(loss,image)    
    opt.apply_gradients([(gradients,image)])
    #To change a Variable in tensorflow use assign
    image.assign(clip_0_1(image))
    
#if one wants to test
'''
train_step(image)
train_step(image)
train_step(image)
plt.imshow(image.numpy()[0])
'''

##MAIN CODE

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

import time
start = time.time()

epochs = 10
steps_per_epoch = 50

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))


